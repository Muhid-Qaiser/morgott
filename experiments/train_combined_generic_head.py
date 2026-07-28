"""Train update-matched frozen-encoder generic instruction-subversion heads.

The two conditions differ only in their second matched training half:

* control: M1 + M2
* combined: M1 + leakage-filtered PromptShield

No subtype target is constructed or consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from strict_normalize import strict_normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SELECTION = REPO_ROOT / "artifacts/combined_generic/selection_s42"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/combined_generic/runs"
MODEL_REVISIONS = {
    "jhu-clsp/mmBERT-base": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
    "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
}
TARGET = "instruction_subversion"
VALIDATION_FEATURE_RECORD_CHUNK = 256
VALIDATION_PREDICTION_BATCH_SIZE = 512


def resolve_model_revision(model_id: str, requested_revision: str | None) -> str:
    """Resolve only immutable revisions accepted by downstream evaluation."""
    pinned = MODEL_REVISIONS.get(model_id)
    if pinned is not None:
        if requested_revision not in {None, pinned}:
            raise ValueError(f"{model_id} must use pinned revision {pinned}")
        return pinned
    if (
        not model_id
        or requested_revision is None
        or re.fullmatch(r"[0-9a-f]{40}", requested_revision) is None
    ):
        raise ValueError("unknown models require a lowercase 40-hex commit revision")
    return requested_revision


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source_hashes(
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"source changed during run: {name}: {path}")


def load_records(path: Path, expected_sha256: str) -> list[dict]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"input hash mismatch: {path}")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    for record in records:
        if record.get("generic_target") != TARGET:
            raise ValueError(f"unexpected target in {record.get('id')}")
        label = record.get("generic_label")
        if type(label) is not int or label not in {0, 1}:
            raise ValueError(f"invalid generic label in {record.get('id')}")
        forbidden = {
            "direct_instruction_subversion",
            "indirect_instruction_subversion",
            "jailbreak",
            "routing_label",
        }
        if forbidden & record.keys():
            raise ValueError(f"subtype or routing target found in {record.get('id')}")
    return records


def _label_counts(records: list[dict]) -> Counter:
    return Counter(record["generic_label"] for record in records)


def validate_populations(
    m1: list[dict],
    m2: list[dict],
    promptshield: list[dict],
    validation_morgott: list[dict],
    validation_promptshield: list[dict],
) -> None:
    train_counts = [_label_counts(rows) for rows in (m1, m2, promptshield)]
    if len({tuple(sorted(counts.items())) for counts in train_counts}) != 1:
        raise ValueError(f"training label counts differ: {train_counts}")
    if len({len(rows) for rows in (m1, m2, promptshield)}) != 1:
        raise ValueError("training half row counts differ")
    validation_counts = [
        _label_counts(rows) for rows in (validation_morgott, validation_promptshield)
    ]
    if any(set(counts) != {0, 1} for counts in validation_counts):
        raise ValueError(f"validation domain lost a label: {validation_counts}")
    if any(record["dataset"] != "morgott" for record in [*m1, *m2]):
        raise ValueError("canonical halves contain a non-morgott row")
    for record in promptshield:
        if (
            record["dataset"] != "promptshield"
            or record.get("channel") is not None
            or record.get("subtype_training_eligible") is not False
        ):
            raise ValueError(
                f"PromptShield provenance or subtype contract failed: {record['id']}"
            )
    hashes = {
        "m1": {record["strict_text_sha256"] for record in m1},
        "m2": {record["strict_text_sha256"] for record in m2},
        "promptshield": {record["strict_text_sha256"] for record in promptshield},
    }
    if hashes["m1"] & hashes["m2"]:
        raise ValueError("M1 and M2 strict text hashes overlap")
    if (hashes["m1"] | hashes["m2"]) & hashes["promptshield"]:
        raise ValueError("canonical and PromptShield strict text hashes overlap")


def new_head(hidden_size: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.LayerNorm(hidden_size * 3),
        nn.Linear(hidden_size * 3, 384),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(384, 1),
    )


def _pool(hidden, attention_mask):
    import torch

    expanded = attention_mask.bool().unsqueeze(-1)
    cls = hidden[:, 0] * expanded[:, 0]
    mean = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1)
    maximum = hidden.masked_fill(
        ~expanded,
        torch.finfo(hidden.dtype).min,
    ).amax(dim=1)
    return torch.cat((cls, mean, maximum), dim=-1)


def _feature_chunk(
    encoder,
    tokenizer,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
):
    import torch

    token_ids = tokenizer(
        [strict_normalize(record["text"]) for record in records],
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
    )["input_ids"]
    features = torch.empty(
        (len(records), encoder.config.hidden_size * 3),
        dtype=torch.bfloat16,
    )
    order = sorted(range(len(records)), key=lambda index: len(token_ids[index]))
    batches = []
    batch = []
    maximum = 0
    for index in order:
        candidate_maximum = max(maximum, len(token_ids[index]))
        if batch and candidate_maximum * (len(batch) + 1) > token_budget:
            batches.append(batch)
            batch = []
            maximum = 0
        batch.append(index)
        maximum = max(maximum, len(token_ids[index]))
    if batch:
        batches.append(batch)

    with torch.no_grad():
        for indices in batches:
            width = max(len(token_ids[index]) for index in indices)
            inputs = torch.full(
                (len(indices), width),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device="cuda",
            )
            mask = torch.zeros_like(inputs)
            for slot, index in enumerate(indices):
                values = token_ids[index]
                inputs[slot, : len(values)] = torch.tensor(
                    values,
                    dtype=torch.long,
                    device="cuda",
                )
                mask[slot, : len(values)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encoder(
                    input_ids=inputs,
                    attention_mask=mask,
                ).last_hidden_state
                pooled = _pool(hidden, mask)
            features[indices] = pooled.to(device="cpu", dtype=features.dtype)
    return features


def extract_features(
    encoder,
    tokenizer,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
    record_chunk: int = 256,
):
    import torch

    chunks = []
    for start in range(0, len(records), record_chunk):
        chunks.append(
            _feature_chunk(
                encoder,
                tokenizer,
                records[start : start + record_chunk],
                max_tokens=max_tokens,
                token_budget=token_budget,
            )
        )
    if not chunks:
        return torch.empty(
            (0, encoder.config.hidden_size * 3),
            dtype=torch.bfloat16,
        )
    return torch.cat(chunks)


def predict_logits(head, features, *, batch_size: int = 512) -> np.ndarray:
    import torch

    head.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(features[start : start + batch_size].to("cuda"))
            values.append(logits[:, 0].float().cpu().numpy())
    return np.concatenate(values)


def _scores(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    scores = np.empty_like(logits)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    scores[~positive] = exponent / (1.0 + exponent)
    return scores


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    return {
        "rows": len(labels),
        "negative": int((labels == 0).sum()),
        "positive": int((labels == 1).sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "bce": float(
            np.mean(-labels * np.log(clipped) - (1 - labels) * np.log1p(-clipped))
        ),
    }


def _bce_from_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def train_head(
    first_features,
    first_labels: np.ndarray,
    second_features,
    second_labels: np.ndarray,
    validation_morgott_features,
    validation_morgott_labels: np.ndarray,
    validation_promptshield_features,
    validation_promptshield_labels: np.ndarray,
    *,
    hidden_size: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    import torch

    if len(first_features) != len(second_features):
        raise ValueError("training halves must have equal rows")
    half_batch = batch_size // 2
    head = new_head(hidden_size, seed).to("cuda")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator().manual_seed(seed)
    curve = []
    best = None
    updates = 0
    for epoch in range(epochs):
        first_order = torch.randperm(
            len(first_features),
            generator=generator,
        ).tolist()
        second_order = torch.randperm(
            len(second_features),
            generator=generator,
        ).tolist()
        losses = []
        head.train()
        for start in range(0, len(first_order), half_batch):
            first_indices = first_order[start : start + half_batch]
            second_indices = second_order[start : start + half_batch]
            first_batch = first_features[first_indices].to("cuda")
            second_batch = second_features[second_indices].to("cuda")
            first_targets = torch.from_numpy(first_labels[first_indices]).to(
                device="cuda",
                dtype=torch.float32,
            )
            second_targets = torch.from_numpy(second_labels[second_indices]).to(
                device="cuda",
                dtype=torch.float32,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                first_logits = head(first_batch)[:, 0]
                second_logits = head(second_batch)[:, 0]
                first_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    first_logits,
                    first_targets,
                )
                second_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    second_logits,
                    second_targets,
                )
                loss = 0.5 * (first_loss + second_loss)
            loss.backward()
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))

        morgott_logits = predict_logits(
            head,
            validation_morgott_features,
        )
        promptshield_logits = predict_logits(
            head,
            validation_promptshield_features,
        )
        morgott_bce = _bce_from_logits(
            validation_morgott_labels,
            morgott_logits,
        )
        promptshield_bce = _bce_from_logits(
            validation_promptshield_labels,
            promptshield_logits,
        )
        macro_bce = 0.5 * (morgott_bce + promptshield_bce)
        curve.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(losses)),
                "validation_morgott_bce": morgott_bce,
                "validation_promptshield_bce": promptshield_bce,
                "validation_macro_bce": macro_bce,
            }
        )
        key = (macro_bce, epoch + 1)
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "epoch": epoch + 1,
                "state": {
                    name: value.detach().contiguous().cpu().clone()
                    for name, value in head.state_dict().items()
                },
            }
    expected_updates = math.ceil(len(first_features) / half_batch) * epochs
    if updates != expected_updates:
        raise ValueError(f"expected {expected_updates} updates, found {updates}")
    head.load_state_dict(best["state"])
    return head, {
        "epochs": epochs,
        "batch_size": batch_size,
        "half_batch_size": half_batch,
        "learning_rate": learning_rate,
        "updates": updates,
        "selected_epoch": best["epoch"],
        "checkpoint_selection": (
            "minimum equal-domain mean of matched Morgott and PromptShield "
            "validation BCE"
        ),
        "curve": curve,
    }


def _save_head(head, path: Path) -> str:
    from safetensors.torch import save_file

    save_file(
        {
            name: value.detach().contiguous().cpu()
            for name, value in head.state_dict().items()
        },
        path,
    )
    return file_sha256(path)


def _artifact_path(selection_dir: Path, spec: dict) -> Path:
    path = (REPO_ROOT / spec["path"]).resolve()
    if not path.is_relative_to(selection_dir):
        raise ValueError(f"selection artifact escapes selection directory: {path}")
    return path


def _load_validation_population(
    report: dict,
    selection_dir: Path,
    name: str,
) -> list[dict]:
    try:
        spec = report["outputs"][name]
        records = load_records(
            _artifact_path(selection_dir, spec),
            spec["sha256"],
        )
        labels = {
            str(label): count for label, count in sorted(_label_counts(records).items())
        }
        if len(records) != spec["rows"] or labels != spec["labels"]:
            raise ValueError("row or label counts differ")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"validation population {name} failed verification") from error
    return records


def _negative_component_evidence(records: list[dict]) -> dict:
    negatives = [record for record in records if record["generic_label"] == 0]
    components_by_channel = defaultdict(set)
    components_by_source = defaultdict(set)
    for record in negatives:
        component = record["validation_component_id"]
        components_by_channel[str(record["channel"])].add(component)
        components_by_source[str(record["source"])].add(component)
    return {
        "rows_by_channel": dict(
            sorted(Counter(str(record["channel"]) for record in negatives).items())
        ),
        "components_by_channel": {
            key: len(values) for key, values in sorted(components_by_channel.items())
        },
        "rows_by_source": dict(
            sorted(Counter(str(record["source"]) for record in negatives).items())
        ),
        "components_by_source": {
            key: len(values) for key, values in sorted(components_by_source.items())
        },
    }


def _validate_update_validation_partition(
    report: dict,
    selection_dir: Path,
) -> None:
    checkpoint = _load_validation_population(
        report,
        selection_dir,
        "validation_morgott_selection",
    )
    calibration = _load_validation_population(
        report,
        selection_dir,
        "validation_morgott_calibration",
    )
    _load_validation_population(
        report,
        selection_dir,
        "validation_promptshield",
    )
    partition = report.get("validation_partition", {})
    disjointness = partition.get("disjointness")
    expected_disjointness = {
        "row",
        "normalized",
        "strict",
        "lineage_group",
        "near",
        "validation_component",
    }
    total = len(checkpoint) + len(calibration)
    actual_fraction = len(checkpoint) / total if total else None
    if (
        partition.get("target_checkpoint_fraction") != 0.2
        or partition.get("actual_checkpoint_fraction") != actual_fraction
        or partition.get("total_rows") != total
        or partition.get("checkpoint_selection_rows") != len(checkpoint)
        or partition.get("calibration_rows") != len(calibration)
        or type(disjointness) is not dict
        or set(disjointness) != expected_disjointness
        or not all(disjointness.values())
        or partition.get("checkpoint_selection")
        != [
            "morgott_validation_checkpoint_selection",
            "promptshield_validation",
        ]
        or partition.get("threshold_calibration")
        != "morgott_validation_calibration_only"
        or partition.get("promptshield_used_for_threshold") is not False
    ):
        raise ValueError("selection validation partition contract failed")

    component_sets = {}
    for role, records in (
        ("checkpoint_selection", checkpoint),
        ("calibration", calibration),
    ):
        components = {record.get("validation_component_id") for record in records}
        if any(type(value) is not str for value in components) or any(
            re.fullmatch(r"validation-component:[0-9a-f]{64}", value) is None
            for value in components
        ):
            raise ValueError("selection validation component identifiers failed")
        component_sets[role] = components
    if component_sets["checkpoint_selection"] & component_sets["calibration"]:
        raise ValueError("selection validation components overlap")

    calibration_contract = partition.get("component_calibration", {})
    fixed_contract = {
        "component_id_field": "validation_component_id",
        "component_id_definition": (
            "SHA-256 over sorted row id and strict-text SHA-256 pairs"
        ),
        "target_unit": ("lineage-and-near validation component within trusted channel"),
        "score_aggregation": (
            "maximum negative score per component within trusted channel"
        ),
        "family_confidence": 0.95,
        "per_channel_confidence": 0.975,
        "multiplicity_correction": "Bonferroni",
        "family_scope": (
            "the two trusted channels, with a separate family for each target"
        ),
        "trusted_channels": ["direct_user", "untrusted_content"],
        "pooled_negative_role": "empirical diagnostic only",
        "inference_caveat": (
            "Components and recurring source families are not IID or sampled "
            "from a deployment distribution; confidence bounds are development "
            "evidence, not production guarantees."
        ),
    }
    components_by_role = {
        role: len(components) for role, components in component_sets.items()
    }
    evidence_by_role = {
        role: _negative_component_evidence(records)
        for role, records in (
            ("checkpoint_selection", checkpoint),
            ("calibration", calibration),
        )
    }
    if (
        partition.get("component_basis")
        != ["source+group_id", "conservative_near_overlap"]
        or partition.get("components") != sum(components_by_role.values())
        or set(calibration_contract)
        != {
            *fixed_contract,
            "components_by_role",
            "negative_evidence_by_role",
        }
        or any(
            calibration_contract.get(key) != value
            for key, value in fixed_contract.items()
        )
        or calibration_contract.get("components_by_role") != components_by_role
        or calibration_contract.get("negative_evidence_by_role") != evidence_by_role
    ):
        raise ValueError("selection validation component calibration contract failed")


def _validate_full_selection_report(report: dict) -> None:
    try:
        base_spec = report["inputs"]["base_update_matched_selection"]
        base_path = (REPO_ROOT / base_spec["path"]).resolve()
        if (
            base_path.name != "selection_report.json"
            or not base_path.is_relative_to((REPO_ROOT / "artifacts").resolve())
            or file_sha256(base_path) != base_spec["sha256"]
        ):
            raise ValueError("base report path or hash mismatch")
        base_report = json.loads(base_path.read_text())
        validate_selection_report(
            base_report,
            selection_dir=base_path.parent,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError("full selection base report failed verification") from error

    validation = report.get("validation", {})
    expected_specs = {
        "morgott": base_report["outputs"]["validation_morgott_selection"],
        "morgott_calibration": base_report["outputs"]["validation_morgott_calibration"],
        "promptshield": base_report["outputs"]["validation_promptshield"],
    }
    if (
        report.get("training_recipe", {}).get("pair_atoms_preserved") is not True
        or report.get("training_recipe", {}).get("pair_ranking_capable") is not True
        or validation.get("selection_report") != base_spec["path"]
        or any(validation.get(name) != spec for name, spec in expected_specs.items())
        or validation.get("checkpoint_selection_only") != ["morgott", "promptshield"]
        or validation.get("threshold_calibration_only") != "morgott_calibration"
        or validation.get("promptshield_used_for_threshold") is not False
        or validation.get("component_calibration")
        != base_report["validation_partition"]["component_calibration"]
    ):
        raise ValueError("full selection validation contract failed")


def validate_selection_report(
    report: dict,
    *,
    selection_dir: Path,
    full: bool = False,
) -> None:
    """Fail before model allocation when a preparation report is stale."""
    if report.get("schema_version") != 2:
        raise ValueError("selection report must use schema version 2")
    purpose = (
        "artifact-only full generic instruction-subversion training recipe"
        if full
        else "artifact-only update-matched generic instruction-subversion experiment"
    )
    eligibility = (
        {
            "canonical_pool": "all retained rows after eligibility and leakage filters",
            "routing_training_eligible": True,
            "input_channel": ["direct_user", "untrusted_content"],
            "label_field": "injection_label",
            "labels": [0, 1],
            "routing_label_used": False,
            "promptshield_subtypes_assigned": False,
            "matched_pairs_are_weak_supervision": True,
        }
        if full
        else {
            "routing_training_eligible": True,
            "input_channel": ["direct_user", "untrusted_content"],
            "label_field": "injection_label",
            "labels": [0, 1],
            "exclude_security_label": "uncertain",
            "exclude_if_all_origins_are_weak_or_unverified": True,
            "routing_label_used": False,
        }
    )
    if (
        report.get("purpose") != purpose
        or report.get("generic_target") != TARGET
        or report.get("eligibility") != eligibility
    ):
        raise ValueError("selection report target contract failed")
    manifest_path = REPO_ROOT / "data/manifest.json"
    expected_manifest = {
        "path": str(manifest_path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(manifest_path),
    }
    if report.get("inputs", {}).get("manifest") != expected_manifest:
        raise ValueError("selection report manifest is stale")
    provenance = {
        "runner_sha256": file_sha256(
            REPO_ROOT
            / (
                "experiments/prepare_full_combined_generic.py"
                if full
                else "experiments/prepare_combined_generic.py"
            )
        ),
        "strict_normalizer_sha256": file_sha256(
            REPO_ROOT / "experiments/strict_normalize.py"
        ),
        "overlap_module_sha256": file_sha256(REPO_ROOT / "src/morgott/overlap.py"),
        "canonical_text_helper_sha256": file_sha256(REPO_ROOT / "src/morgott/data.py"),
    }
    if full:
        provenance.update(
            {
                "base_preparation_runner_sha256": file_sha256(
                    REPO_ROOT / "experiments/prepare_combined_generic.py"
                ),
            }
        )
    if report.get("provenance") != provenance:
        raise ValueError("selection report preparation provenance is stale")
    if full:
        _validate_full_selection_report(report)
    else:
        _validate_update_validation_partition(report, selection_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-dir",
        default=str(DEFAULT_SELECTION.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--model-id", default="jhu-clsp/mmBERT-base")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--condition",
        choices=("control", "combined", "both"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    try:
        revision = resolve_model_revision(args.model_id, args.model_revision)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.epochs < 1
        or args.batch_size < 2
        or args.batch_size % 2
        or args.learning_rate <= 0
    ):
        parser.error("epochs, even batch size, and learning rate must be positive")
    if args.max_tokens < 2 or args.token_budget < args.max_tokens:
        parser.error("token budget must be at least max tokens")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    source_paths = {
        "runner_sha256": Path(__file__).resolve(),
        "strict_normalizer_sha256": (
            Path(__file__).resolve().parent / "strict_normalize.py"
        ),
    }
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    selection_dir = (REPO_ROOT / args.selection_dir).resolve()
    selection_report_path = selection_dir / "selection_report.json"
    selection_report = json.loads(selection_report_path.read_text())
    validate_selection_report(selection_report, selection_dir=selection_dir)

    specs = selection_report["outputs"]
    paths = {
        name: _artifact_path(selection_dir, specs[name])
        for name in (
            "m1",
            "m2",
            "promptshield",
            "validation_morgott_selection",
            "validation_promptshield",
        )
    }
    records = {
        name: load_records(paths[name], specs[name]["sha256"])
        for name in (
            "m1",
            "m2",
            "promptshield",
            "validation_morgott_selection",
            "validation_promptshield",
        )
    }
    for name, values in records.items():
        if len(values) != specs[name]["rows"]:
            raise ValueError(
                f"{name} row count mismatch: expected {specs[name]['rows']}, "
                f"found {len(values)}"
            )
        labels_found = {
            str(label): count for label, count in sorted(_label_counts(values).items())
        }
        if labels_found != specs[name]["labels"]:
            raise ValueError(
                f"{name} label count mismatch: expected {specs[name]['labels']}, "
                f"found {labels_found}"
            )
    validate_populations(
        records["m1"],
        records["m2"],
        records["promptshield"],
        records["validation_morgott_selection"],
        records["validation_promptshield"],
    )

    conditions = (
        ("control", "combined") if args.condition == "both" else (args.condition,)
    )
    model_tag = re.sub(r"[^a-z0-9]+", "-", args.model_id.casefold()).strip("-")
    output_root = Path(args.output_root).resolve()
    if not output_root.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        parser.error("--output-root must be inside the artifacts directory")
    outputs = {
        condition: output_root / f"{model_tag}_{condition}_s{args.seed}"
        for condition in conditions
    }
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(f"refusing to replace existing output: {output}")

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    encoder = AutoModel.from_pretrained(
        args.model_id,
        revision=revision,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to("cuda")
    encoder.eval()
    encoder.gradient_checkpointing_disable()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    features = {}
    labels = {}
    for name in (
        "m1",
        "m2",
        "promptshield",
        "validation_morgott_selection",
        "validation_promptshield",
    ):
        features[name] = extract_features(
            encoder,
            tokenizer,
            records[name],
            max_tokens=args.max_tokens,
            token_budget=args.token_budget,
            record_chunk=VALIDATION_FEATURE_RECORD_CHUNK,
        )
        labels[name] = np.asarray(
            [record["generic_label"] for record in records[name]],
            dtype=np.int64,
        )

    heads = {}
    training = {}
    for condition in conditions:
        second = "m2" if condition == "control" else "promptshield"
        heads[condition], training[condition] = train_head(
            features["m1"],
            labels["m1"],
            features[second],
            labels[second],
            features["validation_morgott_selection"],
            labels["validation_morgott_selection"],
            features["validation_promptshield"],
            labels["validation_promptshield"],
            hidden_size=encoder.config.hidden_size,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    if len({values["updates"] for values in training.values()}) != 1:
        raise ValueError("conditions do not have update-matched training")

    morgott_labels = labels["validation_morgott_selection"]
    promptshield_labels = labels["validation_promptshield"]
    morgott_logits = {
        condition: predict_logits(
            head,
            features["validation_morgott_selection"],
            batch_size=VALIDATION_PREDICTION_BATCH_SIZE,
        )
        for condition, head in heads.items()
    }
    promptshield_logits = {
        condition: predict_logits(
            head,
            features["validation_promptshield"],
            batch_size=VALIDATION_PREDICTION_BATCH_SIZE,
        )
        for condition, head in heads.items()
    }
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    elapsed = time.perf_counter() - started
    output_root.mkdir(parents=True, exist_ok=True)

    for condition, output in outputs.items():
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_root))
        try:
            head_path = temporary / "head.safetensors"
            published_head_path = output / head_path.name
            head_sha256 = _save_head(heads[condition], head_path)
            morgott_scores = _scores(morgott_logits[condition])
            promptshield_scores = _scores(promptshield_logits[condition])

            arrays = {
                "validation_morgott_selection_scores.npy": morgott_scores,
                "validation_morgott_selection_labels.npy": morgott_labels,
                "validation_promptshield_scores.npy": promptshield_scores,
                "validation_promptshield_labels.npy": promptshield_labels,
            }
            for name, values in arrays.items():
                np.save(temporary / name, values)

            probe = features["m1"][:64]
            probe_logits = predict_logits(heads[condition], probe)
            reloaded = new_head(encoder.config.hidden_size, args.seed).to("cuda")
            reloaded.load_state_dict(load_file(str(head_path)))
            reloaded_logits = predict_logits(reloaded, probe)
            roundtrip_delta = float(np.max(np.abs(probe_logits - reloaded_logits)))
            if roundtrip_delta > 1e-6:
                raise ValueError(f"head roundtrip mismatch: {roundtrip_delta}")

            result = {
                "schema_version": 1,
                "purpose": (
                    "artifact-only update-matched generic instruction-subversion "
                    "frozen-encoder experiment"
                ),
                "condition": condition,
                "generic_target": TARGET,
                "model_id": args.model_id,
                "model_revision": revision,
                "attention_implementation": "sdpa",
                "normalization": "strict",
                "max_tokens": args.max_tokens,
                "token_budget": args.token_budget,
                "validation_feature_record_chunk": VALIDATION_FEATURE_RECORD_CHUNK,
                "validation_prediction_batch_size": (VALIDATION_PREDICTION_BATCH_SIZE),
                "seed": args.seed,
                "training": {
                    **training[condition],
                    "first_half": "m1",
                    "second_half": ("m2" if condition == "control" else "promptshield"),
                    "rows_per_half": len(records["m1"]),
                    "labels_per_half": {
                        str(label): count
                        for label, count in sorted(_label_counts(records["m1"]).items())
                    },
                    "loss": (
                        "0.5 * mean_BCE(first_half) + 0.5 * mean_BCE(second_half)"
                    ),
                    "encoder_frozen": True,
                },
                "validation": {
                    "checkpoint_selection_rows": {
                        "morgott": len(records["validation_morgott_selection"]),
                        "promptshield": len(records["validation_promptshield"]),
                    },
                    "morgott_selection": _binary_metrics(
                        morgott_labels,
                        morgott_scores,
                    ),
                    "promptshield": _binary_metrics(
                        promptshield_labels,
                        promptshield_scores,
                    ),
                },
                "runtime": {
                    "shared_extraction_and_training_seconds": elapsed,
                    "peak_reserved_bytes": peak_reserved_bytes,
                },
                "artifact": {
                    "head": str(published_head_path.relative_to(REPO_ROOT)),
                    "head_sha256": head_sha256,
                    "roundtrip_probe_rows": len(probe),
                    "roundtrip_max_abs_logit_delta": roundtrip_delta,
                    "arrays": {name: file_sha256(temporary / name) for name in arrays},
                },
                "provenance": {
                    "selection_report": str(
                        selection_report_path.relative_to(REPO_ROOT)
                    ),
                    "selection_report_sha256": file_sha256(selection_report_path),
                    "selection_inputs": {
                        name: {
                            "path": specs[name]["path"],
                            "sha256": specs[name]["sha256"],
                            "rows": specs[name]["rows"],
                            "labels": specs[name]["labels"],
                        }
                        for name in specs
                    },
                    **source_hashes,
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in (
                            "numpy",
                            "safetensors",
                            "scikit-learn",
                            "torch",
                            "transformers",
                        )
                    },
                },
                "limitations": [
                    "No subtype or PromptShield input-channel label is inferred.",
                    "PromptShield validation has only 497 negatives before joint "
                    "filtering and cannot support a precise 0.1% FPR claim.",
                    "This trainer selects checkpoints on validation but does not "
                    "calibrate an operating threshold.",
                    "No held-out test is scored by this trainer.",
                    "The learned score is advisory and is not approved for blocking.",
                ],
            }
            (temporary / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            _verify_source_hashes(source_paths, source_hashes)
            if output.exists():
                raise FileExistsError(f"refusing to replace existing output: {output}")
            os.replace(temporary, output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(
            f"{condition}: epoch {training[condition]['selected_epoch']}; "
            f"validation AUC morgott "
            f"{result['validation']['morgott_selection']['roc_auc']:.4f}, "
            f"PromptShield "
            f"{result['validation']['promptshield']['roc_auc']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
