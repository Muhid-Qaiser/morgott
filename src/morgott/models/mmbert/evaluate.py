"""Evaluate a maintained mmBERT run without promoting it into authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..detector import choose_threshold
from .core import (
    ATTENTION_IMPLEMENTATION,
    INSTRUCTION_SUBVERSION_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
    load_base_model,
    new_head,
    score_texts,
    source_provenance,
)
from .data import (
    batches,
    canonical_rows,
    external_rows,
    routing_views,
)
from .train import prepare_training_data

_REAL_FINANCE_SOURCES = frozenset(
    {
        "banking77",
        "financebench",
        "harper_valley_bank",
        "tatqa",
    }
)


def _fpr_upper_bound(
    false_positive: int,
    negative: int,
    *,
    confidence: float,
) -> float:
    from scipy.stats import beta

    if negative < 1 or not 0 <= false_positive <= negative:
        raise ValueError("invalid false-positive evidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if false_positive == negative:
        return 1.0
    return float(
        beta.ppf(
            confidence,
            false_positive + 1,
            negative - false_positive,
        )
    )


def _supported_false_positive_budget(
    negative: int,
    target: float,
    *,
    confidence: float,
) -> int | None:
    if negative < 1 or not 0 < target < 1:
        raise ValueError("negative count and target FPR must be positive")
    for false_positive in range(math.floor(target * negative), -1, -1):
        if (
            _fpr_upper_bound(
                false_positive,
                negative,
                confidence=confidence,
            )
            <= target
        ):
            return false_positive
    return None


def _component_evidence(
    scores: np.ndarray,
    *,
    threshold: float | None,
    target: float,
    confidence: float,
) -> dict:
    negative = len(scores)
    if not negative:
        return {
            "status": "underpowered",
            "negative_components": 0,
            "false_positive_component_budget": None,
        }
    budget = _supported_false_positive_budget(
        negative,
        target,
        confidence=confidence,
    )
    zero_upper = _fpr_upper_bound(0, negative, confidence=confidence)
    if budget is None:
        return {
            "status": "underpowered",
            "negative_components": negative,
            "false_positive_component_budget": None,
            "zero_false_positive_component_upper_bound": zero_upper,
        }
    if threshold is None:
        return {
            "status": "powered",
            "negative_components": negative,
            "false_positive_component_budget": budget,
            "zero_false_positive_component_upper_bound": zero_upper,
        }
    false_positive = int((scores >= threshold).sum())
    upper = _fpr_upper_bound(
        false_positive,
        negative,
        confidence=confidence,
    )
    return {
        "status": "satisfies_bound" if upper <= target else "exceeds_bound",
        "negative_components": negative,
        "false_positive_component_budget": budget,
        "false_positive_components": false_positive,
        "component_false_alarm_rate": false_positive / negative,
        "upper_confidence_bound": upper,
        "zero_false_positive_component_upper_bound": zero_upper,
    }


def _select_component_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    records: list[dict],
    *,
    targets: tuple[float, ...] = (0.001, 0.01),
    confidence: float = 0.95,
) -> tuple[dict[str, float], dict]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        scores.ndim != 1
        or labels.ndim != 1
        or len(scores) != len(labels)
        or len(records) != len(labels)
        or not np.isfinite(scores).all()
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("invalid calibration rows")
    negative_mask = labels == 0
    negative_scores = scores[negative_mask]
    negative_records = [
        row for row, selected in zip(records, negative_mask, strict=True) if selected
    ]
    if not len(negative_scores):
        raise ValueError("calibration requires negatives")
    channels = ("direct_user", "untrusted_content")
    observed = {row.get("input_channel") for row in negative_records}
    if not observed <= set(channels):
        raise ValueError("calibration rows have an unsupported trusted channel")
    per_channel_confidence = 1.0 - (1.0 - confidence) / len(channels)
    component_scores = {channel: {} for channel in channels}
    for score, row in zip(negative_scores, negative_records, strict=True):
        channel = row["input_channel"]
        component = row.get("validation_component_id")
        if not isinstance(component, str) or not component:
            raise ValueError("calibration row has no validation component identity")
        previous = component_scores[channel].get(component)
        component_scores[channel][component] = (
            float(score) if previous is None else max(previous, float(score))
        )
    channel_scores = {
        channel: np.asarray(
            [values[key] for key in sorted(values)],
            dtype=np.float64,
        )
        for channel, values in component_scores.items()
    }

    thresholds = {}
    evidence = {}
    for target in targets:
        if not 0 < target < 1:
            raise ValueError("target FPR must be between zero and one")
        name = f"{target:.4%}"
        power = {
            channel: _component_evidence(
                values,
                threshold=None,
                target=target,
                confidence=per_channel_confidence,
            )
            for channel, values in channel_scores.items()
        }
        base = {
            "target_component_false_alarm_probability": target,
            "target_unit": (
                "lineage-and-near validation component within trusted channel"
            ),
            "component_score": "maximum negative row score",
            "family_confidence": confidence,
            "per_channel_confidence": per_channel_confidence,
            "multiplicity_correction": "Bonferroni",
            "negative_rows": len(negative_scores),
        }
        underpowered = [
            channel
            for channel, values in power.items()
            if values["false_positive_component_budget"] is None
        ]
        if underpowered:
            evidence[name] = {
                **base,
                "status": "unavailable",
                "underpowered_channels": underpowered,
                "by_channel": power,
            }
            continue
        candidates = {}
        for channel, values in channel_scores.items():
            budget = power[channel]["false_positive_component_budget"]
            candidates[f"channel:{channel}"] = float(
                np.nextafter(np.sort(values)[::-1][budget], np.inf)
            )
        threshold = max(candidates.values())
        thresholds[name] = threshold
        by_channel = {
            channel: _component_evidence(
                values,
                threshold=threshold,
                target=target,
                confidence=per_channel_confidence,
            )
            for channel, values in channel_scores.items()
        }
        if any(values["status"] != "satisfies_bound" for values in by_channel.values()):
            raise ValueError("component threshold violates a confidence bound")
        evidence[name] = {
            **base,
            "status": "available",
            "threshold": threshold,
            "candidate_thresholds": candidates,
            "by_channel": by_channel,
        }
    return thresholds, evidence


def _load_run(run: Path) -> tuple[dict, object, object, object]:
    from safetensors.torch import load_file

    run = run.resolve()
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mode = result.get("adaptation")
    artifact = result.get("artifact", {})
    head_name = artifact.get("head")
    if not isinstance(head_name, str):
        raise ValueError("run has no head artifact")
    head_path = (run / head_name).resolve()
    if (
        result.get("purpose") != "maintained full-data advisory mmBERT training"
        or result.get("model_id") != MODEL_ID
        or result.get("model_revision") != MODEL_REVISION
        or result.get("attention_implementation") != ATTENTION_IMPLEMENTATION
        or result.get("normalization") != "strict"
        or mode not in {"frozen", "lora"}
        or not head_path.is_relative_to(run)
        or file_sha256(head_path) != artifact.get("head_sha256")
    ):
        raise ValueError("run contract failed")

    encoder, tokenizer = load_base_model()
    if mode == "lora":
        from peft import PeftModel, get_peft_model_state_dict

        adapter_name = artifact.get("adapter")
        if not isinstance(adapter_name, str) or not isinstance(
            artifact.get("adapter_files"), dict
        ):
            raise ValueError("run has no LoRA adapter artifact")
        adapter = (run / adapter_name).resolve()
        if not adapter.is_relative_to(run) or not adapter.is_dir():
            raise ValueError("adapter path escapes the run")
        actual = {
            path.name: file_sha256(path)
            for path in sorted(adapter.iterdir())
            if path.is_file()
        }
        if actual != artifact["adapter_files"]:
            raise ValueError("adapter hash mismatch")
        encoder = PeftModel.from_pretrained(encoder, adapter, is_trainable=False)
        modules = sorted(
            name
            for name, module in encoder.named_modules()
            if hasattr(module, "lora_A")
        )
        parameters = sum(
            value.numel() for value in get_peft_model_state_dict(encoder).values()
        )
        if (
            modules != result["lora"]["targeted_modules"]
            or parameters != result["lora"]["adapter_parameters"]
        ):
            raise ValueError("LoRA identity mismatch")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    head = new_head(encoder.config.hidden_size, result["seed"]).to("cuda")
    head.load_state_dict(load_file(str(head_path)), strict=True)
    head.eval()
    return result, encoder, tokenizer, head


def _score(
    rows,
    encoder,
    tokenizer,
    head,
    *,
    batch_size: int,
) -> dict:
    labels = []
    scores = []
    sources = []
    channels = []
    pair_ids = []
    tags = []
    records = []
    for batch in batches(rows, 512):
        values = score_texts(
            encoder,
            tokenizer,
            head,
            [row["text"] for row in batch],
            batch_size=batch_size,
        )
        labels.extend(row["label"] for row in batch)
        scores.extend(values)
        sources.extend(row["source"] for row in batch)
        channels.extend(row["input_channel"] for row in batch)
        pair_ids.extend(row.get("pair_id") for row in batch)
        tags.extend(row.get("security_tags", ()) for row in batch)
        records.extend(batch)
    if not labels:
        raise ValueError("evaluation population is empty")
    return {
        "labels": np.asarray(labels, dtype=np.int8),
        "scores": np.asarray(scores, dtype=np.float64),
        "sources": np.asarray(sources),
        "channels": np.asarray(channels),
        "pair_ids": pair_ids,
        "tags": tags,
        "records": records,
    }


def _identity_sha256(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(row["id"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _real_finance_mask(scored: dict) -> np.ndarray:
    selected = (
        (scored["labels"] == 0)
        & (scored["channels"] == "direct_user")
        & np.isin(
            scored["sources"],
            list(_REAL_FINANCE_SOURCES),
        )
    )
    if int(selected.sum()) != 7_054:
        raise ValueError("real-finance negative population changed")
    return selected


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    result = {
        "rows": len(labels),
        "positive": int(positives.sum()),
        "negative": int(negatives.sum()),
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "recall": true_positive / positives.sum() if positives.any() else None,
        "fpr": false_positive / negatives.sum() if negatives.any() else None,
        "precision": (true_positive / predictions.sum() if predictions.any() else 0.0),
        "roc_auc": None,
        "pr_auc": None,
        "descriptive_same_test": None,
    }
    if positives.any() and negatives.any():
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
        result["descriptive_same_test"] = {}
        for target in (0.001, 0.01):
            name = f"{target:.4%}"
            descriptive_threshold = choose_threshold(labels, scores, target)
            descriptive_predictions = scores >= descriptive_threshold
            descriptive_true_positive = int(np.sum(descriptive_predictions & positives))
            descriptive_false_positive = int(
                np.sum(descriptive_predictions & negatives)
            )
            result["descriptive_same_test"][name] = {
                "target_fpr": target,
                "threshold": descriptive_threshold,
                "fpr": descriptive_false_positive / negatives.sum(),
                "recall": descriptive_true_positive / positives.sum(),
                "false_positive": descriptive_false_positive,
                "true_positive": descriptive_true_positive,
            }
    return result


def _by_value(scored: dict, key: str, threshold: float) -> dict:
    result = {}
    for value in sorted(set(scored[key])):
        selected = scored[key] == value
        result[str(value)] = _metrics(
            scored["labels"][selected],
            scored["scores"][selected],
            threshold,
        )
    return result


def _pair_metrics(scored: dict, threshold: float | None = None) -> dict:
    grouped = defaultdict(dict)
    for pair_id, label, score in zip(
        scored["pair_ids"],
        scored["labels"],
        scored["scores"],
        strict=True,
    ):
        if pair_id is not None:
            grouped[pair_id][int(label)] = float(score)
    complete = [pair for pair in grouped.values() if set(pair) == {0, 1}]
    result = {
        "pairs": len(complete),
        "attack_scores_higher": (
            float(np.mean([pair[1] > pair[0] for pair in complete]))
            if complete
            else None
        ),
        "mean_attack_minus_benign": (
            float(np.mean([pair[1] - pair[0] for pair in complete]))
            if complete
            else None
        ),
    }
    if threshold is not None and complete:
        benign = np.asarray([pair[0] for pair in complete])
        attack = np.asarray([pair[1] for pair in complete])
        result["applied_threshold"] = {
            "threshold": threshold,
            "benign_fpr": float((benign >= threshold).mean()),
            "attack_recall": float((attack >= threshold).mean()),
            "both_correct": float(
                ((benign < threshold) & (attack >= threshold)).mean()
            ),
        }
    return result


def _by_subtype(scored: dict, threshold: float) -> dict:
    result = {}
    for tag in INSTRUCTION_SUBVERSION_TAGS:
        selected = np.asarray([tag in tags for tags in scored["tags"]])
        if selected.any():
            result[tag] = _metrics(
                scored["labels"][selected],
                scored["scores"][selected],
                threshold,
            )
    return result


def evaluate(
    run: Path,
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    output: Path,
    batch_size: int,
) -> Path:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    result, encoder, tokenizer, head = _load_run(run)
    views = routing_views(data_dir)
    external, _ = external_rows(external_dir)
    prepared = prepare_training_data(
        data_dir,
        external_dir,
        pairs,
        seed=result["seed"],
    )
    provenance = result.get("provenance", {})
    if (
        provenance.get("data_manifest_sha256") != prepared.data_manifest_sha256
        or provenance.get("external_manifest_sha256")
        != prepared.external_manifest_sha256
        or provenance.get("pair_archive_sha256") != file_sha256(pairs)
    ):
        raise ValueError("evaluation data digest differs from the training run")
    calibration_rows = prepared.calibration
    del prepared
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    calibration = _score(
        calibration_rows,
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    thresholds, threshold_evidence = _select_component_thresholds(
        calibration["scores"],
        calibration["labels"],
        calibration["records"],
    )
    if "1.0000%" not in thresholds:
        raise ValueError("the one-percent component threshold is unavailable")
    threshold = thresholds["1.0000%"]
    dev_path, dev_spec = views["dev_test"]
    dev = _score(
        canonical_rows(dev_path, dev_spec, split="dev_test"),
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    promptshield = _score(
        external["promptshield_test"],
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    sep = _score(
        external["sep"],
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    finance = _real_finance_mask(dev)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".mmbert-eval-"))
    try:
        scored_sets = {
            "calibration": calibration,
            "dev_test": dev,
            "promptshield": promptshield,
            "sep": sep,
        }
        score_slices = {}
        arrays = []
        offset = 0
        for name, scored in scored_sets.items():
            stop = offset + len(scored["labels"])
            score_slices[name] = [offset, stop]
            arrays.append(np.column_stack((scored["labels"], scored["scores"])))
            offset = stop
        arrays_path = temporary / "scores.npy"
        np.save(arrays_path, np.concatenate(arrays), allow_pickle=False)
        report = {
            "schema_version": 1,
            "purpose": "advisory mmBERT development evaluation",
            "advisory_only": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": result["adaptation"],
            "run_result_sha256": file_sha256(run / "result.json"),
            "inputs": {
                "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
                "external_manifest_sha256": file_sha256(external_dir / "manifest.json"),
                "pair_archive_sha256": file_sha256(pairs),
                "routing_views": {
                    split: {
                        "sha256": spec["sha256"],
                        "rows": spec["rows"],
                    }
                    for split, (_, spec) in views.items()
                },
            },
            "provenance": source_provenance(
                Path(__file__),
                Path(__file__).with_name("core.py"),
                Path(__file__).with_name("data.py"),
                Path(__file__).with_name("train.py"),
                Path(__file__).resolve().parents[1] / "detector.py",
                Path(__file__).resolve().parents[2] / "data.py",
                Path(__file__).resolve().parents[2] / "normalization.py",
                Path(__file__).resolve().parents[2] / "overlap.py",
            ),
            "calibration": {
                "selection": (
                    "canonical lineage-and-near components disjoint from "
                    "checkpoint selection"
                ),
                "row_identity_sha256": _identity_sha256(calibration["records"]),
                "component_thresholds": threshold_evidence,
                "metrics": _metrics(
                    calibration["labels"],
                    calibration["scores"],
                    threshold,
                ),
            },
            "canonical_dev_test": {
                "row_identity_sha256": _identity_sha256(dev["records"]),
                "metrics": _metrics(dev["labels"], dev["scores"], threshold),
                "by_source": _by_value(dev, "sources", threshold),
                "by_channel": _by_value(dev, "channels", threshold),
                "by_instruction_subtype": _by_subtype(dev, threshold),
            },
            "promptshield_test": {
                "row_identity_sha256": _identity_sha256(promptshield["records"]),
                "metrics": _metrics(
                    promptshield["labels"],
                    promptshield["scores"],
                    threshold,
                ),
            },
            "sep": {
                "row_identity_sha256": _identity_sha256(sep["records"]),
                "metrics": _metrics(sep["labels"], sep["scores"], threshold),
                "pairs": _pair_metrics(sep, threshold),
            },
            "real_finance_negatives": {
                "sources": sorted(_REAL_FINANCE_SOURCES),
                "metrics": _metrics(
                    np.zeros(int(finance.sum()), dtype=np.int8),
                    dev["scores"][finance],
                    threshold,
                ),
                "by_source": {
                    source: _metrics(
                        np.zeros(
                            int((finance & (dev["sources"] == source)).sum()),
                            dtype=np.int8,
                        ),
                        dev["scores"][finance & (dev["sources"] == source)],
                        threshold,
                    )
                    for source in sorted(_REAL_FINANCE_SOURCES)
                },
            },
            "thresholds": {
                "source": "canonical calibration components only",
                "selected": thresholds,
            },
            "scores": {
                "path": "scores.npy",
                "sha256": file_sha256(arrays_path),
                "columns": ["label", "score"],
                "slices": score_slices,
            },
            "runtime": {
                "seconds": time.perf_counter() - started,
                "batch_size": batch_size,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "limitations": [
                "PromptShield test and SEP are already-open development evidence.",
                "This is not a prospective final test or a source-held-out retrain.",
                "The threshold is descriptive and is not approved for blocking.",
            ],
        }
        (temporary / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("artifacts/mmbert/data"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data-archive/matched_pairs_20260726.jsonl.gz"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.batch_size):
        raise ValueError("batch size must be positive")
    output = args.output or args.run / "evaluation"
    print(
        evaluate(
            args.run,
            data_dir=args.data_dir,
            external_dir=args.external_dir,
            pairs=args.pairs,
            output=output,
            batch_size=args.batch_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
