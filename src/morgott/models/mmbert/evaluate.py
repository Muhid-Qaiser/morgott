"""Evaluate a maintained mmBERT run without promoting it into authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ...normalization import strict_normalize
from ..detector import choose_threshold
from .core import (
    ATTENTION_IMPLEMENTATION,
    INSTRUCTION_SUBVERSION_TAGS,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
    pool,
    score_texts,
    source_provenance,
)
from .data import (
    batches,
    canonical_rows,
    external_rows,
    routing_views,
)
from .head_contract import new_head_for_result, resolve_head_contract
from .score_journal import (
    ScoreJournal,
    ScoreJournalSpec,
    require_disjoint_paths,
)
from .train import SUPPORTED_MAX_TOKENS, _usable_cpus, prepare_training_data

_REAL_FINANCE_SOURCES = frozenset(
    {
        "banking77",
        "financebench",
        "harper_valley_bank",
        "tatqa",
    }
)
_SNAPSHOT_NAME = re.compile(r"update-([0-9]+)\.pt")
_HARMFUL_INTENT_TAG = "harmful_intent"
_BENIGN_TAG = "benign"
EVALUATION_SCHEMA_VERSION = 2
EVALUATION_IDENTITY_SCHEMA_VERSION = 2
TRAINING_IDENTITY_SCHEMA_VERSION = 5
EVALUATION_PANEL_ORDER = (
    "calibration",
    "dev_test",
    "promptshield_test",
    "sep",
)
_EVALUATION_REPORT_PANELS = (
    ("calibration", "calibration"),
    ("dev_test", "canonical_dev_test"),
    ("promptshield_test", "promptshield_test"),
    ("sep", "sep"),
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODEL_REGISTRY = Path(__file__).resolve().parents[4] / "model-artifacts.json"


def _read_run_result(run: Path) -> dict:
    run = run.resolve()
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("run result must be an object")
    return result


def _training_max_tokens(result: dict) -> int:
    """Resolve historical implicit-512 and new explicit-cap run records."""
    max_tokens = result.get("max_tokens", MAX_TOKENS)
    if type(max_tokens) is not int or max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(f"run max_tokens must be one of {SUPPORTED_MAX_TOKENS}")
    identity = result.get("training_identity")
    if identity is not None and not isinstance(identity, dict):
        raise ValueError("run training identity must be an object")
    identity_schema = identity.get("schema_version", 0) if identity else 0
    if type(identity_schema) is not int or identity_schema < 0:
        raise ValueError("run training identity schema version must be an integer")
    if identity_schema > TRAINING_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported future training identity schema {identity_schema}"
        )
    if identity_schema == TRAINING_IDENTITY_SCHEMA_VERSION:
        identity_microbatch = identity.get("microbatch_size")
        if (
            identity.get("max_tokens") != max_tokens
            or type(identity_microbatch) is not int
            or identity.get("token_budget") != identity_microbatch * max_tokens
        ):
            raise ValueError("run training identity context contract failed")
    training = result.get("training")
    if isinstance(training, dict):
        recorded_cap = training.get("max_tokens", max_tokens)
        if recorded_cap != max_tokens:
            raise ValueError("run training context cap disagrees with the result")
        microbatch_size = training.get("microbatch_size")
        token_budget = result.get("token_budget")
        if token_budget is not None and (
            type(microbatch_size) is not int
            or type(token_budget) is not int
            or token_budget != microbatch_size * max_tokens
        ):
            raise ValueError("run token budget disagrees with its context cap")
        if (
            identity_schema == TRAINING_IDENTITY_SCHEMA_VERSION
            and training.get("token_budget") != token_budget
        ):
            raise ValueError("run training token budget disagrees with the result")
    return max_tokens


def _configure_tokenizer_execution(workers: int | None = None) -> dict:
    """Enable a quota-bounded Rust tokenizer pool before its first use.

    The evaluator is a single process and does not fork after loading the fast
    tokenizer, so the fork/deadlock reason for disabling tokenizer parallelism
    does not apply here.  Make this entry point authoritative over inherited
    shell state: one earlier run silently inherited ``false`` and scored at
    roughly one third of the normal throughput.  ``--tokenizer-workers 1`` is
    the explicit serial opt-out.

    Rayon fixes its global pool size on first use, so this must run before
    ``_load_run`` constructs and calls the tokenizer.
    """
    automatic = workers is None
    workers = _usable_cpus() if automatic else workers
    if type(workers) is not int or workers < 1:
        raise ValueError("tokenizer workers must be a positive integer")
    parallel = workers > 1
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if parallel else "false"
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    return {
        "parallelism": parallel,
        "rayon_threads": workers,
        "selection": "automatic_cgroup_budget" if automatic else "explicit",
    }


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


def _verified_base_model_identity() -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    manifest = json.loads(_MODEL_REGISTRY.read_text(encoding="utf-8"))
    base_model = manifest.get("base_model")
    fields = (
        ("config.json", "config_sha256"),
        ("pytorch_model.bin", "pytorch_model_sha256"),
        ("special_tokens_map.json", "special_tokens_map_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
        ("tokenizer.json", "tokenizer_json_sha256"),
    )
    if (
        manifest.get("schema_version") != 2
        or not isinstance(base_model, dict)
        or base_model.get("id") != MODEL_ID
        or base_model.get("revision") != MODEL_REVISION
        or any(
            not isinstance(base_model.get(field), str)
            or _SHA256.fullmatch(base_model[field]) is None
            for _, field in fields
        )
    ):
        raise ValueError("base model registry contract failed")

    identity = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        **{field: base_model[field] for _, field in fields},
    }
    for filename, field in fields:
        path = Path(
            hf_hub_download(
                MODEL_ID,
                filename,
                revision=MODEL_REVISION,
            )
        )
        if file_sha256(path) != identity[field]:
            raise ValueError(f"base model hash mismatch: {filename}")
    return identity


def _load_pytorch_base_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("mmBERT requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    encoder = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        dtype=torch.bfloat16,
        use_safetensors=False,
    ).to("cuda")
    return encoder, tokenizer


def _load_run(
    run: Path,
) -> tuple[dict, object, object, object, dict[str, str]]:
    from safetensors.torch import load_file

    run = run.resolve()
    result = _read_run_result(run)
    _training_max_tokens(result)
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
        or mode not in {"frozen", "lora", "lpft"}
        or not head_path.is_relative_to(run)
        or file_sha256(head_path) != artifact.get("head_sha256")
    ):
        raise ValueError("run contract failed")

    base_model = _verified_base_model_identity()
    encoder, tokenizer = _load_pytorch_base_model()
    if _verified_base_model_identity() != base_model:
        raise ValueError("base model changed during loading")
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
    elif mode == "lpft":
        encoder_name = artifact.get("encoder")
        if not isinstance(encoder_name, str):
            raise ValueError("run has no LP-FT encoder artifact")
        encoder_path = (run / encoder_name).resolve()
        if not encoder_path.is_relative_to(run) or file_sha256(
            encoder_path
        ) != artifact.get("encoder_sha256"):
            raise ValueError("LP-FT encoder hash mismatch")
        state = load_file(str(encoder_path))
        if set(state) != set(result.get("lpft", {}).get("trainable_names", ())):
            raise ValueError("LP-FT encoder identity mismatch")
        unexpected = encoder.load_state_dict(state, strict=False).unexpected_keys
        if (
            unexpected
            or sum(value.numel() for value in state.values())
            != result["lpft"]["trainable_parameters"]
        ):
            raise ValueError("LP-FT encoder state mismatch")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    head = new_head_for_result(
        encoder.config.hidden_size,
        result["seed"],
        result,
    ).to("cuda")
    head.load_state_dict(load_file(str(head_path)), strict=True)
    head.eval()
    return result, encoder, tokenizer, head, base_model


def _snapshot_update_from_path(snapshot: Path) -> int:
    match = _SNAPSHOT_NAME.fullmatch(snapshot.name)
    if match is None:
        raise ValueError("snapshot must use the retained update-N.pt name")
    update = int(match.group(1))
    if update < 1:
        raise ValueError("snapshot update must be positive")
    return update


def _validate_state(
    name: str,
    state: object,
    expected: dict,
) -> dict:
    import torch

    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError(f"{name} state keys do not match the completed run")
    for key, value in state.items():
        reference = expected[key]
        if (
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
        ):
            raise ValueError(f"{name} state tensor contract failed: {key!r}")
    return state


def _assert_restored_state(name: str, restored: dict, selected: dict) -> None:
    import torch

    if set(restored) != set(selected) or any(
        not torch.equal(restored[key].detach().cpu(), selected[key].detach().cpu())
        for key in restored
    ):
        raise ValueError(f"restored {name} differs from the retained snapshot")


def _restore_snapshot_state(
    *,
    result: dict,
    payload: dict,
    encoder,
    head,
) -> None:
    mode = result["adaptation"]
    head_state = _validate_state("head", payload["head"], head.state_dict())
    head.load_state_dict(head_state, strict=True)
    _assert_restored_state("head", head.state_dict(), head_state)

    if mode == "frozen":
        if payload["adapter"] is not None or payload["encoder"] is not None:
            raise ValueError("frozen snapshot contains encoder adaptation state")
        return

    if mode == "lora":
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        if payload["encoder"] is not None:
            raise ValueError("LoRA snapshot contains LP-FT encoder state")
        current = get_peft_model_state_dict(encoder)
        selected = _validate_state("LoRA adapter", payload["adapter"], current)
        set_peft_model_state_dict(encoder, dict(selected))
        _assert_restored_state(
            "LoRA adapter",
            get_peft_model_state_dict(encoder),
            selected,
        )
        return

    if payload["adapter"] is not None:
        raise ValueError("LP-FT snapshot contains LoRA adapter state")
    lpft = result.get("lpft")
    names = lpft.get("trainable_names") if isinstance(lpft, dict) else None
    if (
        not isinstance(names, list)
        or not names
        or names != sorted(set(names))
        or any(not isinstance(name, str) for name in names)
    ):
        raise ValueError("completed run has no strict LP-FT state identity")
    current = encoder.state_dict()
    if any(name not in current for name in names):
        raise ValueError("LP-FT state names do not exist in the completed run")
    expected = {name: current[name] for name in names}
    selected = _validate_state("LP-FT encoder", payload["encoder"], expected)
    incompatible = encoder.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError("LP-FT snapshot has unexpected encoder state")
    restored = encoder.state_dict()
    _assert_restored_state(
        "LP-FT encoder",
        {name: restored[name] for name in names},
        selected,
    )


def _load_snapshot(
    snapshot: Path,
    *,
    result: dict,
    encoder,
    head,
) -> dict:
    """Load one retained validation snapshot against its completed-run identity."""

    import torch

    snapshot = snapshot.resolve()
    filename_update = _snapshot_update_from_path(snapshot)
    if not snapshot.is_file():
        raise FileNotFoundError(f"snapshot does not exist: {snapshot}")
    with snapshot.open("rb") as handle:
        snapshot_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)

    required = {
        "loss",
        "epoch",
        "updates",
        "head",
        "adapter",
        "encoder",
        "training_identity",
        "metrics",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("snapshot payload contract failed")
    training_identity = result.get("training_identity")
    if not isinstance(training_identity, dict):
        raise ValueError("completed run has no training identity for snapshot use")
    if payload["training_identity"] != training_identity:
        raise ValueError("snapshot training identity differs from the completed run")

    update = payload["updates"]
    epoch = payload["epoch"]
    metrics = payload["metrics"]
    loss = payload["loss"]
    if (
        type(update) is not int
        or update < 1
        or update != filename_update
        or type(epoch) is not int
        or epoch < 1
        or not isinstance(metrics, dict)
        or metrics.get("updates") != update
        or metrics.get("epoch") != epoch
        or not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
        or metrics.get("selection_loss") != loss
        or type(metrics.get("pre_registered_comparison")) is not bool
        or ("interim" in metrics and type(metrics["interim"]) is not bool)
    ):
        raise ValueError("snapshot validation-point contract failed")
    training = result.get("training")
    curve = training.get("curve") if isinstance(training, dict) else None
    if not isinstance(curve, list) or sum(row == metrics for row in curve) != 1:
        raise ValueError("snapshot validation point is absent from the completed run")

    _restore_snapshot_state(
        result=result,
        payload=payload,
        encoder=encoder,
        head=head,
    )
    head.eval()
    encoder.eval()
    if metrics["pre_registered_comparison"]:
        role = "pre_registered_comparison"
    elif metrics.get("interim", False):
        role = "periodic_validation"
    else:
        role = "epoch_final"
    return {
        "sha256": snapshot_sha256,
        "update": update,
        "epoch": epoch,
        "role": role,
    }


def _evaluation_model_sha256(
    run_result_sha256: str,
    snapshot_sha256: str | None,
    *,
    base_model: dict[str, str] | None = None,
) -> str:
    if base_model is None:
        if snapshot_sha256 is None:
            return run_result_sha256
        identity = {
            "run_result_sha256": run_result_sha256,
            "snapshot_sha256": snapshot_sha256,
        }
    else:
        if (
            set(base_model)
            != {
                "config_sha256",
                "id",
                "pytorch_model_sha256",
                "revision",
                "special_tokens_map_sha256",
                "tokenizer_config_sha256",
                "tokenizer_json_sha256",
            }
            or base_model.get("id") != MODEL_ID
            or base_model.get("revision") != MODEL_REVISION
            or any(
                not isinstance(base_model.get(name), str)
                or _SHA256.fullmatch(base_model[name]) is None
                for name in (
                    "config_sha256",
                    "pytorch_model_sha256",
                    "special_tokens_map_sha256",
                    "tokenizer_config_sha256",
                    "tokenizer_json_sha256",
                )
            )
        ):
            raise ValueError("base model identity contract failed")
        identity = {
            "base_model": base_model,
            "run_result_sha256": run_result_sha256,
            "snapshot_sha256": snapshot_sha256,
        }
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scores = np.empty_like(values)
    positive = values >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    scores[~positive] = exponent / (1.0 + exponent)
    return scores


def _score_multitask_texts(
    encoder,
    tokenizer,
    head,
    texts: list[str],
    *,
    batch_size: int,
    max_tokens: int = MAX_TOKENS,
) -> np.ndarray:
    """Score both heads in one encoder pass while preserving primary math."""

    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if type(max_tokens) is not int or max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(f"max tokens must be one of {SUPPORTED_MAX_TOKENS}")
    encoder.eval()
    head.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                [strict_normalize(text) for text in texts[start : start + batch_size]],
                add_special_tokens=True,
                max_length=max_tokens,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encoder(**encoded).last_hidden_state
                features = pool(hidden, encoded["attention_mask"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = head(features)
            if values.ndim != 2 or values.shape != (len(encoded["input_ids"]), 2):
                raise ValueError("multitask head did not return two logits per row")
            logits.append(values.float().cpu().numpy())
    if not logits:
        return np.empty((0, 2), dtype=np.float64)
    return _sigmoid(np.concatenate(logits, axis=0))


def _score_single_texts(
    encoder,
    tokenizer,
    head,
    texts: list[str],
    *,
    batch_size: int,
    max_tokens: int,
) -> np.ndarray:
    """Score a single-output head at a non-historical runtime context cap."""
    import torch

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    encoder.eval()
    head.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                [strict_normalize(text) for text in texts[start : start + batch_size]],
                add_special_tokens=True,
                max_length=max_tokens,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encoder(**encoded).last_hidden_state
                features = pool(hidden, encoded["attention_mask"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = head(features)
            if values.ndim != 2 or values.shape != (len(encoded["input_ids"]), 1):
                raise ValueError("single-output head did not return one logit per row")
            logits.append(values[:, 0].float().cpu().numpy())
    if not logits:
        return np.empty(0, dtype=np.float64)
    return _sigmoid(np.concatenate(logits, axis=0))


def _score(
    rows,
    encoder,
    tokenizer,
    head,
    *,
    batch_size: int,
    journal: ScoreJournal | None = None,
    score_columns: tuple[str, ...] = ("score",),
    max_tokens: int = MAX_TOKENS,
) -> dict:
    if type(max_tokens) is not int or max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(f"max tokens must be one of {SUPPORTED_MAX_TOKENS}")
    if score_columns not in {
        ("score",),
        ("score", "harmful_intent_score"),
    }:
        raise ValueError("unsupported evaluation score columns")

    def score_batch(texts: list[str]) -> np.ndarray:
        if len(score_columns) == 1:
            scores = (
                score_texts(
                    encoder,
                    tokenizer,
                    head,
                    texts,
                    batch_size=batch_size,
                )
                if max_tokens == MAX_TOKENS
                else _score_single_texts(
                    encoder,
                    tokenizer,
                    head,
                    texts,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                )
            )
            return np.asarray(scores, dtype=np.float64)[:, np.newaxis]
        return _score_multitask_texts(
            encoder,
            tokenizer,
            head,
            texts,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )

    if journal is not None:
        rows = list(rows)
        if len(rows) != journal.spec.rows or journal.spec.columns != score_columns:
            raise ValueError("score journal does not match the evaluation population")
        start = journal.completed_rows
        for batch in batches(rows[start:], 512):
            values = score_batch([row["text"] for row in batch])
            journal.append(values, start=start)
            start += len(batch)
        values = journal.scores()
        if not rows:
            raise ValueError("evaluation population is empty")
        return {
            "labels": np.asarray([row["label"] for row in rows], dtype=np.int8),
            "scores": values[:, 0].astype(np.float64, copy=False),
            "head_scores": values.astype(np.float64, copy=False),
            "sources": np.asarray([row["source"] for row in rows]),
            "channels": np.asarray([row["input_channel"] for row in rows]),
            "pair_ids": [row.get("pair_id") for row in rows],
            "tags": [row.get("security_tags", ()) for row in rows],
            "records": rows,
        }

    labels = []
    head_scores = []
    sources = []
    channels = []
    pair_ids = []
    tags = []
    records = []
    for batch in batches(rows, 512):
        values = score_batch([row["text"] for row in batch])
        labels.extend(row["label"] for row in batch)
        head_scores.append(values)
        sources.extend(row["source"] for row in batch)
        channels.extend(row["input_channel"] for row in batch)
        pair_ids.extend(row.get("pair_id") for row in batch)
        tags.extend(row.get("security_tags", ()) for row in batch)
        records.extend(batch)
    if not labels:
        raise ValueError("evaluation population is empty")
    values = np.concatenate(head_scores, axis=0).astype(np.float64, copy=False)
    return {
        "labels": np.asarray(labels, dtype=np.int8),
        "scores": values[:, 0],
        "head_scores": values,
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


def _score_panel_sha256(records: list[dict]) -> str:
    """Hash ordered scoring inputs without persisting row identities or text."""

    digest = hashlib.sha256()
    for row in records:
        tags = row.get("security_tags") or ()
        if isinstance(tags, str) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("invalid security tags in scoring population")
        optional_identity = {
            name: row.get(name)
            for name in (
                "group_id",
                "pair_id",
                "validation_component_id",
            )
        }
        if any(
            value is not None and not isinstance(value, str)
            for value in optional_identity.values()
        ):
            raise ValueError("invalid group identity in scoring population")
        metadata = {
            "id": row["id"],
            "label": row["label"],
            "source": row["source"],
            "input_channel": row["input_channel"],
            "security_tags": sorted(tags),
            **optional_identity,
        }
        digest.update(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(hashlib.sha256(row["text"].encode("utf-8")).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _scoring_sha256(max_tokens: int = MAX_TOKENS) -> str:
    if type(max_tokens) is not int or max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(f"max tokens must be one of {SUPPORTED_MAX_TOKENS}")
    digest = hashlib.sha256()
    digest.update(f"evaluation_max_tokens={max_tokens}\n".encode("ascii"))
    for path in (
        Path(__file__),
        Path(__file__).with_name("core.py"),
        Path(__file__).with_name("head_contract.py"),
        Path(__file__).resolve().parents[2] / "normalization.py",
        Path(__file__).resolve().parents[4] / "uv.lock",
    ):
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _evaluation_input_sha256(
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    additional_pairs: Path | None,
) -> dict[str, str | None]:
    return {
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "external_manifest_sha256": file_sha256(external_dir / "manifest.json"),
        "pair_archive_sha256": file_sha256(pairs),
        "additional_pair_archive_sha256": (
            file_sha256(additional_pairs) if additional_pairs is not None else None
        ),
    }


def _require_unchanged_evaluation_inputs(
    expected: dict[str, str | None],
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    additional_pairs: Path | None,
) -> None:
    if (
        _evaluation_input_sha256(
            data_dir=data_dir,
            external_dir=external_dir,
            pairs=pairs,
            additional_pairs=additional_pairs,
        )
        != expected
    ):
        raise ValueError("evaluation inputs changed during evaluation")


def _evaluation_identity_document(
    *,
    model_sha256: str,
    scoring_sha256: str,
    training_max_tokens: int,
    evaluation_max_tokens: int,
    ordered_panel_sha256: tuple[tuple[str, str], ...],
) -> dict:
    for name, value in (
        ("model_sha256", model_sha256),
        ("scoring_sha256", scoring_sha256),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"evaluation identity {name} must be a SHA-256 digest")
    if (
        type(training_max_tokens) is not int
        or training_max_tokens < 1
        or type(evaluation_max_tokens) is not int
        or evaluation_max_tokens < 1
    ):
        raise ValueError("evaluation identity context caps must be positive integers")
    if not isinstance(ordered_panel_sha256, (list, tuple)) or len(
        ordered_panel_sha256
    ) != len(EVALUATION_PANEL_ORDER):
        raise ValueError("evaluation identity requires every ordered score panel")

    normalized = []
    for expected_name, entry in zip(
        EVALUATION_PANEL_ORDER,
        ordered_panel_sha256,
        strict=True,
    ):
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or entry[0] != expected_name
            or not isinstance(entry[1], str)
            or _SHA256.fullmatch(entry[1]) is None
        ):
            raise ValueError("evaluation score-panel identity or order is invalid")
        normalized.append({"name": expected_name, "sha256": entry[1]})
    return {
        "schema_version": EVALUATION_IDENTITY_SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "scoring_sha256": scoring_sha256,
        "training_max_tokens": training_max_tokens,
        "evaluation_max_tokens": evaluation_max_tokens,
        "ordered_score_panels": normalized,
    }


def _evaluation_identity_sha256(
    *,
    model_sha256: str,
    scoring_sha256: str,
    training_max_tokens: int,
    evaluation_max_tokens: int,
    identity_schema_version: int = 1,
    ordered_panel_sha256: tuple[tuple[str, str], ...] | None = None,
) -> str:
    if identity_schema_version == 1:
        if ordered_panel_sha256 is not None:
            raise ValueError("legacy evaluation identity cannot bind score panels")
        for name, value in (
            ("model_sha256", model_sha256),
            ("scoring_sha256", scoring_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"evaluation identity {name} must be a SHA-256 digest")
        if (
            type(training_max_tokens) is not int
            or training_max_tokens < 1
            or type(evaluation_max_tokens) is not int
            or evaluation_max_tokens < 1
        ):
            raise ValueError(
                "evaluation identity context caps must be positive integers"
            )
        identity = {
            "model_sha256": model_sha256,
            "scoring_sha256": scoring_sha256,
            "training_max_tokens": training_max_tokens,
            "evaluation_max_tokens": evaluation_max_tokens,
        }
    elif identity_schema_version == EVALUATION_IDENTITY_SCHEMA_VERSION:
        if ordered_panel_sha256 is None:
            raise ValueError("current evaluation identity requires score panels")
        identity = _evaluation_identity_document(
            model_sha256=model_sha256,
            scoring_sha256=scoring_sha256,
            training_max_tokens=training_max_tokens,
            evaluation_max_tokens=evaluation_max_tokens,
            ordered_panel_sha256=ordered_panel_sha256,
        )
    else:
        raise ValueError("unsupported evaluation identity schema version")
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def _expected_evaluation_identity_sha256(
    report: dict,
    *,
    model_sha256: str,
    scoring_sha256: str,
    training_max_tokens: int,
    evaluation_max_tokens: int,
) -> str:
    report_schema_version = report.get("schema_version")
    if report_schema_version == 1:
        if "evaluation_identity" in report or any(
            isinstance(report.get(report_name), dict)
            and "score_panel_sha256" in report[report_name]
            for _, report_name in _EVALUATION_REPORT_PANELS
        ):
            raise ValueError(
                "schema-1 evaluation has partial current identity metadata"
            )
        return _evaluation_identity_sha256(
            model_sha256=model_sha256,
            scoring_sha256=scoring_sha256,
            training_max_tokens=training_max_tokens,
            evaluation_max_tokens=evaluation_max_tokens,
        )
    if report_schema_version != EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported full-evaluation schema version")

    ordered_panel_sha256 = []
    for panel_name, report_name in _EVALUATION_REPORT_PANELS:
        panel = report.get(report_name)
        digest = panel.get("score_panel_sha256") if isinstance(panel, dict) else None
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                f"full evaluation has no valid {panel_name} score-panel identity"
            )
        ordered_panel_sha256.append((panel_name, digest))
    ordered_panel_sha256 = tuple(ordered_panel_sha256)
    expected_document = _evaluation_identity_document(
        model_sha256=model_sha256,
        scoring_sha256=scoring_sha256,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
        ordered_panel_sha256=ordered_panel_sha256,
    )
    if report.get("evaluation_identity") != expected_document:
        raise ValueError("full-evaluation identity document mismatch")
    return _evaluation_identity_sha256(
        model_sha256=model_sha256,
        scoring_sha256=scoring_sha256,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
        identity_schema_version=EVALUATION_IDENTITY_SCHEMA_VERSION,
        ordered_panel_sha256=ordered_panel_sha256,
    )


def _real_finance_mask(scored: dict) -> np.ndarray:
    selected = (
        (scored["labels"] == 0)
        & (scored["channels"] == "direct_user")
        & np.isin(
            scored["sources"],
            list(_REAL_FINANCE_SOURCES),
        )
    )
    if int(selected.sum()) != 7_043:
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


def _harmful_labels(tags: list) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(len(tags), dtype=np.int8)
    known = np.zeros(len(tags), dtype=bool)
    for index, values in enumerate(tags):
        values = values or ()
        if isinstance(values, str) or any(not isinstance(tag, str) for tag in values):
            raise ValueError("invalid security tags in harmful-intent evaluation")
        if _HARMFUL_INTENT_TAG in values:
            labels[index] = 1
            known[index] = True
        elif _BENIGN_TAG in values:
            known[index] = True
    return labels, known


def _harmful_metrics(labels: np.ndarray, known: np.ndarray, scores: np.ndarray) -> dict:
    if (
        labels.shape != known.shape
        or labels.shape != scores.shape
        or labels.ndim != 1
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("invalid harmful-intent evaluation arrays")
    selected_labels = labels[known]
    selected_scores = scores[known]
    positives = selected_labels == 1
    negatives = selected_labels == 0
    result = {
        "counts": {
            "rows": len(labels),
            "known": int(known.sum()),
            "unknown_masked": int((~known).sum()),
            "positive": int(positives.sum()),
            "negative": int(negatives.sum()),
        },
        "binary_cross_entropy": None,
        "roc_auc": None,
        "average_precision": None,
        "positive_score_mean": (
            float(selected_scores[positives].mean()) if positives.any() else None
        ),
        "negative_score_mean": (
            float(selected_scores[negatives].mean()) if negatives.any() else None
        ),
    }
    if len(selected_labels):
        epsilon = np.finfo(np.float64).eps
        clipped = np.clip(selected_scores, epsilon, 1.0 - epsilon)
        result["binary_cross_entropy"] = float(
            -np.mean(
                selected_labels * np.log(clipped)
                + (1 - selected_labels) * np.log1p(-clipped)
            )
        )
    if positives.any() and negatives.any():
        result["roc_auc"] = float(roc_auc_score(selected_labels, selected_scores))
        result["average_precision"] = float(
            average_precision_score(selected_labels, selected_scores)
        )
    return result


def _harmful_population(scored: dict) -> dict:
    if scored["head_scores"].shape[1] != 2:
        raise ValueError("harmful-intent evidence requires a two-output head")
    labels, known = _harmful_labels(scored["tags"])
    scores = scored["head_scores"][:, 1]
    by_source = {}
    for source in sorted(set(scored["sources"])):
        selected = scored["sources"] == source
        by_source[str(source)] = _harmful_metrics(
            labels[selected],
            known[selected],
            scores[selected],
        )
    return {
        "aggregate": _harmful_metrics(labels, known, scores),
        "by_source": by_source,
    }


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
    snapshot: Path | None = None,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    additional_pairs: Path | None,
    output: Path,
    batch_size: int,
    prep_cache: Path | None = None,
    score_journal: Path | None = None,
    tokenizer_workers: int | None = None,
    evaluation_max_tokens: int | None = None,
) -> Path:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    tokenizer_execution = _configure_tokenizer_execution(tokenizer_workers)
    result, encoder, tokenizer, head, base_model = _load_run(run)
    training_max_tokens = _training_max_tokens(result)
    if evaluation_max_tokens is None:
        evaluation_max_tokens = training_max_tokens
    if (
        type(evaluation_max_tokens) is not int
        or evaluation_max_tokens not in SUPPORTED_MAX_TOKENS
    ):
        raise ValueError(f"evaluation max tokens must be one of {SUPPORTED_MAX_TOKENS}")
    native_context = evaluation_max_tokens == training_max_tokens
    context_suffix = f"trainctx{training_max_tokens}-evalctx{evaluation_max_tokens}"
    if (
        training_max_tokens != MAX_TOKENS or evaluation_max_tokens != MAX_TOKENS
    ) and context_suffix not in output.name:
        raise ValueError(
            "non-default or cross-cap evaluation output name must include "
            + context_suffix
        )
    head_contract = resolve_head_contract(result)
    checkpoint = (
        _load_snapshot(
            snapshot,
            result=result,
            encoder=encoder,
            head=head,
        )
        if snapshot is not None
        else None
    )
    if score_journal is not None:
        require_disjoint_paths(output, score_journal)
    run_result_sha256 = file_sha256(run / "result.json")
    model_sha256 = _evaluation_model_sha256(
        run_result_sha256,
        checkpoint["sha256"] if checkpoint is not None else None,
        base_model=base_model,
    )
    scoring_sha256 = _scoring_sha256(evaluation_max_tokens)
    score_columns = (
        ("score", "harmful_intent_score") if head_contract.outputs == 2 else ("score",)
    )
    input_sha256 = _evaluation_input_sha256(
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        additional_pairs=additional_pairs,
    )

    def score_population(name: str, rows) -> dict:
        records = list(rows)
        if not records:
            raise ValueError("evaluation population is empty")
        journal = (
            ScoreJournal(
                score_journal / name,
                ScoreJournalSpec(
                    model_sha256=model_sha256,
                    panel_sha256=score_panel_sha256[name],
                    scoring_sha256=scoring_sha256,
                    rows=len(records),
                    batch_size=batch_size,
                    columns=score_columns,
                ),
            )
            if score_journal is not None
            else None
        )
        return _score(
            records,
            encoder,
            tokenizer,
            head,
            batch_size=batch_size,
            journal=journal,
            score_columns=score_columns,
            max_tokens=evaluation_max_tokens,
        )

    views = routing_views(data_dir)
    external, _ = external_rows(external_dir)
    prepared = prepare_training_data(
        data_dir,
        external_dir,
        pairs,
        seed=result["seed"],
        additional_pair_archive=additional_pairs,
        cache_dir=prep_cache,
    )
    provenance = result.get("provenance", {})
    if (
        provenance.get("data_manifest_sha256") != prepared.data_manifest_sha256
        or provenance.get("external_manifest_sha256")
        != prepared.external_manifest_sha256
        or any(provenance.get(name) != digest for name, digest in input_sha256.items())
    ):
        raise ValueError("evaluation data digest differs from the training run")
    dev_path, dev_spec = views["dev_test"]
    population_rows = {
        "calibration": list(prepared.calibration),
        "dev_test": list(canonical_rows(dev_path, dev_spec, split="dev_test")),
        "promptshield_test": list(external["promptshield_test"]),
        "sep": list(external["sep"]),
    }
    del prepared
    _require_unchanged_evaluation_inputs(
        input_sha256,
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        additional_pairs=additional_pairs,
    )
    if tuple(population_rows) != EVALUATION_PANEL_ORDER:
        raise ValueError("evaluation score-panel order changed")
    ordered_panel_sha256 = tuple(
        (name, _score_panel_sha256(population_rows[name]))
        for name in EVALUATION_PANEL_ORDER
    )
    score_panel_sha256 = dict(ordered_panel_sha256)
    evaluation_identity = _evaluation_identity_document(
        model_sha256=model_sha256,
        scoring_sha256=scoring_sha256,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
        ordered_panel_sha256=ordered_panel_sha256,
    )
    evaluation_identity_sha256 = _evaluation_identity_sha256(
        model_sha256=model_sha256,
        scoring_sha256=scoring_sha256,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
        identity_schema_version=EVALUATION_IDENTITY_SCHEMA_VERSION,
        ordered_panel_sha256=ordered_panel_sha256,
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    calibration = score_population(
        "calibration",
        population_rows["calibration"],
    )
    thresholds, threshold_evidence = _select_component_thresholds(
        calibration["scores"],
        calibration["labels"],
        calibration["records"],
    )
    if "1.0000%" not in thresholds:
        raise ValueError("the one-percent component threshold is unavailable")
    threshold = thresholds["1.0000%"]
    dev = score_population(
        "dev_test",
        population_rows["dev_test"],
    )
    promptshield = score_population(
        "promptshield_test",
        population_rows["promptshield_test"],
    )
    sep = score_population(
        "sep",
        population_rows["sep"],
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
            arrays.append(np.column_stack((scored["labels"], scored["head_scores"])))
            offset = stop
        arrays_path = temporary / "scores.npy"
        np.save(arrays_path, np.concatenate(arrays), allow_pickle=False)
        report = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "purpose": "advisory mmBERT development evaluation",
            "advisory_only": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "base_model": base_model,
            "adaptation": result["adaptation"],
            "training_max_tokens": training_max_tokens,
            "evaluation_max_tokens": evaluation_max_tokens,
            "native_context_evaluation": native_context,
            "evaluation_identity_sha256": evaluation_identity_sha256,
            "evaluation_identity": evaluation_identity,
            "head_contract": {
                "architecture": head_contract.architecture,
                "outputs": head_contract.outputs,
                "columns": {
                    str(index): name for index, name in enumerate(head_contract.columns)
                },
                "primary_column": head_contract.primary_column,
            },
            "run_result_sha256": run_result_sha256,
            **(
                {
                    "evaluation_model_sha256": model_sha256,
                    "evaluated_checkpoint": checkpoint,
                }
                if checkpoint is not None
                else {}
            ),
            "inputs": {
                **input_sha256,
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
                Path(__file__).with_name("head_contract.py"),
                Path(__file__).with_name("score_journal.py"),
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
                "score_panel_sha256": score_panel_sha256["calibration"],
                "component_thresholds": threshold_evidence,
                "metrics": _metrics(
                    calibration["labels"],
                    calibration["scores"],
                    threshold,
                ),
            },
            "canonical_dev_test": {
                "row_identity_sha256": _identity_sha256(dev["records"]),
                "score_panel_sha256": score_panel_sha256["dev_test"],
                "metrics": _metrics(dev["labels"], dev["scores"], threshold),
                "by_source": _by_value(dev, "sources", threshold),
                "by_channel": _by_value(dev, "channels", threshold),
                "by_instruction_subtype": _by_subtype(dev, threshold),
            },
            "promptshield_test": {
                "row_identity_sha256": _identity_sha256(promptshield["records"]),
                "score_panel_sha256": score_panel_sha256["promptshield_test"],
                "metrics": _metrics(
                    promptshield["labels"],
                    promptshield["scores"],
                    threshold,
                ),
            },
            "sep": {
                "row_identity_sha256": _identity_sha256(sep["records"]),
                "score_panel_sha256": score_panel_sha256["sep"],
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
            **(
                {
                    "harmful_intent": {
                        "role": "masked diagnostic only; never used for threshold selection",
                        "label_contract": (
                            "harmful_intent security tag is positive, benign security "
                            "tag is negative, and every other row is masked unknown"
                        ),
                        "source_confounded": True,
                        "source_confounding_limitation": (
                            "tag availability and class balance differ by source, so "
                            "aggregate auxiliary metrics do not establish cross-source "
                            "harmfulness generalization"
                        ),
                        "calibration": _harmful_population(calibration),
                        "canonical_dev_test": _harmful_population(dev),
                    }
                }
                if head_contract.outputs == 2
                else {}
            ),
            "scores": {
                "path": "scores.npy",
                "sha256": file_sha256(arrays_path),
                "scoring_sha256": scoring_sha256,
                "evaluation_identity_sha256": evaluation_identity_sha256,
                "training_max_tokens": training_max_tokens,
                "evaluation_max_tokens": evaluation_max_tokens,
                "columns": ["label", *score_columns],
                "slices": score_slices,
            },
            "runtime": {
                "seconds": time.perf_counter() - started,
                "batch_size": batch_size,
                "training_max_tokens": training_max_tokens,
                "evaluation_max_tokens": evaluation_max_tokens,
                "native_context_evaluation": native_context,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "resumable_score_journal": score_journal is not None,
                "tokenizer": tokenizer_execution,
            },
            "limitations": [
                "PromptShield test and SEP are already-open development evidence.",
                "This is not a prospective final test or a source-held-out retrain.",
                "The threshold is descriptive and is not approved for blocking.",
                (
                    f"Scores use the run-native {evaluation_max_tokens}-token cap."
                    if native_context
                    else (
                        f"Cross-cap diagnostic: weights trained at {training_max_tokens} "
                        f"tokens were evaluated at {evaluation_max_tokens} tokens."
                    )
                ),
            ],
        }
        (temporary / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _require_unchanged_evaluation_inputs(
            input_sha256,
            data_dir=data_dir,
            external_dir=external_dir,
            pairs=pairs,
            additional_pairs=additional_pairs,
        )
        os.replace(temporary, output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _evaluation_output_name(
    *,
    snapshot_update: int | None,
    training_max_tokens: int,
    evaluation_max_tokens: int,
) -> str:
    name = (
        f"evaluation-update-{snapshot_update}"
        if snapshot_update is not None
        else "evaluation"
    )
    if training_max_tokens != MAX_TOKENS or evaluation_max_tokens != MAX_TOKENS:
        name += f"-trainctx{training_max_tokens}-evalctx{evaluation_max_tokens}"
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="explicit retained update-N.pt snapshot from this completed run",
    )
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
    parser.add_argument("--additional-pairs", type=Path)
    parser.add_argument(
        "--prep-cache",
        type=Path,
        default=Path("artifacts/mmbert/prep-cache"),
        help="reuse the digest-keyed prepared corpus rather than rebuilding it; "
        "evaluation still verifies the run's recorded provenance digests against "
        "what it loads, so a mismatched corpus fails the same way either way",
    )
    parser.add_argument(
        "--no-prep-cache", dest="prep_cache", action="store_const", const=None
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--evaluation-max-tokens",
        type=int,
        choices=SUPPORTED_MAX_TOKENS,
        help="override the run-native context cap for an explicit cross-cap diagnostic",
    )
    parser.add_argument(
        "--tokenizer-workers",
        type=int,
        help=(
            "Rust tokenizer threads; defaults to the container CPU budget. "
            "Use 1 for deterministic serial execution"
        ),
    )
    parser.add_argument(
        "--score-journal",
        type=Path,
        help="text-free resumable score scratch directory; defaults beside output",
    )
    parser.add_argument(
        "--no-score-journal",
        action="store_true",
        help="disable resumable numeric score shards",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.batch_size):
        raise ValueError("batch size must be positive")
    if args.tokenizer_workers is not None and args.tokenizer_workers < 1:
        raise ValueError("tokenizer workers must be positive")
    snapshot_update = (
        _snapshot_update_from_path(args.snapshot) if args.snapshot is not None else None
    )
    training_max_tokens = _training_max_tokens(_read_run_result(args.run))
    evaluation_max_tokens = args.evaluation_max_tokens or training_max_tokens
    output_name = _evaluation_output_name(
        snapshot_update=snapshot_update,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
    )
    output = args.output or args.run / output_name
    context_suffix = f"trainctx{training_max_tokens}-evalctx{evaluation_max_tokens}"
    if (
        args.output is not None
        and output_name
        != (
            f"evaluation-update-{snapshot_update}"
            if snapshot_update is not None
            else "evaluation"
        )
        and context_suffix not in output.name
    ):
        raise ValueError(
            "non-default or cross-cap --output name must include " + context_suffix
        )
    score_journal = (
        None
        if args.no_score_journal
        else args.score_journal or output.parent / f".{output.name}.score-journal"
    )
    print(
        evaluate(
            args.run,
            snapshot=args.snapshot,
            data_dir=args.data_dir,
            external_dir=args.external_dir,
            pairs=args.pairs,
            additional_pairs=args.additional_pairs,
            output=output,
            batch_size=args.batch_size,
            prep_cache=args.prep_cache,
            score_journal=score_journal,
            tokenizer_workers=args.tokenizer_workers,
            evaluation_max_tokens=args.evaluation_max_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
