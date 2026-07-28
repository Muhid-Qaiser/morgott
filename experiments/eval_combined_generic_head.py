"""Fail-closed evaluation for update-matched and full-combined generic heads.

Checkpoints use Morgott and PromptShield checkpoint-selection diagnostics.
Thresholds use only component-max negatives from the disjoint canonical
calibration role, with simultaneous Bonferroni-adjusted channel bounds.
The reported same-test operating points are explicitly descriptive.
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
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from prepare_combined_generic import (
    PROMPTSHIELD_TEST,
    PROMPTSHIELD_TEST_SHA256,
    REPO_ROOT,
    SEP,
    SEP_SHA256,
    TARGET,
    canonical_is_eligible,
    file_sha256,
    strict_hash,
)
from prepare_full_combined_generic import PAIR_ARCHIVE, PAIR_ARCHIVE_SHA256
from train_combined_generic_head import (
    VALIDATION_FEATURE_RECORD_CHUNK,
    VALIDATION_PREDICTION_BATCH_SIZE,
    _bce_from_logits,
    _binary_metrics,
    _scores,
    extract_features,
    load_records,
    new_head,
    predict_logits,
    resolve_model_revision,
)
from train_combined_generic_lora import (
    ADAPTER_LEARNING_RATE,
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    HEAD_LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_MODULES,
    LORA_PARAMETERS,
    LORA_RANK,
    MAX_TOKENS,
    MICROBATCH_SIZE,
    ROWS_PER_HALF,
    TARGET_MODULES,
    TOKEN_BUDGET,
    combined_lora_schedule,
    lora_run_directory_name,
    select_checkpoint_epoch,
)
from train_combined_generic_lora import (
    MODEL_ID as LORA_MODEL_ID,
)
from train_combined_generic_lora import (
    MODEL_REVISION as LORA_MODEL_REVISION,
)
from train_full_combined_generic_head import (
    FEATURE_CACHE_DATA,
    FEATURE_CACHE_REPORT,
    OBJECTIVES,
    _stable_json_sha256,
    canonical_feature_cache_spec,
    objective_spec,
)
from train_full_combined_generic_head import (
    run_directory_name as full_run_directory_name,
)

from morgott.data import text_hash
from morgott.detector import choose_threshold

OPERATING_FPRS = (0.001, 0.01)
EVALUATION_DIRECTORY = "evaluation_generic_v3"
EVALUATION_FEATURE_CACHE = (
    REPO_ROOT / "artifacts/combined_generic/evaluation_feature_cache_v1"
)
EVALUATION_TOKEN_BUDGET = 8192
EXPECTED_EXTERNAL = {
    "promptshield": {"rows": 23_516, "positive": 6_486},
    "sep": {"rows": 18_320, "positive": 9_160},
}
FINANCE_SOURCES = {
    "banking77",
    "financebench",
    "harper_valley_bank",
    "tatqa",
}


def _verify_source_hashes(
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"source changed during run: {name}: {path}")


def _evaluator_source_paths(
    *,
    full: bool,
    adaptation: str = "frozen",
) -> dict[str, Path]:
    paths = {
        "evaluator": Path(__file__).resolve(),
        "generic_preparation_helper": (
            REPO_ROOT / "experiments/prepare_combined_generic.py"
        ),
        "full_preparation_helper": (
            REPO_ROOT / "experiments/prepare_full_combined_generic.py"
        ),
        "training_head_helper": (
            REPO_ROOT / "experiments/train_combined_generic_head.py"
        ),
        "strict_normalizer": REPO_ROOT / "experiments/strict_normalize.py",
        "descriptive_threshold_helper": REPO_ROOT / "src/morgott/detector.py",
        "canonical_text_helper": REPO_ROOT / "src/morgott/data.py",
    }
    if full:
        paths["full_training_helper"] = (
            REPO_ROOT / "experiments/train_full_combined_generic_head.py"
        )
    if adaptation == "lora":
        paths["lora_training_runner"] = (
            REPO_ROOT / "experiments/train_combined_generic_lora.py"
        )
    return paths


def _validate_model_revision(model_id: object, revision: object) -> None:
    if not isinstance(model_id, str) or not isinstance(revision, str):
        raise ValueError("run model revision contract failed")
    try:
        resolved = resolve_model_revision(model_id, revision)
    except ValueError as error:
        raise ValueError(f"run model revision contract failed: {error}") from error
    if resolved != revision:
        raise ValueError("run model revision contract failed")


def _validate_lora_run_contract(result: dict) -> None:
    lora = result.get("lora", {})
    training = result.get("training", {})
    curve = training.get("curve")
    schedule = combined_lora_schedule(
        rows_per_half=ROWS_PER_HALF,
        epochs=EPOCHS,
        microbatch_size=MICROBATCH_SIZE,
        effective_batch_size=EFFECTIVE_BATCH_SIZE,
    )
    targeted_modules = lora.get("targeted_modules")
    fixed = (
        result.get("adaptation") == "lora"
        and result.get("condition") == "combined"
        and result.get("model_id") == LORA_MODEL_ID
        and result.get("model_revision") == LORA_MODEL_REVISION
        and result.get("max_tokens") == MAX_TOKENS
        and result.get("token_budget") == TOKEN_BUDGET
        and lora.get("rank") == LORA_RANK
        and lora.get("alpha") == LORA_ALPHA
        and lora.get("dropout") == LORA_DROPOUT
        and lora.get("bias") == "none"
        and lora.get("target_modules_regex") == TARGET_MODULES
        and isinstance(targeted_modules, list)
        and len(targeted_modules) == LORA_MODULES
        and len(set(targeted_modules)) == LORA_MODULES
        and all(re.search(TARGET_MODULES, name) for name in targeted_modules)
        and lora.get("adapter_parameters") == LORA_PARAMETERS
        and training.get("epochs") == EPOCHS
        and training.get("microbatch_size") == MICROBATCH_SIZE
        and training.get("effective_batch_size") == EFFECTIVE_BATCH_SIZE
        and training.get("half_batch_size") == schedule["half_batch_size"]
        and training.get("updates_per_epoch") == schedule["updates_per_epoch"]
        and training.get("updates") == schedule["updates"]
        and training.get("forward_backward_microsteps")
        == schedule["forward_backward_microsteps"]
        and training.get("adapter_learning_rate") == ADAPTER_LEARNING_RATE
        and training.get("head_learning_rate") == HEAD_LEARNING_RATE
        and training.get("scheduler") == "constant"
        and training.get("checkpoint_selection")
        == (
            "minimum equal-domain mean of matched Morgott and PromptShield "
            "validation BCE"
        )
        and training.get("first_half") == "m1"
        and training.get("second_half") == "promptshield"
        and training.get("rows_per_half") == ROWS_PER_HALF
        and training.get("loss")
        == "0.5 * mean_BCE(first_half) + 0.5 * mean_BCE(second_half)"
        and training.get("base_encoder_frozen") is True
        and training.get("adapter_trainable") is True
        and isinstance(curve, list)
        and len(curve) == EPOCHS
        and [row.get("epoch") for row in curve] == [1, 2, 3]
        and all(
            all(
                type(row.get(name)) is float and math.isfinite(row[name])
                for name in (
                    "validation_morgott_bce",
                    "validation_promptshield_bce",
                    "validation_macro_bce",
                )
            )
            and abs(
                row["validation_macro_bce"]
                - 0.5
                * (row["validation_morgott_bce"] + row["validation_promptshield_bce"])
            )
            <= 1e-12
            for row in curve
        )
        and training.get("selected_epoch") == select_checkpoint_epoch(curve)
    )
    if not fixed:
        raise ValueError("LoRA gate recipe contract failed")


def _verify_lora_adapter(run_directory: Path, result: dict) -> Path:
    artifact = result.get("artifact", {})
    adapter_path = _artifact_path({"path": artifact.get("adapter")})
    if adapter_path != run_directory / "adapter":
        raise ValueError("LoRA adapter path is not inside the run directory")
    recorded_files = artifact.get("adapter_files")
    if not isinstance(recorded_files, dict):
        raise ValueError("LoRA adapter file hashes are missing")
    actual_files = {
        path.name: path for path in adapter_path.iterdir() if path.is_file()
    }
    if set(recorded_files) != set(actual_files) or not {
        "adapter_config.json",
        "adapter_model.safetensors",
    } <= set(actual_files):
        raise ValueError("LoRA adapter file set mismatch")
    for name, path in actual_files.items():
        _verify_recorded_hash(
            recorded_files.get(name),
            path,
            name=f"LoRA adapter {name}",
        )
    config = json.loads(actual_files["adapter_config.json"].read_text())
    expected = {
        "base_model_name_or_path": result.get("model_id"),
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "target_modules": TARGET_MODULES,
        "task_type": "FEATURE_EXTRACTION",
        "peft_type": "LORA",
        "revision": None,
        "inference_mode": True,
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "lora_bias": False,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("LoRA adapter configuration mismatch")
    return adapter_path


def _validate_lora_validation_artifacts(
    result: dict,
    scores: dict[str, np.ndarray],
    logits: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> None:
    computed = {
        name: _binary_metrics(labels[name], scores[name])
        for name in ("morgott", "promptshield")
    }
    selection_bces = {
        name: _bce_from_logits(labels[name], logits[name])
        for name in ("morgott", "promptshield")
    }
    if any(
        np.max(np.abs(_scores(logits[name]) - scores[name])) > 1e-12
        for name in ("morgott", "promptshield")
    ):
        raise ValueError("LoRA validation scores do not match saved logits")
    recorded = result.get("validation", {})
    selected_epoch = result.get("training", {}).get("selected_epoch")
    curve = result.get("training", {}).get("curve", [])
    macro_bce = 0.5 * sum(selection_bces.values())
    try:
        selected_curve = curve[selected_epoch - 1]
        recorded_bces = (
            recorded["morgott_selection"]["bce"],
            recorded["promptshield"]["bce"],
            recorded["morgott_selection"]["selection_bce"],
            recorded["promptshield"]["selection_bce"],
            recorded["macro_bce"],
            selected_curve["validation_morgott_bce"],
            selected_curve["validation_promptshield_bce"],
            selected_curve["validation_macro_bce"],
        )
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("LoRA selected validation BCE contract failed") from error
    expected_bces = (
        computed["morgott"]["bce"],
        computed["promptshield"]["bce"],
        selection_bces["morgott"],
        selection_bces["promptshield"],
        macro_bce,
        selection_bces["morgott"],
        selection_bces["promptshield"],
        macro_bce,
    )
    rows = recorded.get("checkpoint_selection_rows")
    if rows != {
        "morgott": len(labels["morgott"]),
        "promptshield": len(labels["promptshield"]),
    } or any(
        not math.isfinite(actual) or abs(actual - expected) > 1e-7
        for actual, expected in zip(recorded_bces, expected_bces, strict=True)
    ):
        raise ValueError("LoRA selected validation BCE contract failed")


def _training_checkpoint_provenance(result: dict) -> dict:
    training = result.get("training", {})
    seed = result.get("seed")
    epochs = training.get("epochs")
    updates = training.get("updates")
    selected_epoch = training.get("selected_epoch")
    curve = training.get("curve")
    if (
        type(seed) is not int
        or seed < 0
        or type(epochs) is not int
        or epochs < 1
        or type(updates) is not int
        or updates < 1
        or type(selected_epoch) is not int
        or not 1 <= selected_epoch <= epochs
        or not isinstance(curve, list)
        or [row.get("epoch") for row in curve] != list(range(1, epochs + 1))
        or updates % epochs
    ):
        raise ValueError("run checkpoint provenance contract failed")
    provenance = {
        "seed": seed,
        "training_epochs": epochs,
        "training_updates": updates,
        "selected_epoch": selected_epoch,
        "selected_checkpoint_updates": updates // epochs * selected_epoch,
    }
    if result.get("adaptation") == "lora":
        microsteps = training.get("forward_backward_microsteps")
        if type(microsteps) is not int or microsteps < 1 or microsteps % epochs:
            raise ValueError("run checkpoint provenance contract failed")
        provenance.update(
            {
                "training_forward_backward_microsteps": microsteps,
                "selected_checkpoint_forward_backward_microsteps": (
                    microsteps // epochs * selected_epoch
                ),
            }
        )
    return provenance


def _validate_run_directory_identity(
    run_directory: Path,
    result: dict,
    *,
    full: bool,
) -> None:
    try:
        if full:
            expected = full_run_directory_name(
                result["model_id"],
                objective=result["objective"]["name"],
                pair_ranking_weight=result["training"]["pair_ranking_weight"],
                seed=result["seed"],
            )
        else:
            condition = result["condition"]
            if condition not in {"control", "combined"}:
                raise ValueError("invalid update-matched condition")
            adaptation = result.get("adaptation", "frozen")
            if adaptation == "lora":
                if condition != "combined":
                    raise ValueError("LoRA gate must use the combined condition")
                expected = lora_run_directory_name(
                    result["model_id"],
                    result["seed"],
                )
            elif adaptation == "frozen":
                model_tag = re.sub(
                    r"[^a-z0-9]+",
                    "-",
                    result["model_id"].casefold(),
                ).strip("-")
                expected = f"{model_tag}_{condition}_s{result['seed']}"
            else:
                raise ValueError("invalid adaptation")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run directory identity inputs are invalid") from error
    if run_directory.name != expected:
        raise ValueError(
            f"run directory identity mismatch: expected {expected}, "
            f"found {run_directory.name}"
        )


def _validate_full_objective(result: dict, report: dict | None = None) -> None:
    objective = result.get("objective")
    if not isinstance(objective, dict):
        raise ValueError("full run has no objective spec")
    recorded = result.get("provenance", {}).get("objective_spec_sha256")
    if recorded != _stable_json_sha256(objective):
        raise ValueError("full run objective spec hash mismatch")
    name = objective.get("name")
    coefficients = objective.get("domain_bce_coefficients")
    if (
        name not in OBJECTIVES
        or not isinstance(coefficients, dict)
        or set(coefficients) != {"morgott", "promptshield", "matched_pairs"}
        or any(
            type(value) not in {int, float} or value < 0
            for value in coefficients.values()
        )
        or not coefficients["morgott"]
    ):
        raise ValueError("full run objective contract failed")
    if report is None:
        return
    promptshield_spec = report["outputs"]["promptshield"]
    labels = np.concatenate(
        [
            np.full(
                int(promptshield_spec["labels"][str(label)]),
                label,
                dtype=np.int64,
            )
            for label in (0, 1)
        ]
    )
    expected = objective_spec(
        name,
        canonical_rows=report["outputs"]["morgott_train_index"]["rows"],
        promptshield_labels=labels,
        matched_pair_rows=report["outputs"]["matched_pairs"]["rows"],
    )
    if objective != expected:
        raise ValueError("full run objective spec differs from selected populations")


def _fitted_domains(run: dict) -> dict[str, bool]:
    if run["full"]:
        coefficients = run["result"]["objective"]["domain_bce_coefficients"]
        return {
            domain: float(coefficients[domain]) > 0
            for domain in ("morgott", "promptshield", "matched_pairs")
        }
    condition = run["result"].get("condition")
    if condition not in {"control", "combined"}:
        raise ValueError(f"unsupported update-matched condition: {condition}")
    return {
        "morgott": True,
        "promptshield": condition == "combined",
        "matched_pairs": False,
    }


def _promptshield_evaluation_scope(run: dict) -> str:
    fitted = _fitted_domains(run)
    if fitted["promptshield"] and not fitted["morgott"]:
        return (
            "PromptShield-internal source-OOD development benchmark; its published "
            "train and test component source datasets are mutually exclusive but "
            "share benchmark curation; this already-open test is not a pristine "
            "final evaluation"
        )
    if fitted["promptshield"]:
        return (
            "PromptShield's internally source-disjoint split is used as development "
            "evidence; PromptShield's own train and test component sources are "
            "mutually exclusive, but this model also fits Morgott; known "
            "LMSYS-family overlap and missing row-level PromptShield provenance "
            "mean the test is not source-OOD relative to the complete fit; this "
            "already-open test is not a pristine final evaluation"
        )
    return (
        "PromptShield-validation-informed development diagnostic; PromptShield "
        "train was not fitted, but the model fits Morgott and PromptShield "
        "validation selected checkpoints; known LMSYS-family overlap and missing "
        "row-level PromptShield provenance mean the test is not source-OOD "
        "relative to the complete model-selection pipeline; this already-open "
        "test is not a pristine final evaluation"
    )


def applied_point(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    predicted = scores >= threshold
    negative = labels == 0
    positive = labels == 1
    return {
        "threshold": float(threshold),
        "fpr": float(predicted[negative].mean()) if negative.any() else None,
        "tpr": float(predicted[positive].mean()) if positive.any() else None,
        "false_positive": int((predicted & negative).sum()),
        "true_positive": int((predicted & positive).sum()),
        "negative": int(negative.sum()),
        "positive": int(positive.sum()),
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
    maximum = math.floor(target * negative)
    for false_positive in range(maximum, -1, -1):
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


def _stratum_threshold_evidence(
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
            "false_positive_component_budget": budget,
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


def select_calibration_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    records: list[dict],
    *,
    targets: tuple[float, ...] = OPERATING_FPRS,
    confidence: float = 0.95,
) -> tuple[dict[str, float], dict]:
    """Select component-level thresholds with simultaneous channel coverage."""
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
        record
        for record, selected in zip(records, negative_mask, strict=True)
        if selected
    ]
    if not len(negative_scores):
        raise ValueError("calibration requires negatives")
    trusted_channels = {"direct_user", "untrusted_content"}
    observed_channels = {str(record.get("channel")) for record in negative_records}
    if not observed_channels <= trusted_channels:
        raise ValueError("calibration rows have an unsupported trusted channel")
    channels = sorted(trusted_channels)
    per_channel_confidence = 1.0 - (1.0 - confidence) / len(channels)
    component_scores = {channel: {} for channel in channels}
    for score, record in zip(negative_scores, negative_records, strict=True):
        channel = str(record.get("channel"))
        component_id = record.get("validation_component_id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("calibration row has no validation component identity")
        previous = component_scores[channel].get(component_id)
        component_scores[channel][component_id] = (
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
        channel_power = {
            channel: _stratum_threshold_evidence(
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
            "family_scope": (
                "the two trusted channels, with a separate family for each target"
            ),
            "negative_rows": len(negative_scores),
            "pooled_negative_role": "empirical diagnostic only",
            "inference_caveat": (
                "Components and recurring source families are not IID or sampled "
                "from a deployment distribution; this is development evidence, "
                "not a production guarantee."
            ),
        }
        underpowered_channels = [
            channel
            for channel, values in channel_power.items()
            if values["false_positive_component_budget"] is None
        ]
        if underpowered_channels:
            evidence[name] = {
                **base,
                "status": "unavailable",
                "reason": "one or more trusted channels are underpowered",
                "underpowered_channels": underpowered_channels,
                "by_channel": channel_power,
            }
            continue
        candidate_thresholds = {}
        for channel, values in channel_scores.items():
            channel_budget = channel_power[channel]["false_positive_component_budget"]
            ordered_channel = np.sort(values)[::-1]
            candidate_thresholds[f"channel:{channel}"] = float(
                np.nextafter(ordered_channel[channel_budget], np.inf)
            )
        threshold = max(candidate_thresholds.values())
        thresholds[name] = threshold

        by_channel = {}
        for channel, values in channel_scores.items():
            component_evidence = _stratum_threshold_evidence(
                values,
                threshold=threshold,
                target=target,
                confidence=per_channel_confidence,
            )
            selected = np.asarray(
                [str(record.get("channel")) == channel for record in negative_records],
                dtype=bool,
            )
            row_false_positive = int((negative_scores[selected] >= threshold).sum())
            component_evidence["row_empirical"] = {
                "negative": int(selected.sum()),
                "false_positive": row_false_positive,
                "fpr": row_false_positive / int(selected.sum()),
            }
            if component_evidence["status"] != "satisfies_bound":
                raise ValueError(
                    "calibration threshold violates a channel confidence bound"
                )
            by_channel[channel] = component_evidence
        pooled_false_positive = int((negative_scores >= threshold).sum())
        by_source = {}
        for source in sorted(
            {str(record.get("source")) for record in negative_records}
        ):
            selected = np.asarray(
                [str(record.get("source")) == source for record in negative_records],
                dtype=bool,
            )
            false_positive = int((negative_scores[selected] >= threshold).sum())
            by_source[source] = {
                "status": "empirical_only",
                "negative": int(selected.sum()),
                "false_positive": false_positive,
                "fpr": false_positive / int(selected.sum()),
            }
        evidence[name] = {
            **base,
            "status": "available",
            "threshold": threshold,
            "selection": (
                "maximum trusted-channel component-max tie-aware order statistics "
                "with Bonferroni-adjusted one-sided Clopper-Pearson upper bounds"
            ),
            "candidate_thresholds": candidate_thresholds,
            "pooled_row_empirical": {
                "negative": len(negative_scores),
                "false_positive": pooled_false_positive,
                "fpr": pooled_false_positive / len(negative_scores),
            },
            "by_channel": by_channel,
            "by_source": by_source,
        }
    return thresholds, evidence


def binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if (
        scores.ndim != 1
        or labels.ndim != 1
        or len(scores) != len(labels)
        or not len(scores)
        or not np.isfinite(scores).all()
        or not np.isin(labels, (0, 1)).all()
        or set(labels.tolist()) != {0, 1}
    ):
        raise ValueError("binary metrics require finite aligned scores and both labels")
    descriptive = {}
    for target in OPERATING_FPRS:
        name = f"{target:.4%}"
        threshold = choose_threshold(labels, scores, target)
        point = applied_point(scores, labels, threshold)
        if point["fpr"] > target + 1e-15:
            raise ValueError(f"same-test threshold exceeds target FPR {target}")
        descriptive[name] = {
            "target_fpr": target,
            **point,
        }
    return {
        "rows": len(labels),
        "negative": int((labels == 0).sum()),
        "positive": int((labels == 1).sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "validation_threshold_applied": {
            name: applied_point(scores, labels, threshold)
            for name, threshold in thresholds.items()
        },
        "descriptive_same_test": descriptive,
    }


def negative_metrics(
    scores: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("negative metrics require finite one-dimensional scores")
    return {
        "rows": len(scores),
        "validation_threshold_applied": {
            name: {
                "threshold": threshold,
                "false_positive": int((scores >= threshold).sum()),
                "fpr": float((scores >= threshold).mean()),
            }
            for name, threshold in thresholds.items()
        },
    }


def pair_metrics(
    benign: np.ndarray,
    attack: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    from sklearn.metrics import roc_auc_score

    benign = np.asarray(benign)
    attack = np.asarray(attack)
    if (
        benign.ndim != 1
        or attack.ndim != 1
        or len(benign) != len(attack)
        or not len(benign)
        or not np.isfinite(benign).all()
        or not np.isfinite(attack).all()
    ):
        raise ValueError("pair metrics require aligned finite score arrays")
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), len(benign))
    scores = np.column_stack((benign, attack)).reshape(-1)
    return {
        "pairs": len(benign),
        "rows": len(scores),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "ordering_rate": float((attack > benign).mean()),
        "ties": int((attack == benign).sum()),
        "mean_margin": float((attack - benign).mean()),
        "validation_threshold_applied": {
            name: {
                "threshold": threshold,
                "benign_fpr": float((benign >= threshold).mean()),
                "attack_tpr": float((attack >= threshold).mean()),
                "both_correct": float(
                    ((benign < threshold) & (attack >= threshold)).mean()
                ),
            }
            for name, threshold in thresholds.items()
        },
    }


class CollisionChecker:
    """Normalized- or strict-exact fitting collision checker."""

    def __init__(
        self,
        *,
        normalized: set[str] | None = None,
        strict: set[str] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        self.normalized = normalized or set()
        self.strict = strict or set()
        self.connection = connection

    @classmethod
    def from_records(cls, records: list[dict]) -> CollisionChecker:
        return cls(
            normalized={
                row.get("normalized_text_sha256") or text_hash(row["text"])
                for row in records
            },
            strict={
                row.get("strict_text_sha256") or strict_hash(row["text"])
                for row in records
            },
        )

    def add_records(self, records: list[dict]) -> None:
        for row in records:
            self.normalized.add(
                row.get("normalized_text_sha256") or text_hash(row["text"])
            )
            self.strict.add(row.get("strict_text_sha256") or strict_hash(row["text"]))

    def _database_seen(self, column: str, values: list[str]) -> set[str]:
        if self.connection is None:
            return set()
        seen = set()
        for start in range(0, len(values), 500):
            unique = sorted(set(values[start : start + 500]))
            placeholders = ",".join("?" for _ in unique)
            rows = self.connection.execute(
                f"""
                SELECT {column}
                FROM candidates
                WHERE {column} IN ({placeholders})
                """,
                unique,
            )
            seen.update(row[0] for row in rows)
        return seen

    def unseen_mask(self, texts: list[str]) -> np.ndarray:
        normalized = [text_hash(text) for text in texts]
        strict = [strict_hash(text) for text in texts]
        database_normalized = self._database_seen(
            "normalized_text_sha256",
            normalized,
        )
        database_strict = self._database_seen("strict_text_sha256", strict)
        return np.asarray(
            [
                normalized_value not in self.normalized
                and normalized_value not in database_normalized
                and strict_value not in self.strict
                and strict_value not in database_strict
                for normalized_value, strict_value in zip(
                    normalized,
                    strict,
                    strict=True,
                )
            ],
            dtype=bool,
        )


def _read_jsonl(path: Path, expected_sha256: str) -> list[dict]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"input hash mismatch for {path}: expected {expected_sha256}, "
            f"found {actual}"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _artifact_path(spec: dict) -> Path:
    path = (REPO_ROOT / spec["path"]).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"artifact path escapes repository: {path}")
    return path


def _add_scored_input_provenance(
    run: dict,
    canonical_dev_spec: dict,
    *,
    pairs_are_held_out: bool,
) -> None:
    sources = {
        "canonical_dev_test": (
            _artifact_path(canonical_dev_spec),
            canonical_dev_spec["sha256"],
        ),
        "promptshield_test": (PROMPTSHIELD_TEST, PROMPTSHIELD_TEST_SHA256),
        "sep": (SEP, SEP_SHA256),
    }
    if pairs_are_held_out:
        sources["matched_pairs"] = (PAIR_ARCHIVE, PAIR_ARCHIVE_SHA256)
    for name, (path, digest) in sources.items():
        run["causal_paths"][f"scored:{name}"] = path
        run["causal_hashes"][f"scored:{name}"] = digest


def _load_generic_spec(spec: dict) -> list[dict]:
    path = _artifact_path(spec)
    records = load_records(path, spec["sha256"])
    if len(records) != spec["rows"]:
        raise ValueError(f"row count mismatch for {path}")
    labels = Counter(str(row["generic_label"]) for row in records)
    if dict(sorted(labels.items())) != spec["labels"]:
        raise ValueError(f"label count mismatch for {path}")
    return records


def _verify_recorded_hash(
    recorded: str | None,
    path: Path,
    *,
    name: str,
) -> None:
    if not recorded:
        raise ValueError(f"missing recorded {name} hash")
    actual = file_sha256(path)
    if actual != recorded:
        raise ValueError(f"{name} hash mismatch: {path}")


def _verify_selection_report(report: dict, *, full: bool) -> None:
    if (
        report.get("schema_version") != 2
        or report.get("generic_target") != TARGET
        or report["eligibility"].get("label_field") != "injection_label"
        or report["eligibility"].get("routing_label_used") is not False
    ):
        raise ValueError("selection report target contract failed")
    manifest_spec = report["inputs"]["manifest"]
    manifest_path = _artifact_path(manifest_spec)
    _verify_recorded_hash(
        manifest_spec.get("sha256"),
        manifest_path,
        name="data manifest",
    )
    provenance = report["provenance"]
    if full:
        checks = {
            "runner": (
                provenance.get("runner_sha256"),
                REPO_ROOT / "experiments/prepare_full_combined_generic.py",
            ),
            "base preparation runner": (
                provenance.get("base_preparation_runner_sha256"),
                REPO_ROOT / "experiments/prepare_combined_generic.py",
            ),
            "strict normalizer": (
                provenance.get("strict_normalizer_sha256"),
                REPO_ROOT / "experiments/strict_normalize.py",
            ),
            "overlap helper": (
                provenance.get("overlap_module_sha256"),
                REPO_ROOT / "src/morgott/overlap.py",
            ),
            "canonical text helper": (
                provenance.get("canonical_text_helper_sha256"),
                REPO_ROOT / "src/morgott/data.py",
            ),
        }
    else:
        checks = {
            "runner": (
                provenance.get("runner_sha256"),
                REPO_ROOT / "experiments/prepare_combined_generic.py",
            ),
            "strict normalizer": (
                provenance.get("strict_normalizer_sha256"),
                REPO_ROOT / "experiments/strict_normalize.py",
            ),
            "overlap helper": (
                provenance.get("overlap_module_sha256"),
                REPO_ROOT / "src/morgott/overlap.py",
            ),
            "canonical text helper": (
                provenance.get("canonical_text_helper_sha256"),
                REPO_ROOT / "src/morgott/data.py",
            ),
        }
    for name, (recorded, path) in checks.items():
        _verify_recorded_hash(recorded, path, name=name)
    if full:
        validation = report.get("validation", {})
        component_calibration = validation.get("component_calibration", {})
        if (
            validation.get("promptshield_used_for_threshold") is not False
            or validation.get("threshold_calibration_only") != "morgott_calibration"
            or "morgott_calibration" not in validation
            or component_calibration.get("component_id_field")
            != "validation_component_id"
            or component_calibration.get("family_confidence") != 0.95
            or component_calibration.get("per_channel_confidence") != 0.975
            or component_calibration.get("multiplicity_correction") != "Bonferroni"
            or component_calibration.get("family_scope")
            != "the two trusted channels, with a separate family for each target"
            or component_calibration.get("pooled_negative_role")
            != "empirical diagnostic only"
        ):
            raise ValueError("full selection validation partition contract failed")
    else:
        partition = report.get("validation_partition", {})
        disjointness = partition.get("disjointness", {})
        component_calibration = partition.get("component_calibration", {})
        outputs = report.get("outputs", {})
        checkpoint = outputs.get("validation_morgott_selection", {})
        calibration = outputs.get("validation_morgott_calibration", {})
        if (
            partition.get("target_checkpoint_fraction") != 0.2
            or partition.get("promptshield_used_for_threshold") is not False
            or not disjointness
            or not all(disjointness.values())
            or disjointness.get("validation_component") is not True
            or component_calibration.get("component_id_field")
            != "validation_component_id"
            or component_calibration.get("family_confidence") != 0.95
            or component_calibration.get("per_channel_confidence") != 0.975
            or component_calibration.get("multiplicity_correction") != "Bonferroni"
            or component_calibration.get("family_scope")
            != "the two trusted channels, with a separate family for each target"
            or component_calibration.get("pooled_negative_role")
            != "empirical diagnostic only"
            or checkpoint.get("rows", 0) + calibration.get("rows", 0)
            != partition.get("total_rows")
            or checkpoint.get("rows") != partition.get("checkpoint_selection_rows")
            or calibration.get("rows") != partition.get("calibration_rows")
        ):
            raise ValueError("selection validation partition contract failed")
    if full:
        base_spec = report["inputs"]["base_update_matched_selection"]
        base_path = _artifact_path(base_spec)
        _verify_recorded_hash(
            base_spec.get("sha256"),
            base_path,
            name="base update-matched selection report",
        )
        base_report = json.loads(base_path.read_text())
        _verify_selection_report(base_report, full=False)
        if (
            report["inputs"]["promptshield_train"]
            != base_report["outputs"]["promptshield"]
            or report["inputs"]["promptshield_validation"]
            != base_report["outputs"]["validation_promptshield"]
            or report["validation"]["morgott"]
            != base_report["outputs"]["validation_morgott_selection"]
            or report["validation"]["morgott_calibration"]
            != base_report["outputs"]["validation_morgott_calibration"]
            or report["validation"]["promptshield"]
            != base_report["outputs"]["validation_promptshield"]
            or report["validation"]["component_calibration"]
            != base_report["validation_partition"]["component_calibration"]
        ):
            raise ValueError("full report PromptShield artifacts differ from its base")


def _expected_full_feature_cache_spec(
    result: dict,
    selection_report: dict,
    selection_report_path: Path,
) -> dict:
    feature_width = result.get("feature_width")
    if type(feature_width) is not int or feature_width < 3 or feature_width % 3:
        raise ValueError("full run has an invalid feature width")
    provenance = result["provenance"]
    return canonical_feature_cache_spec(
        selection_report_path=selection_report_path,
        selection_report_sha256=provenance["full_selection_report_sha256"],
        canonical_spec=selection_report["outputs"]["morgott_train_index"],
        canonical_input_spec=selection_report["inputs"]["canonical_train"],
        model_id=result["model_id"],
        model_revision=result["model_revision"],
        hidden_size=feature_width // 3,
        max_tokens=result["max_tokens"],
        token_budget=result["token_budget"],
        feature_record_chunk=result["canonical_feature_record_chunk"],
        runner_sha256=provenance["runner_sha256"],
        head_helper_sha256=provenance["head_helper_sha256"],
        strict_normalizer_sha256=provenance["strict_normalizer_sha256"],
        canonical_projection_sha256=provenance["canonical_projection_sha256"],
        packages=provenance["packages"],
    )


def _validate_recorded_full_feature_cache_provenance(
    result: dict,
    selection_report: dict,
    selection_report_path: Path,
) -> dict:
    provenance = result["provenance"]
    recorded = provenance.get("canonical_feature_cache")
    if not isinstance(recorded, dict):
        raise ValueError("full run has no canonical feature-cache provenance")
    report_path = _artifact_path({"path": recorded["report"]})
    data_path = _artifact_path({"path": recorded["data"]})
    artifacts = (REPO_ROOT / "artifacts").resolve()
    if (
        report_path.name != FEATURE_CACHE_REPORT
        or data_path.name != FEATURE_CACHE_DATA
        or report_path.parent != data_path.parent
        or not report_path.is_relative_to(artifacts)
    ):
        raise ValueError("canonical feature-cache report path is invalid")
    expected_spec = _expected_full_feature_cache_spec(
        result,
        selection_report,
        selection_report_path,
    )
    digests = {
        "report_sha256": recorded.get("report_sha256"),
        "data_sha256": recorded.get("data_sha256"),
        "spec_sha256": recorded.get("spec_sha256"),
    }
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in digests.values()
    ):
        raise ValueError("canonical feature-cache provenance has an invalid digest")
    if digests["spec_sha256"] != _stable_json_sha256(expected_spec):
        raise ValueError("canonical feature-cache provenance mismatch")
    return {
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": digests["report_sha256"],
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": digests["data_sha256"],
        "spec_sha256": digests["spec_sha256"],
        "files_reverified": False,
    }


def _negative_component_evidence(records: list[dict]) -> dict:
    negatives = [row for row in records if row["generic_label"] == 0]
    rows_by_channel = Counter(str(row["channel"]) for row in negatives)
    rows_by_source = Counter(str(row["source"]) for row in negatives)
    components_by_channel = defaultdict(set)
    components_by_source = defaultdict(set)
    for row in negatives:
        component_id = row.get("validation_component_id")
        if not isinstance(component_id, str) or not component_id.startswith(
            "validation-component:"
        ):
            raise ValueError("validation row has no deterministic component identity")
        components_by_channel[str(row["channel"])].add(component_id)
        components_by_source[str(row["source"])].add(component_id)
    return {
        "rows_by_channel": dict(sorted(rows_by_channel.items())),
        "components_by_channel": {
            key: len(values) for key, values in sorted(components_by_channel.items())
        },
        "rows_by_source": dict(sorted(rows_by_source.items())),
        "components_by_source": {
            key: len(values) for key, values in sorted(components_by_source.items())
        },
    }


def discover_run(run_directory: Path) -> dict:
    run_directory = run_directory.resolve()
    if not run_directory.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError("run must be inside the repository artifacts directory")
    result_path = run_directory / "result.json"
    result_bytes = result_path.read_bytes()
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    result = json.loads(result_bytes)
    if (
        result.get("generic_target") != TARGET
        or result.get("normalization") != "strict"
        or result.get("attention_implementation") != "sdpa"
        or result.get("validation_feature_record_chunk")
        != VALIDATION_FEATURE_RECORD_CHUNK
        or result.get("validation_prediction_batch_size")
        != VALIDATION_PREDICTION_BATCH_SIZE
    ):
        raise ValueError("run target or inference contract failed")
    model_id = result.get("model_id")
    revision = result.get("model_revision")
    _validate_model_revision(model_id, revision)
    purpose = result.get("purpose", "")
    full = "full-combined" in purpose
    update_matched = "update-matched" in purpose
    if full == update_matched:
        raise ValueError("cannot identify exactly one combined-run recipe")
    adaptation = result.get("adaptation", "frozen")
    if adaptation not in {"frozen", "lora"} or (full and adaptation != "frozen"):
        raise ValueError("run adaptation contract failed")
    if adaptation == "lora":
        _validate_lora_run_contract(result)
    _validate_run_directory_identity(run_directory, result, full=full)

    provenance = result["provenance"]
    if full:
        report_path = _artifact_path({"path": provenance["full_selection_report"]})
        expected_report_sha256 = provenance["full_selection_report_sha256"]
        trainer_path = REPO_ROOT / "experiments/train_full_combined_generic_head.py"
        _verify_recorded_hash(
            provenance.get("head_helper_sha256"),
            REPO_ROOT / "experiments/train_combined_generic_head.py",
            name="head helper",
        )
        _verify_recorded_hash(
            provenance.get("canonical_projection_sha256"),
            REPO_ROOT / "experiments/prepare_combined_generic.py",
            name="canonical projection helper",
        )
    else:
        report_path = _artifact_path({"path": provenance["selection_report"]})
        expected_report_sha256 = provenance["selection_report_sha256"]
        trainer_path = REPO_ROOT / (
            "experiments/train_combined_generic_lora.py"
            if adaptation == "lora"
            else "experiments/train_combined_generic_head.py"
        )
        if adaptation == "lora":
            _verify_recorded_hash(
                provenance.get("head_helper_sha256"),
                REPO_ROOT / "experiments/train_combined_generic_head.py",
                name="head helper",
            )
    _verify_recorded_hash(
        expected_report_sha256,
        report_path,
        name="selection report",
    )
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != expected_report_sha256:
        raise ValueError("selection report changed during discovery")
    report = json.loads(report_bytes)
    _verify_selection_report(report, full=full)
    if full:
        _validate_full_objective(result, report)
    elif adaptation == "lora":
        m1 = report.get("outputs", {}).get("m1", {})
        promptshield = report.get("outputs", {}).get("promptshield", {})
        if (
            m1.get("rows") != ROWS_PER_HALF
            or promptshield.get("rows") != ROWS_PER_HALF
            or m1.get("labels") != promptshield.get("labels")
            or result["training"].get("labels_per_half") != m1.get("labels")
        ):
            raise ValueError("LoRA gate fitted-half population contract failed")
    _verify_recorded_hash(
        provenance.get("runner_sha256"),
        trainer_path,
        name="training runner",
    )
    _verify_recorded_hash(
        provenance.get("strict_normalizer_sha256"),
        REPO_ROOT / "experiments/strict_normalize.py",
        name="strict normalizer",
    )
    feature_cache = (
        _validate_recorded_full_feature_cache_provenance(result, report, report_path)
        if full
        else None
    )

    head_path = _artifact_path({"path": result["artifact"]["head"]})
    if head_path.parent != run_directory:
        raise ValueError("head path is not inside the run directory")
    _verify_recorded_hash(
        result["artifact"].get("head_sha256"),
        head_path,
        name="head",
    )
    adapter_path = (
        _verify_lora_adapter(run_directory, result) if adaptation == "lora" else None
    )
    checkpoint_provenance = _training_checkpoint_provenance(result)

    array_names = [
        "validation_morgott_selection_scores.npy",
        "validation_morgott_selection_labels.npy",
        "validation_promptshield_scores.npy",
        "validation_promptshield_labels.npy",
    ]
    if adaptation == "lora":
        array_names.extend(
            (
                "validation_morgott_selection_logits.npy",
                "validation_promptshield_logits.npy",
            )
        )
    array_paths = {}
    for name in array_names:
        expected = result["artifact"]["arrays"].get(name)
        path = run_directory / name
        _verify_recorded_hash(expected, path, name=name)
        array_paths[name] = path
    validation_morgott_scores = np.load(
        array_paths["validation_morgott_selection_scores.npy"],
        allow_pickle=False,
    )
    validation_morgott_labels = np.load(
        array_paths["validation_morgott_selection_labels.npy"],
        allow_pickle=False,
    )
    validation_promptshield_scores = np.load(
        array_paths["validation_promptshield_scores.npy"],
        allow_pickle=False,
    )
    validation_promptshield_labels = np.load(
        array_paths["validation_promptshield_labels.npy"],
        allow_pickle=False,
    )
    validation_logits = (
        {
            "morgott": np.load(
                array_paths["validation_morgott_selection_logits.npy"],
                allow_pickle=False,
            ),
            "promptshield": np.load(
                array_paths["validation_promptshield_logits.npy"],
                allow_pickle=False,
            ),
        }
        if adaptation == "lora"
        else {}
    )
    for name, scores, labels in (
        (
            "morgott",
            validation_morgott_scores,
            validation_morgott_labels,
        ),
        (
            "promptshield",
            validation_promptshield_scores,
            validation_promptshield_labels,
        ),
    ):
        if (
            scores.ndim != 1
            or labels.ndim != 1
            or len(scores) != len(labels)
            or not np.isfinite(scores).all()
            or not np.isin(labels, (0, 1)).all()
            or (
                adaptation == "lora"
                and (
                    validation_logits[name].ndim != 1
                    or len(validation_logits[name]) != len(labels)
                    or not np.isfinite(validation_logits[name]).all()
                )
            )
        ):
            raise ValueError(f"invalid saved {name} validation arrays")

    if full:
        morgott_spec = report["validation"]["morgott"]
        calibration_spec = report["validation"]["morgott_calibration"]
        promptshield_spec = report["validation"]["promptshield"]
    else:
        morgott_spec = report["outputs"]["validation_morgott_selection"]
        calibration_spec = report["outputs"]["validation_morgott_calibration"]
        promptshield_spec = report["outputs"]["validation_promptshield"]
    morgott_records = _load_generic_spec(morgott_spec)
    calibration_records = _load_generic_spec(calibration_spec)
    promptshield_records = _load_generic_spec(promptshield_spec)
    if any(
        row.get("experiment_role") != "checkpoint_selection" for row in morgott_records
    ):
        raise ValueError("Morgott checkpoint rows have the wrong experiment role")
    if any(row.get("experiment_role") != "calibration" for row in calibration_records):
        raise ValueError("Morgott calibration rows have the wrong experiment role")
    component_calibration = (
        report["validation"]["component_calibration"]
        if full
        else report["validation_partition"]["component_calibration"]
    )
    checkpoint_components = {
        row.get("validation_component_id") for row in morgott_records
    }
    calibration_components = {
        row.get("validation_component_id") for row in calibration_records
    }
    if (
        any(
            not isinstance(component_id, str)
            or not component_id.startswith("validation-component:")
            for component_id in checkpoint_components | calibration_components
        )
        or not checkpoint_components.isdisjoint(calibration_components)
        or component_calibration.get("components_by_role")
        != {
            "checkpoint_selection": len(checkpoint_components),
            "calibration": len(calibration_components),
        }
        or component_calibration.get("negative_evidence_by_role")
        != {
            "checkpoint_selection": _negative_component_evidence(morgott_records),
            "calibration": _negative_component_evidence(calibration_records),
        }
    ):
        raise ValueError("validation component artifact contract failed")
    if not np.array_equal(
        validation_morgott_labels,
        np.asarray([row["generic_label"] for row in morgott_records]),
    ):
        raise ValueError("saved Morgott validation labels do not match source rows")
    if not np.array_equal(
        validation_promptshield_labels,
        np.asarray([row["generic_label"] for row in promptshield_records]),
    ):
        raise ValueError(
            "saved PromptShield validation labels do not match source rows"
        )
    if adaptation == "lora":
        _validate_lora_validation_artifacts(
            result,
            {
                "morgott": validation_morgott_scores,
                "promptshield": validation_promptshield_scores,
            },
            validation_logits,
            {
                "morgott": validation_morgott_labels,
                "promptshield": validation_promptshield_labels,
            },
        )
    causal_paths = {
        "run_result": result_path,
        "selection_report": report_path,
        "head": head_path,
        **{f"validation_array:{name}": path for name, path in array_paths.items()},
    }
    causal_hashes = {
        "run_result": result_sha256,
        "selection_report": expected_report_sha256,
        "head": result["artifact"]["head_sha256"],
        **{
            f"validation_array:{name}": result["artifact"]["arrays"][name]
            for name in array_paths
        },
    }
    population_specs = (
        {
            "m1": report["outputs"]["m1"],
            "m2": report["outputs"]["m2"],
            "promptshield": report["outputs"]["promptshield"],
            "validation_morgott": morgott_spec,
            "validation_calibration": calibration_spec,
            "validation_promptshield": promptshield_spec,
        }
        if not full
        else {
            "morgott_train_index": report["outputs"]["morgott_train_index"],
            "promptshield": report["outputs"]["promptshield"],
            "matched_pairs": report["outputs"]["matched_pairs"],
            "validation_morgott": morgott_spec,
            "validation_calibration": calibration_spec,
            "validation_promptshield": promptshield_spec,
        }
    )
    for name, spec in population_specs.items():
        causal_paths[f"population:{name}"] = _artifact_path(spec)
        causal_hashes[f"population:{name}"] = spec["sha256"]
    if adaptation == "lora":
        for name, digest in result["artifact"]["adapter_files"].items():
            causal_paths[f"adapter:{name}"] = adapter_path / name
            causal_hashes[f"adapter:{name}"] = digest
    return {
        "directory": run_directory,
        "result_path": result_path,
        "result": result,
        "full": full,
        "adaptation": adaptation,
        "report_path": report_path,
        "report": report,
        "head_path": head_path,
        "adapter_path": adapter_path,
        "feature_cache": feature_cache,
        "checkpoint_provenance": checkpoint_provenance,
        "thresholds": {},
        "threshold_evidence": {},
        "validation": {
            "morgott": binary_metrics(
                validation_morgott_scores,
                validation_morgott_labels,
                {},
            ),
            "promptshield": binary_metrics(
                validation_promptshield_scores,
                validation_promptshield_labels,
                {},
            ),
        },
        "validation_records": {
            "morgott": morgott_records,
            "promptshield": promptshield_records,
        },
        "validation_saved_scores": {
            "morgott": validation_morgott_scores,
            "promptshield": validation_promptshield_scores,
        },
        "validation_saved_logits": validation_logits,
        "validation_labels": {
            "morgott": validation_morgott_labels,
            "promptshield": validation_promptshield_labels,
        },
        "calibration_records": calibration_records,
        "causal_paths": causal_paths,
        "causal_hashes": causal_hashes,
    }


def _fit_collision_checker(run: dict) -> tuple[CollisionChecker, list, bool]:
    report = run["report"]
    fitted = _fitted_domains(run)
    if not run["full"]:
        condition = run["result"].get("condition")
        second = "m2" if condition == "control" else "promptshield"
        records = [
            *_load_generic_spec(report["outputs"]["m1"]),
            *_load_generic_spec(report["outputs"][second]),
        ]
        return CollisionChecker.from_records(records), [], True

    canonical_spec = report["outputs"]["morgott_train_index"]
    database_path = _artifact_path(canonical_spec)
    _verify_recorded_hash(
        canonical_spec["sha256"],
        database_path,
        name="canonical training index",
    )
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
    )
    fitted_records = []
    if fitted["promptshield"]:
        fitted_records.extend(_load_generic_spec(report["outputs"]["promptshield"]))
    if fitted["matched_pairs"]:
        fitted_records.extend(_load_generic_spec(report["outputs"]["matched_pairs"]))
    checker = CollisionChecker.from_records(fitted_records)
    checker.connection = connection
    return checker, [connection], not fitted["matched_pairs"]


def _load_model(run: dict):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    result = run["result"]
    tokenizer = AutoTokenizer.from_pretrained(
        result["model_id"],
        revision=result["model_revision"],
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    encoder = AutoModel.from_pretrained(
        result["model_id"],
        revision=result["model_revision"],
        attn_implementation=result["attention_implementation"],
        dtype=torch.bfloat16,
    )
    if run["adaptation"] == "lora":
        from peft import PeftModel, get_peft_model_state_dict

        encoder = PeftModel.from_pretrained(
            encoder,
            run["adapter_path"],
            is_trainable=False,
        )
        loaded_modules = sorted(
            name
            for name, module in encoder.named_modules()
            if hasattr(module, "lora_A")
        )
        loaded_parameters = sum(
            value.numel() for value in get_peft_model_state_dict(encoder).values()
        )
        if (
            loaded_modules != sorted(result["lora"]["targeted_modules"])
            or loaded_parameters != result["lora"]["adapter_parameters"]
        ):
            raise ValueError("loaded LoRA adapter identity does not match the run")
    encoder = encoder.to("cuda")
    encoder.eval()
    encoder.gradient_checkpointing_disable()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    head = new_head(encoder.config.hidden_size, run["result"]["seed"]).to("cuda")
    head.load_state_dict(load_file(str(run["head_path"])), strict=True)
    head.eval()
    return encoder, tokenizer, head


def _verify_validation_against_head(
    encoder,
    tokenizer,
    head,
    run: dict,
) -> dict:
    rescored_logits = {}
    rescored_scores = {}
    score_deltas = {}
    logit_deltas = {}
    for name in ("morgott", "promptshield"):
        rescored_logits[name] = _score_validation_records(
            encoder,
            tokenizer,
            head,
            run["validation_records"][name],
            max_tokens=run["result"]["max_tokens"],
            token_budget=run["result"]["token_budget"],
            record_chunk=run["result"]["validation_feature_record_chunk"],
            prediction_batch_size=run["result"]["validation_prediction_batch_size"],
            cache_identity=run.get("evaluation_feature_cache_identity"),
        )
        rescored_scores[name] = _scores(rescored_logits[name])
        score_delta = float(
            np.max(np.abs(rescored_scores[name] - run["validation_saved_scores"][name]))
        )
        if score_delta > 1e-5:
            raise ValueError(
                f"{name} validation scores do not match the recorded head: "
                f"{score_delta}"
            )
        score_deltas[name] = score_delta
        if run.get("adaptation") == "lora":
            logit_delta = float(
                np.max(
                    np.abs(rescored_logits[name] - run["validation_saved_logits"][name])
                )
            )
            if logit_delta > 1e-5:
                raise ValueError(
                    f"{name} validation logits do not match the recorded head: "
                    f"{logit_delta}"
                )
            logit_deltas[name] = logit_delta
    run["validation"] = {
        name: binary_metrics(
            rescored_scores[name],
            run["validation_labels"][name],
            {},
        )
        for name in ("morgott", "promptshield")
    }
    return {
        "maximum_absolute_score_delta": score_deltas,
        "maximum_absolute_logit_delta": logit_deltas or None,
        "tolerance": 1e-5,
    }


def _score_validation_records(
    encoder,
    tokenizer,
    head,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
    record_chunk: int,
    prediction_batch_size: int,
    cache_identity: dict | None = None,
) -> np.ndarray:
    if record_chunk < 1 or prediction_batch_size < 1:
        raise ValueError("validation feature and prediction batches must be positive")
    features = _extract_features_with_cache(
        encoder,
        tokenizer,
        records,
        max_tokens=max_tokens,
        token_budget=token_budget,
        record_chunk=record_chunk,
        cache_identity=cache_identity,
    )
    return predict_logits(
        head,
        features,
        batch_size=prediction_batch_size,
    )


def _extract_features_with_cache(
    encoder,
    tokenizer,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
    record_chunk: int,
    cache_identity: dict | None,
):
    if cache_identity is None:
        return extract_features(
            encoder,
            tokenizer,
            records,
            max_tokens=max_tokens,
            token_budget=token_budget,
            record_chunk=record_chunk,
        )

    import torch

    text_digest = hashlib.sha256()
    for record in records:
        encoded = record["text"].encode()
        text_digest.update(len(encoded).to_bytes(8, "big"))
        text_digest.update(encoded)
    spec = {
        "schema_version": 1,
        "cache_identity": cache_identity,
        "max_tokens": max_tokens,
        "token_budget": token_budget,
        "record_chunk": record_chunk,
        "rows": len(records),
        "feature_width": encoder.config.hidden_size * 3,
        "ordered_raw_text_sha256": text_digest.hexdigest(),
    }
    key = _stable_json_sha256(spec)
    parent = EVALUATION_FEATURE_CACHE / key[:2]
    directory = parent / key
    data_path = directory / "features.npy"
    report_path = directory / "report.json"

    def load():
        if not data_path.is_file() or not report_path.is_file():
            raise ValueError(f"incomplete evaluation feature cache: {directory}")
        report = json.loads(report_path.read_text())
        if report.get("spec") != spec:
            raise ValueError("evaluation feature cache spec mismatch")
        if file_sha256(data_path) != report.get("data_sha256"):
            raise ValueError("evaluation feature cache data digest mismatch")
        stored = np.load(data_path, allow_pickle=False)
        if stored.dtype != np.uint16 or list(stored.shape) != [
            len(records),
            encoder.config.hidden_size * 3,
        ]:
            raise ValueError("evaluation feature cache array contract mismatch")
        return torch.from_numpy(stored.copy()).view(torch.bfloat16)

    if directory.exists():
        return load()

    features = extract_features(
        encoder,
        tokenizer,
        records,
        max_tokens=max_tokens,
        token_budget=token_budget,
        record_chunk=record_chunk,
    )
    if (
        features.device.type != "cpu"
        or features.dtype != torch.bfloat16
        or list(features.shape) != [len(records), encoder.config.hidden_size * 3]
    ):
        raise ValueError("evaluation features do not match the cache contract")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=parent))
    try:
        temporary_data = temporary / data_path.name
        stored = features.detach().contiguous().view(torch.uint16).numpy()
        np.save(temporary_data, stored, allow_pickle=False)
        (temporary / report_path.name).write_text(
            json.dumps(
                {
                    "spec": spec,
                    "data_sha256": file_sha256(temporary_data),
                },
                indent=2,
            )
            + "\n"
        )
        try:
            os.replace(temporary, directory)
        except OSError:
            if not directory.exists():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return load()


def _score_records(
    encoder,
    tokenizer,
    head,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
    record_chunk: int = 512,
    cache_identity: dict | None = None,
) -> np.ndarray:
    if record_chunk < 1:
        raise ValueError("feature record chunk must be positive")
    chunks = []
    for start in range(0, len(records), record_chunk):
        features = _extract_features_with_cache(
            encoder,
            tokenizer,
            records[start : start + record_chunk],
            max_tokens=max_tokens,
            token_budget=token_budget,
            record_chunk=record_chunk,
            cache_identity=cache_identity,
        )
        chunks.append(predict_logits(head, features))
    return _scores(np.concatenate(chunks))


def _identity_update(digest, row: dict) -> None:
    digest.update(str(row["id"]).encode())
    digest.update(b"\0")
    digest.update(
        str(row.get("normalized_text_sha256") or text_hash(row["text"])).encode()
    )
    digest.update(b"\n")


def _canonical_dev_spec(run: dict) -> dict:
    spec = run["report"]["inputs"].get("routing_views", {}).get("dev_test")
    if spec is not None:
        return spec
    manifest = json.loads((REPO_ROOT / "data/manifest.json").read_text())
    manifest_spec = manifest["routing_views"]["dev_test"]
    return {
        **manifest_spec,
        "path": str(
            (REPO_ROOT / "data" / manifest_spec["path"]).relative_to(REPO_ROOT)
        ),
    }


def _score_canonical_dev(
    encoder,
    tokenizer,
    head,
    checker: CollisionChecker,
    run: dict,
    spec: dict,
) -> dict:
    path = _artifact_path(spec)
    _verify_recorded_hash(spec["sha256"], path, name="canonical dev-test")
    score_chunks = []
    labels = []
    sources = []
    channels = []
    unseen = []
    finance = []
    rows = []
    identity = hashlib.sha256()

    def flush() -> None:
        if not rows:
            return
        score_chunks.append(
            _score_records(
                encoder,
                tokenizer,
                head,
                rows,
                max_tokens=run["result"]["max_tokens"],
                token_budget=EVALUATION_TOKEN_BUDGET,
                cache_identity=run.get("evaluation_feature_cache_identity"),
            )
        )
        unseen.extend(checker.unseen_mask([row["text"] for row in rows]).tolist())
        rows.clear()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not canonical_is_eligible(row):
                continue
            _identity_update(identity, row)
            rows.append(row)
            labels.append(row["injection_label"])
            sources.append(row["source"])
            channels.append(row["input_channel"])
            finance.append(
                row["input_channel"] == "direct_user"
                and row["source"] in FINANCE_SOURCES
                and row["injection_label"] == 0
            )
            if len(rows) == 512:
                flush()
    flush()
    scores = np.concatenate(score_chunks)
    labels_array = np.asarray(labels, dtype=np.int64)
    unseen_array = np.asarray(unseen, dtype=bool)
    finance_array = np.asarray(finance, dtype=bool)
    if not (
        len(scores)
        == len(labels_array)
        == len(sources)
        == len(channels)
        == len(unseen_array)
    ):
        raise ValueError("canonical dev-test scoring lost row alignment")
    finance_counts = Counter(
        source for source, selected in zip(sources, finance, strict=True) if selected
    )
    if set(finance_counts) != FINANCE_SOURCES or any(
        count < 1 for count in finance_counts.values()
    ):
        raise ValueError(f"finance population mismatch: {dict(finance_counts)}")
    return {
        "scores": scores,
        "labels": labels_array,
        "sources": np.asarray(sources),
        "channels": np.asarray(channels),
        "unseen": unseen_array,
        "finance": finance_array,
        "ordered_identity_sha256": identity.hexdigest(),
    }


def _external_records(
    path: Path,
    expected_sha256: str,
    *,
    text_field: str,
    expected: dict,
) -> list[dict]:
    rows = _read_jsonl(path, expected_sha256)
    labels = Counter(row["label"] for row in rows)
    if (
        len(rows) != expected["rows"]
        or labels[1] != expected["positive"]
        or labels[0] != expected["rows"] - expected["positive"]
    ):
        raise ValueError(f"external population mismatch for {path}: {dict(labels)}")
    return [
        {
            "id": row["id"],
            "text": row[text_field],
            "generic_label": row["label"],
            "source": row.get("source"),
        }
        for row in rows
    ]


def _slice_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    if set(labels.tolist()) == {0, 1}:
        return binary_metrics(scores, labels, thresholds)
    if set(labels.tolist()) == {0}:
        return negative_metrics(scores, thresholds)
    return {
        "rows": len(labels),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
        "validation_threshold_applied": {
            name: applied_point(scores, labels, threshold)
            for name, threshold in thresholds.items()
        },
        "descriptive_same_test": None,
    }


def _external_evaluation(
    encoder,
    tokenizer,
    head,
    checker: CollisionChecker,
    run: dict,
    *,
    name: str,
    records: list[dict],
) -> tuple[dict, np.ndarray, np.ndarray]:
    scores = _score_records(
        encoder,
        tokenizer,
        head,
        records,
        max_tokens=run["result"]["max_tokens"],
        token_budget=EVALUATION_TOKEN_BUDGET,
        cache_identity=run.get("evaluation_feature_cache_identity"),
    )
    labels = np.asarray(
        [row["generic_label"] for row in records],
        dtype=np.int64,
    )
    unseen = checker.unseen_mask([row["text"] for row in records])
    if not unseen.any() or set(labels[unseen].tolist()) != {0, 1}:
        raise ValueError(f"{name} collision-masked population lacks both classes")
    return (
        {
            "full": binary_metrics(scores, labels, run["thresholds"]),
            "fitting_collision_definition": (
                "normalized or strict exact match to a fitted row"
            ),
            "fitting_collision_rows": int((~unseen).sum()),
            "collision_masked": binary_metrics(
                scores[unseen],
                labels[unseen],
                run["thresholds"],
            ),
        },
        scores,
        unseen,
    )


def _pair_evaluation(
    encoder,
    tokenizer,
    head,
    checker: CollisionChecker,
    run: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    if _fitted_domains(run)["matched_pairs"]:
        return (
            {
                "scored": False,
                "reason": (
                    "the selected objective fits generated pairs from every "
                    "generator lab; no genuine pair holdout exists"
                ),
            },
            {},
        )
    pairs = _read_jsonl(PAIR_ARCHIVE, PAIR_ARCHIVE_SHA256)
    records = [
        {"id": f"pair:{index}:{role}", "text": pair[role]}
        for index, pair in enumerate(pairs)
        for role in ("benign", "attack")
    ]
    scores = _score_records(
        encoder,
        tokenizer,
        head,
        records,
        max_tokens=run["result"]["max_tokens"],
        token_budget=EVALUATION_TOKEN_BUDGET,
        cache_identity=run.get("evaluation_feature_cache_identity"),
    ).reshape(-1, 2)
    unseen_rows = checker.unseen_mask([row["text"] for row in records]).reshape(-1, 2)
    unseen_pairs = unseen_rows.all(axis=1)
    if not unseen_pairs.any():
        raise ValueError("no genuinely unfitted generated pairs remain")
    benign = scores[unseen_pairs, 0]
    attack = scores[unseen_pairs, 1]
    by_lab = {}
    labs = np.asarray([pair["generator_lab"] for pair in pairs])
    for lab in sorted(set(labs[unseen_pairs].tolist())):
        selected = unseen_pairs & (labs == lab)
        by_lab[lab] = pair_metrics(
            scores[selected, 0],
            scores[selected, 1],
            run["thresholds"],
        )
    return (
        {
            "scored": True,
            "independence": (
                "not fitted by this objective; weak-synthesis diagnostic, "
                "not an external benchmark"
            ),
            "archive_pairs": len(pairs),
            "fitting_collision_pairs": int((~unseen_pairs).sum()),
            "genuinely_unfitted": pair_metrics(
                benign,
                attack,
                run["thresholds"],
            ),
            "by_generator_lab": by_lab,
        },
        {
            "matched_pair_scores": scores,
            "matched_pair_unseen": unseen_pairs,
        },
    )


def _save_array(
    directory: Path,
    published_directory: Path,
    name: str,
    values: np.ndarray,
) -> dict:
    path = directory / f"{name}.npy"
    np.save(path, values)
    return {
        "path": str((published_directory / path.name).relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def _publish_verified_evaluation(
    temporary: Path,
    output: Path,
    run: dict,
    *,
    source_paths: dict[str, Path],
    source_hashes: dict[str, str],
) -> None:
    _verify_source_hashes(source_paths, source_hashes)
    _verify_source_hashes(run["causal_paths"], run["causal_hashes"])
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    snapshot_paths = {
        **_evaluator_source_paths(full=True),
        **_evaluator_source_paths(full=False, adaptation="lora"),
    }
    snapshot_hashes = {name: file_sha256(path) for name, path in snapshot_paths.items()}
    run = discover_run(Path(args.run))
    source_paths = _evaluator_source_paths(
        full=run["full"],
        adaptation=run["adaptation"],
    )
    source_hashes = {name: snapshot_hashes[name] for name in source_paths}
    if args.preflight_only:
        print(f"preflight passed: {run['directory']}; adaptation {run['adaptation']}")
        return 0
    checker, resources, pairs_are_held_out = _fit_collision_checker(run)
    try:
        canonical_dev_spec = _canonical_dev_spec(run)
        _add_scored_input_provenance(
            run,
            canonical_dev_spec,
            pairs_are_held_out=pairs_are_held_out,
        )
        encoder, tokenizer, head = _load_model(run)
        if run["adaptation"] == "frozen":
            import torch

            device = torch.cuda.get_device_properties(torch.cuda.current_device())
            run["evaluation_feature_cache_identity"] = {
                "schema_version": 1,
                "model_id": run["result"]["model_id"],
                "model_revision": run["result"]["model_revision"],
                "attention_implementation": run["result"]["attention_implementation"],
                "normalization": run["result"]["normalization"],
                "evaluator_sha256": source_hashes["evaluator"],
                "feature_extractor_sha256": source_hashes["training_head_helper"],
                "strict_normalizer_sha256": source_hashes["strict_normalizer"],
                "torch": importlib.metadata.version("torch"),
                "transformers": importlib.metadata.version("transformers"),
                "cuda": torch.version.cuda,
                "device": {
                    "name": device.name,
                    "compute_capability": [device.major, device.minor],
                },
            }
        else:
            run["evaluation_feature_cache_identity"] = None
        validation_roundtrip = _verify_validation_against_head(
            encoder,
            tokenizer,
            head,
            run,
        )
        calibration_records = run["calibration_records"]
        calibration_scores = _score_records(
            encoder,
            tokenizer,
            head,
            calibration_records,
            max_tokens=run["result"]["max_tokens"],
            token_budget=run["result"]["token_budget"],
            record_chunk=VALIDATION_FEATURE_RECORD_CHUNK,
            cache_identity=run["evaluation_feature_cache_identity"],
        )
        calibration_labels = np.asarray(
            [row["generic_label"] for row in calibration_records],
            dtype=np.int64,
        )
        calibration_channels = np.asarray(
            [row["channel"] for row in calibration_records]
        )
        calibration_sources = np.asarray([row["source"] for row in calibration_records])
        thresholds, threshold_evidence = select_calibration_thresholds(
            calibration_scores,
            calibration_labels,
            calibration_records,
        )
        run["thresholds"] = thresholds
        run["threshold_evidence"] = threshold_evidence
        canonical = _score_canonical_dev(
            encoder,
            tokenizer,
            head,
            checker,
            run,
            canonical_dev_spec,
        )
        promptshield_records = _external_records(
            PROMPTSHIELD_TEST,
            PROMPTSHIELD_TEST_SHA256,
            text_field="prompt",
            expected=EXPECTED_EXTERNAL["promptshield"],
        )
        sep_records = _external_records(
            SEP,
            SEP_SHA256,
            text_field="text",
            expected=EXPECTED_EXTERNAL["sep"],
        )
        promptshield, promptshield_scores, promptshield_unseen = _external_evaluation(
            encoder,
            tokenizer,
            head,
            checker,
            run,
            name="PromptShield",
            records=promptshield_records,
        )
        sep, sep_scores, sep_unseen = _external_evaluation(
            encoder,
            tokenizer,
            head,
            checker,
            run,
            name="SEP",
            records=sep_records,
        )
        pairs, pair_arrays = _pair_evaluation(
            encoder,
            tokenizer,
            head,
            checker,
            run,
        )
    finally:
        for resource in resources:
            resource.close()

    thresholds = run["thresholds"]
    promptshield["evaluation_scope"] = _promptshield_evaluation_scope(run)
    sep["evaluation_scope"] = "cross-distribution transfer development test"
    calibration_result = {
        "evaluation_scope": (
            "canonical validation calibration role disjoint from checkpoint "
            "selection; component-max negatives select thresholds and positives "
            "report row recall"
        ),
        "full": binary_metrics(
            calibration_scores,
            calibration_labels,
            thresholds,
        ),
        "by_channel": {
            channel: _slice_metrics(
                calibration_scores[calibration_channels == channel],
                calibration_labels[calibration_channels == channel],
                thresholds,
            )
            for channel in sorted(set(calibration_channels.tolist()))
        },
        "by_source": {
            source: _slice_metrics(
                calibration_scores[calibration_sources == source],
                calibration_labels[calibration_sources == source],
                thresholds,
            )
            for source in sorted(set(calibration_sources.tolist()))
        },
    }
    canonical_result = {
        "evaluation_scope": (
            "group-held-out repeated canonical development comparison; "
            "source families recur in fitting"
        ),
        "full": binary_metrics(
            canonical["scores"],
            canonical["labels"],
            thresholds,
        ),
        "fitting_collision_rows": int((~canonical["unseen"]).sum()),
        "collision_masked": binary_metrics(
            canonical["scores"][canonical["unseen"]],
            canonical["labels"][canonical["unseen"]],
            thresholds,
        ),
        "by_channel": {
            channel: _slice_metrics(
                canonical["scores"][canonical["channels"] == channel],
                canonical["labels"][canonical["channels"] == channel],
                thresholds,
            )
            for channel in sorted(set(canonical["channels"].tolist()))
        },
        "by_source": {
            source: _slice_metrics(
                canonical["scores"][canonical["sources"] == source],
                canonical["labels"][canonical["sources"] == source],
                thresholds,
            )
            for source in sorted(set(canonical["sources"].tolist()))
        },
        "ordered_identity_sha256": canonical["ordered_identity_sha256"],
    }
    finance_scores = canonical["scores"][canonical["finance"]]
    finance_sources = canonical["sources"][canonical["finance"]]
    finance_result = {
        "independence": (
            "group-held-out canonical dev-test negatives; source families also "
            "occur in fitting"
        ),
        "label_basis": "canonical injection_label == 0",
        "source_counts": {
            source: int((finance_sources == source).sum())
            for source in sorted(FINANCE_SOURCES)
        },
        "all": negative_metrics(finance_scores, thresholds),
        "by_source": {
            source: negative_metrics(
                finance_scores[finance_sources == source],
                thresholds,
            )
            for source in sorted(FINANCE_SOURCES)
        },
    }

    output = run["directory"] / EVALUATION_DIRECTORY
    if output.exists():
        raise FileExistsError(f"refusing to replace existing evaluation: {output}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{EVALUATION_DIRECTORY}.", dir=run["directory"])
    )
    try:
        arrays = {
            "canonical_calibration_scores": _save_array(
                temporary,
                output,
                "canonical_calibration_scores",
                calibration_scores,
            ),
            "canonical_calibration_labels": _save_array(
                temporary,
                output,
                "canonical_calibration_labels",
                calibration_labels,
            ),
            "canonical_dev_scores": _save_array(
                temporary,
                output,
                "canonical_dev_scores",
                canonical["scores"],
            ),
            "canonical_dev_labels": _save_array(
                temporary,
                output,
                "canonical_dev_labels",
                canonical["labels"],
            ),
            "canonical_dev_unseen": _save_array(
                temporary,
                output,
                "canonical_dev_unseen",
                canonical["unseen"],
            ),
            "promptshield_scores": _save_array(
                temporary,
                output,
                "promptshield_scores",
                promptshield_scores,
            ),
            "promptshield_unseen": _save_array(
                temporary,
                output,
                "promptshield_unseen",
                promptshield_unseen,
            ),
            "sep_scores": _save_array(temporary, output, "sep_scores", sep_scores),
            "sep_unseen": _save_array(temporary, output, "sep_unseen", sep_unseen),
        }
        arrays.update(
            {
                name: _save_array(temporary, output, name, values)
                for name, values in pair_arrays.items()
            }
        )
        evaluation = {
            "schema_version": 2,
            "purpose": (
                "fail-closed evaluation of a generic instruction-subversion head"
            ),
            "run": str(run["directory"].relative_to(REPO_ROOT)),
            "recipe": "full_combined" if run["full"] else "update_matched",
            "adaptation": run["adaptation"],
            "condition": run["result"].get("condition"),
            "objective": run["result"].get("objective"),
            "objective_spec_sha256": run["result"]["provenance"].get(
                "objective_spec_sha256"
            ),
            "pair_ranking_weight": run["result"]["training"].get(
                "pair_ranking_weight",
                0.0,
            ),
            "model_id": run["result"]["model_id"],
            "model_revision": run["result"]["model_revision"],
            "attention_implementation": run["result"]["attention_implementation"],
            "normalization": run["result"]["normalization"],
            "max_tokens": run["result"]["max_tokens"],
            "validation_feature_record_chunk": run["result"][
                "validation_feature_record_chunk"
            ],
            "validation_prediction_batch_size": run["result"][
                "validation_prediction_batch_size"
            ],
            "evaluation_token_budget": EVALUATION_TOKEN_BUDGET,
            "evaluation_feature_cache": {
                "enabled": run["evaluation_feature_cache_identity"] is not None,
                "identity": run["evaluation_feature_cache_identity"],
                "integrity": (
                    "content-addressed exact raw-text order and verified array digest"
                ),
                "causal_input": False,
            },
            "head_sha256": run["result"]["artifact"]["head_sha256"],
            "adapter_files_sha256": (
                run["result"]["artifact"]["adapter_files"]
                if run["adaptation"] == "lora"
                else None
            ),
            **run["checkpoint_provenance"],
            "operating_fprs": list(OPERATING_FPRS),
            "operating_target_unit": (
                "lineage-and-near validation component false-alarm probability "
                "within trusted channel"
            ),
            "thresholds": {
                "source": "canonical calibration role only",
                "selection": (
                    "maximum trusted-channel component-max tie-aware order "
                    "statistics with 97.5% per-channel one-sided Clopper-Pearson "
                    "upper bounds and 95% Bonferroni family confidence"
                ),
                "pooled_negative_role": "empirical diagnostic only",
                "promptshield_validation_used": False,
                "selected": run["threshold_evidence"],
            },
            "checkpoint_selection": run["validation"],
            "calibration": calibration_result,
            "validation_head_roundtrip": validation_roundtrip,
            "canonical_dev_test": canonical_result,
            "promptshield_test": promptshield,
            "sep": sep,
            "real_finance_negatives": finance_result,
            "generated_pairs": {
                **pairs,
                "eligible_only_if_genuinely_held_out": True,
                "run_has_genuine_pair_holdout": pairs_are_held_out,
            },
            "training_provenance_only": {
                "canonical_feature_cache": run["feature_cache"],
            },
            "arrays": arrays,
            "input_sha256": {
                "run_result": file_sha256(run["result_path"]),
                "selection_report": file_sha256(run["report_path"]),
                "canonical_calibration": (
                    run["report"]["validation"]["morgott_calibration"]["sha256"]
                    if run["full"]
                    else run["report"]["outputs"]["validation_morgott_calibration"][
                        "sha256"
                    ]
                ),
                "head": file_sha256(run["head_path"]),
                "adapter_files": (
                    {
                        name: file_sha256(run["adapter_path"] / name)
                        for name in sorted(run["result"]["artifact"]["adapter_files"])
                    }
                    if run["adaptation"] == "lora"
                    else None
                ),
                "canonical_dev_test": canonical_dev_spec["sha256"],
                "promptshield_test": PROMPTSHIELD_TEST_SHA256,
                "sep": SEP_SHA256,
                "matched_pairs": PAIR_ARCHIVE_SHA256 if pairs_are_held_out else None,
                **source_hashes,
                "calibration_threshold_helper": source_hashes["evaluator"],
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn")
            },
            "interpretation": [
                "Only canonical calibration negatives select applied thresholds.",
                "Threshold confidence targets component false-alarm probability "
                "within each trusted channel; applied row FPR and TPR remain "
                "separate empirical measurements.",
                "PromptShield validation is checkpoint-selection evidence and "
                "never selects an operating threshold.",
                "A target is unavailable when either trusted channel lacks enough "
                "calibration components for its Bonferroni-adjusted bound.",
                "Validation components and recurring source families are not IID "
                "or sampled from a deployment distribution, so these are "
                "development bounds rather than production guarantees.",
                "Every same-test threshold is descriptive and must not be "
                "presented as deployable performance.",
                "PromptShield and SEP are already-open development sets, not "
                "prospective final tests.",
                "The learned score is advisory and is not approved for blocking.",
            ],
        }
        (temporary / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2) + "\n"
        )
        _publish_verified_evaluation(
            temporary,
            output,
            run,
            source_paths=source_paths,
            source_hashes=source_hashes,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"wrote {output}; PromptShield AUC "
        f"{promptshield['full']['roc_auc']:.4f}; SEP AUC "
        f"{sep['full']['roc_auc']:.4f}"
    )
    print(
        "Only canonical calibration thresholds are applied. Same-test thresholds "
        "are descriptive."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
