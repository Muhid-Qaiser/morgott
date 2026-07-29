#!/usr/bin/env python3
"""Summarize the frozen OpenRouter downstream evaluation without raw text."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts" / "openrouter_downstream_eval"
SELECTED_CONFIGURATIONS = (
    "safeguard_default",
    "qwen_default",
    "deepseek_off",
)
ABLATION_PAIRS = (
    ("safeguard_default", "safeguard_low"),
    ("qwen_off", "qwen_default"),
    ("deepseek_off", "deepseek_high"),
    ("deepseek_pro_off", "deepseek_pro_high"),
    ("deepseek_pro_high", "deepseek_pro_high_strict"),
    ("deepseek_off", "deepseek_pro_off"),
    ("deepseek_off", "deepseek_pro_high_strict"),
    ("deepseek_off", "deepseek_flash_fp8_off"),
    ("deepseek_pro_off", "deepseek_pro_fp8_off"),
    ("deepseek_pro_high_strict", "deepseek_pro_fp8_high_strict"),
    ("deepseek_flash_fp8_off", "deepseek_pro_fp8_off"),
    ("deepseek_flash_fp8_off", "deepseek_pro_fp8_high_strict"),
)
RECALL_TARGETS = (0.95, 0.99, 0.995, 1.0)
ENCODER_REPORTS = {
    "mmbert_frozen_full_s42": (
        ROOT / "artifacts" / "models" / "mmbert-frozen-s42" / "evaluation.json"
    ),
    "mmbert_lora_partial_s42": (
        ROOT / "artifacts" / "models" / "mmbert-lora-s42" / "evaluation.json"
    ),
}
SETUP_FAILURE_FILES = (
    "pro_alias_route_failures.jsonl",
    "pro_dated_native_route_failures.jsonl",
    "fp8_baidu_route_failures.jsonl",
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    stored_path = path if path.exists() else Path(f"{path}.gz")
    opener = gzip.open if stored_path.suffix == ".gz" else Path.open
    with opener(stored_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {stored_path}")
    return rows


def _file_sha256(path: Path) -> str:
    stored_path = path if path.exists() else Path(f"{path}.gz")
    opener = gzip.open if stored_path.suffix == ".gz" else Path.open
    digest = hashlib.sha256()
    with opener(stored_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    if labels.shape != predictions.shape:
        raise ValueError("label and prediction shapes differ")
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    tn = int(np.sum(~predictions & negatives))
    return {
        "rows": int(len(labels)),
        "positives": int(np.sum(positives)),
        "negatives": int(np.sum(negatives)),
        "predicted_positive": int(np.sum(predictions)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": _ratio(tp + tn, len(labels)),
        "recall": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
        "precision": _ratio(tp, tp + fp),
        "specificity": _ratio(tn, tn + fp),
    }


def _score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    result = _binary_metrics(labels, scores >= 0.5)
    result["threshold"] = 0.5
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["average_precision"] = float(average_precision_score(labels, scores))
    else:
        result["auroc"] = None
        result["average_precision"] = None
    return result


def _runtime_metrics(records: list[dict]) -> dict:
    seconds = np.asarray(
        [
            float(record["client_seconds"])
            for record in records
            if isinstance(record.get("client_seconds"), int | float)
        ],
        dtype=np.float64,
    )

    def mean_field(name: str) -> float | None:
        values = [
            float(record[name])
            for record in records
            if isinstance(record.get(name), int | float)
        ]
        return float(np.mean(values)) if values else None

    return {
        "cost_usd": sum(
            float(record.get("cost") or 0)
            for record in records
            if isinstance(record.get("cost"), int | float)
        ),
        "latency_seconds_mean": float(np.mean(seconds)) if len(seconds) else None,
        "latency_seconds_p50": (
            float(np.quantile(seconds, 0.5)) if len(seconds) else None
        ),
        "latency_seconds_p95": (
            float(np.quantile(seconds, 0.95)) if len(seconds) else None
        ),
        "prompt_tokens_mean": mean_field("prompt_tokens"),
        "completion_tokens_mean": mean_field("completion_tokens"),
        "reasoning_tokens_mean": mean_field("reasoning_tokens"),
        "attempts_mean": mean_field("attempts"),
    }


def _record_map(records: list[dict]) -> dict[str, dict]:
    result = {}
    for record in records:
        panel_id = record["panel_id"]
        if panel_id in result:
            raise ValueError(f"duplicate result for {panel_id}")
        result[panel_id] = record
    return result


def _llm_predictions(
    panel: list[dict],
    records: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(
        [records[row["panel_id"]]["status"] == "ok" for row in panel],
        dtype=bool,
    )
    fail_safe = np.asarray(
        [
            (
                bool(records[row["panel_id"]]["subversion"])
                if records[row["panel_id"]]["status"] == "ok"
                else True
            )
            for row in panel
        ],
        dtype=bool,
    )
    return valid, fail_safe


def _llm_group_metrics(
    panel: list[dict],
    records: dict[str, dict],
    indices: np.ndarray,
) -> dict:
    labels = np.asarray([panel[index]["label"] for index in indices], dtype=np.int8)
    selected = [records[panel[index]["panel_id"]] for index in indices]
    valid = np.asarray([record["status"] == "ok" for record in selected], dtype=bool)
    valid_predictions = np.asarray(
        [bool(record["subversion"]) for record in selected if record["status"] == "ok"],
        dtype=bool,
    )
    fail_safe_predictions = np.asarray(
        [
            bool(record["subversion"]) if record["status"] == "ok" else True
            for record in selected
        ],
        dtype=bool,
    )
    return {
        "statuses": dict(
            sorted(Counter(record["status"] for record in selected).items())
        ),
        "valid_output_rate": _ratio(int(np.sum(valid)), len(valid)),
        "valid_outputs_only": _binary_metrics(labels[valid], valid_predictions),
        "failure_routes_to_review": _binary_metrics(labels, fail_safe_predictions),
    }


def _llm_summary(panel: list[dict], records: list[dict]) -> dict:
    mapped = _record_map(records)
    if set(mapped) != {row["panel_id"] for row in panel}:
        raise ValueError("LLM results do not cover the supplied panel exactly")
    all_indices = np.arange(len(panel), dtype=np.int64)
    datasets = defaultdict(list)
    channels = defaultdict(list)
    sources = defaultdict(list)
    tags = defaultdict(list)
    for index, row in enumerate(panel):
        datasets[row["dataset"]].append(index)
        channels[row["input_channel"]].append(index)
        if row["dataset"] == "canonical":
            sources[row["source"]].append(index)
            for tag in row.get("security_tags", []):
                tags[tag].append(index)
    harmful_indices = [
        index
        for index, row in enumerate(panel)
        if row["dataset"] == "canonical"
        and (
            ("harmful_intent" in row.get("security_tags", []))
            ^ ("benign" in row.get("security_tags", []))
        )
    ]
    harmful_records = [mapped[panel[index]["panel_id"]] for index in harmful_indices]
    harmful_valid = np.asarray(
        [record["status"] == "ok" for record in harmful_records],
        dtype=bool,
    )
    harmful_labels = np.asarray(
        [
            "harmful_intent" in panel[index].get("security_tags", [])
            for index in harmful_indices
        ],
        dtype=np.int8,
    )
    harmful_predictions = np.asarray(
        [
            bool(record["harmful_request"])
            for record in harmful_records
            if record["status"] == "ok"
        ],
        dtype=bool,
    )
    harmful_non_injection_indices = [
        index
        for index, row in enumerate(panel)
        if "harmful_non_injection" in row.get("security_tags", [])
    ]
    harmful_non_injection_records = [
        mapped[panel[index]["panel_id"]] for index in harmful_non_injection_indices
    ]
    harmful_non_injection_valid = [
        record for record in harmful_non_injection_records if record["status"] == "ok"
    ]
    return {
        "overall": _llm_group_metrics(panel, mapped, all_indices),
        "runtime": _runtime_metrics(records),
        "datasets": {
            key: _llm_group_metrics(panel, mapped, np.asarray(value, dtype=np.int64))
            for key, value in sorted(datasets.items())
        },
        "channels": {
            key: _llm_group_metrics(panel, mapped, np.asarray(value, dtype=np.int64))
            for key, value in sorted(channels.items())
        },
        "canonical_sources": {
            key: _llm_group_metrics(panel, mapped, np.asarray(value, dtype=np.int64))
            for key, value in sorted(sources.items())
        },
        "canonical_security_tags": {
            key: _llm_group_metrics(panel, mapped, np.asarray(value, dtype=np.int64))
            for key, value in sorted(tags.items())
        },
        "harmful_request_masked": {
            "mask": "canonical harmful_intent positives versus canonical benign negatives",
            "eligible_rows": len(harmful_indices),
            "statuses": dict(
                sorted(Counter(record["status"] for record in harmful_records).items())
            ),
            "valid_outputs_only": _binary_metrics(
                harmful_labels[harmful_valid],
                harmful_predictions,
            ),
        },
        "harmful_non_injection": {
            "rows": len(harmful_non_injection_indices),
            "statuses": dict(
                sorted(
                    Counter(
                        record["status"] for record in harmful_non_injection_records
                    ).items()
                )
            ),
            "valid_outputs": len(harmful_non_injection_valid),
            "subversion_positive": sum(
                bool(record["subversion"]) for record in harmful_non_injection_valid
            ),
            "subversion_positive_rate": _ratio(
                sum(
                    bool(record["subversion"]) for record in harmful_non_injection_valid
                ),
                len(harmful_non_injection_valid),
            ),
            "harmful_request_positive": sum(
                bool(record["harmful_request"])
                for record in harmful_non_injection_valid
            ),
            "harmful_request_positive_rate": _ratio(
                sum(
                    bool(record["harmful_request"])
                    for record in harmful_non_injection_valid
                ),
                len(harmful_non_injection_valid),
            ),
        },
    }


def _threshold_for_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    target: float,
) -> float:
    positives = np.sort(scores[labels == 1])[::-1]
    if not len(positives) or not 0 < target <= 1:
        raise ValueError("invalid recall target")
    required = math.ceil(target * len(positives))
    return float(positives[required - 1])


def _cascade_metrics(
    panel: list[dict],
    scores: np.ndarray,
    threshold: float,
    records: dict[str, dict],
) -> dict:
    labels = np.asarray([row["label"] for row in panel], dtype=np.int8)
    escalated = scores >= threshold
    valid, llm_review = _llm_predictions(panel, records)
    predictions = escalated & llm_review
    seconds = np.asarray(
        [float(records[row["panel_id"]].get("client_seconds") or 0) for row in panel],
        dtype=np.float64,
    )
    costs = np.asarray(
        [float(records[row["panel_id"]].get("cost") or 0) for row in panel],
        dtype=np.float64,
    )
    positives = labels == 1
    negatives = labels == 0
    escalated_count = int(np.sum(escalated))
    return {
        "threshold": threshold,
        "first_stage": _binary_metrics(labels, escalated),
        "final": _binary_metrics(labels, predictions),
        "escalated_rows": escalated_count,
        "escalation_rate": _ratio(escalated_count, len(panel)),
        "auto_pass_rate": _ratio(int(np.sum(~escalated)), len(panel)),
        "attacks_missed_below_threshold": int(np.sum(positives & ~escalated)),
        "attacks_cleared_by_llm": int(
            np.sum(positives & escalated & valid & ~llm_review)
        ),
        "benign_escalated": int(np.sum(negatives & escalated)),
        "benign_cleared_by_llm": int(
            np.sum(negatives & escalated & valid & ~llm_review)
        ),
        "llm_operational_failures_in_escalated_zone": int(np.sum(escalated & ~valid)),
        "observed_llm_cost_usd": float(np.sum(costs[escalated])),
        "observed_llm_cost_usd_per_1000_inputs": (
            float(np.sum(costs[escalated]) * 1000 / len(panel))
        ),
        "mean_added_llm_seconds_per_input": float(
            np.sum(seconds[escalated]) / len(panel)
        ),
        "mean_llm_seconds_per_escalated_input": (
            float(np.mean(seconds[escalated])) if escalated_count else None
        ),
    }


def _three_zone_metrics(
    panel: list[dict],
    scores: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    records: dict[str, dict],
) -> dict:
    if low_threshold >= high_threshold:
        raise ValueError("three-zone low threshold must be below its high threshold")
    labels = np.asarray([row["label"] for row in panel], dtype=np.int8)
    auto_pass = scores < low_threshold
    llm_zone = (scores >= low_threshold) & (scores < high_threshold)
    retained_review = scores >= high_threshold
    valid, llm_review = _llm_predictions(panel, records)
    predictions = retained_review | (llm_zone & llm_review)
    seconds = np.asarray(
        [float(records[row["panel_id"]].get("client_seconds") or 0) for row in panel],
        dtype=np.float64,
    )
    costs = np.asarray(
        [float(records[row["panel_id"]].get("cost") or 0) for row in panel],
        dtype=np.float64,
    )
    positives = labels == 1
    llm_count = int(np.sum(llm_zone))
    return {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "first_stage": _binary_metrics(labels, ~auto_pass),
        "final": _binary_metrics(labels, predictions),
        "auto_pass_rows": int(np.sum(auto_pass)),
        "auto_pass_rate": _ratio(int(np.sum(auto_pass)), len(panel)),
        "llm_zone_rows": llm_count,
        "llm_zone_rate": _ratio(llm_count, len(panel)),
        "retained_review_rows": int(np.sum(retained_review)),
        "retained_review_rate": _ratio(int(np.sum(retained_review)), len(panel)),
        "attacks_missed_below_low_threshold": int(np.sum(positives & auto_pass)),
        "attacks_cleared_by_llm_in_middle_zone": int(
            np.sum(positives & llm_zone & valid & ~llm_review)
        ),
        "llm_operational_failures_in_middle_zone": int(np.sum(llm_zone & ~valid)),
        "observed_llm_cost_usd": float(np.sum(costs[llm_zone])),
        "observed_llm_cost_usd_per_1000_inputs": float(
            np.sum(costs[llm_zone]) * 1000 / len(panel)
        ),
        "mean_added_llm_seconds_per_input": float(
            np.sum(seconds[llm_zone]) / len(panel)
        ),
        "mean_llm_seconds_per_middle_input": (
            float(np.mean(seconds[llm_zone])) if llm_count else None
        ),
    }


def _encoder_summary(
    panel: list[dict],
    primary_by_configuration: dict[str, list[dict]],
) -> dict:
    labels = np.asarray([row["label"] for row in panel], dtype=np.int8)
    datasets = defaultdict(list)
    for index, row in enumerate(panel):
        datasets[row["dataset"]].append(index)
    result = {}
    encoder_names = sorted(panel[0]["mmbert_scores"])
    for encoder in encoder_names:
        report = _load_json(ENCODER_REPORTS[encoder])
        validation_threshold = report["thresholds"]["selected"]["1.0000%"]
        if validation_threshold["status"] != "available":
            raise ValueError(
                f"canonical validation threshold is unavailable: {encoder}"
            )
        scores = np.asarray(
            [row["mmbert_scores"][encoder] for row in panel],
            dtype=np.float64,
        )
        panel_thresholds = {
            f"panel_recall_{target:.4f}": _threshold_for_recall(
                labels,
                scores,
                target,
            )
            for target in RECALL_TARGETS
        }
        thresholds = {
            "untouched_0.5": 0.5,
            "canonical_validation_1pct_component_fpr": float(
                validation_threshold["threshold"]
            ),
            **panel_thresholds,
        }
        result[encoder] = {
            "evaluation_report": {
                "path": str(ENCODER_REPORTS[encoder].relative_to(ROOT)),
                "sha256": _file_sha256(ENCODER_REPORTS[encoder]),
            },
            "standalone": _score_metrics(labels, scores),
            "datasets": {
                dataset: _score_metrics(
                    labels[np.asarray(indices, dtype=np.int64)],
                    scores[np.asarray(indices, dtype=np.int64)],
                )
                for dataset, indices in sorted(datasets.items())
            },
            "operating_thresholds": thresholds,
            "panel_recall_target_thresholds": panel_thresholds,
            "cascades": {
                threshold_name: {
                    configuration: _cascade_metrics(
                        panel,
                        scores,
                        threshold,
                        _record_map(primary_by_configuration[configuration]),
                    )
                    for configuration in SELECTED_CONFIGURATIONS
                }
                for threshold_name, threshold in thresholds.items()
            },
            "three_zone_diagnostic": {
                threshold_name: {
                    configuration: _three_zone_metrics(
                        panel,
                        scores,
                        low_threshold,
                        float(validation_threshold["threshold"]),
                        _record_map(primary_by_configuration[configuration]),
                    )
                    for configuration in SELECTED_CONFIGURATIONS
                }
                for threshold_name, low_threshold in panel_thresholds.items()
            },
        }
    return result


def _reasoning_ablation(panel: list[dict], records: list[dict]) -> dict:
    audit_ids = {record["panel_id"] for record in records}
    audit_panel = [row for row in panel if row["panel_id"] in audit_ids]
    by_configuration = defaultdict(list)
    for record in records:
        by_configuration[record["configuration"]].append(record)
    summaries = {
        configuration: _llm_summary(audit_panel, config_records)
        for configuration, config_records in sorted(by_configuration.items())
    }
    pairs = {}
    labels = {row["panel_id"]: row["label"] for row in audit_panel}
    for left, right in ABLATION_PAIRS:
        left_map = _record_map(by_configuration[left])
        right_map = _record_map(by_configuration[right])
        paired = [
            panel_id
            for panel_id in sorted(left_map)
            if left_map[panel_id]["status"] == "ok"
            and right_map[panel_id]["status"] == "ok"
        ]
        disagreements = [
            panel_id
            for panel_id in paired
            if left_map[panel_id]["subversion"] != right_map[panel_id]["subversion"]
        ]
        pairs[f"{left}_vs_{right}"] = {
            "paired_valid_rows": len(paired),
            "subversion_disagreements": len(disagreements),
            "left_correct_right_wrong": sum(
                left_map[panel_id]["subversion"] == bool(labels[panel_id])
                and right_map[panel_id]["subversion"] != bool(labels[panel_id])
                for panel_id in disagreements
            ),
            "right_correct_left_wrong": sum(
                right_map[panel_id]["subversion"] == bool(labels[panel_id])
                and left_map[panel_id]["subversion"] != bool(labels[panel_id])
                for panel_id in disagreements
            ),
        }
    return {"configurations": summaries, "paired": pairs}


def _audit_summary(
    panel: list[dict],
    primary_by_configuration: dict[str, list[dict]],
    audit_records: list[dict],
) -> dict:
    labels = {row["panel_id"]: bool(row["label"]) for row in panel}
    by_configuration = defaultdict(list)
    for record in audit_records:
        by_configuration[record["configuration"]].append(record)
    result = {}
    for configuration in SELECTED_CONFIGURATIONS:
        primary = _record_map(primary_by_configuration[configuration])
        variants = defaultdict(dict)
        for record in by_configuration[configuration]:
            variants[record["variant"]][record["panel_id"]] = record
        audit_ids = sorted(variants["repeat_2"])
        repeated_ids = [
            panel_id
            for panel_id in audit_ids
            if primary[panel_id]["status"] == "ok"
            and variants["repeat_2"][panel_id]["status"] == "ok"
            and variants["repeat_3"][panel_id]["status"] == "ok"
        ]
        unanimous_subversion = sum(
            len(
                {
                    primary[panel_id]["subversion"],
                    variants["repeat_2"][panel_id]["subversion"],
                    variants["repeat_3"][panel_id]["subversion"],
                }
            )
            == 1
            for panel_id in repeated_ids
        )
        unanimous_harmful = sum(
            len(
                {
                    primary[panel_id]["harmful_request"],
                    variants["repeat_2"][panel_id]["harmful_request"],
                    variants["repeat_3"][panel_id]["harmful_request"],
                }
            )
            == 1
            for panel_id in repeated_ids
        )
        prompt_ids = [
            panel_id
            for panel_id in audit_ids
            if primary[panel_id]["status"] == "ok"
            and variants["subversion_only"][panel_id]["status"] == "ok"
        ]
        prompt_disagreements = [
            panel_id
            for panel_id in prompt_ids
            if primary[panel_id]["subversion"]
            != variants["subversion_only"][panel_id]["subversion"]
        ]
        result[configuration] = {
            "statuses": {
                variant: dict(
                    sorted(
                        Counter(record["status"] for record in records.values()).items()
                    )
                )
                for variant, records in sorted(variants.items())
            },
            "repeatability": {
                "three_valid_outputs": len(repeated_ids),
                "subversion_unanimous": unanimous_subversion,
                "subversion_unanimous_rate": _ratio(
                    unanimous_subversion,
                    len(repeated_ids),
                ),
                "harmful_request_unanimous": unanimous_harmful,
                "harmful_request_unanimous_rate": _ratio(
                    unanimous_harmful,
                    len(repeated_ids),
                ),
            },
            "harmful_field_prompt_interference": {
                "paired_valid_outputs": len(prompt_ids),
                "subversion_disagreements": len(prompt_disagreements),
                "with_harmful_field_correct_only": sum(
                    primary[panel_id]["subversion"] == labels[panel_id]
                    and variants["subversion_only"][panel_id]["subversion"]
                    != labels[panel_id]
                    for panel_id in prompt_disagreements
                ),
                "subversion_only_correct_only": sum(
                    variants["subversion_only"][panel_id]["subversion"]
                    == labels[panel_id]
                    and primary[panel_id]["subversion"] != labels[panel_id]
                    for panel_id in prompt_disagreements
                ),
            },
        }
    return result


def _validate_counts(
    panel: list[dict],
    primary: list[dict],
    ablation: list[dict],
    audits: list[dict],
) -> None:
    if len(panel) != 20_000:
        raise ValueError("expected a 20,000-row panel")
    if len({record["job_id"] for record in primary}) != len(primary):
        raise ValueError("duplicate primary job ID")
    if len({record["job_id"] for record in ablation}) != len(ablation):
        raise ValueError("duplicate ablation job ID")
    if len({record["job_id"] for record in audits}) != len(audits):
        raise ValueError("duplicate audit job ID")
    primary_counts = Counter(record["configuration"] for record in primary)
    audit_counts = Counter(record["configuration"] for record in audits)
    ablation_counts = Counter(record["configuration"] for record in ablation)
    if primary_counts != Counter({name: 20_000 for name in SELECTED_CONFIGURATIONS}):
        raise ValueError(f"incomplete primary results: {primary_counts}")
    if audit_counts != Counter({name: 600 for name in SELECTED_CONFIGURATIONS}):
        raise ValueError(f"incomplete audit results: {audit_counts}")
    if set(ablation_counts.values()) != {200} or len(ablation_counts) != 12:
        raise ValueError(f"incomplete reasoning ablation: {ablation_counts}")


def _analyze(input_dir: Path) -> dict:
    paths = {
        "manifest": input_dir / "manifest.json",
        "panel": input_dir / "panel.jsonl",
        "primary": input_dir / "results.jsonl",
        "reasoning_ablation": input_dir / "ablation_results.jsonl",
        "audits": input_dir / "audits_results.jsonl",
    }
    for name in SETUP_FAILURE_FILES:
        paths[f"excluded_setup_{name.removesuffix('.jsonl')}"] = input_dir / name
    manifest = _load_json(paths["manifest"])
    panel = _load_jsonl(paths["panel"])
    if any("text" in row for row in panel):
        raise ValueError("analysis panel unexpectedly contains raw text")
    primary = _load_jsonl(paths["primary"])
    ablation = _load_jsonl(paths["reasoning_ablation"])
    audits = _load_jsonl(paths["audits"])
    setup_failures = {}
    for name in SETUP_FAILURE_FILES:
        records = _load_jsonl(input_dir / name)
        if any(record["status"] == "ok" for record in records):
            raise ValueError(f"setup-failure ledger contains a valid output: {name}")
        setup_failures[name] = {
            "rows": len(records),
            "statuses": dict(
                sorted(Counter(record["status"] for record in records).items())
            ),
            "http_statuses": dict(
                sorted(
                    Counter(
                        str(record.get("http_status")) for record in records
                    ).items()
                )
            ),
            "sha256": _file_sha256(input_dir / name),
        }
    _validate_counts(panel, primary, ablation, audits)
    primary_by_configuration = defaultdict(list)
    for record in primary:
        primary_by_configuration[record["configuration"]].append(record)
    return {
        "schema_version": 1,
        "panel_sha256": manifest["panel"]["sha256"],
        "input_hashes": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
        "selected_configurations": list(SELECTED_CONFIGURATIONS),
        "limitations": manifest["limitations"]
        + [
            "Recall-target thresholds are selected and evaluated on the same open panel.",
            "Operational failures route to review in fail-safe and cascade metrics.",
            "Latency is client-observed under concurrent research traffic, not a production SLA.",
        ],
        "excluded_setup_attempts": setup_failures,
        "reasoning_ablation": _reasoning_ablation(panel, ablation),
        "llm_standalone": {
            configuration: _llm_summary(
                panel,
                primary_by_configuration[configuration],
            )
            for configuration in SELECTED_CONFIGURATIONS
        },
        "mmbert_and_cascades": _encoder_summary(
            panel,
            primary_by_configuration,
        ),
        "audits": _audit_summary(
            panel,
            primary_by_configuration,
            audits,
        ),
    }


def _self_check() -> None:
    metrics = _binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([False, True, False, True]),
    )
    assert metrics["tp"] == metrics["fp"] == metrics["fn"] == metrics["tn"] == 1
    threshold = _threshold_for_recall(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.9, 0.1, 0.8, 0.7]),
        1.0,
    )
    assert threshold == 0.7
    print("self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return 0
    output = args.output or args.input / "summary.json"
    summary = _analyze(args.input)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _file_sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
