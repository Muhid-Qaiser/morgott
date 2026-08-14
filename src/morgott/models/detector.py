from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from ..data import (
    deduplicate,
    manifest_output_path,
    normalize_text,
    read_verified_jsonl,
    split_is_validation,
)

KEYWORD_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bignore (?:all |any |the )?(?:previous|prior|above)",
        r"\b(?:disregard|override) (?:all |any |the )?(?:previous|prior|above|instructions?)",
        r"\bforget (?:all |any |the )?(?:previous|prior|above)",
        r"\b(?:do anything now|dan mode|jailbreak)\b",
        r"\b(?:reveal|show|print|repeat).{0,40}\b(?:system|developer|hidden) prompt",
    )
]

DIRECT_OPERATING_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
DIRECT_PRECISION_FLOORS = (0.80, 0.85, 0.90, 0.95)
DIRECT_REVIEW_PRECISION_FLOOR = 0.85
DIRECT_EXPECTED_ATTACK_PREVALENCES = (0.001, 0.01, 0.05)


def validation_mask(rows: list[dict]) -> np.ndarray:
    return np.asarray([split_is_validation(row) for row in rows])


def split_fit_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    selected = validation_mask(rows)
    fit = [row for row, value in zip(rows, selected, strict=True) if not value]
    validation = [row for row, value in zip(rows, selected, strict=True) if value]
    return fit, validation


def choose_threshold(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    max_fpr: float,
) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("threshold selection requires both classes")

    fallback = float(np.nextafter(scores.max(), np.inf))
    thresholds = np.unique(scores)
    nan_thresholds = np.isnan(thresholds)
    real_thresholds = thresholds[~nan_thresholds]
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    sorted_negatives = np.sort(negatives[~np.isnan(negatives)], axis=None)
    false_positives = len(sorted_negatives) - np.searchsorted(
        sorted_negatives,
        real_thresholds,
        side="left",
    )
    eligible = np.flatnonzero(false_positives / negatives.size <= max_fpr)
    if len(eligible):
        threshold = real_thresholds[eligible[0]]
        if not nan_thresholds.any() or np.mean(positives >= threshold) > 0:
            return float(threshold)
    return fallback


def choose_threshold_for_precision(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    min_precision: float,
) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("threshold selection requires both classes")
    if not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= min_precision)
    if not len(eligible):
        raise ValueError("no observed threshold meets min_precision")
    best_recall = recall[eligible].max()
    best_index = eligible[recall[eligible] == best_recall][-1]
    return float(thresholds[best_index])


def _expected_precision(recall: float, fpr: float, attack_prevalence: float) -> float:
    true_signal = recall * attack_prevalence
    false_signal = fpr * (1 - attack_prevalence)
    return true_signal / (true_signal + false_signal) if true_signal else 0.0


def _wilson_upper(successes: int, trials: int, z: float = 1.96) -> float | None:
    if not trials:
        return None
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = rate + z**2 / (2 * trials)
    margin = z * np.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
    return float((center + margin) / denominator)


def _rates(
    rows: list[dict], scores: np.ndarray, threshold: float
) -> dict[str, float | int | None]:
    labels = np.asarray([row["label"] for row in rows])
    predictions = scores >= threshold
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    false_negative = positives - true_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )

    result: dict[str, float | int | None] = {
        "rows": len(rows),
        "positive": positives,
        "negative": negatives,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": true_positive / positives if positives else None,
        "fpr": false_positive / negatives if negatives else None,
        "fpr_95_upper": _wilson_upper(false_positive, negatives),
        "false_signals_per_10k": (
            false_positive / negatives * 10_000 if negatives else None
        ),
        "brier": float(brier_score_loss(labels, scores)),
        "pr_auc": None,
        "tpr_at_fpr_0_001": None,
        "tpr_at_fpr_0_01": None,
    }
    if positives and negatives:
        result["pr_auc"] = float(average_precision_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        for target, key in (
            (0.001, "tpr_at_fpr_0_001"),
            (0.01, "tpr_at_fpr_0_01"),
        ):
            eligible = tpr[fpr <= target]
            result[key] = float(eligible.max()) if len(eligible) else 0.0

    grouped: dict[str, list[bool]] = defaultdict(list)
    group_labels = {}
    for row, predicted in zip(rows, predictions, strict=True):
        grouped[row["group_id"]].append(bool(predicted))
        group_labels[row["group_id"]] = row["label"]
    positive_groups = [
        np.mean(grouped[group]) for group in grouped if group_labels[group] == 1
    ]
    negative_groups = [
        np.mean(grouped[group]) for group in grouped if group_labels[group] == 0
    ]
    result["groups"] = len(grouped)
    result["cluster_weighted_recall"] = (
        float(np.mean(positive_groups)) if positive_groups else None
    )
    result["cluster_weighted_fpr"] = (
        float(np.mean(negative_groups)) if negative_groups else None
    )
    return result


def _evaluate(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    result = _rates(rows, scores, threshold)
    subgroups = {}
    for status in sorted({row["goal_policy_status"] for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["goal_policy_status"] == status
        ]
        subgroups[status] = _rates(
            [rows[index] for index in indices], scores[indices], threshold
        )
    result["by_goal_policy_status"] = subgroups
    channels = {}
    for channel in sorted({row["input_channel"] for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if row["input_channel"] == channel
        ]
        channels[channel] = _rates(
            [rows[index] for index in indices], scores[indices], threshold
        )
    result["by_input_channel"] = channels
    categories = {}
    for category in sorted({row["category"] or "(none)" for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if (row["category"] or "(none)") == category
        ]
        categories[category] = _rates(
            [rows[index] for index in indices], scores[indices], threshold
        )
    result["by_category"] = categories
    return result


def _latency(score, texts: list[str]) -> float:
    step = max(1, len(texts) // 2_048)
    sample = texts[::step][:2_048]
    score(sample[:1])
    start = time.perf_counter()
    for _ in range(3):
        score(sample)
    return (time.perf_counter() - start) * 1_000_000 / (3 * len(sample))


def _char_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _score_paragraphs(model: Pipeline, texts: list[str]) -> np.ndarray:
    candidates = []
    spans = []
    for text in texts:
        parts = [text] + [
            part for part in re.split(r"\n\s*\n", text) if len(part.strip()) >= 8
        ]
        start = len(candidates)
        candidates.extend(normalize_text(part) for part in parts)
        spans.append((start, len(candidates)))
    scores = model.predict_proba(candidates)[:, 1]
    return np.asarray([scores[start:end].max() for start, end in spans])


def _fit_sensor(
    rows: list[dict],
    max_fpr: float | None = None,
    *,
    min_precision: float | None = None,
    score_paragraphs: bool = False,
) -> tuple[Pipeline, dict, list, np.ndarray]:
    if (max_fpr is None) == (min_precision is None):
        raise ValueError("select exactly one threshold constraint")
    fit, validation = split_fit_validation(rows)
    model = _char_pipeline()
    start = time.perf_counter()
    model.fit(
        [normalize_text(row["text"]) for row in fit],
        [row["label"] for row in fit],
    )
    fit_seconds = time.perf_counter() - start
    validation_texts = [row["text"] for row in validation]
    validation_score = (
        _score_paragraphs(model, validation_texts)
        if score_paragraphs
        else model.predict_proba([normalize_text(text) for text in validation_texts])[
            :, 1
        ]
    )
    validation_labels = [row["label"] for row in validation]
    threshold = (
        choose_threshold(validation_labels, validation_score, max_fpr)
        if max_fpr is not None
        else choose_threshold_for_precision(
            validation_labels, validation_score, min_precision
        )
    )
    training = {
        "fit_rows": len(fit),
        "fit_positive": sum(row["label"] for row in fit),
        "validation_rows": len(validation),
        "validation_positive": sum(row["label"] for row in validation),
        "threshold_target_fpr": max_fpr,
        "threshold_min_precision": min_precision,
        "threshold": threshold,
        "fit_seconds": fit_seconds,
        "validation": _rates(validation, validation_score, threshold),
    }
    return model, training, validation, validation_score


def run_benchmark(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    reports_dir: Path = Path("reports"),
) -> dict:
    manifest_path = data_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    def read(output: dict) -> list[dict]:
        return read_verified_jsonl(
            manifest_output_path(data_dir, output), output["sha256"]
        )

    train = read(manifest["injection_views"]["direct_train"]) + read(
        manifest["injection_views"]["direct_validation"]
    )
    model, training, validation, validation_score = _fit_sensor(
        train, min_precision=DIRECT_REVIEW_PRECISION_FLOOR
    )
    threshold = training["threshold"]
    indirect_train = read(manifest["injection_views"]["indirect_train"]) + read(
        manifest["injection_views"]["indirect_validation"]
    )
    indirect_model, indirect_training, _, _ = _fit_sensor(
        indirect_train, 0.0, score_paragraphs=True
    )
    indirect_threshold = indirect_training["threshold"]

    physical_evaluation = {
        name: read(output)
        for name, output in manifest["injection_views"].items()
        if name
        not in {
            "direct_train",
            "direct_validation",
            "indirect_train",
            "indirect_validation",
        }
    }
    evaluation = dict(physical_evaluation)
    evaluation["harmful_nonattack"], _ = deduplicate(
        physical_evaluation["harmbench"] + physical_evaluation["do_not_answer"]
    )
    evaluation["external_hard_negatives"], _ = deduplicate(
        physical_evaluation["oasst1_chat"]
        + physical_evaluation["oasst1_position_stress"]
        + physical_evaluation["xstest"]
        + physical_evaluation["harmbench"]
        + physical_evaluation["do_not_answer"]
        + physical_evaluation["notinject"]
    )
    normalized = {
        name: [normalize_text(row["text"]) for row in rows]
        for name, rows in evaluation.items()
    }
    positive_matches = {
        normalize_text(row["text"]) for row in train if row["label"] == 1
    }
    scorers = {
        "no_guard": lambda texts: np.zeros(len(texts)),
        "exact_match": lambda texts: np.asarray(
            [float(text in positive_matches) for text in texts]
        ),
        "keyword_rules": lambda texts: np.asarray(
            [
                float(any(pattern.search(text) for pattern in KEYWORD_PATTERNS))
                for text in texts
            ]
        ),
    }
    thresholds = {
        "no_guard": 0.5,
        "exact_match": 0.5,
        "keyword_rules": 0.5,
    }
    all_text = [
        text
        for name, texts in normalized.items()
        if name in physical_evaluation
        for text in texts
    ]

    results = {
        "schema_version": 3,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "training": training,
        "indirect_training": indirect_training,
        "detectors": {},
    }
    for name, score in scorers.items():
        results["detectors"][name] = {
            "threshold": thresholds[name],
            "latency_us_per_sample": _latency(score, all_text),
            "sets": {
                dataset: _rates(rows, score(normalized[dataset]), thresholds[name])
                for dataset, rows in evaluation.items()
            },
        }

    direct_scores = {
        dataset: model.predict_proba(texts)[:, 1]
        for dataset, texts in normalized.items()
    }
    results["detectors"]["char_ngram_logreg"] = {
        "threshold": threshold,
        "input_channel": "direct_user",
        "latency_us_per_sample": _latency(
            lambda texts: model.predict_proba(texts)[:, 1], all_text
        ),
        "sets": {
            dataset: _evaluate(rows, direct_scores[dataset], threshold)
            for dataset, rows in evaluation.items()
        },
    }

    validation_labels = [row["label"] for row in validation]

    def direct_point(point_threshold: float) -> dict:
        return {
            "validation": _rates(validation, validation_score, point_threshold),
            "external_hard_negatives": _rates(
                evaluation["external_hard_negatives"],
                direct_scores["external_hard_negatives"],
                point_threshold,
            ),
            "toxic_chat": _rates(
                evaluation["toxic_chat"],
                direct_scores["toxic_chat"],
                point_threshold,
            ),
            "prompt_injections": _rates(
                evaluation["prompt_injections"],
                direct_scores["prompt_injections"],
                point_threshold,
            ),
            "multi_turn": _rates(
                evaluation["multi_turn"],
                direct_scores["multi_turn"],
                point_threshold,
            ),
            "jailbreaks_over_time": _rates(
                evaluation["jailbreaks_over_time"],
                direct_scores["jailbreaks_over_time"],
                point_threshold,
            ),
        }

    results["direct_precision_profiles"] = []
    for precision_floor in DIRECT_PRECISION_FLOORS:
        profile_threshold = choose_threshold_for_precision(
            validation_labels, validation_score, precision_floor
        )
        profile = {
            "role": (
                "recommended shadow review"
                if precision_floor == DIRECT_REVIEW_PRECISION_FLOOR
                else "precision diagnostic"
            ),
            "min_validation_precision": precision_floor,
            "threshold": profile_threshold,
            **direct_point(profile_threshold),
        }
        validation_metrics = profile["validation"]
        profile["expected_precision"] = [
            {
                "attack_prevalence": prevalence,
                "point_estimate": _expected_precision(
                    validation_metrics["recall"],
                    validation_metrics["fpr"],
                    prevalence,
                ),
                "fpr_upper_stress_estimate": _expected_precision(
                    validation_metrics["recall"],
                    validation_metrics["fpr_95_upper"],
                    prevalence,
                ),
            }
            for prevalence in DIRECT_EXPECTED_ATTACK_PREVALENCES
        ]
        results["direct_precision_profiles"].append(profile)

    results["direct_operating_points"] = []
    for fpr_budget in DIRECT_OPERATING_FPR_BUDGETS:
        point_threshold = choose_threshold(
            validation_labels, validation_score, fpr_budget
        )
        results["direct_operating_points"].append(
            {
                "role": (
                    "stringent FPR diagnostic"
                    if fpr_budget == DIRECT_OPERATING_FPR_BUDGETS[0]
                    else "FPR diagnostic"
                ),
                "validation_fpr_budget": fpr_budget,
                "threshold": point_threshold,
                **direct_point(point_threshold),
            }
        )
    recommended = next(
        point
        for point in results["direct_precision_profiles"]
        if point["min_validation_precision"] == DIRECT_REVIEW_PRECISION_FLOOR
    )
    looser = results["direct_precision_profiles"][0]
    stricter = results["direct_precision_profiles"][2]
    results["direct_review_recommendation"] = {
        "mode": "shadow_review",
        "profile": "validation_precision_floor",
        "min_validation_precision": DIRECT_REVIEW_PRECISION_FLOOR,
        "threshold": recommended["threshold"],
        "selection_basis": "grouped validation high-precision practical knee",
        "delta_from_80_percent_floor": {
            "true_positive": recommended["validation"]["true_positive"]
            - looser["validation"]["true_positive"],
            "false_positive": recommended["validation"]["false_positive"]
            - looser["validation"]["false_positive"],
        },
        "delta_to_90_percent_floor": {
            "true_positive": stricter["validation"]["true_positive"]
            - recommended["validation"]["true_positive"],
            "false_positive": stricter["validation"]["false_positive"]
            - recommended["validation"]["false_positive"],
        },
        "not_for": ["blocking", "authorization", "production calibration"],
    }

    indirect_sets = {
        name: physical_evaluation[name]
        for name in (
            "bipia_clean_context",
            "bipia_payload",
            "bipia_context",
            "tensor_trust_context",
        )
    }
    indirect_texts = {
        name: [row["text"] for row in rows] for name, rows in indirect_sets.items()
    }
    indirect_scores = {
        name: _score_paragraphs(indirect_model, indirect_texts[name])
        for name in indirect_sets
    }
    results["detectors"]["indirect_char_ngram_logreg"] = {
        "threshold": indirect_threshold,
        "input_channel": "untrusted_content",
        "latency_us_per_sample": _latency(
            lambda texts: _score_paragraphs(indirect_model, texts),
            [text for texts in indirect_texts.values() for text in texts],
        ),
        "sets": {
            name: _evaluate(rows, indirect_scores[name], indirect_threshold)
            for name, rows in indirect_sets.items()
        },
    }
    combined_sets = {}
    for name, rows in indirect_sets.items():
        direct_elevated = direct_scores[name] >= threshold
        indirect_elevated = indirect_scores[name] >= indirect_threshold
        metrics = _rates(
            rows,
            np.logical_or(direct_elevated, indirect_elevated).astype(float),
            0.5,
        )
        metrics["direct_elevated"] = int(direct_elevated.sum())
        metrics["indirect_elevated"] = int(indirect_elevated.sum())
        metrics["both_elevated"] = int(
            np.logical_and(direct_elevated, indirect_elevated).sum()
        )
        combined_sets[name] = metrics
    results["untrusted_content_combined_shadow"] = {
        "combination": "binary OR after channel-specific locked thresholds",
        "metric_note": "binary decision metrics; Brier is error rate, not calibration",
        "sets": combined_sets,
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 2,
            "operating_mode": "shadow",
            "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "channels": {
                "direct_user": {
                    "target": "direct jailbreak or prompt-injection attack attempt",
                    "threshold": threshold,
                    "threshold_selection": {
                        "metric": "validation_precision_floor",
                        "value": DIRECT_REVIEW_PRECISION_FLOOR,
                    },
                    "model": model,
                },
                "untrusted_content": {
                    "target": "indirect prompt-injection attempt in untrusted content",
                    "threshold": indirect_threshold,
                    "model": indirect_model,
                    "scoring": "max_paragraph",
                },
            },
        },
        artifacts_dir / "guard_bundle.joblib",
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "baseline.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results


def scan(
    text: str,
    model_path: Path = Path("artifacts/guard_bundle.joblib"),
    channel: str = "direct_user",
) -> dict:
    artifact = joblib.load(model_path)
    sensor = artifact["channels"][channel]

    def score(selected: dict) -> float:
        if selected.get("scoring") == "max_paragraph":
            return float(_score_paragraphs(selected["model"], [text])[0])
        return float(selected["model"].predict_proba([normalize_text(text)])[0, 1])

    component_names = [channel]
    if channel == "untrusted_content" and "direct_user" in artifact["channels"]:
        component_names.append("direct_user")
    components = {}
    for name in component_names:
        selected = artifact["channels"][name]
        component_score = score(selected)
        components[name] = {
            "score": component_score,
            "threshold": selected["threshold"],
            "elevated": component_score >= selected["threshold"],
        }
    elevated = any(component["elevated"] for component in components.values())
    return {
        "target": sensor["target"],
        "channel": channel,
        "score": components[channel]["score"],
        "threshold": sensor["threshold"],
        "components": components,
        "triggered_by": [
            name for name, component in components.items() if component["elevated"]
        ],
        "signal": "elevated" if elevated else "low",
        "review_recommended": elevated,
        "decision": "allow",
        "mode": artifact.get("operating_mode", "shadow"),
        "note": "Advisory sensor only; enforce capabilities and data flow separately.",
    }
