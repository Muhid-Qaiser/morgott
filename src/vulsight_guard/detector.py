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

from .data import deduplicate, normalize_text, read_jsonl


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


def split_fit_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    fit, validation = [], []
    for row in rows:
        digest = hashlib.sha256(
            row.get("split_group_id", row["group_id"]).encode()
        ).digest()
        (validation if int.from_bytes(digest[:2]) % 5 == 0 else fit).append(row)
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

    best_threshold, best_recall = float(np.nextafter(scores.max(), np.inf)), -1.0
    for threshold in np.unique(scores)[::-1]:
        predictions = scores >= threshold
        fpr = predictions[labels == 0].mean()
        recall = predictions[labels == 1].mean()
        if fpr <= max_fpr and (
            recall > best_recall
            or (recall == best_recall and threshold < best_threshold)
        ):
            best_threshold, best_recall = float(threshold), float(recall)
    return best_threshold


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


def _format(value: float | int | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


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


def _write_report(path: Path, manifest: dict, results: dict) -> None:
    profiles_by_floor = {
        profile["min_validation_precision"]: profile
        for profile in results["direct_precision_profiles"]
    }
    looser = profiles_by_floor[0.80]["validation"]
    recommended = profiles_by_floor[DIRECT_REVIEW_PRECISION_FLOOR]["validation"]
    stricter = profiles_by_floor[0.90]["validation"]
    lines = [
        "# Broad jailbreak-sensor baseline",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "This is a P0 shadow-mode text sensor for direct jailbreak and prompt-injection "
        "attempts. It is not a harmful-content classifier, a block decision, or an "
        "authorization boundary.",
        "",
        "## Data",
        "",
        "| Partition | Rows | Attack | Non-attack |",
        "|---|---:|---:|---:|",
    ]
    for name, output in manifest["outputs"].items():
        lines.append(
            f"| {name} | {output['rows']} | {output['positive']} | {output['negative']} |"
        )
    lines += [
        "",
        "Exact normalized duplicates are removed, evaluation text duplicated in training "
        "is blocked, OASST1 conversations are grouped by tree, and the multi-turn corpus "
        "is grouped by attacker goal.",
        "",
        "## Shadow-review operating point",
        "",
        f"The character n-gram model threshold is {results['training']['threshold']:.6f}. "
        "It is the recommended high-precision starting point for shadow review, "
        "selected only on deterministic validation groups at an 85% minimum precision "
        "floor; the official test partitions were not used for threshold selection.",
        f"Training used {results['training']['fit_rows']} rows; "
        f"{results['training']['validation_rows']} rows across "
        f"{results['training']['validation']['groups']} lineage groups were reserved for "
        f"threshold selection. The selected profile observes "
        f"{_format(results['training']['validation']['precision'])} precision, "
        f"{_format(results['training']['validation']['recall'])} recall, and "
        f"{_format(results['training']['validation']['fpr'])} FPR on that source mixture.",
        f"The separate untrusted-content sensor threshold is "
        f"{results['indirect_training']['threshold']:.6f}; it requires zero false "
        f"positives on its {results['indirect_training']['validation_rows']}-row "
        "BIPIA training holdout and scores the maximum whole-document/paragraph signal. "
        "Both sensors remain shadow-only.",
        "",
        "| Detector | Evaluation | Recall | FPR | False signals / 10k | Precision | PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for detector, detector_result in results["detectors"].items():
        for dataset, metrics in detector_result["sets"].items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        detector,
                        dataset,
                        _format(metrics["recall"]),
                        _format(metrics["fpr"]),
                        _format(metrics["false_signals_per_10k"]),
                        _format(metrics["precision"]),
                        _format(metrics["pr_auc"]),
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Direct-chat precision profiles",
        "",
        "The 85% floor is the practical high-precision knee. Compared with the 80% "
        f"floor, it gives up {looser['true_positive'] - recommended['true_positive']} "
        f"validation attacks while removing {looser['false_positive'] - recommended['false_positive']} "
        "false signals. Tightening from 85% to 90% removes only "
        f"{recommended['false_positive'] - stricter['false_positive']} more false "
        f"signals while losing {recommended['true_positive'] - stricter['true_positive']} "
        "more attacks. All profiles remain advisory and were selected without official "
        "test results.",
        "",
        "Observed precision reflects the validation source mixture (66 attacks among "
        "7,186 rows), not product traffic. Expected-precision cells are prevalence "
        "scenarios calculated from validation recall and FPR. Each cell is `point / "
        "FPR-upper stress estimate`; neither value is production calibration, and the "
        "stress estimate is not a full confidence interval.",
        "",
        "| Role | Minimum validation precision | Threshold | TP / attacks | FP / non-attacks | Recall | Observed precision | Observed FPR | Expected precision @ 0.1% attacks | @ 1% | @ 5% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in results["direct_precision_profiles"]:
        validation = profile["validation"]
        expected = profile["expected_precision"]
        lines.append(
            "| "
            + " | ".join(
                (
                    profile["role"],
                    f"{profile['min_validation_precision']:.0%}",
                    f"{profile['threshold']:.6f}",
                    f"{validation['true_positive']} / {validation['positive']}",
                    f"{validation['false_positive']} / {validation['negative']}",
                    _format(validation["recall"]),
                    _format(validation["precision"]),
                    _format(validation["fpr"]),
                    *(
                        f"{_format(scenario['point_estimate'])} / "
                        f"{_format(scenario['fpr_upper_stress_estimate'])}"
                        for scenario in expected
                    ),
                )
            )
            + " |"
        )
    lines += [
        "",
        "Frozen-suite transfer at each precision profile is shown below. It did not "
        "choose the recommendation.",
        "",
        "| Minimum validation precision | External FPR | ToxicChat recall / precision / FPR | deepset recall / precision / FPR | Obfuscated recall | Temporal-source recall / precision / FPR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in results["direct_precision_profiles"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{point['min_validation_precision']:.0%}",
                    _format(point["external_hard_negatives"]["fpr"]),
                    " / ".join(
                        _format(point["toxic_chat"][key])
                        for key in ("recall", "precision", "fpr")
                    ),
                    " / ".join(
                        _format(point["prompt_injections"][key])
                        for key in ("recall", "precision", "fpr")
                    ),
                    _format(point["multi_turn"]["recall"]),
                    " / ".join(
                        _format(point["jailbreaks_over_time"][key])
                        for key in ("recall", "precision", "fpr")
                    ),
                )
            )
            + " |"
        )
    lines += [
        "",
        "## FPR-budget diagnostics",
        "",
        "There is no universal target FPR. These validation-selected rows preserve the "
        "0.1%, 0.5%, 1%, 2%, and 5% diagnostics for comparison, but none is the default "
        "review profile or calibrated for blocking or production traffic.",
        "",
        "| Role | Validation FPR budget | Threshold | TP / attacks | FP / non-attacks | Recall | Precision | Observed FPR | False signals / 10k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in results["direct_operating_points"]:
        validation = point["validation"]
        lines.append(
            "| "
            + " | ".join(
                (
                    point["role"],
                    f"{point['validation_fpr_budget']:.3%}",
                    f"{point['threshold']:.6f}",
                    f"{validation['true_positive']} / {validation['positive']}",
                    f"{validation['false_positive']} / {validation['negative']}",
                    _format(validation["recall"]),
                    _format(validation["precision"]),
                    _format(validation["fpr"]),
                    _format(validation["false_signals_per_10k"]),
                )
            )
            + " |"
        )
    lines += [
        "",
        "| Validation FPR budget | External FPR | ToxicChat recall / precision / FPR | deepset recall / precision / FPR | Obfuscated recall | Temporal-source recall / precision / FPR |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for point in results["direct_operating_points"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{point['validation_fpr_budget']:.3%}",
                    _format(point["external_hard_negatives"]["fpr"]),
                    " / ".join(
                        _format(point["toxic_chat"][key])
                        for key in ("recall", "precision", "fpr")
                    ),
                    " / ".join(
                        _format(point["prompt_injections"][key])
                        for key in ("recall", "precision", "fpr")
                    ),
                    _format(point["multi_turn"]["recall"]),
                    " / ".join(
                        _format(point["jailbreaks_over_time"][key])
                        for key in ("recall", "precision", "fpr")
                    ),
                )
            )
            + " |"
        )
    char_result = results["detectors"]["char_ngram_logreg"]
    indirect_result = results["detectors"]["indirect_char_ngram_logreg"]
    lines += [
        "",
        "## Runtime",
        "",
        "Latency uses three warm-process passes over at most 2,048 rows sampled "
        "deterministically across the evaluation order.",
        "",
        "| Detector | Mean batch latency per sample (µs) |",
        "|---|---:|",
    ]
    for detector, detector_result in results["detectors"].items():
        lines.append(f"| {detector} | {detector_result['latency_us_per_sample']:.1f} |")

    xstest = char_result["sets"]["xstest"]
    ordinary = char_result["sets"]["oasst1_chat"]
    position_stress = char_result["sets"]["oasst1_position_stress"]
    harmful = char_result["sets"]["harmful_nonattack"]
    external = char_result["sets"]["external_hard_negatives"]
    notinject = char_result["sets"]["notinject"]
    obfuscated = char_result["sets"]["multi_turn"]
    direct_bipia_payload = char_result["sets"]["bipia_payload"]
    direct_bipia_context = char_result["sets"]["bipia_context"]
    temporal = char_result["sets"]["jailbreaks_over_time"]
    tensor_attack = char_result["sets"]["tensor_trust_attack"]
    bipia_clean = indirect_result["sets"]["bipia_clean_context"]
    bipia_payload = indirect_result["sets"]["bipia_payload"]
    bipia_context = indirect_result["sets"]["bipia_context"]
    tensor_context = indirect_result["sets"]["tensor_trust_context"]
    tensor_combined = results["untrusted_content_combined_shadow"]["sets"][
        "tensor_trust_context"
    ]
    xstest_safe = xstest["by_goal_policy_status"]["safe"]
    xstest_unsafe = xstest["by_goal_policy_status"]["unsafe"]
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Across external direct-user hard negatives, FPR is "
        f"{_format(external['fpr'])} ({external['false_positive']}/"
        f"{external['negative']}), or {_format(external['false_signals_per_10k'])} "
        f"signals per 10k prompts. The Wilson 95% upper bound is "
        f"{_format(external['fpr_95_upper'])}.",
        f"- Multilingual OASST1 human-chat FPR is {_format(ordinary['fpr'])} "
        f"({ordinary['false_positive']}/{ordinary['negative']}); held-out harmful but "
        f"non-injection FPR is {_format(harmful['fpr'])} "
        f"({harmful['false_positive']}/{harmful['negative']}).",
        f"- Two-prompt OASST1 position-stress FPR is "
        f"{_format(position_stress['fpr'])} "
        f"({position_stress['false_positive']}/{position_stress['negative']}).",
        f"- XSTest hard-negative FPR: {_format(xstest['fpr'])} "
        f"({xstest['false_positive']}/{xstest['negative']}); Wilson 95% upper bound "
        f"{_format(xstest['fpr_95_upper'])}.",
        f"- XSTest safe/unsafe hard-negative FPR: {_format(xstest_safe['fpr'])} "
        f"({xstest_safe['false_positive']}/{xstest_safe['negative']}) / "
        f"{_format(xstest_unsafe['fpr'])} "
        f"({xstest_unsafe['false_positive']}/{xstest_unsafe['negative']}).",
        f"- NotInject trigger-word hard-negative FPR: {_format(notinject['fpr'])} "
        f"({notinject['false_positive']}/{notinject['negative']}); Wilson 95% upper "
        f"bound {_format(notinject['fpr_95_upper'])}.",
        f"- Out-of-source obfuscated-jailbreak recall: {_format(obfuscated['recall'])} "
        f"({obfuscated['true_positive']}/{obfuscated['positive']}; "
        f"cluster-weighted {_format(obfuscated['cluster_weighted_recall'])}).",
        f"- JailbreaksOverTime source-shift recall/FPR is "
        f"{_format(temporal['recall'])}/{_format(temporal['fpr'])} "
        f"({temporal['true_positive']}/{temporal['positive']} attacks; "
        f"{temporal['false_positive']}/{temporal['negative']} source-labeled "
        "negatives). Source and time are confounded, so this is not a clean temporal "
        "causal estimate.",
        f"- Human-authored Tensor Trust attack-only recall is "
        f"{_format(tensor_attack['recall'])} "
        f"({tensor_attack['true_positive']}/{tensor_attack['positive']}); the "
        f"provenance-scoped sensor gets {_format(tensor_context['recall'])} "
        f"({tensor_context['true_positive']}/{tensor_context['positive']}) on the "
        "same attacks embedded between benchmark defenses. Running the direct-override "
        f"fallback as designed raises combined recall to "
        f"{_format(tensor_combined['recall'])} "
        f"({tensor_combined['true_positive']}/{tensor_combined['positive']}). This "
        "source is evaluation only and has no explicit standard dataset license.",
        f"- The direct-chat model gets {_format(direct_bipia_payload['recall'])} BIPIA "
        f"payload recall and {_format(direct_bipia_context['recall'])} poisoned-context "
        "recall. Lowering its threshold does not solve this without conflating ordinary "
        "questions with attacks.",
        f"- The provenance-scoped indirect model gets {_format(bipia_payload['recall'])} "
        f"payload recall and {_format(bipia_context['recall'])} poisoned-context recall "
        f"(cluster-weighted {_format(bipia_context['cluster_weighted_recall'])}), with "
        f"clean-context FPR {_format(bipia_clean['fpr'])} "
        f"({bipia_clean['false_positive']}/{bipia_clean['negative']}).",
        "- Every classifier result is advisory. Trusted input provenance selects the "
        "sensor set; untrusted content also runs the direct-override fallback. The CLI "
        "always returns allow, while the reference monitor independently constrains "
        "actions and data flow.",
        "- Same-source scores measure a cheap known-distribution sensor. The obfuscated "
        "and indirect holdouts test transfer but remain benchmark attack families.",
        "",
        "## Obfuscated holdout by variant",
        "",
        "| Variant | Rows | Recall |",
        "|---|---:|---:|",
    ]
    for category, metrics in obfuscated["by_category"].items():
        lines.append(
            f"| {category} | {metrics['rows']} | {_format(metrics['recall'])} |"
        )
    plain_random = obfuscated["by_category"]["word_mapping_random"]
    plain_filtered = obfuscated["by_category"]["word_mapping_perp_filter"]
    perfect_encoded = sum(
        metrics["recall"] == 1.0
        for category, metrics in obfuscated["by_category"].items()
        if "/" in category
    )
    lines += [
        "",
        f"The {_format(obfuscated['recall'])} aggregate hides a wide range: "
        f"{perfect_encoded} encoded variants reach 100% recall. The two plain "
        f"word-mapping variants reach only {_format(plain_random['recall'])} and "
        f"{_format(plain_filtered['recall'])}, which is the actionable weakness.",
        "",
        "## Label audit",
        "",
        "A qualitative Codex-agent inspection of the highest-confidence errors found "
        "obvious jailbreak-style "
        "ToxicChat rows labeled non-jailbreak, harmful framing labeled jailbreak without "
        "an explicit override, context-dependent deepset positives that look benign as "
        "standalone text, and jailbreak/DAN-style WildChat rows labeled negative in "
        "JailbreaksOverTime. This was not independent human annotation. These metrics "
        "measure the source labels; they are not clean ground truth. See "
        "[label-audit.md](label-audit.md).",
        "",
        "## Limits",
        "",
        "- ToxicChat contributes only 204 jailbreak positives across its official splits.",
        "- OASST1 has no injection label; accepted human prompts are weak negatives and may contain attacks.",
        "- Held-out data estimates benchmark FPR, not the production traffic base rate or user impact.",
        "- Source, language, family, and time are confounded; this is a baseline, not a robustness claim.",
        "- ToxicChat and deepset expose no template lineage here, so their groups are row-level after exact deduplication.",
        "- The cipher holdout scores only its single Prompt field; it is not a multi-turn session evaluation.",
        "- BIPIA context rows are a deterministic three-position slice, not its full task/target-model evaluation.",
        f"- The indirect sensor has only {bipia_clean['negative']} held-out clean BIPIA contexts; its FPR confidence interval is wide.",
        "- No adaptive attacker, guard transformer, LLM judge, or live target model is included in headline metrics.",
        "- ToxicChat and Do-Not-Answer have non-commercial licenses, so this model is research-only.",
        "- Classifier misses must be contained by deterministic action and egress policy.",
        "",
        "Source details and pinned revisions are in the generated data manifest.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    reports_dir: Path = Path("reports"),
) -> dict:
    manifest = json.loads(
        (data_dir / "processed" / "manifest.json").read_text(encoding="utf-8")
    )
    train = read_jsonl(data_dir / "processed" / "train.jsonl")
    model, training, validation, validation_score = _fit_sensor(
        train, min_precision=DIRECT_REVIEW_PRECISION_FLOOR
    )
    threshold = training["threshold"]
    indirect_train = read_jsonl(data_dir / "processed" / "indirect_train.jsonl")
    indirect_model, indirect_training, _, _ = _fit_sensor(
        indirect_train, 0.0, score_paragraphs=True
    )
    indirect_threshold = indirect_training["threshold"]

    physical_evaluation = {
        name: read_jsonl(data_dir / "processed" / f"{name}.jsonl")
        for name in manifest["outputs"]
        if name not in {"train", "indirect_train"}
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
    (reports_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports_dir / "baseline.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(reports_dir / "baseline.md", manifest, results)
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
