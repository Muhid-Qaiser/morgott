from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from morgott.data import normalize_text, read_jsonl
from morgott.detector import (
    _score_paragraphs,
    choose_threshold,
    split_fit_validation,
)
from morgott.policy import REFERENCE_POLICY, SCENARIOS, authorize


SOURCE = "nemotron_agentic_ipi"
FIT_PARTITIONS = {"train", "indirect_train"}
GROUP_FIELDS = ("domain", "attack_category", "injection_vector", "target_tool")
NEAR_THRESHOLD = 0.90


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(row[field] for row in rows).items()))
        for field in GROUP_FIELDS
    }


def _score_summary(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    elevated = scores >= threshold
    result = {
        "rows": len(rows),
        "elevated": int(elevated.sum()),
        "recall": float(elevated.mean()) if len(rows) else None,
        "threshold": float(threshold),
        "score_quantiles": {
            str(quantile): float(np.quantile(scores, quantile))
            for quantile in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "by_group": {},
    }
    for field in GROUP_FIELDS:
        result["by_group"][field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = np.asarray(
                [index for index, row in enumerate(rows) if row[field] == value]
            )
            selected = elevated[indices]
            result["by_group"][field][value] = {
                "rows": len(indices),
                "elevated": int(selected.sum()),
                "recall": float(selected.mean()),
            }
    return result


def _validation_rates(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray([row["label"] for row in rows])
    elevated = scores >= threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "recall": float(elevated[positive].mean()),
        "fpr": float(elevated[negative].mean()),
    }


def _operating_point_diagnostics(
    direct_scores: np.ndarray,
    indirect_scores: np.ndarray,
    references: dict[str, list[dict]],
    artifact: dict,
    baseline: dict,
) -> dict:
    direct_precision = []
    for point in baseline["direct_precision_profiles"]:
        direct_precision.append(
            {
                "min_validation_precision": point["min_validation_precision"],
                "threshold": point["threshold"],
                "validation": point["validation"],
                "source_recall": float((direct_scores >= point["threshold"]).mean()),
            }
        )
    direct_fpr = []
    for point in baseline["direct_operating_points"]:
        direct_fpr.append(
            {
                "validation_fpr_budget": point["validation_fpr_budget"],
                "threshold": point["threshold"],
                "validation": point["validation"],
                "source_recall": float((direct_scores >= point["threshold"]).mean()),
            }
        )

    _, indirect_validation = split_fit_validation(references["indirect_train"])
    sensor = artifact["channels"]["untrusted_content"]
    validation_scores = _score_paragraphs(
        sensor["model"], [row["text"] for row in indirect_validation]
    )
    validation_labels = [row["label"] for row in indirect_validation]
    indirect_fpr = []
    for budget in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        threshold = choose_threshold(validation_labels, validation_scores, budget)
        indirect_fpr.append(
            {
                "validation_fpr_budget": budget,
                "threshold": threshold,
                "validation": _validation_rates(
                    indirect_validation, validation_scores, threshold
                ),
                "source_recall": float((indirect_scores >= threshold).mean()),
            }
        )

    indirect_by_budget = {
        point["validation_fpr_budget"]: point for point in indirect_fpr
    }
    combined = []
    for direct_point in direct_fpr:
        budget = direct_point["validation_fpr_budget"]
        if budget not in indirect_by_budget:
            continue
        indirect_point = indirect_by_budget[budget]
        elevated = np.logical_or(
            direct_scores >= direct_point["threshold"],
            indirect_scores >= indirect_point["threshold"],
        )
        combined.append(
            {
                "component_validation_fpr_budget": budget,
                "direct_threshold": direct_point["threshold"],
                "indirect_threshold": indirect_point["threshold"],
                "source_recall": float(elevated.mean()),
                "combined_fpr_not_estimated": True,
            }
        )
    return {
        "selection_note": (
            "Every threshold was selected on the original grouped direct or BIPIA "
            "validation rows before scoring this positive-only source."
        ),
        "direct_precision_profiles": direct_precision,
        "direct_fpr_diagnostics": direct_fpr,
        "indirect_fpr_diagnostics": indirect_fpr,
        "combined_component_budget_diagnostics": combined,
    }


def _validate_policy_references(rows: list[dict], scenarios: list[dict]) -> None:
    by_id = {row["source_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate processed Nemotron source IDs")
    for scenario in scenarios:
        reference = scenario["source_reference"]
        row = by_id.get(reference["source_id"])
        if row is None:
            raise ValueError(f"missing Nemotron source ID {reference['source_id']}")
        for field in GROUP_FIELDS:
            if reference[field] != row[field]:
                raise ValueError(
                    f"Nemotron source reference mismatch for {reference['source_id']}: "
                    f"{field}"
                )


def _overlap_audit(source_rows: list[dict], references: dict[str, list[dict]]) -> dict:
    source = [normalize_text(row["text"]) for row in source_rows]
    reference = {
        name: [normalize_text(row["text"]) for row in rows]
        for name, rows in references.items()
    }
    reference_sets = {name: set(texts) for name, texts in reference.items()}
    exact_matches = {
        name: {
            index for index, text in enumerate(source) if text in reference_sets[name]
        }
        for name in reference
    }
    exact_any = set().union(*exact_matches.values())

    corpus = source + [text for texts in reference.values() for text in texts]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(corpus)
    source_matrix = matrix[: len(source)]
    offset = len(source)
    near_by_partition = {}
    near_matches = {}
    near_any: set[int] = set()
    for name, texts in reference.items():
        partition_matrix = matrix[offset : offset + len(texts)]
        offset += len(texts)
        if not texts:
            near_by_partition[name] = 0
            near_matches[name] = set()
            continue
        distances, _ = (
            NearestNeighbors(
                n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1
            )
            .fit(partition_matrix)
            .kneighbors(source_matrix)
        )
        matches = {
            index
            for index, distance in enumerate(distances[:, 0])
            if index not in exact_matches[name]
            and 1.0 - float(distance) >= NEAR_THRESHOLD
        }
        near_matches[name] = matches
        near_by_partition[name] = len(matches)
        near_any.update(matches)

    fit_exact = set().union(
        *(matches for name, matches in exact_matches.items() if name in FIT_PARTITIONS)
    )
    fit_near = set().union(
        *(matches for name, matches in near_matches.items() if name in FIT_PARTITIONS)
    )
    return {
        "source_unique_texts": len(source),
        "reference_partitions": {name: len(texts) for name, texts in reference.items()},
        "normalized_exact": {
            "source_texts_with_any_match": len(exact_any),
            "source_texts_with_fit_match": len(fit_exact),
            "by_partition": {
                name: len(matches) for name, matches in exact_matches.items()
            },
        },
        "near": {
            "method": "word 1-2 gram TF-IDF cosine on normalized text",
            "threshold": NEAR_THRESHOLD,
            "excludes_exact_matches": True,
            "source_texts_with_any_match": len(near_any),
            "source_texts_with_fit_match": len(fit_near),
            "by_partition": near_by_partition,
        },
    }


def run(
    data_dir: Path = Path("data/processed"),
    artifact_path: Path = Path("artifacts/guard_bundle.joblib"),
    reports_dir: Path = Path("reports"),
) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(data_dir / f"{SOURCE}.jsonl")
    references = {
        path.stem: read_jsonl(path)
        for path in sorted(data_dir.glob("*.jsonl"))
        if path.stem != SOURCE
    }
    artifact = joblib.load(artifact_path)
    baseline = json.loads((reports_dir / "baseline.json").read_text(encoding="utf-8"))
    direct = artifact["channels"]["direct_user"]
    indirect = artifact["channels"]["untrusted_content"]
    texts = [row["text"] for row in rows]
    start = time.perf_counter()
    direct_scores = direct["model"].predict_proba(
        [normalize_text(text) for text in texts]
    )[:, 1]
    direct_seconds = time.perf_counter() - start
    start = time.perf_counter()
    indirect_scores = _score_paragraphs(indirect["model"], texts)
    indirect_seconds = time.perf_counter() - start
    direct_elevated = direct_scores >= direct["threshold"]
    indirect_elevated = indirect_scores >= indirect["threshold"]
    combined = np.logical_or(direct_elevated, indirect_elevated).astype(float)

    source_scenarios = [
        scenario
        for scenario in SCENARIOS
        if scenario.get("source_reference", {}).get("dataset") == SOURCE
    ]
    _validate_policy_references(rows, source_scenarios)
    policy = []
    for scenario in source_scenarios:
        allowed, reason = authorize(
            REFERENCE_POLICY, scenario["action"], scenario["context"]
        )
        policy.append(
            {
                "source_reference": scenario["source_reference"],
                "reference_monitor_committed": allowed,
                "reason": reason,
            }
        )

    profile = manifest["source_profiles"][SOURCE]
    output = manifest["outputs"][SOURCE]
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            **manifest["sources"][SOURCE],
            "download_sha256": manifest["download_sha256"][
                "nemotron_agentic_ipi/train.jsonl"
            ],
        },
        "projection": {
            **profile,
            "retained_rows": len(rows),
            "removed_exact_duplicates": manifest["deduplication"][SOURCE]["duplicates"],
            "blocked_by_fit_exact_match": manifest["deduplication"][SOURCE][
                "blocked_by_train"
            ],
            "processed_sha256": output["sha256"],
            "retained_group_counts": _group_counts(rows),
        },
        "overlap": _overlap_audit(rows, references),
        "artifact_sha256": _sha256(artifact_path),
        "detectors": {
            "direct_user_fallback_diagnostic": {
                **_score_summary(rows, direct_scores, direct["threshold"]),
                "seconds": direct_seconds,
                "not_direct_user_fit": True,
            },
            "indirect_sensor": {
                **_score_summary(rows, indirect_scores, indirect["threshold"]),
                "seconds": indirect_seconds,
            },
            "combined_untrusted_shadow": _score_summary(rows, combined, 0.5),
        },
        "operating_point_diagnostics": _operating_point_diagnostics(
            direct_scores,
            indirect_scores,
            references,
            artifact,
            baseline,
        ),
        "reference_monitor_scenarios": policy,
        "limitations": [
            "The suite is fully synthetic, positive-only, and filtered for attacks that succeeded against the source defender.",
            "It has no benign controls and cannot estimate false-positive rate, precision, benign task utility, or production safety.",
            "Recall measures the projected injection text only, not execution of the original agent environment or deterministic trace verifier.",
            "The direct-user sensor is reported only as the fallback used for untrusted content, not as evidence for direct-user model fit.",
            "Near-overlap is heuristic; fuzzy matches are disclosed and retained as a separate evaluation suite.",
            "Relaxed operating points are diagnostics selected on existing validation rows, not recommendations or tuning on this source.",
        ],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "nemotron-agentic-ipi.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(reports_dir / "nemotron-agentic-ipi.md", result)
    return result


def _write_markdown(path: Path, result: dict) -> None:
    projection = result["projection"]
    overlap = result["overlap"]
    lines = [
        "# Nemotron agentic indirect-injection audit",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "This is evaluation-only system/indirect evidence. The source is fully "
        "synthetic, positive-only, and filtered for attacks that succeeded against "
        "its defender; it is not benign-utility, false-positive, or production evidence.",
        "",
        "## Pinned source and safe projection",
        "",
        f"- Revision: `{result['source']['revision']}`",
        f"- License: {result['source']['license']}",
        f"- File SHA-256: `{result['source']['download_sha256']}`",
        f"- Raw source rows / exact-unique retained texts: "
        f"{projection['raw_rows']} / {projection['retained_rows']}",
        f"- Exact duplicates removed / exact fit overlaps blocked: "
        f"{projection['removed_exact_duplicates']} / "
        f"{projection['blocked_by_fit_exact_match']}",
        "- Persisted content: `injection.injection_text` plus domain, attack category, "
        "injection vector, and target-tool grouping metadata.",
        "- Omitted: source environment and its synthetic identity records, system/user "
        "prompts, tool schemas, injection goal, target arguments, and provider/model "
        "responses. The public synthetic injection text itself is retained because it "
        "is the detector input.",
        "",
        "## Leakage audit",
        "",
        f"Across every other processed fit/evaluation output, "
        f"{overlap['normalized_exact']['source_texts_with_any_match']} retained source "
        "texts have an exact normalized match and "
        f"{overlap['near']['source_texts_with_any_match']} have a non-exact near match "
        f"at TF-IDF cosine >= {overlap['near']['threshold']:.2f}. Fit-only counts are "
        f"{overlap['normalized_exact']['source_texts_with_fit_match']} exact and "
        f"{overlap['near']['source_texts_with_fit_match']} near. Exact fit overlaps are "
        "blocked by the data builder; fuzzy/evaluation matches remain disclosed and the "
        "suite stays separate.",
        "",
        "## Detector recall",
        "",
        "| Signal | Elevated / rows | Recall |",
        "|---|---:|---:|",
    ]
    for name, metrics in result["detectors"].items():
        lines.append(
            f"| {name} | {metrics['elevated']} / {metrics['rows']} | "
            f"{metrics['recall']:.4f} |"
        )
    diagnostics = result["operating_point_diagnostics"]
    lines += [
        "",
        "## Validation-anchored operating-point diagnostics",
        "",
        diagnostics["selection_note"],
        "",
        "| Direct validation FPR budget | Threshold | Observed validation FPR | Source recall |",
        "|---:|---:|---:|---:|",
    ]
    for point in diagnostics["direct_fpr_diagnostics"]:
        lines.append(
            f"| {point['validation_fpr_budget']:.1%} | {point['threshold']:.6f} | "
            f"{point['validation']['fpr']:.4f} | {point['source_recall']:.4f} |"
        )
    lines += [
        "",
        "| Indirect BIPIA-validation FPR budget | Threshold | Observed validation FPR | Validation recall | Source recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    for point in diagnostics["indirect_fpr_diagnostics"]:
        lines.append(
            f"| {point['validation_fpr_budget']:.1%} | {point['threshold']:.6f} | "
            f"{point['validation']['fpr']:.4f} | "
            f"{point['validation']['recall']:.4f} | {point['source_recall']:.4f} |"
        )
    lines += [
        "",
        "| Per-component validation FPR budget | Combined source recall |",
        "|---:|---:|",
    ]
    for point in diagnostics["combined_component_budget_diagnostics"]:
        lines.append(
            f"| {point['component_validation_fpr_budget']:.1%} | "
            f"{point['source_recall']:.4f} |"
        )
    lines += [
        "",
        "The combined rows do not estimate a combined FPR: the two component budgets "
        "come from different validation mixtures and OR aggregation accumulates false "
        "signals. The source did not choose any threshold.",
    ]
    for detector, metrics in result["detectors"].items():
        lines += ["", f"### {detector} by source grouping", ""]
        for field, groups in metrics["by_group"].items():
            lines += [
                "",
                f"#### {field}",
                "",
                "| Value | Elevated / rows | Recall |",
                "|---|---:|---:|",
            ]
            for value, group in groups.items():
                lines.append(
                    f"| {value} | {group['elevated']} / {group['rows']} | "
                    f"{group['recall']:.4f} |"
                )
    policy = result["reference_monitor_scenarios"]
    lines += [
        "",
        "## Deterministic containment scenarios",
        "",
        f"The reference monitor commits {sum(item['reference_monitor_committed'] for item in policy)}/"
        f"{len(policy)} representative unauthorized actions. These scenarios copy only "
        "safe source IDs and categorical grouping metadata, using local synthetic "
        "canaries instead of source environment data.",
        "",
        "## Limits",
        "",
        *[f"- {limitation}" for limitation in result["limitations"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/guard_bundle.joblib")
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    result = run(args.data_dir, args.artifact, args.reports_dir)
    print(
        json.dumps(
            {
                "rows": result["projection"]["retained_rows"],
                "indirect_recall": result["detectors"]["indirect_sensor"]["recall"],
                "combined_recall": result["detectors"]["combined_untrusted_shadow"][
                    "recall"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
