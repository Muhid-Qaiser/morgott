"""Build a compact, provenance-preserving encoder pilot comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULT_PATHS = {
    "modernbert": HERE / "modernbert_pilot.json",
    "deberta": HERE / "deberta_pilot.json",
}
BASELINE_PATH = ROOT / "reports" / "baseline.json"
COMMON_SETS = (
    "external_hard_negatives",
    "toxic_chat",
    "prompt_injections",
    "multi_turn",
    "jailbreaks_over_time",
    "tensor_trust_attack",
    "tensor_trust_context",
    "bipia_payload",
    "bipia_context",
    "bipia_clean_context",
    "notinject",
    "oasst1_chat",
    "oasst1_position_stress",
)
METRIC_KEYS = (
    "rows",
    "positive",
    "negative",
    "true_positive",
    "false_positive",
    "recall",
    "precision",
    "fpr",
    "fpr_95_upper",
    "pr_auc",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_metrics(metrics: dict) -> dict:
    return {key: metrics.get(key) for key in METRIC_KEYS}


def assert_fair_protocol(results: dict[str, dict]) -> None:
    modernbert = results["modernbert"]
    deberta = results["deberta"]
    checks = {
        "protocol": (modernbert["protocol"], deberta["protocol"]),
        "training_subset": (
            modernbert["training_subset"],
            deberta["training_subset"],
        ),
        "input_sha256": (modernbert["input_sha256"], deberta["input_sha256"]),
        "default_precision_floor": (
            modernbert["default_precision_floor"],
            deberta["default_precision_floor"],
        ),
    }
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(f"unfair comparison; mismatched {', '.join(mismatches)}")


def _candidate_summary(result: dict) -> dict:
    profiles = {}
    for profile in result["direct_precision_profiles"]:
        floor = f"{int(profile['validation_precision_floor'] * 100)}"
        profiles[floor] = {
            "attained": profile["attained"],
            "threshold": profile["threshold"],
            "validation": (
                compact_metrics(profile["validation"])
                if profile["validation"] is not None
                else None
            ),
            "sets": {
                name: compact_metrics(profile["sets"][name]) for name in COMMON_SETS
            },
        }
    return {
        "model": result["model"],
        "precision_profiles": profiles,
        "fpr_diagnostics": {
            f"{point['validation_fpr_budget']:.3f}": {
                "threshold": point["threshold"],
                "validation": compact_metrics(point["validation"]),
            }
            for point in result["direct_fpr_diagnostics"]
        },
        "training": result["training"],
        "memory_preflight": result["memory_preflight"],
        "inference": {
            "batch_size": result["inference"]["batch_size"],
            "evaluation_peak_allocated_mib": result["inference"][
                "evaluation_peak_allocated_mib"
            ],
            "latency": result["inference"]["latency"],
        },
    }


def _control_summary(baseline: dict) -> dict:
    detector = baseline["detectors"]["char_ngram_logreg"]
    profiles = {}
    for profile in baseline["direct_precision_profiles"]:
        floor = f"{int(profile['min_validation_precision'] * 100)}"
        profiles[floor] = {
            "threshold": profile["threshold"],
            "validation": compact_metrics(profile["validation"]),
        }
    return {
        "name": "character 3-5 gram TF-IDF + balanced logistic regression",
        "threshold": detector["threshold"],
        "precision_profiles": profiles,
        "fpr_diagnostics": {
            f"{point['validation_fpr_budget']:.3f}": {
                "threshold": point["threshold"],
                "validation": compact_metrics(point["validation"]),
            }
            for point in baseline["direct_operating_points"]
        },
        "default_sets": {
            name: compact_metrics(detector["sets"][name]) for name in COMMON_SETS
        },
        "latency_us_per_sample": detector["latency_us_per_sample"],
    }


def build_comparison() -> dict:
    results = {name: _load(path) for name, path in RESULT_PATHS.items()}
    assert_fair_protocol(results)
    baseline = _load(BASELINE_PATH)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pilot_only": True,
        "source_sha256": {
            **{path.name: _sha256(path) for path in RESULT_PATHS.values()},
            str(BASELINE_PATH.relative_to(ROOT)): _sha256(BASELINE_PATH),
        },
        "fairness_checks": {
            "identical_protocol": True,
            "identical_training_subset": True,
            "identical_input_sha256": True,
            "training_row_order_sha256": results["modernbert"]["training_subset"][
                "ordered_row_ids_sha256"
            ],
            "validation_rows_untouched": results["modernbert"]["training_subset"][
                "validation_rows_untouched"
            ],
        },
        "control": _control_summary(baseline),
        "candidates": {
            name: _candidate_summary(result) for name, result in results.items()
        },
        "decision": {
            "promotion": "none",
            "modernbert": (
                "underfit in this one-epoch subsampled screen; this does not rule "
                "out a fuller end-to-end run"
            ),
            "deberta": (
                "interesting continuation candidate, but not promoted from one "
                "seed/epoch: modest validation gains coexist with hard-negative "
                "and multi-turn regressions and thresholds concentrated near 1"
            ),
            "architecture_claim": (
                "this pilot does not establish intrinsic backbone superiority"
            ),
        },
    }


def main() -> None:
    comparison = build_comparison()
    output = HERE / "comparison.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
