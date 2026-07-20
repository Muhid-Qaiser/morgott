#!/usr/bin/env python3
"""Create deterministic, source-weighted WildChat ablation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from vulsight_guard.detector import split_fit_validation


SEED = "vulsight-wildchat-ablation-v1"
TARGETS = (5_000, 20_000, 50_000)


def sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def report_path(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_accepted(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if (
            row.get("weak_label") is not True
            or row.get("label") != 0
            or row.get("label_basis") != "cross_family_model_weak_label"
            or not isinstance(row.get("sample_id"), str)
            or not isinstance(row.get("text"), str)
        ):
            raise ValueError("accepted weak-label row is malformed")
        if row["sample_id"] in seen:
            raise ValueError("duplicate accepted sample id")
        seen.add(row["sample_id"])
        provenance = row.get("judge_provenance")
        if not isinstance(provenance, list) or len(provenance) not in {2, 3}:
            raise ValueError("accepted row has invalid judge provenance")
        if any(
            item.get("label") != "benign" or item.get("confidence") != "high"
            for item in provenance
            if isinstance(item, dict)
        ) or not all(isinstance(item, dict) for item in provenance):
            raise ValueError("accepted row lacks unanimous high-confidence benign")
        if bool(row.get("third_audited")) != (len(provenance) == 3):
            raise ValueError("third-audit provenance mismatch")


def select(rows: list[dict], count: int, seed: str = SEED) -> list[dict]:
    if count > len(rows):
        raise ValueError("insufficient accepted rows")
    return sorted(rows, key=lambda row: sha256(f"{seed}\0{row['sample_id']}"))[:count]


def weights(base_positive: int, base_negative: int, weak_rows: int) -> dict[str, float]:
    if base_positive <= 0 or base_negative <= 0 or weak_rows < 0:
        raise ValueError("invalid class counts")
    total = base_positive + base_negative
    weak_mass = 0.05 if weak_rows else 0.0
    base_negative_mass = 0.5 - weak_mass
    return {
        "base_positive_per_row": 0.5 * total / base_positive,
        "base_negative_per_row": base_negative_mass * total / base_negative,
        "weak_negative_per_row": weak_mass * total / weak_rows if weak_rows else 0.0,
        "base_positive_total_mass": 0.5 * total,
        "base_negative_total_mass": base_negative_mass * total,
        "weak_negative_total_mass": weak_mass * total,
        "total_mass": float(total),
    }


def manifest(
    name: str,
    base_path: Path,
    base_rows: list[dict],
    accepted_path: Path,
    selected: list[dict],
) -> dict:
    fit_rows, validation_rows = split_fit_validation(base_rows)
    positive = sum(row.get("label") == 1 for row in fit_rows)
    negative = sum(row.get("label") == 0 for row in fit_rows)
    return {
        "schema_version": 1,
        "name": name,
        "base": {
            "path": str(base_path),
            "sha256": sha256(base_path.read_bytes()),
            "rows": len(base_rows),
            "fit_rows": len(fit_rows),
            "fit_positive": positive,
            "fit_negative": negative,
            "validation_rows": len(validation_rows),
        },
        "weak_source": {
            "path": str(accepted_path),
            "sha256": sha256(accepted_path.read_bytes()),
            "available_rows": len(read_jsonl(accepted_path)),
            "selected_rows": len(selected),
            "selected_sample_ids": [row["sample_id"] for row in selected],
            "selection_seed": SEED,
            "label_status": "model-only weak negatives",
        },
        "weights": {
            **weights(positive, negative, len(selected)),
            "instruction": "use these sample weights with class_weight=None",
            "weak_share_of_total_negative_mass": 0.1 if selected else 0.0,
        },
        "evaluation_status": "not_run",
        "promotion_status": "development only; never production FPR",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = Path(__file__).with_name("outputs") / "ablations"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=root / "data/processed/train.jsonl"
    )
    parser.add_argument(
        "--accepted",
        type=Path,
        default=Path(__file__).with_name("outputs") / "accepted.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument(
        "--report", type=Path, default=root / "reports/wildchat-ablation-plan.json"
    )
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="write ignored deterministic manifests; otherwise only validate and plan",
    )
    args = parser.parse_args()
    base_rows = read_jsonl(args.base)
    fit_rows, validation_rows = split_fit_validation(base_rows)
    accepted = read_jsonl(args.accepted)
    validate_accepted(accepted)
    plan = {
        "mode": "write-manifests" if args.write_manifests else "offline-plan",
        "base_rows": len(base_rows),
        "base_positive": sum(row.get("label") == 1 for row in base_rows),
        "base_fit_rows": len(fit_rows),
        "base_fit_positive": sum(row.get("label") == 1 for row in fit_rows),
        "base_validation_rows": len(validation_rows),
        "accepted_negative_rows": len(accepted),
        "first_comparison": [0, len(accepted)],
        "future_accepted_targets": list(TARGETS),
        "available_targets": [target for target in TARGETS if target <= len(accepted)],
        "source_weight_policy": "weak negatives receive 10% of total negative mass",
    }
    if not args.write_manifests:
        print(json.dumps(plan, sort_keys=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [("zero", [])]
    if accepted:
        candidates.append(
            (f"pilot_all_{len(accepted)}", select(accepted, len(accepted)))
        )
    candidates.extend(
        (f"accepted_{target}", select(accepted, target))
        for target in TARGETS
        if target <= len(accepted)
    )
    outputs = []
    for name, selected_rows in candidates:
        value = manifest(name, args.base, base_rows, args.accepted, selected_rows)
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        outputs.append(
            {
                "name": name,
                "rows": len(selected_rows),
                "path": report_path(path),
                "sha256": sha256(path.read_bytes()),
            }
        )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **plan,
        "manifests": outputs,
        "evaluation_note": (
            "Run zero versus all accepted-from-pilot first. Frozen public suites become "
            "repeated development evidence after they determine scaling."
        ),
        "metric_status": "no LLM-labelled row is evaluation ground truth",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {**plan, "report": str(args.report), "manifests": len(outputs)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
