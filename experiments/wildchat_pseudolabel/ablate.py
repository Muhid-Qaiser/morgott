"""Deterministic selection and source weights for WildChat ablations."""

from __future__ import annotations

import hashlib


SEED = "vulsight-wildchat-ablation-v1"
TARGETS = (5_000, 20_000, 50_000)


def sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


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
