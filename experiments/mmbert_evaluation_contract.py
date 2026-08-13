"""Shared artifact checks for disposable mmBERT snapshot evaluations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from morgott.models.mmbert.core import file_sha256


def _invalid_constant(value: str):
    raise ValueError(f"JSON contains non-finite constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON contains duplicate key: {key!r}")
        value[key] = item
    return value


def strict_json_loads(value: str | bytes):
    """Decode strict JSON while rejecting duplicate keys and non-finite values."""

    return json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )


def read_json_object(path: Path, *, max_bytes: int) -> tuple[dict, str]:
    """Read a bounded strict-JSON object and return its byte digest."""

    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact does not exist: {path}")
    raw = path.read_bytes()
    if len(raw) < 2 or len(raw) > max_bytes:
        raise ValueError(f"JSON artifact has invalid size: {path}")
    try:
        value = strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"artifact is not strict UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def score_artifact(
    full_evaluation_path: Path,
    report: dict,
    *,
    score_columns: tuple[str, ...] | None = None,
    slice_names: tuple[str, ...] | None = None,
) -> tuple[Path, str]:
    """Resolve and hash a score artifact, with optional shape metadata checks."""

    scores = report.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("full evaluation has no score-artifact contract")
    relative = scores.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("full evaluation score path is invalid")
    parent = full_evaluation_path.resolve().parent
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent) or not path.is_file():
        raise ValueError("full evaluation score artifact is missing or escapes output")
    digest = file_sha256(path)
    if digest != scores.get("sha256"):
        raise ValueError("full evaluation score artifact hash mismatch")

    if score_columns is not None and scores.get("columns") != ["label", *score_columns]:
        raise ValueError("full evaluation score columns do not match the head")
    if slice_names is not None:
        slices = scores.get("slices")
        if not isinstance(slices, dict) or set(slices) != set(slice_names):
            raise ValueError("full evaluation score slices are invalid")
        expected_start = 0
        for name in slice_names:
            span = slices[name]
            if (
                not isinstance(span, list)
                or len(span) != 2
                or any(type(item) is not int for item in span)
                or span[0] != expected_start
                or span[1] <= span[0]
            ):
                raise ValueError("full evaluation score slices are not contiguous")
            expected_start = span[1]
    return path, digest


def transported_threshold(
    report: dict,
    *,
    target: str,
    allow_one: bool,
    require_canonical_recall: bool,
) -> float:
    """Validate one transported canonical-calibration operating threshold."""

    thresholds = report.get("thresholds")
    calibration = report.get("calibration")
    canonical = report.get("canonical_dev_test")
    if (
        not isinstance(thresholds, dict)
        or thresholds.get("source") != "canonical calibration components only"
        or not isinstance(thresholds.get("selected"), dict)
        or not isinstance(calibration, dict)
        or not isinstance(canonical, dict)
    ):
        raise ValueError("full evaluation threshold protocol changed")

    threshold = thresholds["selected"].get(target)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
    ):
        raise ValueError("full evaluation has no finite one-percent threshold")
    valid_upper_bound = threshold <= 1.0 if allow_one else threshold < 1.0
    if not 0.0 < threshold or not valid_upper_bound:
        raise ValueError("full evaluation has no finite one-percent threshold")

    component = calibration.get("component_thresholds", {}).get(target)
    calibration_metrics = calibration.get("metrics")
    canonical_metrics = canonical.get("metrics")
    canonical_recall = (
        canonical_metrics.get("recall") if isinstance(canonical_metrics, dict) else None
    )
    inconsistent = (
        not isinstance(component, dict)
        or component.get("status") != "available"
        or component.get("threshold") != threshold
        or not isinstance(calibration_metrics, dict)
        or calibration_metrics.get("threshold") != threshold
        or not isinstance(canonical_metrics, dict)
        or canonical_metrics.get("threshold") != threshold
    )
    if require_canonical_recall:
        inconsistent = inconsistent or (
            not isinstance(canonical_recall, (int, float))
            or isinstance(canonical_recall, bool)
            or not math.isfinite(canonical_recall)
            or not 0.0 <= canonical_recall <= 1.0
        )
    if inconsistent:
        raise ValueError(
            "full evaluation threshold evidence is internally inconsistent"
        )
    return float(threshold)
