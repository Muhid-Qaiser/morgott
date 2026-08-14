"""Run a bounded, deterministic throughput and polarity canary.

This never launches the full baseline panel. It samples exactly 4,096 rows
across the four standard slices, records no prompt text or row IDs, and reports
a conservative projection against the 60-minute full-panel gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from morgott.models.mmbert.core import file_sha256
from morgott.models.mmbert.evaluate import (
    _identity_sha256,
    _metrics,
    _score_panel_sha256,
)

from .adapters import SMOKE_ATTACK, SMOKE_BENIGN, build_baseline
from .run import REDTEAM_PATH, build_panel, score_rows

CANARY_SCHEMA_VERSION = 2
CANARY_ROWS = 4_096
CANARY_SEED = 42
FULL_RUNTIME_GATE_SECONDS = 60 * 60
SLICE_NAMES = (
    "calibration",
    "canonical_dev_test",
    "promptshield_test",
    "sep",
)


def _sample(rows: list[dict], *, count: int, label: str) -> list[dict]:
    """Select a stable hash-ranked sample without persisting private rows."""
    if count < 1 or count > len(rows):
        raise ValueError(f"invalid {label} sample size {count} for {len(rows)} rows")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"guard-canary-v1\0{CANARY_SEED}\0{label}\0{row['id']}".encode()
        ).digest(),
    )
    return ranked[:count]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quality_at_reference_cutoff(baseline, scored: dict) -> dict:
    native = baseline.spec.native_threshold
    threshold = 0.5 if native is None else float(native)
    metrics = {
        name: _metrics(value["labels"], value["scores"], threshold)
        for name, value in scored.items()
    }
    if native is None:
        return {"quality_at_fixed_cutoff_0_5": metrics}
    if threshold == 0.5:
        # Preserve the archived ProtectAI schema-1 field exactly.
        return {"quality_at_native_cutoff_0_5": metrics}
    return {
        "quality_at_native_cutoff": {
            "threshold": threshold,
            "by_slice": metrics,
        }
    }


def _legacy_single_weight_sha256(model_identity: object) -> str | None:
    if not isinstance(model_identity, dict):
        return None
    files = model_identity.get("files")
    record = files.get("model.safetensors") if isinstance(files, dict) else None
    value = record.get("sha256") if isinstance(record, dict) else None
    return value if isinstance(value, str) else None


def _source_hashes(
    *, data_dir: Path, external_dir: Path, pairs: Path, redteam: Path
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    mmbert = root / "src/morgott/models/mmbert"
    paths = {
        "data_manifest": data_dir / "manifest.json",
        "external_manifest": external_dir / "manifest.json",
        "matched_pairs": pairs,
        "redteam_reserve": redteam,
        "canary_source": Path(__file__),
        "adapter_source": Path(__file__).with_name("adapters.py"),
        "harness_source": Path(__file__).with_name("run.py"),
        "core_source": mmbert / "core.py",
        "data_source": mmbert / "data.py",
        "detector_source": root / "src/morgott/models/detector.py",
        "evaluation_source": mmbert / "evaluate.py",
        "inference_source": mmbert / "inference.py",
        "training_source": mmbert / "train.py",
        "normalization_source": root / "src/morgott/normalization.py",
        "overlap_source": root / "src/morgott/overlap.py",
        "lockfile": root / "uv.lock",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def run_canary(
    slug: str,
    *,
    output: Path,
    batch_size: int | None,
    require_panel_sha256: str,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    redteam: Path,
) -> tuple[Path, bool]:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the guard canary")

    panel = build_panel(
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        redteam=redteam,
        allow_drift=False,
    )
    if panel["panel_sha256"] != require_panel_sha256:
        raise ValueError(
            "panel identity mismatch: expected "
            f"{require_panel_sha256}, assembled {panel['panel_sha256']}"
        )

    per_slice = CANARY_ROWS // len(SLICE_NAMES)
    sampled = {
        name: _sample(panel["slices"][name], count=per_slice, label=name)
        for name in SLICE_NAMES
    }
    baseline = build_baseline(slug, batch_size=batch_size)
    batch_size = baseline.batch_size
    if baseline.spec.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported; stop and review before changing dtype")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}-"))
    try:
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        baseline.load()
        load_seconds = time.perf_counter() - load_started

        smoke_scores, smoke_overflow = baseline.score([SMOKE_BENIGN, SMOKE_ATTACK])
        if not smoke_scores[1] > smoke_scores[0]:
            raise RuntimeError("explicit canary polarity check failed")

        score_started = time.perf_counter()
        scored = {
            name: score_rows(
                baseline,
                rows,
                batch_size=batch_size,
                label=f"canary_{name}",
            )
            for name, rows in sampled.items()
        }
        score_seconds = time.perf_counter() - score_started

        projected_standard_seconds = sum(
            len(panel["slices"][name]) / scored[name]["runtime"]["rows_per_second"]
            for name in SLICE_NAMES
        )
        slowest_rate = min(
            value["runtime"]["rows_per_second"] for value in scored.values()
        )
        projected_redteam_seconds = len(panel["redteam"]) / slowest_rate
        projected_full_seconds = (
            load_seconds + projected_standard_seconds + projected_redteam_seconds
        )
        passes = projected_full_seconds <= FULL_RUNTIME_GATE_SECONDS

        baseline_description = baseline.describe()
        model_identity = baseline_description["model_identity"]
        report = {
            "schema_version": CANARY_SCHEMA_VERSION,
            "purpose": "bounded guard throughput, identity, and polarity canary",
            "advisory_only": True,
            "full_evaluation_launched": False,
            "baseline": baseline_description,
            "canary": {
                "sampling": "lowest SHA-256 rank of guard-canary-v1, seed, slice, and row ID",
                "seed": CANARY_SEED,
                "rows": CANARY_ROWS,
                "rows_per_slice": per_slice,
                "ordered_row_identity_sha256": {
                    name: _identity_sha256(rows) for name, rows in sampled.items()
                },
                "ordered_scoring_input_sha256": {
                    name: _score_panel_sha256(rows) for name, rows in sampled.items()
                },
                "label_counts": {
                    name: {
                        "negative": sum(row["label"] == 0 for row in rows),
                        "positive": sum(row["label"] == 1 for row in rows),
                    }
                    for name, rows in sampled.items()
                },
            },
            "polarity_smoke": {
                "benign_score": float(smoke_scores[0]),
                "attack_score": float(smoke_scores[1]),
                "attack_greater_than_benign": True,
                "overflow": [bool(value) for value in smoke_overflow],
            },
            **_quality_at_reference_cutoff(baseline, scored),
            "runtime": {
                "device": torch.cuda.get_device_name(),
                "batch_size": batch_size,
                "load_seconds": load_seconds,
                "score_seconds": score_seconds,
                "rows_per_second": CANARY_ROWS / score_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "by_slice": {name: value["runtime"] for name, value in scored.items()},
            },
            "full_runtime_gate": {
                "maximum_seconds": FULL_RUNTIME_GATE_SECONDS,
                "projected_standard_seconds": projected_standard_seconds,
                "projected_redteam_seconds_conservative": projected_redteam_seconds,
                "projected_total_seconds_including_load": projected_full_seconds,
                "passes": passes,
                "full_panel_rows": sum(len(rows) for rows in panel["slices"].values())
                + len(panel["redteam"]),
                "decision": "report_to_owner_before_full_scoring",
            },
            "panel": {
                "panel_sha256": panel["panel_sha256"],
                "row_identity_sha256": panel["row_identity_sha256"],
                "redteam_row_identity_sha256": _identity_sha256(panel["redteam"]),
                "rows": {name: len(rows) for name, rows in panel["slices"].items()},
                "redteam_rows": len(panel["redteam"]),
            },
            "model_identity_sha256": _canonical_sha256(model_identity),
            "model_weights_sha256": _legacy_single_weight_sha256(model_identity),
            "source_sha256": _source_hashes(
                data_dir=data_dir,
                external_dir=external_dir,
                pairs=pairs,
                redteam=redteam,
            ),
            "limitations": [
                "This hash-ranked sample is a throughput and polarity canary, not a comparable full-panel result.",
                (
                    "Its native-cutoff metrics"
                    if baseline.spec.native_threshold is not None
                    else "Its fixed 0.5-cutoff metrics"
                )
                + " are descriptive and do not use the registered shared-threshold protocol.",
                "Already-open development data and undisclosed training overlap preclude a prospective claim.",
            ],
        }
        (temporary / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        baseline.unload()
        gc.collect()
        torch.cuda.empty_cache()
        os.replace(temporary, output)
        return output, passes
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="protectai-deberta-v3-prompt-injection-v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--require-panel-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--external-dir", type=Path, default=Path("artifacts/mmbert/data")
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data-archive/matched_pairs_20260726.jsonl.gz"),
    )
    parser.add_argument("--redteam", type=Path, default=REDTEAM_PATH)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    output, passes = run_canary(
        args.baseline,
        output=args.output,
        batch_size=args.batch_size,
        require_panel_sha256=args.require_panel_sha256,
        data_dir=args.data_dir,
        external_dir=args.external_dir,
        pairs=args.pairs,
        redteam=args.redteam,
    )
    print(output)
    return 0 if passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
