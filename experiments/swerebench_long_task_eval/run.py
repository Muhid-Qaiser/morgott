#!/usr/bin/env python3
"""Freeze and score the pinned SWE-rebench legitimate-task denominator."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from morgott.data import file_sha256, text_hash
from morgott.models.downstream import MMBERT_HIGH, MMBERT_LOW_BY_CHANNEL
from morgott.models.mmbert.data import filter_small_training_sets
from morgott.models.mmbert.serving import MmbertRuntime
from morgott.sources.tasks import _sensitive_text_reasons

ROOT = Path(__file__).resolve().parents[2]
_BASE_SPEC = importlib.util.spec_from_file_location(
    "morgott_swebench_long_benign_eval",
    ROOT / "experiments" / "swebench_long_benign_eval" / "run.py",
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError("cannot load the retained SWE-bench evaluation helpers")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)

DEFAULT_OUTPUT = ROOT / "artifacts" / "swerebench_long_task_eval"
REVISION = "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"
SOURCE_FILES = {
    "test-00000-of-00002.parquet": {
        "bytes": 101_556_119,
        "sha256": "39d4791f12cf5ee2a2e56d47eeef559642a800534ff053e1ae3acab0a0c87067",
    },
    "test-00001-of-00002.parquet": {
        "bytes": 109_723_276,
        "sha256": "c50af8bffbfe70fc3a89b2e47825f299bc04c1058318fe2263b7e4857f8193d7",
    },
}
EXPECTED_ROWS = 21_336
PANEL_FIELDS = {
    "base_commit",
    "created_at",
    "instance_id",
    "length_bucket",
    "normalized_text_sha256",
    "panel_id",
    "repository",
    "split_group_id",
    "text_chars",
    "text_sha256",
}
EXPERIMENT_FILES = (
    "experiments/swerebench_long_task_eval/README.md",
    "experiments/swerebench_long_task_eval/run.py",
    "experiments/swerebench_long_task_eval/test_run.py",
    "experiments/swebench_long_benign_eval/run.py",
)


def _source_contract(source_dir: Path) -> dict:
    contract = {}
    for name, expected in SOURCE_FILES.items():
        path = source_dir / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"missing or wrong-size pinned source file: {name}")
        digest = file_sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"pinned source digest mismatch: {name}")
        contract[name] = {**expected, "sha256": digest}
    return contract


def _source_rows(source_dir: Path) -> list[dict]:
    columns = ("instance_id", "repo", "base_commit", "created_at", "problem_statement")
    rows = []
    for name in SOURCE_FILES:
        parquet = pq.ParquetFile(source_dir / name)
        for batch in parquet.iter_batches(columns=columns, batch_size=256):
            rows.extend(batch.to_pylist())
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} SWE-rebench test rows, found {len(rows)}"
        )
    ids = [row.get("instance_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(
        set(ids)
    ):
        raise ValueError("SWE-rebench instance IDs are invalid or duplicated")
    for row in rows:
        if not isinstance(row.get("repo"), str) or not row["repo"]:
            raise ValueError("SWE-rebench row has no repository")
        if (
            not isinstance(row.get("problem_statement"), str)
            or not row["problem_statement"].strip()
        ):
            raise ValueError("SWE-rebench row has no problem statement")
    return rows


def _length_bucket(characters: int) -> str:
    if characters < 2_048:
        return "under_2048_chars"
    if characters < 4_096:
        return "2048_to_4095_chars"
    if characters < 8_192:
        return "4096_to_8191_chars"
    return "8192_chars_or_more"


def _panel_row(row: dict) -> dict:
    text = row["problem_statement"]
    instance_id = row["instance_id"]
    return {
        "panel_id": f"swerebench:{instance_id}",
        "instance_id": instance_id,
        "repository": row["repo"],
        "base_commit": row.get("base_commit"),
        "created_at": str(row.get("created_at")),
        "split_group_id": f"swerebench:repo:{row['repo']}",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "normalized_text_sha256": text_hash(text),
        "text_chars": len(text),
        "length_bucket": _length_bucket(len(text)),
    }


def _prepare(source_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(
            "SWE-rebench evaluation is write-once; use a fresh output"
        )
    source_contract = _source_contract(source_dir)
    source_rows = _source_rows(source_dir)
    privacy = Counter()
    privacy_excluded = 0
    unique = {}
    exact_duplicates = 0
    for row in sorted(source_rows, key=lambda value: value["instance_id"]):
        text = row["problem_statement"]
        reasons = _sensitive_text_reasons(text)
        if reasons:
            privacy_excluded += 1
            privacy.update(reasons)
            continue
        digest = text_hash(text)
        if digest in unique:
            exact_duplicates += 1
            continue
        unique[digest] = row
    candidates = {
        "swerebench": [
            {
                "id": f"swerebench:{row['instance_id']}",
                "text": row["problem_statement"],
                "source": "swerebench",
                "label": 0,
            }
            for row in unique.values()
        ]
    }
    reference_counts = Counter()
    kept, removed = filter_small_training_sets(
        candidates,
        base._reference_rows(base._historical_manifest(), reference_counts),
    )
    kept_ids = {row["id"] for row in kept["swerebench"]}
    panel = [
        _panel_row(row)
        for row in unique.values()
        if f"swerebench:{row['instance_id']}" in kept_ids
    ]
    panel.sort(key=lambda row: row["panel_id"])
    if not panel:
        raise ValueError("privacy and fit-overlap filtering removed the complete panel")
    panel_spec = base._write_gzip_jsonl(output / "panel.jsonl.gz", panel)
    manifest = {
        "schema_version": 1,
        "purpose": "frozen repository-grouped long legitimate-task workload diagnostic",
        "source": {
            "name": "nebius/SWE-rebench",
            "revision": REVISION,
            "files": source_contract,
            "projection": "problem_statement only",
            "raw_text_retained_in_artifacts": False,
        },
        "selection": {
            "raw_test_rows": len(source_rows),
            "privacy_excluded_rows": privacy_excluded,
            "privacy_reason_matches": dict(sorted(privacy.items())),
            "normalized_exact_duplicates_excluded": exact_duplicates,
            "fit_overlap_removed": removed["swerebench"],
            "fit_reference_rows": dict(sorted(reference_counts.items())),
            "retained": len(panel),
        },
        "population": {
            "repositories": len({row["repository"] for row in panel}),
            "length_buckets": dict(
                sorted(Counter(row["length_bucket"] for row in panel).items())
            ),
        },
        "analysis_contract": {
            "primary": [
                "registered-gate restriction and review load",
                "score distribution and repository macro load",
                "token and window counts by fixed length bucket",
            ],
            "threshold_selection": "none",
            "claim_boundary": "legitimate-task workload, not a production false-positive estimate",
        },
        "experiment_contract": base._file_contract(EXPERIMENT_FILES),
        "panel": panel_spec,
    }
    base._write_json(output / "manifest.json", manifest)
    return manifest


def _load(output: Path) -> tuple[list[dict], dict, str]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get(
        "experiment_contract"
    ) != base._file_contract(EXPERIMENT_FILES):
        raise ValueError("SWE-rebench frozen contract changed")
    compressed = (output / "panel.jsonl.gz").read_bytes()
    spec = manifest["panel"]
    if (
        hashlib.sha256(compressed).hexdigest() != spec["sha256"]
        or hashlib.sha256(gzip.decompress(compressed)).hexdigest()
        != spec["content_sha256"]
    ):
        raise ValueError("SWE-rebench panel hash mismatch")
    panel = base._read_jsonl(output / "panel.jsonl.gz")
    if (
        len(panel) != spec["rows"]
        or len({row.get("panel_id") for row in panel}) != len(panel)
        or any(set(row) != PANEL_FIELDS for row in panel)
    ):
        raise ValueError("SWE-rebench panel schema mismatch")
    return panel, manifest, file_sha256(manifest_path)


def _texts(source_dir: Path, panel: list[dict]) -> dict[str, str]:
    wanted = {row["instance_id"]: row for row in panel}
    texts = {}
    for source_row in _source_rows(source_dir):
        panel_row = wanted.get(source_row["instance_id"])
        if panel_row is None:
            continue
        text = source_row["problem_statement"]
        if hashlib.sha256(text.encode()).hexdigest() != panel_row["text_sha256"]:
            raise ValueError("SWE-rebench source text changed after panel freeze")
        texts[panel_row["panel_id"]] = text
    if len(texts) != len(panel):
        raise ValueError("SWE-rebench frozen source rows are incomplete")
    return texts


def _run(source_dir: Path, output: Path) -> dict:
    result_path = output / "local_results.jsonl.gz"
    if result_path.exists():
        raise FileExistsError("SWE-rebench result is write-once; use analyze")
    _source_contract(source_dir)
    panel, _, manifest_sha256 = _load(output)
    texts = _texts(source_dir, panel)
    runtime = MmbertRuntime.from_artifacts(ROOT / "model-artifacts.json")
    results = []
    for index, row in enumerate(panel, 1):
        started = time.perf_counter()
        prepared = runtime.prepare(texts[row["panel_id"]])
        scores = runtime.score(prepared.windows)
        low = sum(score < MMBERT_LOW_BY_CHANNEL["direct_user"] for score in scores)
        high = sum(score >= MMBERT_HIGH for score in scores)
        results.append(
            {
                "manifest_sha256": manifest_sha256,
                "panel_id": row["panel_id"],
                "assessment": {
                    "token_count": prepared.token_count,
                    "window_count": len(scores),
                    "low_windows": low,
                    "middle_windows": len(scores) - low - high,
                    "high_windows": high,
                    "max_mmbert_score": max(scores),
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                },
            }
        )
        if index % 250 == 0 or index == len(panel):
            print(f"scored {index}/{len(panel)}", flush=True)
    return base._write_gzip_jsonl(result_path, results)


def _valid_assessment(assessment: object) -> bool:
    fields = {
        "high_windows",
        "latency_ms",
        "low_windows",
        "max_mmbert_score",
        "middle_windows",
        "token_count",
        "window_count",
    }
    if not isinstance(assessment, dict) or set(assessment) != fields:
        return False
    if any(
        type(assessment[field]) is not int or assessment[field] < 0
        for field in (
            "high_windows",
            "low_windows",
            "middle_windows",
            "token_count",
            "window_count",
        )
    ):
        return False
    score = assessment["max_mmbert_score"]
    latency = assessment["latency_ms"]
    return (
        assessment["window_count"] >= 1
        and assessment["window_count"]
        == assessment["low_windows"]
        + assessment["middle_windows"]
        + assessment["high_windows"]
        and isinstance(score, int | float)
        and not isinstance(score, bool)
        and math.isfinite(score)
        and 0 <= score <= 1
        and isinstance(latency, int | float)
        and not isinstance(latency, bool)
        and math.isfinite(latency)
        and latency >= 0
    )


def _analyze(output: Path) -> dict:
    panel, manifest, manifest_sha256 = _load(output)
    results = base._read_jsonl(output / "local_results.jsonl.gz")
    expected = {row["panel_id"]: row for row in panel}
    by_id = {}
    for row in results:
        assessment = row.get("assessment")
        if (
            row.get("manifest_sha256") != manifest_sha256
            or row.get("panel_id") not in expected
            or row["panel_id"] in by_id
            or not _valid_assessment(assessment)
        ):
            raise ValueError("invalid SWE-rebench result row")
        by_id[row["panel_id"]] = assessment
    if set(by_id) != set(expected):
        raise ValueError("SWE-rebench result is incomplete")
    joined = [(row, by_id[row["panel_id"]]) for row in panel]
    overall = base._rate(joined)
    grouped = defaultdict(list)
    for panel_row, assessment in joined:
        grouped[panel_row["repository"]].append((panel_row, assessment))
    repository_rates = [base._rate(rows) for rows in grouped.values()]
    maxima = sorted(assessment["max_mmbert_score"] for _, assessment in joined)
    quantiles = statistics.quantiles(maxima, n=100, method="inclusive")
    summary = {
        "schema_version": 1,
        "purpose": manifest["purpose"],
        "panel_rows": len(joined),
        "repositories": len(grouped),
        "registered_gate": overall,
        "repository_macro": {
            "hard_restriction_rate": statistics.fmean(
                row["hard_restriction_rate"] for row in repository_rates
            ),
            "review_required_rate": statistics.fmean(
                row["review_required_rate"] for row in repository_rates
            ),
        },
        "by_length": base._by(joined, "length_bucket"),
        "maximum_score_quantiles": {
            "p50": quantiles[49],
            "p90": quantiles[89],
            "p95": quantiles[94],
            "p99": quantiles[98],
        },
        "operation": {
            "total_tokens": sum(row["token_count"] for row in by_id.values()),
            "total_windows": sum(row["window_count"] for row in by_id.values()),
            "maximum_tokens": max(row["token_count"] for row in by_id.values()),
            "local_latency_seconds": sum(row["latency_ms"] for row in by_id.values())
            / 1_000,
        },
        "limitations": [
            "Public repository tasks are a legitimate-task workload, not a broad benignity adjudication.",
            "The result measures restriction and review load, not production FPR.",
            "The dataset is Python-repository concentrated and already open development evidence.",
        ],
    }
    base._write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command in {"prepare", "run"} and args.source_dir is None:
        parser.error("--source-dir is required for prepare and run")
    if args.command == "prepare":
        result = _prepare(args.source_dir.resolve(), output)
    elif args.command == "run":
        result = _run(args.source_dir.resolve(), output)
    else:
        result = _analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
