#!/usr/bin/env python3
"""Freeze and score a long-benign SWE-bench Verified development panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

from morgott.data import file_sha256, iter_verified_jsonl
from morgott.models.downstream import (
    MMBERT_HIGH,
    MMBERT_LOW_BY_CHANNEL,
    THRESHOLD_SHA256,
)
from morgott.models.mmbert.data import (
    external_rows,
    filter_small_training_sets,
    matched_pairs,
)
from morgott.models.mmbert.serving import MmbertRuntime

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "swebench_long_benign_eval"
MODEL_PROVENANCE_COMMIT = "91e8c829c8b39c8ff37a6ca2479c8fc057168d39"
MODEL_DATA_MANIFEST_SHA256 = (
    "27bdd9c244fbf479d699cb7c8d826385c0bd0f2f39e5154051db10e927c58f81"
)
SOURCE = "swebench_verified"
MAX_REMOTE_WINDOWS = 4_000
PANEL_FIELDS = {
    "input_channel",
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
    "experiments/swebench_long_benign_eval/README.md",
    "experiments/swebench_long_benign_eval/run.py",
    "experiments/swebench_long_benign_eval/test_run.py",
)
SELECTION_FILES = (
    "src/morgott/data.py",
    "src/morgott/normalization.py",
    "src/morgott/overlap.py",
    "src/morgott/models/mmbert/data.py",
)
MODEL_FILES = (
    "model-artifacts.json",
    "src/morgott/models/downstream.py",
    "src/morgott/models/mmbert/core.py",
    "src/morgott/models/mmbert/serving.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> dict:
    content = b"".join(_canonical_json(row) for row in rows)
    compressed = gzip.compress(content, compresslevel=9, mtime=0)
    _atomic_bytes(path, compressed)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "sha256": _sha256_bytes(compressed),
        "content_sha256": _sha256_bytes(content),
    }


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _file_contract(paths: tuple[str, ...]) -> dict:
    return {path: file_sha256(ROOT / path) for path in paths}


def _historical_manifest() -> dict:
    content = subprocess.check_output(
        ["git", "show", f"{MODEL_PROVENANCE_COMMIT}:data/manifest.json"],
        cwd=ROOT,
    )
    if _sha256_bytes(content) != MODEL_DATA_MANIFEST_SHA256:
        raise ValueError("registered model data manifest changed")
    return json.loads(content)


def _source_rows() -> tuple[list[dict], dict, dict]:
    manifest_path = ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest.get("source_outputs", {}).get(SOURCE)
    if not isinstance(spec, dict):
        raise ValueError(
            "SWE-bench Verified source is absent from the canonical corpus"
        )
    rows = list(iter_verified_jsonl(ROOT / "data" / spec["path"], spec["sha256"]))
    if len(rows) != spec["rows"]:
        raise ValueError("SWE-bench Verified source row count changed")
    ids = set()
    for row in rows:
        if (
            row.get("source") != SOURCE
            or row.get("source_role") != "dev_test"
            or row.get("routing_training_eligible") is not True
            or row.get("routing_label") != 0
            or row.get("security_label") != "benign"
            or row.get("input_channel") != "direct_user"
            or not isinstance(row.get("id"), str)
            or row["id"] in ids
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or not isinstance(row.get("source_repository"), str)
            or not row["source_repository"]
            or row.get("split_group_id")
            != f"swebench_verified:repo:{row.get('source_repository')}"
        ):
            raise ValueError("invalid SWE-bench Verified canonical row")
        ids.add(row["id"])
    return rows, manifest, spec


def _length_bucket(characters: int) -> str:
    if characters < 1_024:
        return "under_1024_chars"
    if characters < 2_048:
        return "1024_to_2047_chars"
    if characters < 4_096:
        return "2048_to_4095_chars"
    return "4096_chars_or_more"


def _panel_row(row: dict) -> dict:
    text = row["text"]
    return {
        "panel_id": row["id"],
        "text_sha256": _sha256_bytes(text.encode()),
        "normalized_text_sha256": row["normalized_text_sha256"],
        "text_chars": len(text),
        "repository": row["source_repository"],
        "instance_id": row["source_instance_id"],
        "split_group_id": row["split_group_id"],
        "length_bucket": _length_bucket(len(text)),
        "input_channel": "direct_user",
    }


def _reference_rows(historical: dict, counts: Counter):
    def canonical():
        for source, spec in sorted(historical["source_outputs"].items()):
            seen = 0
            for row in iter_verified_jsonl(
                ROOT / "data" / spec["path"], spec["sha256"]
            ):
                counts[f"canonical:{source}"] += 1
                seen += 1
                yield row
            if seen != spec["rows"]:
                raise ValueError(f"historical canonical row count changed: {source}")

    external = external_rows(ROOT / "artifacts" / "mmbert" / "data")[0]

    def counted(name: str, rows):
        for row in rows:
            counts[name] += 1
            yield row

    return chain(
        canonical(),
        counted("promptshield_train", external["promptshield_train"]),
        counted(
            "matched_pairs",
            chain.from_iterable(
                matched_pairs(ROOT / "data-archive" / "matched_pairs_20260726.jsonl.gz")
            ),
        ),
    )


def _prepare(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    panel_path = output / "panel.jsonl.gz"
    if manifest_path.exists() or panel_path.exists():
        raise FileExistsError("SWE-bench panel is write-once; use a fresh output")
    source_rows, data_manifest, source_spec = _source_rows()
    historical = _historical_manifest()
    if historical["source_outputs"] != {
        source: data_manifest["source_outputs"][source]
        for source in historical["source_outputs"]
    }:
        raise ValueError("preexisting canonical source outputs changed")
    candidates = {
        SOURCE: [
            {
                "id": row["id"],
                "text": row["text"],
                "source": SOURCE,
                "label": 0,
            }
            for row in source_rows
        ]
    }
    reference_counts = Counter()
    kept, removed = filter_small_training_sets(
        candidates,
        _reference_rows(historical, reference_counts),
    )
    kept_ids = {row["id"] for row in kept[SOURCE]}
    panel = [_panel_row(row) for row in source_rows if row["id"] in kept_ids]
    if not panel:
        raise ValueError("fit-overlap filtering removed the complete panel")
    panel_spec = _write_gzip_jsonl(panel_path, panel)
    repositories = Counter(row["repository"] for row in panel)
    lengths = Counter(row["length_bucket"] for row in panel)
    manifest = {
        "schema_version": 1,
        "purpose": "prospectively frozen canonical long-benign local-gate diagnostic",
        "source": {
            "name": SOURCE,
            "revision": data_manifest["sources"][SOURCE]["revision"],
            "source_output_sha256": source_spec["sha256"],
            "source_rows": source_spec["rows"],
            "data_manifest_sha256": file_sha256(ROOT / "data" / "manifest.json"),
            "projection": "problem_statement only",
            "input_channel": "direct_user",
            "raw_text_retained_in_artifacts": False,
            "label_basis": (
                "human-verified solvable software issue, not a safety annotation"
            ),
        },
        "selection": {
            "method": (
                "normalized, audit-strict, and conservative near-overlap removal "
                "against every source row present at retained-model fit time, "
                "PromptShield train, and retained matched pairs"
            ),
            "model_provenance_commit": MODEL_PROVENANCE_COMMIT,
            "model_data_manifest_sha256": MODEL_DATA_MANIFEST_SHA256,
            "reference_rows": dict(sorted(reference_counts.items())),
            "removed_by_reason": removed[SOURCE],
            "retained": len(panel),
        },
        "population": {
            "repositories": dict(sorted(repositories.items())),
            "length_buckets": dict(sorted(lengths.items())),
        },
        "analysis_contract": {
            "primary": [
                "document hard-restriction rate from the registered mmBERT high gate",
                "document review-required rate and total middle-window count",
            ],
            "hard_restriction_gate": "at most 1% of retained documents",
            "remote_continuation_rule": (
                "only when the hard-restriction gate passes and at most 4000 "
                "windows require DeepSeek review"
            ),
            "secondary": [
                "repository macro and per-repository route rates",
                "fixed character-length and single/multi-window route rates",
                "local scores, token and window counts, and latency",
            ],
            "threshold_selection": "none; use the registered local gate unchanged",
            "claim_boundary": (
                "benign-only public software issues cannot establish attack recall, "
                "balanced long-context robustness, prevalence, or production FPR"
            ),
        },
        "model_contract": {
            "threshold_sha256": THRESHOLD_SHA256,
            "low_direct_user": MMBERT_LOW_BY_CHANNEL["direct_user"],
            "high": MMBERT_HIGH,
            "files": _file_contract(MODEL_FILES),
        },
        "selection_contract": _file_contract(SELECTION_FILES),
        "experiment_contract": _file_contract(EXPERIMENT_FILES),
        "panel": panel_spec,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_panel(output: Path) -> tuple[list[dict], dict, str]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported SWE-bench evaluation manifest")
    panel_path = output / "panel.jsonl.gz"
    compressed = panel_path.read_bytes()
    spec = manifest["panel"]
    if (
        _sha256_bytes(compressed) != spec["sha256"]
        or _sha256_bytes(gzip.decompress(compressed)) != spec["content_sha256"]
    ):
        raise ValueError("SWE-bench panel hash mismatch")
    panel = _read_jsonl(panel_path)
    ids = [row.get("panel_id") for row in panel]
    if (
        len(panel) != spec["rows"]
        or len(ids) != len(set(ids))
        or any(
            set(row) != PANEL_FIELDS
            or row["input_channel"] != "direct_user"
            or row["length_bucket"] != _length_bucket(row["text_chars"])
            or row["split_group_id"] != f"swebench_verified:repo:{row['repository']}"
            for row in panel
        )
    ):
        raise ValueError("SWE-bench panel schema mismatch")
    return panel, manifest, file_sha256(manifest_path)


def _validate_contract(manifest: dict) -> None:
    if (
        manifest.get("source", {}).get("data_manifest_sha256")
        != file_sha256(ROOT / "data" / "manifest.json")
        or manifest.get("model_contract", {}).get("threshold_sha256")
        != THRESHOLD_SHA256
        or manifest["model_contract"].get("files") != _file_contract(MODEL_FILES)
        or manifest.get("selection_contract") != _file_contract(SELECTION_FILES)
        or manifest.get("experiment_contract") != _file_contract(EXPERIMENT_FILES)
    ):
        raise ValueError("SWE-bench frozen contract changed")


def _texts_for_panel(panel: list[dict]) -> dict[str, str]:
    source_rows, _, _ = _source_rows()
    source_by_id = {row["id"]: row for row in source_rows}
    texts = {}
    for panel_row in panel:
        row = source_by_id.get(panel_row["panel_id"])
        if (
            row is None
            or _sha256_bytes(row["text"].encode()) != panel_row["text_sha256"]
            or row["normalized_text_sha256"] != panel_row["normalized_text_sha256"]
            or row["source_repository"] != panel_row["repository"]
        ):
            raise ValueError("canonical source no longer matches the frozen panel")
        texts[panel_row["panel_id"]] = row["text"]
    return texts


def _run(output: Path) -> dict:
    result_path = output / "local_results.jsonl.gz"
    if result_path.exists():
        raise FileExistsError("local result is write-once; use analyze")
    panel, manifest, manifest_sha256 = _load_panel(output)
    _validate_contract(manifest)
    texts = _texts_for_panel(panel)
    runtime = MmbertRuntime.from_artifacts(ROOT / "model-artifacts.json")
    results = []
    for index, row in enumerate(panel, 1):
        started = time.perf_counter()
        prepared = runtime.prepare(texts[row["panel_id"]])
        scores = runtime.score(prepared.windows)
        low = sum(score < MMBERT_LOW_BY_CHANNEL["direct_user"] for score in scores)
        high = sum(score >= MMBERT_HIGH for score in scores)
        assessment = {
            "token_count": prepared.token_count,
            "window_count": len(scores),
            "low_windows": low,
            "middle_windows": len(scores) - low - high,
            "high_windows": high,
            "max_mmbert_score": max(scores),
            "latency_ms": (time.perf_counter() - started) * 1_000,
        }
        results.append(
            {
                "schema_version": 1,
                "manifest_sha256": manifest_sha256,
                "panel_id": row["panel_id"],
                "assessment": assessment,
            }
        )
        if index % 50 == 0 or index == len(panel):
            print(f"scored {index}/{len(panel)}", flush=True)
    return {"results": _write_gzip_jsonl(result_path, results)}


def _validated_results(
    output: Path, panel: list[dict], manifest_sha256: str
) -> list[tuple[dict, dict]]:
    rows = _read_jsonl(output / "local_results.jsonl.gz")
    by_id = {}
    expected = {row["panel_id"]: row for row in panel}
    fields = {
        "high_windows",
        "latency_ms",
        "low_windows",
        "max_mmbert_score",
        "middle_windows",
        "token_count",
        "window_count",
    }
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid SWE-bench local result")
        assessment = row.get("assessment")
        panel_id = row.get("panel_id")
        if (
            set(row) != {"schema_version", "manifest_sha256", "panel_id", "assessment"}
            or row["schema_version"] != 1
            or row["manifest_sha256"] != manifest_sha256
            or panel_id not in expected
            or panel_id in by_id
            or not isinstance(assessment, dict)
            or set(assessment) != fields
            or any(
                type(assessment[field]) is not int or assessment[field] < 0
                for field in (
                    "high_windows",
                    "low_windows",
                    "middle_windows",
                    "token_count",
                    "window_count",
                )
            )
            or assessment["window_count"] < 1
            or assessment["window_count"]
            != assessment["low_windows"]
            + assessment["middle_windows"]
            + assessment["high_windows"]
            or not isinstance(assessment["max_mmbert_score"], int | float)
            or isinstance(assessment["max_mmbert_score"], bool)
            or not math.isfinite(assessment["max_mmbert_score"])
            or not 0 <= assessment["max_mmbert_score"] <= 1
            or not isinstance(assessment["latency_ms"], int | float)
            or isinstance(assessment["latency_ms"], bool)
            or not math.isfinite(assessment["latency_ms"])
            or assessment["latency_ms"] < 0
        ):
            raise ValueError("invalid SWE-bench local result")
        by_id[panel_id] = assessment
    if set(by_id) != set(expected):
        raise ValueError("SWE-bench local result is incomplete")
    return [(row, by_id[row["panel_id"]]) for row in panel]


def _rate(rows: list[tuple[dict, dict]]) -> dict:
    hard = sum(assessment["high_windows"] > 0 for _, assessment in rows)
    review = sum(
        assessment["high_windows"] == 0 and assessment["middle_windows"] > 0
        for _, assessment in rows
    )
    total = len(rows)
    return {
        "documents": total,
        "pass": total - hard - review,
        "review_required": review,
        "review_required_rate": review / total if total else None,
        "hard_restricted": hard,
        "hard_restriction_rate": hard / total if total else None,
        "middle_windows": sum(assessment["middle_windows"] for _, assessment in rows),
        "high_windows": sum(assessment["high_windows"] for _, assessment in rows),
    }


def _by(rows: list[tuple[dict, dict]], field: str) -> dict:
    grouped = defaultdict(list)
    for panel_row, assessment in rows:
        grouped[str(panel_row[field])].append((panel_row, assessment))
    return {key: _rate(grouped[key]) for key in sorted(grouped)}


def _decision(total: int, *, hard_restricted: int, middle_windows: int) -> str:
    if total < 1 or not 0 <= hard_restricted <= total or middle_windows < 0:
        raise ValueError("invalid local decision counts")
    if hard_restricted * 100 > total:
        return "reject_registered_local_gate_long_benign"
    if middle_windows > MAX_REMOTE_WINDOWS:
        return "stop_before_remote_window_budget"
    return "eligible_for_bounded_remote_phase"


def _analyze(output: Path) -> dict:
    panel, manifest, manifest_sha256 = _load_panel(output)
    _validate_contract(manifest)
    joined = _validated_results(output, panel, manifest_sha256)
    overall = _rate(joined)
    by_repository = _by(joined, "repository")
    by_length = _by(joined, "length_bucket")
    by_windows = {
        "single_window": _rate(
            [item for item in joined if item[1]["window_count"] == 1]
        ),
        "multiple_windows": _rate(
            [item for item in joined if item[1]["window_count"] > 1]
        ),
    }
    result_path = output / "local_results.jsonl.gz"
    hard_restrictions = [
        {
            **panel_row,
            "token_count": assessment["token_count"],
            "window_count": assessment["window_count"],
            "high_windows": assessment["high_windows"],
            "max_mmbert_score": assessment["max_mmbert_score"],
        }
        for panel_row, assessment in joined
        if assessment["high_windows"] > 0
    ]
    summary = {
        "schema_version": 1,
        "purpose": manifest["purpose"],
        "manifest_sha256": manifest_sha256,
        "panel_sha256": manifest["panel"]["sha256"],
        "result_ledger": {
            "path": str(result_path.relative_to(ROOT)),
            "rows": len(joined),
            "sha256": file_sha256(result_path),
            "content_sha256": _sha256_bytes(gzip.decompress(result_path.read_bytes())),
        },
        "decision": _decision(
            len(joined),
            hard_restricted=overall["hard_restricted"],
            middle_windows=overall["middle_windows"],
        ),
        "primary": overall,
        "repository_macro": {
            "hard_restriction_rate": statistics.fmean(
                value["hard_restriction_rate"] for value in by_repository.values()
            ),
            "review_required_rate": statistics.fmean(
                value["review_required_rate"] for value in by_repository.values()
            ),
        },
        "by_repository": by_repository,
        "by_length": by_length,
        "by_windows": by_windows,
        "hard_restrictions": hard_restrictions,
        "operation": {
            "total_tokens": sum(assessment["token_count"] for _, assessment in joined),
            "total_windows": sum(
                assessment["window_count"] for _, assessment in joined
            ),
            "maximum_tokens": max(
                assessment["token_count"] for _, assessment in joined
            ),
            "maximum_windows": max(
                assessment["window_count"] for _, assessment in joined
            ),
            "local_latency_seconds": sum(
                assessment["latency_ms"] for _, assessment in joined
            )
            / 1_000,
        },
        "limitations": [
            "Human solvability review is not a safety annotation.",
            "The panel is benign-only and cannot measure attack recall.",
            "Repository concentration and related issue styles limit row independence.",
            "This is already-open dev-test evidence, not a prospective final test.",
            "No threshold, prompt, model, or provider may be selected from this panel.",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = _prepare(output)
    elif args.command == "run":
        result = _run(output)
    else:
        result = _analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
