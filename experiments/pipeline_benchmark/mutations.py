#!/usr/bin/env python3
"""Replay the frozen mutation population against the registered 1,024 model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.cascade_mutation_asr import run as retained
from experiments.pipeline_benchmark import local
from morgott.models.downstream import MMBERT_HIGH, MMBERT_LOW_BY_CHANNEL
from morgott.sources.tasks import _sensitive_text_reasons

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "pipeline_benchmark" / "20260816"
RETAINED_OUTPUT = ROOT / "artifacts" / "cascade_mutation_asr"
PREFIX = "mutation_1024"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_gzip_bytes(rows: list[dict[str, Any]]) -> bytes:
    return gzip.compress(
        b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows),
        mtime=0,
    )


def _verify_registered_model() -> dict[str, Any]:
    registry = json.loads(local.MODEL_REGISTRY.read_text(encoding="utf-8"))
    model = registry.get("models", {}).get(local.MODEL_KEY)
    if not isinstance(model, dict):
        raise ValueError("registered 1,024 model is unavailable")
    serving = model.get("serving", {})
    result = model.get("result", {})
    head = model.get("head", {})
    if (
        registry.get("schema_version") != 2
        or registry.get("advisory_only") is not True
        or serving.get("max_tokens") != local.MAX_TOKENS
        or serving.get("window_overlap") != local.WINDOW_OVERLAP
        or result.get("path")
        != str((local.MODEL_DIR / "result.json").relative_to(ROOT))
        or head.get("path")
        != str((local.MODEL_DIR / "head.safetensors").relative_to(ROOT))
        or local.file_sha256(local.MODEL_DIR / "result.json") != result.get("sha256")
        or local.file_sha256(local.MODEL_DIR / "head.safetensors") != head.get("sha256")
    ):
        raise ValueError("registered 1,024 model identity changed")
    return {
        "model_key": local.MODEL_KEY,
        "registry_sha256": local.file_sha256(local.MODEL_REGISTRY),
        "result_sha256": result["sha256"],
        "head_sha256": head["sha256"],
        "max_tokens": local.MAX_TOKENS,
        "window_overlap": local.WINDOW_OVERLAP,
    }


def _verify_retained_contract() -> dict[str, Any]:
    manifest_path = RETAINED_OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_inputs = {
        name: specification.get("sha256")
        for name, specification in manifest.get("inputs", {}).items()
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("frozen_population", {}).get("base_rows") != 300
        or tuple(manifest["frozen_population"].get("aggregate_families", ()))
        != retained.FAMILIES
        or manifest["frozen_population"].get("mutations_per_attack")
        != retained.MUTATIONS_PER_ATTACK
        or expected_inputs != retained.INPUTS
        or manifest.get("canonical_data", {}).get("manifest_sha256")
        != local.file_sha256(ROOT / "data" / "manifest.json")
    ):
        raise ValueError("retained mutation study contract changed")
    retained._verify_inputs()
    return {
        "manifest_sha256": local.file_sha256(manifest_path),
        "summary_sha256": local.file_sha256(RETAINED_OUTPUT / "summary.json"),
        "inputs": {
            name: {
                "path": str((retained.EVALUATION / name).relative_to(ROOT)),
                "sha256": digest,
            }
            for name, digest in retained.INPUTS.items()
        },
    }


def build_population() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Reconstruct the privacy-clean frozen bases and all aggregate mutations."""
    rows, no_op, text_hashes = retained._verified_rows()
    panel: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    excluded = Counter()
    by_channel = Counter()
    by_source = Counter()
    eligible = 0
    for row in rows:
        reasons = _sensitive_text_reasons(row["text"])
        if reasons:
            excluded.update(reasons)
            continue
        eligible += 1
        by_channel[row["input_channel"]] += 1
        by_source[row["source"]] += 1
        base_artifact_id = retained._panel_id(row["base_id"])
        base_hash = _sha256(row["text"].encode())
        base = {
            "panel_id": base_artifact_id,
            "dataset": "mutation_base",
            "source": row["source"],
            "input_channel": row["input_channel"],
            "label": 1,
            "text_sha256": base_hash,
            "kind": "base",
            "base_artifact_id": base_artifact_id,
            "slot": row["slot"],
        }
        panel.append(base)
        texts[base_artifact_id] = row["text"]
        for family in retained.FAMILIES:
            family_index = retained.FAMILY_INDEX[family]
            for repeat in range(retained.MUTATIONS_PER_FAMILY):
                candidate = retained.mutate(
                    family,
                    row["text"],
                    retained.SEED + 1000 * row["slot"] + 17 * family_index + repeat,
                )
                expected_hash = str(text_hashes[row["slot"], family_index, repeat])
                expected_no_op = bool(no_op[row["slot"], family_index, repeat])
                if (
                    _sha256(candidate.encode()) != expected_hash
                    or (candidate == row["text"]) != expected_no_op
                ):
                    raise ValueError("frozen mutation reconstruction changed")
                mutation_id = retained._mutation_id(base_artifact_id, family, repeat)
                panel.append(
                    {
                        "panel_id": mutation_id,
                        "dataset": "mutation",
                        "source": row["source"],
                        "input_channel": row["input_channel"],
                        "label": 1,
                        "text_sha256": expected_hash,
                        "kind": "mutation",
                        "base_artifact_id": base_artifact_id,
                        "slot": row["slot"],
                        "family": family,
                        "repeat": repeat,
                        "no_op": expected_no_op,
                    }
                )
                texts[mutation_id] = candidate
    expected_mutations = eligible * retained.MUTATIONS_PER_ATTACK
    if (
        eligible != 240
        or len(panel) != eligible + expected_mutations
        or len(texts) != len(panel)
        or len({row["panel_id"] for row in panel}) != len(panel)
    ):
        raise ValueError("privacy-clean mutation population changed")
    return (
        panel,
        texts,
        {
            "frozen_base_rows": len(rows),
            "eligible_base_rows": eligible,
            "excluded_base_rows": len(rows) - eligible,
            "excluded_by_reason": dict(sorted(excluded.items())),
            "mutations_per_base": retained.MUTATIONS_PER_ATTACK,
            "mutation_rows": expected_mutations,
            "by_channel": dict(sorted(by_channel.items())),
            "by_source": dict(sorted(by_source.items())),
            "raw_text_retained": False,
        },
    )


def _route(record: dict[str, Any]) -> str:
    return retained.local_document_route(
        record["input_channel"], tuple(record["window_scores"])
    )


def _asr(base_ids: list[str], rows: list[dict[str, Any]], route: str | None) -> dict:
    counts = Counter(
        row["base_artifact_id"]
        for row in rows
        if route is None
        and row["local_route"] != "restrict"
        or route is not None
        and row["local_route"] == route
    )
    return {
        str(k): retained.exact_asr([counts[base_id] for base_id in base_ids], k)
        for k in retained.KS
    }


def summarize(records: list[dict[str, Any]], population: dict[str, Any]) -> dict:
    bases = [row for row in records if row["kind"] == "base"]
    mutations = [row for row in records if row["kind"] == "mutation"]
    if (
        len(bases) != population["eligible_base_rows"]
        or len(mutations) != population["mutation_rows"]
    ):
        raise ValueError("scored mutation population is incomplete")
    caught_ids = [
        row["artifact_id"] for row in bases if row["local_route"] == "restrict"
    ]
    caught = set(caught_ids)
    selected = [row for row in mutations if row["base_artifact_id"] in caught]
    if len(selected) != len(caught_ids) * retained.MUTATIONS_PER_ATTACK:
        raise ValueError("current-model caught-set mutations are incomplete")

    def breakdown(field: str) -> dict[str, Any]:
        values = sorted({str(row[field]) for row in selected})
        result = {}
        for value in values:
            group = [row for row in selected if str(row[field]) == value]
            group_bases = sorted({row["base_artifact_id"] for row in group})
            result[value] = {
                "caught_base_attacks": len(group_bases),
                "mutation_rows": len(group),
                "routes": dict(
                    sorted(Counter(row["local_route"] for row in group).items())
                ),
                "high_gate_asr_at_k": _asr(group_bases, group, None),
                "local_pass_asr_floor_at_k": _asr(group_bases, group, "pass"),
            }
        return result

    base_routes = Counter(row["local_route"] for row in bases)
    mutation_routes = Counter(row["local_route"] for row in mutations)
    caught_routes = Counter(row["local_route"] for row in selected)
    high_asr = _asr(caught_ids, selected, None)
    base_recall = len(caught_ids) / len(bases)
    return {
        "schema_version": 1,
        "evidence_scope": "registered_1024_local_model_only",
        "advisory_only": True,
        "provider_calls": 0,
        "full_cascade_asr": None,
        "full_cascade_asr_status": "not_measured_without_provider_review",
        "thresholds": {
            "low_by_channel": MMBERT_LOW_BY_CHANNEL,
            "local_high": MMBERT_HIGH,
        },
        "population": population,
        "base": {
            "routes": dict(sorted(base_routes.items())),
            "high_gate_caught": len(caught_ids),
            "high_gate_recall": base_recall,
        },
        "all_mutations": {"routes": dict(sorted(mutation_routes.items()))},
        "current_high_gate_caught_set": {
            "base_attacks": len(caught_ids),
            "mutation_rows": len(selected),
            "routes": dict(sorted(caught_routes.items())),
            "high_gate_asr_at_k": high_asr,
            "local_pass_asr_floor_at_k": _asr(caught_ids, selected, "pass"),
            "effective_local_high_recall_at_k": {
                key: base_recall * (1 - value) if value is not None else None
                for key, value in high_asr.items()
            },
        },
        "by_family": breakdown("family"),
        "by_channel": breakdown("input_channel"),
        "windowing": {
            "max_tokens": local.MAX_TOKENS,
            "overlap": local.WINDOW_OVERLAP,
            "artifacts": len(records),
            "windows": sum(row["window_count"] for row in records),
            "maximum_windows_per_artifact": max(row["window_count"] for row in records),
        },
        "limitations": [
            "This is already-open synthetic development evidence, not a final test.",
            "The high-gate ASR measures local restrictions only; review-zone outcomes require the separately frozen provider phase.",
            "Uniform fixed mutations are not adaptive attacks or production traffic.",
        ],
    }


def run(output: Path, *, batch_size: int = 24) -> None:
    paths = {
        "manifest": output / f"{PREFIX}_manifest.json",
        "scores": output / f"{PREFIX}_scores.jsonl.gz",
        "runtime": output / f"{PREFIX}_runtime.json",
        "summary": output / f"{PREFIX}_summary.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"mutation replay is write-once: {existing}")
    model = _verify_registered_model()
    retained_contract = _verify_retained_contract()
    panel, texts, population = build_population()
    ordered = sorted(
        panel, key=lambda row: (len(texts[row["panel_id"]]), row["panel_id"])
    )
    scored, runtime = local.score_cuda(ordered, texts, batch_size=batch_size)
    metadata = {row["panel_id"]: row for row in panel}
    records = []
    for record in scored:
        source = metadata.get(record["artifact_id"])
        if source is None or source["text_sha256"] != record["text_sha256"]:
            raise ValueError("scored mutation identity changed")
        enriched = record | {
            key: source[key]
            for key in (
                "kind",
                "base_artifact_id",
                "slot",
                "family",
                "repeat",
                "no_op",
            )
            if key in source
        }
        enriched["local_route"] = _route(enriched)
        records.append(enriched)
    records.sort(key=lambda row: row["artifact_id"])
    if (
        runtime.get("model_key") != model["model_key"]
        or runtime.get("result_sha256") != model["result_sha256"]
        or runtime.get("head_sha256") != model["head_sha256"]
        or runtime.get("max_tokens") != model["max_tokens"]
        or runtime.get("window_overlap") != model["window_overlap"]
        or runtime.get("runtime", {}).get("dtype") != "bfloat16"
    ):
        raise ValueError("CUDA runtime did not preserve the registered model contract")
    summary = summarize(records, population)
    score_bytes = _jsonl_gzip_bytes(records)
    runtime_document = runtime | {
        "scores_path": str(paths["scores"].relative_to(ROOT)),
        "scores_sha256": _sha256(score_bytes),
    }
    runtime_bytes = _json_bytes(runtime_document)
    summary["model"] = model
    summary["runtime_sha256"] = _sha256(runtime_bytes)
    summary["scores_sha256"] = _sha256(score_bytes)
    summary_bytes = _json_bytes(summary)
    manifest = {
        "schema_version": 1,
        "purpose": "text-free current-1024 replay of the frozen mutation population",
        "advisory_only": True,
        "production_changes": False,
        "provider_calls": 0,
        "model": model,
        "retained_population": retained_contract,
        "code": {
            "mutation_replay_sha256": local.file_sha256(Path(__file__)),
            "local_scorer_sha256": local.file_sha256(Path(local.__file__)),
            "retained_reconstruction_sha256": local.file_sha256(
                Path(retained.__file__)
            ),
        },
        "artifacts": {
            "scores": {
                "path": str(paths["scores"].relative_to(ROOT)),
                "sha256": _sha256(score_bytes),
                "rows": len(records),
            },
            "runtime": {
                "path": str(paths["runtime"].relative_to(ROOT)),
                "sha256": _sha256(runtime_bytes),
            },
            "summary": {
                "path": str(paths["summary"].relative_to(ROOT)),
                "sha256": _sha256(summary_bytes),
            },
        },
        "raw_text_retained": False,
    }
    _write_once(paths["scores"], score_bytes)
    _write_once(paths["runtime"], runtime_bytes)
    _write_once(paths["summary"], summary_bytes)
    _write_once(paths["manifest"], _json_bytes(manifest))
    print(
        json.dumps(
            {
                "artifacts": len(records),
                "summary": summary["current_high_gate_caught_set"],
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args.output.resolve(), batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
