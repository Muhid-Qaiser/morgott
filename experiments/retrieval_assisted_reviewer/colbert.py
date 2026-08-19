#!/usr/bin/env python3
"""Run the optional local mLateOn reranker experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.retrieval_assisted_reviewer import run

MLATEON_MODEL = "lightonai/mLateOn"
MLATEON_REVISION = "edd378f99593c0ac8a15518b97ad89786b02685e"
MLATEON_DIMENSION = 128
MLATEON_SENTENCE_TRANSFORMERS_VERSION = "6.0.0"
MLATEON_TRANSFORMERS_VERSION = "5.14.1"
MLATEON_TORCH_VERSION = "2.13.0"
MLATEON_METHOD = "mlateon_exact_maxsim_hnsw_bm25_union"
MLATEON_SPARSE_TIMEOUT_MS = 1_000.0
MLATEON_RRF_METHOD = f"{run.HNSW_HYBRID_METHOD}_sparse_timeout1000"
MLATEON_FALLBACK_AFTER_MS = 1_000.0
MLATEON_TARGET_P95_MS = 100.0
DEFAULT_SOURCE = (
    run.ROOT / "artifacts" / "retrieval_assisted_reviewer_full_rows_hnsw_sparse"
)
DEFAULT_OUTPUT = run.ROOT / "artifacts" / "retrieval_assisted_reviewer_colbert"
MLATEON_STAGE0_SAMPLES = (
    "Which planet is the Red Planet?",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Mars, connu pour son apparence rougeâtre, est souvent appelé la planète rouge.",
    "Mars, bekannt für sein rötliches Erscheinungsbild, wird oft als der Rote Planet bezeichnet.",
    "Venus is often called Earth's twin because of its similar size and proximity.",
)
MLATEON_STAGE0_REFERENCE_SCORES = (9.6029, 9.5838, 9.5877, 9.4578)
MLATEON_STAGE0_SCORE_TOLERANCE = 0.01
MLATEON_NATIVE_FILES = {
    "1_Dense/config.json": "95ab26aeb95fc3d7cc1694d7a860150d70a952dcaad0b84a698594b83834ee20",
    "1_Dense/model.safetensors": "ea7543f3eb4be49cb9061b45e3e5898786f729b8a90dfe59e01e8bde70ccbc2b",
    "2_Dense/config.json": "5bc031962f62ec27de4a2a6b586ba0658dfff3749fd318c3332c69b0c8e928d2",
    "2_Dense/model.safetensors": "84d89c6b79918d4c944fe4da88e1cccb7daab85bd7b53d16c9619006a656b5a1",
    "3_Dense/config.json": "501861dc63acde2428c2ee3a862978c3cb81f2bf289504d533a5d3d4039cc115",
    "3_Dense/model.safetensors": "d18d32c042cc76f53e3c7dd9fa220a1f380237a066d0c48e6b1139ddcbac8d60",
    "config.json": "636ffacfec7bd4ba79f3cf61fb58dd2f4a6db2f86bc6f68f311f86891d93092d",
    "config_sentence_transformers.json": "5ec5cbd5287ddd4590d0260f8570b8c8466d44f179da2b57527c96c842a5bbba",
    "model.safetensors": "f72148b87da45a9d94cca3dae9571fbd652c6803d381f67a6460418212af5db9",
    "modules.json": "48c40af6b8d1105cefe7174e2c840281cbf14d677e28977327db0791d7de7bbc",
    "sentence_bert_config.json": "a5c9d5001021337092aed614c52e31a6d9a97862b7e8f140a34b1872ee2c7f3e",
    "tokenizer.json": "33b6ce1724e107a3298219a322b4c963fcfcb9c8661bd1f30189bdd0d36c4669",
    "tokenizer_config.json": "fc39c67d6d8076217fc4cb3136cae5f75862ecf23201661df117dafd224c49f5",
}


def _token_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").float().numpy()
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.size or not np.all(np.isfinite(matrix)):
        raise ValueError("mLateOn returned an invalid token matrix")
    return matrix


def maxsim_score(query: np.ndarray, document: np.ndarray) -> float:
    query = _token_matrix(query)
    document = _token_matrix(document)
    if query.shape[1] != document.shape[1]:
        raise ValueError("mLateOn query and document dimensions differ")
    score = float(np.max(query @ document.T, axis=1).sum(dtype=np.float32))
    if not np.isfinite(score):
        raise ValueError("mLateOn returned a non-finite MaxSim score")
    return score


def _snapshot_identity(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    if snapshot.name != MLATEON_REVISION:
        raise ValueError("mLateOn snapshot identity is incomplete")
    records = []
    for name, expected_sha256 in sorted(MLATEON_NATIVE_FILES.items()):
        path = snapshot / name
        if not path.is_file() or run.file_sha256(path) != expected_sha256:
            raise ValueError(f"mLateOn snapshot file changed: {name}")
        records.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": expected_sha256}
        )
    return {
        "model": MLATEON_MODEL,
        "revision": MLATEON_REVISION,
        "files": records,
        "sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _load_mlateon(snapshot: Path, *, device: str) -> Any:
    if (
        importlib.metadata.version("sentence-transformers")
        != MLATEON_SENTENCE_TRANSFORMERS_VERSION
    ):
        raise RuntimeError(
            f"sentence-transformers {MLATEON_SENTENCE_TRANSFORMERS_VERSION} is required"
        )
    if importlib.metadata.version("transformers") != MLATEON_TRANSFORMERS_VERSION:
        raise RuntimeError(f"transformers {MLATEON_TRANSFORMERS_VERSION} is required")
    if importlib.metadata.version("torch") != MLATEON_TORCH_VERSION:
        raise RuntimeError(f"Torch {MLATEON_TORCH_VERSION} is required")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for mLateOn")
        torch.cuda.reset_peak_memory_stats()
    from sentence_transformers import MultiVectorEncoder

    return MultiVectorEncoder(
        model_name_or_path=str(snapshot.resolve()),
        device=device,
        local_files_only=True,
        trust_remote_code=False,
    )


def _runtime_identity(device: str) -> dict[str, Any]:
    import torch

    identity = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "sentence-transformers": importlib.metadata.version("sentence-transformers"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
        "torch_cuda": torch.version.cuda,
        "device": device,
    }
    if device == "cuda":
        identity.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
            }
        )
    return identity


def _memory_telemetry(device: str) -> dict[str, int]:
    result = {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == "cuda":
        import torch

        result.update(
            {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    return result


def _validate_model_contract(model: Any) -> None:
    input_module = model[0]
    masks = [module for module in model if type(module).__name__ == "MultiVectorMask"]
    if (
        model.prompts != {"query": "[Q] ", "document": "[D] "}
        or (input_module.query_length, input_module.document_length) != (8192, 8192)
        or input_module.query_expansion is not None
        or len(masks) != 1
        or masks[0].skiplist_words != []
        or masks[0].skiplist_tasks != ["document"]
        or masks[0].keep_only_token_ids is not None
    ):
        raise ValueError("mLateOn model contract changed")


def _validate_token_limit(model: Any, text: str, *, role: str) -> None:
    features = model.preprocess(
        [text],
        prompt=model.prompts[role],
        task=role,
        processing_kwargs={"text": {"truncation": False, "padding": False}},
    )
    input_ids = features["input_ids"]
    token_count = (
        int(input_ids.numel())
        if hasattr(input_ids, "numel")
        else int(np.asarray(input_ids).size)
    )
    limit = getattr(model[0], f"{role}_length")
    if token_count <= 0 or token_count > limit:
        raise ValueError(f"mLateOn {role} exceeds its token limit")


def _encode_query(model: Any, query_text: str) -> np.ndarray:
    _validate_token_limit(model, query_text, role="query")
    query = _token_matrix(
        model.encode_query(
            query_text,
            convert_to_numpy=False,
            normalize_embeddings=False,
        )
    )
    if query.shape[1] != MLATEON_DIMENSION:
        raise ValueError("mLateOn query dimension changed")
    return query


def _encode_documents(model: Any, documents: list[str]) -> list[np.ndarray]:
    for document in documents:
        _validate_token_limit(model, document, role="document")
    encoded = model.encode_document(
        documents,
        batch_size=32,
        convert_to_numpy=False,
        normalize_embeddings=False,
    )
    if not isinstance(encoded, (list, tuple)) or len(encoded) != len(documents):
        raise ValueError("mLateOn returned the wrong document batch")
    matrices = [_token_matrix(value) for value in encoded]
    if any(value.shape[1] != MLATEON_DIMENSION for value in matrices):
        raise ValueError("mLateOn document dimension changed")
    return matrices


def _encode_scores(model: Any, query_text: str, documents: list[str]) -> list[float]:
    query = _encode_query(model, query_text)
    matrices = _encode_documents(model, documents)
    return [maxsim_score(query, value) for value in matrices]


def stage0_canary(output: Path, *, snapshot: Path, device: str) -> None:
    result_path = output / "mlateon-stage0.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to replace mLateOn canary: {result_path}")
    snapshot_identity = _snapshot_identity(snapshot)
    loaded = time.perf_counter()
    model = _load_mlateon(snapshot, device=device)
    load_seconds = time.perf_counter() - loaded
    _validate_model_contract(model)
    hashes = []
    latencies = []
    similarity_errors = []
    reference_errors = []
    for _ in range(2):
        started = time.perf_counter()
        query = _encode_query(model, MLATEON_STAGE0_SAMPLES[0])
        documents = _encode_documents(model, list(MLATEON_STAGE0_SAMPLES[1:]))
        scores = np.asarray(
            [maxsim_score(query, document) for document in documents],
            dtype=np.float32,
        )
        library_scores = model.similarity(query, documents)
        if hasattr(library_scores, "detach"):
            library_scores = library_scores.detach().to("cpu").float().numpy()
        library_scores = np.asarray(library_scores, dtype=np.float32).reshape(-1)
        if library_scores.shape != scores.shape or not np.all(
            np.isfinite(library_scores)
        ):
            raise ValueError("mLateOn similarity returned invalid Stage 0 scores")
        latencies.append((time.perf_counter() - started) * 1_000)
        hashes.append(hashlib.sha256(scores.tobytes()).hexdigest())
        similarity_errors.append(float(np.max(np.abs(scores - library_scores))))
        reference_errors.append(
            float(
                np.max(
                    np.abs(
                        scores
                        - np.asarray(MLATEON_STAGE0_REFERENCE_SCORES, dtype=np.float32)
                    )
                )
            )
        )
    runtime = _runtime_identity(device)
    artifact = {
        "schema_version": 1,
        "purpose": "local mLateOn contract canary",
        "advisory_only": True,
        "production_changes": False,
        "model": snapshot_identity,
        "runtime": runtime,
        "device": device,
        "dimension": MLATEON_DIMENSION,
        "load_seconds": load_seconds,
        "repeat_score_hashes": hashes,
        "similarity_max_abs_error": max(similarity_errors),
        "reference_max_abs_error": max(reference_errors),
        "reference_tolerance": MLATEON_STAGE0_SCORE_TOLERANCE,
        "repeat_latency_ms": latencies,
        "memory": _memory_telemetry(device),
        "stores_raw_text_or_vectors": False,
        "passed": hashes[0] == hashes[1]
        and max(similarity_errors) <= 1e-4
        and max(reference_errors) <= MLATEON_STAGE0_SCORE_TOLERANCE,
    }
    if not artifact["passed"]:
        raise ValueError("mLateOn repeat scores changed")
    output.mkdir(parents=True, exist_ok=True)
    run._atomic_json(result_path, artifact)
    print(json.dumps({"result": str(result_path), "passed": True}, sort_keys=True))


def _percentile(values: list[float], percentile: int) -> float | None:
    return (
        float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        if values
        else None
    )


def _latency_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["colbert"]["latency_ms"]) for row in rows]
    successful = [
        float(row["colbert"]["latency_ms"])
        for row in rows
        if not row["colbert"]["colbert_fallback"]
    ]
    p95 = _percentile(latencies, 95)
    return {
        "latency_population": "all attempts including post-execution fallbacks",
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": p95,
        "latency_p99_ms": _percentile(latencies, 99),
        "success_only_latency_p95_ms": _percentile(successful, 95),
        "target_added_p95_ms": MLATEON_TARGET_P95_MS,
        "local_sequential_added_p95_under_target": p95 is not None
        and p95 <= MLATEON_TARGET_P95_MS,
        "target_added_p95_passed": False,
        "target_shape_tested": False,
        "latency_is_sequential_local_not_c4": True,
    }


def _source_contract(
    source: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_path = source / "manifest.json"
    retrieval_path = source / "validation-retrieval.jsonl"
    analysis = run._read_json(source / "validation-hnsw-hybrid-analysis.json")
    binding = analysis.get("evidence_binding", {})
    if binding.get("manifest_sha256") != run.file_sha256(manifest_path) or binding.get(
        "retrieval_sha256"
    ) != run.file_sha256(retrieval_path):
        raise ValueError("HNSW hybrid source evidence changed")
    manifest = run._study_manifest(source)
    hybrid = manifest.get("hnsw_hybrid", {})
    sparse_identity = hybrid.get("sparse_identity", {})
    if (
        manifest.get("bank", {}).get("mode") != "full"
        or hybrid.get("post_hoc_consumed_validation") is not True
        or hybrid.get("dense", {}).get("method") != run.HNSW_HYBRID_DENSE_ARM
        or hybrid.get("fusion", {}).get("method") != run.HNSW_HYBRID_METHOD
        or hybrid.get("sparse", {}).get("maximum_terms")
        != run.PARTITIONED_SPARSE_MAX_TERMS
        or sparse_identity.get("path") != run.FULLROW_SPARSE_IDENTITY_PATH
        or sparse_identity.get("index_path") != run.FULLROW_SPARSE_INDEX_PATH
        or sparse_identity.get("sha256")
        != run.file_sha256(source / run.FULLROW_SPARSE_IDENTITY_PATH)
        or sparse_identity.get("index_sha256")
        != run.file_sha256(source / run.FULLROW_SPARSE_INDEX_PATH)
    ):
        raise ValueError("HNSW hybrid source contract changed")
    _, units = run._load_local_evidence(source, "validation")
    dense_rows = [
        row
        for row in run._read_jsonl(retrieval_path)
        if row.get("method") == run.HNSW_HYBRID_DENSE_ARM
    ]
    dense = {row["unit_id"]: row for row in dense_rows}
    if len(dense) != len(dense_rows) or set(dense) != {row["unit_id"] for row in units}:
        raise ValueError("HNSW hybrid dense coverage changed")
    sparse = run._open_fullrow_partitioned_sparse_index(source, manifest)
    sparse.close()
    return manifest, units, dense


def rerank_study(
    output: Path,
    *,
    source: Path,
    snapshot: Path,
    device: str,
) -> None:
    result_path = output / "validation-mlateon-rerank.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to replace mLateOn rerank: {result_path}")
    snapshot_identity = _snapshot_identity(snapshot)
    runtime = _runtime_identity(device)
    canary_path = output / "mlateon-stage0.json"
    canary = run._read_json(canary_path)
    if (
        canary.get("passed") is not True
        or canary.get("model") != snapshot_identity
        or canary.get("device") != device
        or canary.get("runtime") != runtime
    ):
        raise ValueError("a matching passed mLateOn Stage 0 canary is required")
    source = source.resolve()
    manifest, units, dense = _source_contract(source)
    texts = run._reload_unit_texts(source, "validation")
    model = _load_mlateon(snapshot, device=device)
    _validate_model_contract(model)
    sparse = run._open_fullrow_partitioned_sparse_index(source, manifest)
    bank = sqlite3.connect(
        (source / manifest["bank"]["path"]).resolve().as_uri() + "?mode=ro&immutable=1",
        uri=True,
    )
    prepared = []
    try:
        for unit in units:
            query_text = texts[unit["unit_id"]][1]
            sparse_started = time.perf_counter()
            try:
                sparse_rankings, sparse_ms = run._fullrow_partitioned_sparse_rank(
                    sparse,
                    bank,
                    query_text,
                    channel=unit["input_channel"],
                    timeout_ms=MLATEON_SPARSE_TIMEOUT_MS,
                )
                sparse_failure = None
            except (
                TimeoutError,
                sqlite3.Error,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                sparse_rankings = {0: [], 1: []}
                sparse_ms = (time.perf_counter() - sparse_started) * 1_000
                sparse_failure = (
                    "timeout"
                    if isinstance(error, TimeoutError)
                    else type(error).__name__
                )
            _, baseline = run._fullrow_hnsw_hybrid_records(
                bank,
                unit=unit,
                dense_record=dense[unit["unit_id"]],
                sparse_rankings=sparse_rankings,
                sparse_latency_ms=sparse_ms,
                sparse_failure_code=sparse_failure,
            )
            baseline["method"] = MLATEON_RRF_METHOD
            prepared.append(
                {
                    "unit": unit,
                    "query_text": query_text,
                    "sparse_search_ms": sparse_ms,
                    "sparse_failure_code": sparse_failure,
                    "baseline": baseline,
                }
            )

        cache_started = time.perf_counter()
        candidate_texts = {}
        for item in prepared:
            unit = item["unit"]
            union = _validate_baseline(
                bank,
                item["baseline"],
                input_channel=unit["input_channel"],
            )
            candidate_texts.update(_candidate_texts(bank, union))
        document_cache = {}
        cache_failures = []
        ordered_candidates = sorted(candidate_texts)
        for start in range(0, len(ordered_candidates), 256):
            ids = ordered_candidates[start : start + 256]
            try:
                matrices = _encode_documents(
                    model, [candidate_texts[example_id] for example_id in ids]
                )
            except Exception:
                cache_failures.extend(ids)
                continue
            document_cache.update(dict(zip(ids, matrices, strict=True)))
        cache_seconds = time.perf_counter() - cache_started

        rows = []
        for item in prepared:
            unit = item["unit"]
            reranked = rerank_hybrid_record(
                bank,
                model,
                query_text=item["query_text"],
                input_channel=unit["input_channel"],
                baseline=item["baseline"],
                latency_gate_ms=MLATEON_FALLBACK_AFTER_MS,
                document_cache=document_cache,
            )
            rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "source": unit["source"],
                    "input_channel": unit["input_channel"],
                    "review_kind": unit["kind"],
                    "label": unit["label"],
                    "sparse_search_ms": item["sparse_search_ms"],
                    "sparse_failure_code": item["sparse_failure_code"],
                    "rrf": {
                        key: item["baseline"][key]
                        for key in (
                            "status",
                            "failure_code",
                            "selected_ids",
                            "candidate_ids",
                            "branch_candidate_ids",
                            "sparse_fallback",
                        )
                    },
                    "colbert": reranked,
                }
            )
    finally:
        bank.close()
        sparse.close()
    rrf_failures = sum(row["rrf"]["status"] != "ok" for row in rows)
    colbert_failures = sum(row["colbert"]["status"] != "ok" for row in rows)
    summary = {
        "units": len(rows),
        "sparse_timeouts": sum(row["sparse_failure_code"] == "timeout" for row in rows),
        "sparse_fallbacks": sum(row["rrf"]["sparse_fallback"] for row in rows),
        "rrf_packet_failures": rrf_failures,
        "colbert_packet_failures": colbert_failures,
        "colbert_fallbacks": sum(row["colbert"]["colbert_fallback"] for row in rows),
        "changed_packets": sum(
            row["colbert"]["selected_ids"] != row["rrf"]["selected_ids"] for row in rows
        ),
        **_latency_evidence(rows),
        "production_selection_allowed": False,
    }
    artifact = {
        "schema_version": 1,
        "purpose": "post-hoc local mLateOn rerank over HNSW plus BM25 candidates",
        "advisory_only": True,
        "production_changes": False,
        "post_hoc_consumed_validation": True,
        "source": {
            "path": str(source.relative_to(run.ROOT.resolve())),
            "manifest_sha256": run.file_sha256(source / "manifest.json"),
            "retrieval_sha256": run.file_sha256(source / "validation-retrieval.jsonl"),
            "bank_sha256": manifest["bank"]["sha256"],
        },
        "model": snapshot_identity,
        "runtime": runtime,
        "device": device,
        "candidate_generator": {
            "dense": run.HNSW_HYBRID_DENSE_ARM,
            "sparse": run.FULLROW_SPARSE_METHOD,
            "maximum_terms": run.PARTITIONED_SPARSE_MAX_TERMS,
            "sparse_timeout_ms": MLATEON_SPARSE_TIMEOUT_MS,
            "rrf": MLATEON_RRF_METHOD,
            "maximum_union_per_label": 70,
        },
        "reranker": {
            "method": MLATEON_METHOD,
            "score": "exact uncompressed float32 MaxSim",
            "query_prefix_added_by_sentence_transformers": "[Q] ",
            "document_prefix_added_by_sentence_transformers": "[D] ",
            "fallback_after_ms": MLATEON_FALLBACK_AFTER_MS,
            "fallback_after_is_hard_deadline": False,
            "exact_rrf_decision_fields_on_failure": True,
        },
        "document_cache": {
            "scope": "frozen validation candidate union only",
            "persisted": False,
            "unique_candidates": len(ordered_candidates),
            "encoded_candidates": len(document_cache),
            "failed_candidates": len(cache_failures),
            "tokens": sum(value.shape[0] for value in document_cache.values()),
            "bytes": sum(value.nbytes for value in document_cache.values()),
            "build_seconds": cache_seconds,
            "cache_miss_falls_back_to_rrf": True,
        },
        "summary": summary,
        "rows": rows,
        "memory": _memory_telemetry(device),
        "stores_raw_text_or_vectors": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    run._atomic_json(result_path, artifact)
    print(json.dumps({"result": str(result_path), "summary": summary}, sort_keys=True))


def _validate_baseline(
    bank: sqlite3.Connection,
    baseline: dict[str, Any],
    *,
    input_channel: str,
) -> dict[int, list[str]]:
    if (
        baseline.get("method") != MLATEON_RRF_METHOD
        or baseline.get("status") not in {"ok", "failed"}
        or set(baseline.get("candidate_ids", {})) != {"0", "1"}
        or set(baseline.get("branch_candidate_ids", {})) != {"dense", "sparse"}
    ):
        raise ValueError("ColBERT baseline packet is invalid")
    rankings = {label: list(baseline["candidate_ids"][str(label)]) for label in (0, 1)}
    branch_ids = baseline["branch_candidate_ids"]
    if any(
        set(branch_ids.get(branch, {})) != {"0", "1"} for branch in ("dense", "sparse")
    ):
        raise ValueError("ColBERT branch candidates are invalid")
    union = {
        label: list(
            dict.fromkeys(
                list(branch_ids["dense"][str(label)])
                + list(branch_ids["sparse"][str(label)])
            )
        )
        for label in (0, 1)
    }
    if any(not values or len(values) > 70 for values in union.values()):
        raise ValueError("ColBERT candidate union is invalid")
    for label, values in union.items():
        if len(values) != len(set(values)) or any(
            run._example_metadata(bank, example_id)["input_channel"] != input_channel
            or run._example_metadata(bank, example_id)["label"] != label
            for example_id in values
        ):
            raise ValueError("ColBERT candidate crossed its partition")
    selected = list(baseline.get("selected_ids", []))
    try:
        recomputed = run._select_examples(bank, rankings, input_channel=input_channel)
        expected = ("ok", recomputed)
    except ValueError:
        expected = ("failed", [])
    if (baseline["status"], selected) != expected:
        raise ValueError("ColBERT baseline selection changed")
    return union


def _candidate_texts(
    bank: sqlite3.Connection, rankings: dict[int, list[str]]
) -> dict[str, str]:
    ids = list(dict.fromkeys(rankings[0] + rankings[1]))
    placeholders = ",".join("?" for _ in ids)
    rows = dict(
        bank.execute(
            f"SELECT example_id, text FROM examples WHERE example_id IN ({placeholders})",
            ids,
        )
    )
    if set(rows) != set(ids):
        raise ValueError("ColBERT candidate text is missing")
    return {example_id: str(rows[example_id]) for example_id in ids}


def rerank_hybrid_record(
    bank: sqlite3.Connection,
    model: Any,
    *,
    query_text: str,
    input_channel: str,
    baseline: dict[str, Any],
    latency_gate_ms: float,
    document_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if latency_gate_ms <= 0:
        raise ValueError("ColBERT latency gate must be positive")
    union = _validate_baseline(bank, baseline, input_channel=input_channel)
    fallback_candidates = {
        label: list(baseline["candidate_ids"][label]) for label in ("0", "1")
    }
    fallback_selected = list(baseline["selected_ids"])
    started = time.perf_counter()
    try:
        if document_cache is None:
            texts = _candidate_texts(bank, union)
            ordered_ids = list(texts)
            values = _encode_scores(
                model,
                query_text,
                [texts[example_id] for example_id in ordered_ids],
            )
        else:
            ordered_ids = list(dict.fromkeys(union[0] + union[1]))
            documents = [document_cache[value] for value in ordered_ids]
            query = _encode_query(model, query_text)
            values = [maxsim_score(query, document) for document in documents]
        scores = dict(zip(ordered_ids, values, strict=True))
        rankings = {
            label: sorted(union[label], key=lambda value: (-scores[value], value))
            for label in (0, 1)
        }
        selected = run._select_examples(bank, rankings, input_channel=input_channel)
        latency_ms = (time.perf_counter() - started) * 1_000
        if latency_ms > latency_gate_ms:
            raise TimeoutError("mLateOn exceeded its latency gate")
    except Exception as error:
        return {
            "unit_id": baseline["unit_id"],
            "method": MLATEON_METHOD,
            "status": baseline["status"],
            "failure_code": baseline.get("failure_code"),
            "selected_ids": fallback_selected,
            "candidate_ids": fallback_candidates,
            "rrf_method": baseline["method"],
            "rrf_selected_ids": fallback_selected,
            "colbert_fallback": True,
            "colbert_failure_code": type(error).__name__,
            "latency_ms": (time.perf_counter() - started) * 1_000,
            "latency_gate_ms": latency_gate_ms,
            "latency_gate_is_hard_deadline": False,
        }
    return {
        "unit_id": baseline["unit_id"],
        "method": MLATEON_METHOD,
        "status": "ok",
        "failure_code": None,
        "selected_ids": selected,
        "candidate_ids": {str(label): rankings[label] for label in (0, 1)},
        "rrf_method": baseline["method"],
        "rrf_selected_ids": fallback_selected,
        "colbert_fallback": False,
        "colbert_failure_code": None,
        "latency_ms": latency_ms,
        "latency_gate_ms": latency_gate_ms,
        "latency_gate_is_hard_deadline": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage0 = subparsers.add_parser("stage0")
    stage0.add_argument("--snapshot", type=Path, required=True)
    stage0.add_argument("--device", choices=("cpu", "cuda"), required=True)
    rerank = subparsers.add_parser("rerank")
    rerank.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE)
    rerank.add_argument("--snapshot", type=Path, required=True)
    rerank.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "stage0":
        stage0_canary(output, snapshot=args.snapshot, device=args.device)
    else:
        rerank_study(
            output,
            source=args.source_output,
            snapshot=args.snapshot,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
