from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import types
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.retrieval_assisted_reviewer import run


class _EmbeddingResponse:
    def __init__(self, payload: dict) -> None:
        self.status = 200
        self.headers = {}
        self._payload = payload

    async def __aenter__(self) -> _EmbeddingResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self, *, content_type: object = None) -> dict:
        return self._payload


class _EmbeddingSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.request: dict | None = None

    def post(self, url: str, **kwargs: object) -> _EmbeddingResponse:
        self.request = {"url": url, **kwargs}
        return _EmbeddingResponse(self._payload)


class _FakeHnswIndex:
    def __init__(self, dimension: int, m: int, metric: int) -> None:
        self.dimension = dimension
        self.m = m
        self.metric = metric
        self.d = dimension
        self.metric_type = metric
        self.hnsw = types.SimpleNamespace(
            efConstruction=0,
            efSearch=0,
            nb_neighbors=lambda _: m,
        )
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32).copy()

    def search(self, query: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.vectors @ np.asarray(query[0], dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:count]
        return scores[order][None, :], order.astype(np.int64)[None, :]

    def reconstruct_batch(self, positions: np.ndarray) -> np.ndarray:
        return self.vectors[np.asarray(positions, dtype=np.int64)]


class _FakeFaiss:
    METRIC_INNER_PRODUCT = 0
    IndexHNSWFlat = _FakeHnswIndex
    __version__ = "test"

    @staticmethod
    def omp_set_num_threads(_: int) -> None:
        return None

    @staticmethod
    def omp_get_max_threads() -> int:
        return 1

    @staticmethod
    def get_compile_options() -> str:
        return "test"

    @staticmethod
    def write_index(index: _FakeHnswIndex, path: str) -> None:
        with Path(path).open("wb") as handle:
            np.savez(
                handle,
                vectors=index.vectors,
                dimension=index.dimension,
                m=index.m,
                metric=index.metric,
                ef_construction=index.hnsw.efConstruction,
                ef_search=index.hnsw.efSearch,
            )

    @staticmethod
    def read_index(path: str) -> _FakeHnswIndex:
        with np.load(path, allow_pickle=False) as saved:
            index = _FakeHnswIndex(
                int(saved["dimension"]), int(saved["m"]), int(saved["metric"])
            )
            index.add(saved["vectors"])
            index.hnsw.efConstruction = int(saved["ef_construction"])
            index.hnsw.efSearch = int(saved["ef_search"])
        return index


class AnalyzeReviewIdentityTests(unittest.TestCase):
    def test_analyze_rejects_changed_selected_examples(self) -> None:
        unit = {"unit_id": "unit-1"}
        job = {
            "job_id": "job-1",
            "arm": "baseline",
            "row": {"panel_id": "unit-1", "text_sha256": "text-sha"},
            "prompt_sha256": "prompt-sha",
            "selected_ids": [],
            "retrieval_fallback": False,
        }
        record = {
            "job_id": "job-1",
            "arm": "baseline",
            "row_id": "unit-1",
            "status": "ok",
            "prompt_sha256": "prompt-sha",
            "text_sha256": "text-sha",
            "selected_ids": ["changed"],
            "retrieval_fallback": False,
            "transport": "strict_logprob",
            "requested_provider": "cloudflare",
            "requested_model": run.MODEL,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(run, "_study_manifest", return_value={}),
            mock.patch.object(run, "_load_local_evidence", return_value=([], [unit])),
            mock.patch.object(run, "_read_jsonl", return_value=[record]),
            mock.patch.object(run, "_review_jobs", return_value=[job]),
        ):
            with self.assertRaisesRegex(ValueError, "review identity changed"):
                run.analyze(Path(directory), split="validation")


def _bank_rows() -> list[dict]:
    rows = []
    for label in (0, 1):
        for index in range(3):
            rows.append(
                {
                    "id": f"row-{label}-{index}",
                    "text": f"instruction hierarchy example {label} {index}",
                    "label": label,
                    "source": f"source-{label}-{index}",
                    "input_channel": "direct_user",
                    "group_id": f"group-{label}-{index}",
                    "security_tags": (
                        ["benign"] if label == 0 else ["direct_prompt_injection"]
                    ),
                }
            )
    return rows


def _bank() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    run._create_bank_schema(connection)
    rows = _bank_rows()
    run._insert_bank_rows(connection, rows, {row["source"]: "MIT" for row in rows})
    connection.execute("INSERT INTO fts_unicode(fts_unicode) VALUES('rebuild')")
    connection.execute("INSERT INTO fts_trigram(fts_trigram) VALUES('rebuild')")
    return connection


def _lineage_output(path: Path) -> tuple[Path, list[dict]]:
    rows = []
    for channel in ("direct_user", "untrusted_content"):
        for label in (0, 1):
            for index in range(3):
                rows.append(
                    {
                        "id": f"{channel}-{label}-{index}",
                        "text": f"instruction hierarchy {channel} {label} {index}",
                        "label": label,
                        "source": f"source-{channel}-{label}-{index}",
                        "input_channel": channel,
                        "group_id": f"group-{channel}-{label}-{index}",
                        "security_tags": (
                            ["benign"] if label == 0 else ["direct_prompt_injection"]
                        ),
                    }
                )
    bank_path, summary = run._write_bank(
        path,
        rows,
        {row["source"]: "MIT" for row in rows},
    )
    manifest = {
        "schema_version": 1,
        "advisory_only": True,
        "production_changes": False,
        "inputs": {"data_manifest_sha256": "a" * 64},
        "bank": {
            **summary,
            "mode": "full_lineage",
            "path": bank_path.name,
            "sha256": run.file_sha256(bank_path),
            "max_example_bytes": 1024,
            "routing_view_sha256": "b" * 64,
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return path, rows


def _lineage_serving_source(path: Path) -> Path:
    path.mkdir(parents=True)
    source, _ = _lineage_output(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    dense_path = source / "dense-pplx-4b-256.sqlite3"
    dense = sqlite3.connect(dense_path)
    bank = sqlite3.connect(source / "bank.sqlite3")
    try:
        dense.execute(
            "CREATE TABLE vectors (example_rowid INTEGER PRIMARY KEY, vector BLOB NOT NULL)"
        )
        for (rowid,) in bank.execute("SELECT rowid FROM examples ORDER BY rowid"):
            vector = np.zeros(256, dtype=np.float32)
            vector[(int(rowid) - 1) % 256] = 1.0
            dense.execute(
                "INSERT INTO vectors VALUES (?, ?)", (rowid, vector.tobytes())
            )
        dense.commit()
    finally:
        bank.close()
        dense.close()
    (source / "dense-pplx-4b-256.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "document_key": "pplx-4b-256",
                "model": run.DENSE_CONFIGS["pplx-4b"]["document_model"],
                "dimension": 256,
                "input_type": None,
                "bank_sha256": manifest["bank"]["sha256"],
                "rows": manifest["bank"]["rows"],
                "path": dense_path.name,
                "sha256": run.file_sha256(dense_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parameters = {
        f"faiss_hnsw_ef{ef_search}_top{overretrieve}": {
            "parameters": {
                "m": 32,
                "ef_construction": 200,
                "ef_search": ef_search,
                "overretrieve": overretrieve,
                "exact_rescore": 20,
                "exact_rescore_dtype": "float32",
            }
        }
        for ef_search, overretrieve in ((1_024, 160),)
    }
    (source / "validation-hnsw-extension-pplx-4b.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "validation",
                "config": "pplx-4b",
                "bank_mode": "full_lineage",
                "bank_sha256": manifest["bank"]["sha256"],
                "bank_rows": manifest["bank"]["rows"],
                "query_matrix": {
                    "fresh_for_this_run": True,
                    "model": run.DENSE_CONFIGS["pplx-4b"]["query_model"],
                    "input_type": None,
                    "dtype": "float32",
                },
                "backends": {"faiss_hnsw": {"variants": parameters}},
                "variant_decisions": {
                    "faiss_hnsw_ef1024_top160": {"advances_to_cascade": True}
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source


class DenseVectorReuseTests(unittest.TestCase):
    def test_reuses_exact_subset_by_example_identity_without_provider_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _lineage_serving_source(root / "source")
            source_manifest = json.loads(
                (source / "manifest.json").read_text(encoding="utf-8")
            )
            source_bank = sqlite3.connect(source / "bank.sqlite3")
            try:
                rows = [
                    {
                        "id": example_id,
                        "text": text,
                        "label": label,
                        "source": row_source,
                        "input_channel": channel,
                        "group_id": group_id,
                        "security_tags": subtype.split(","),
                    }
                    for example_id, text, label, row_source, channel, group_id, subtype in source_bank.execute(
                        "SELECT example_id, text, label, source, input_channel, group_id, subtype "
                        "FROM examples ORDER BY example_id LIMIT -1 OFFSET 1"
                    )
                ]
            finally:
                source_bank.close()
            target = root / "target"
            target.mkdir()
            bank_path, summary = run._write_bank(
                target,
                rows,
                {row["source"]: "MIT" for row in rows},
            )
            manifest = {
                **source_manifest,
                "bank": {
                    **summary,
                    "mode": "full_lineage",
                    "path": bank_path.name,
                    "sha256": run.file_sha256(bank_path),
                },
            }
            (target / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )

            identity = run.reuse_bank_vectors(
                target,
                source_output=source,
                config_name="pplx-4b",
            )

            self.assertFalse(identity["provider_calls"])
            self.assertEqual(identity["rows"], len(rows))
            dense = sqlite3.connect(target / identity["path"])
            try:
                self.assertEqual(
                    dense.execute("SELECT COUNT(*) FROM vectors").fetchone(),
                    (len(rows),),
                )
            finally:
                dense.close()


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_balanced_quotas_use_the_requested_population(self) -> None:
        quotas = run._equal_quotas(Counter({("a",): 2, ("b",): 10}), 8)

        self.assertEqual(quotas, {("a",): 2, ("b",): 6})

    def test_sparse_queries_are_parameterized_and_select_balanced_examples(
        self,
    ) -> None:
        connection = _bank()
        try:
            rankings, _ = run._sparse_rank(
                connection,
                'instruction " OR label:1 *',
                channel="direct_user",
                tokenizer="unicode",
            )
            selected = run._select_examples(
                connection, rankings, input_channel="direct_user"
            )
        finally:
            connection.close()

        self.assertEqual(len(selected), 4)
        self.assertEqual([int(value.split("-")[1]) for value in selected], [0, 1, 0, 1])

    def test_rrf_combines_ranks_without_score_calibration(self) -> None:
        fused = run._rrf(
            {0: ["a", "b"], 1: ["x", "y"]},
            {0: ["b", "a"], 1: ["x", "z"]},
        )
        dense_weighted = run._rrf(
            {0: ["a"], 1: ["x"]},
            {0: ["b"], 1: ["y"]},
            right_weight=2.0,
        )

        self.assertEqual(fused[0], ["a", "b"])
        self.assertEqual(fused[1][0], "x")
        self.assertEqual(dense_weighted, {0: ["b", "a"], 1: ["y", "x"]})

    def test_lineage_deduplication_keeps_the_first_ranked_example(self) -> None:
        connection = _bank()
        try:
            connection.execute(
                """
                INSERT INTO examples(
                    example_id, text, text_sha256, label, input_channel,
                    source, group_id, subtype, license
                )
                SELECT 'row-0-duplicate', text, text_sha256, label, input_channel,
                       source, group_id, subtype, license
                FROM examples WHERE example_id = 'row-0-0'
                """
            )
            deduplicated = run._deduplicate_lineages(
                connection,
                {
                    0: ["row-0-0", "row-0-duplicate", "row-0-1"],
                    1: ["row-1-0", "row-1-1"],
                },
            )
        finally:
            connection.close()

        self.assertEqual(deduplicated[0], ["row-0-0", "row-0-1"])

    def test_partitioned_sparse_sidecar_is_contentless_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, rows = _lineage_output(Path(temporary))

            identity = run.build_partitioned_sparse_index(output)
            manifest = run._study_manifest(output)
            sparse, rowids = run._open_partitioned_sparse_index(output, manifest)
            try:
                rankings, _ = run._partitioned_sparse_rank(
                    sparse,
                    rowids,
                    "instruction hierarchy",
                    channel="direct_user",
                )
                stored_text = sparse.execute(
                    "SELECT text FROM fts_direct_user_0 LIMIT 1"
                ).fetchone()
            finally:
                sparse.close()

            expected = {
                label: {
                    row["id"]
                    for row in rows
                    if row["input_channel"] == "direct_user" and row["label"] == label
                }
                for label in (0, 1)
            }
            self.assertEqual(
                {label: set(values) for label, values in rankings.items()}, expected
            )
            self.assertEqual(stored_text, (None,))
            self.assertEqual(identity["bank_sha256"], manifest["bank"]["sha256"])
            self.assertEqual(
                identity["sha256"],
                run.file_sha256(output / identity["path"]),
            )
            self.assertFalse(
                (output / f".{run.PARTITIONED_SPARSE_INDEX_PATH}.tmp").exists()
            )

    def test_partitioned_hybrid_replay_falls_back_to_exact_dense_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, rows = _lineage_output(Path(temporary))
            run.build_partitioned_sparse_index(output)
            manifest = run._study_manifest(output)
            sparse, rowids = run._open_partitioned_sparse_index(output, manifest)
            bank = sqlite3.connect(output / manifest["bank"]["path"])
            dense_rankings = {
                label: [
                    row["id"]
                    for row in rows
                    if row["input_channel"] == "direct_user" and row["label"] == label
                ]
                for label in (0, 1)
            }
            dense_selected = run._select_examples(
                bank, dense_rankings, input_channel="direct_user"
            )
            dense_record = {
                "unit_id": "unit",
                "method": run.PARTITIONED_DENSE_REPLAY_METHOD,
                "status": "ok",
                "selected_ids": dense_selected,
                "candidate_ids": {
                    str(label): values for label, values in dense_rankings.items()
                },
                "latency_ms": 12.5,
            }
            unit = {"unit_id": "unit", "input_channel": "direct_user"}
            try:
                successful = run._partitioned_replay_records(
                    bank,
                    sparse,
                    rowids,
                    query_text="instruction hierarchy",
                    unit=unit,
                    dense_record=dense_record,
                )
                sparse.close()
                fallback = run._partitioned_replay_records(
                    bank,
                    sparse,
                    rowids,
                    query_text="instruction hierarchy",
                    unit=unit,
                    dense_record=dense_record,
                )
            finally:
                bank.close()

        self.assertEqual(successful[0]["status"], "ok")
        self.assertFalse(successful[1]["sparse_fallback"])
        self.assertEqual(fallback[0]["status"], "failed")
        self.assertTrue(fallback[1]["sparse_fallback"])
        self.assertEqual(fallback[1]["selected_ids"], dense_selected)
        self.assertEqual(fallback[1]["candidate_ids"], dense_record["candidate_ids"])

    def test_fullrow_sparse_sidecar_overretrieves_then_keeps_unique_lineages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = _lineage_output(Path(temporary))
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bank_path = output / manifest["bank"]["path"]
            with sqlite3.connect(bank_path) as bank:
                bank.execute(
                    """
                    INSERT INTO examples(
                        example_id, text, text_sha256, label, input_channel,
                        source, group_id, subtype, license
                    )
                    SELECT 'direct_user-0-duplicate', text, text_sha256, label,
                           input_channel, source, group_id, subtype, license
                    FROM examples WHERE example_id = 'direct_user-0-0'
                    """
                )
            manifest["bank"]["mode"] = "full"
            manifest["bank"]["rows"] += 1
            manifest["bank"]["sha256"] = run.file_sha256(bank_path)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True), encoding="utf-8"
            )

            identity = run.build_fullrow_partitioned_sparse_index(output)
            sparse = run._open_fullrow_partitioned_sparse_index(output, manifest)
            bank = sqlite3.connect(bank_path)
            try:
                rankings, _ = run._fullrow_partitioned_sparse_rank(
                    sparse,
                    bank,
                    "instruction hierarchy",
                    channel="direct_user",
                )
            finally:
                bank.close()
                sparse.close()

        self.assertEqual(identity["raw_candidates_per_label"], 320)
        self.assertEqual(identity["retained_lineages_per_label"], 50)
        self.assertEqual(len(rankings[0]), 3)
        self.assertNotIn("direct_user-0-duplicate", rankings[0])

    def test_fullrow_hnsw_hybrid_rescues_dense_failure_and_fails_soft(self) -> None:
        bank = _bank()
        unit = {"unit_id": "unit", "input_channel": "direct_user"}
        dense_candidates = {0: ["row-0-0"], 1: ["row-1-0"]}
        dense_failure = {
            "unit_id": "unit",
            "method": run.HNSW_HYBRID_DENSE_ARM,
            "status": "failed",
            "failure_code": "insufficient_balanced_candidates",
            "selected_ids": [],
            "candidate_ids": {
                str(label): values for label, values in dense_candidates.items()
            },
        }
        try:
            _, rescued = run._fullrow_hnsw_hybrid_records(
                bank,
                unit=unit,
                dense_record=dense_failure,
                sparse_rankings={
                    0: ["row-0-1", "row-0-2"],
                    1: ["row-1-1", "row-1-2"],
                },
                sparse_latency_ms=5.0,
            )
            dense_candidates = {
                label: [f"row-{label}-{index}" for index in range(3)]
                for label in (0, 1)
            }
            dense_selected = run._select_examples(
                bank, dense_candidates, input_channel="direct_user"
            )
            dense_success = {
                **dense_failure,
                "status": "ok",
                "failure_code": None,
                "selected_ids": dense_selected,
                "candidate_ids": {
                    str(label): values for label, values in dense_candidates.items()
                },
            }
            _, fallback = run._fullrow_hnsw_hybrid_records(
                bank,
                unit=unit,
                dense_record=dense_success,
                sparse_rankings={0: [], 1: []},
                sparse_latency_ms=250.0,
                sparse_failure_code="timeout",
            )
        finally:
            bank.close()

        self.assertEqual(rescued["status"], "ok")
        self.assertTrue(rescued["dense_failure_rescued"])
        self.assertFalse(rescued["sparse_fallback"])
        self.assertEqual(len(rescued["selected_ids"]), 4)
        self.assertEqual(fallback["selected_ids"], dense_selected)
        self.assertEqual(fallback["candidate_ids"], dense_success["candidate_ids"])
        self.assertTrue(fallback["sparse_fallback"])
        self.assertEqual(fallback["sparse_failure_code"], "timeout")

    def test_embedding_contract_normalizes_and_rejects_invalid_dimensions(self) -> None:
        payload = {
            "model": "Vendor/Model",
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 4.0]},
            ],
        }

        matrix = run._embedding_vectors(
            payload,
            expected_model="vendor/model",
            expected_rows=2,
            dimension=2,
        )

        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "dimension"):
            run._embedding_vectors(
                payload,
                expected_model="vendor/model",
                expected_rows=2,
                dimension=3,
            )

    def test_dense_loader_rejects_a_stale_embedding_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = _lineage_serving_source(Path(temporary) / "source")
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            identity_path = output / "dense-pplx-4b-256.json"
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            identity["model"] = "different/model"
            identity_path.write_text(json.dumps(identity), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dense bank identity changed"):
                run._load_dense_index(output, manifest, run.DENSE_CONFIGS["pplx-4b"])

    def test_qwen_stage0_request_is_pinned_and_only_queries_are_prefixed(self) -> None:
        document = "A customer asks how to update a delivery address."
        query = run._qwen_stage0_text(document, query=True)
        self.assertEqual(run._qwen_stage0_text(document, query=False), document)
        self.assertEqual(
            query,
            "Instruct: Given text from an LLM application, retrieve labeled examples "
            "with similar instruction-subversion behavior.\nQuery:" + document,
        )
        with self.assertRaisesRegex(ValueError, "over-length"):
            run._qwen_stage0_text("x" * run.QWEN_STAGE0_MAX_INPUT_BYTES, query=True)

        vector = [1.0, *([0.0] * 255)]
        session = _EmbeddingSession(
            {
                "model": "qwen3-embedding-8b",
                "provider": "Nebius",
                "data": [{"index": 0, "embedding": vector}],
                "usage": {
                    "prompt_tokens": 40,
                    "total_tokens": 40,
                    "cost": 0.000001,
                },
            }
        )
        matrix, metadata = asyncio.run(
            run._call_embeddings(
                session,
                "secret",
                texts=[query],
                model=run.QWEN_STAGE0_MODEL,
                dimension=256,
                input_type=None,
                provider="nebius",
            )
        )

        assert session.request is not None
        body = session.request["json"]
        self.assertEqual(
            body["provider"],
            {
                "order": ["nebius"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        self.assertNotIn("zdr", json.dumps(body).casefold())
        self.assertNotIn("data_collection", body)
        self.assertEqual(body["input"], [query])
        self.assertEqual(matrix.shape, (1, 256))
        self.assertEqual(metadata["requested_provider"], "nebius")
        self.assertEqual(metadata["response_provider"], "Nebius")
        self.assertEqual(metadata["cost_usd"], "0.000001")

    def test_qwen_stage0_canary_records_only_hashes_and_contract_metrics(self) -> None:
        calls = []

        async def fake_call(
            _session: object,
            _api_key: str,
            *,
            texts: list[str],
            model: str,
            dimension: int,
            input_type: str | None,
            provider: str | None = None,
        ) -> tuple[np.ndarray, dict]:
            calls.append(
                {
                    "texts": texts,
                    "model": model,
                    "dimension": dimension,
                    "input_type": input_type,
                    "provider": provider,
                }
            )
            matrix = np.zeros((len(texts), dimension), dtype=np.float32)
            for index, text in enumerate(texts):
                sample = text.removeprefix(run.QWEN_STAGE0_QUERY_PREFIX)
                matrix[index, run.QWEN_STAGE0_SAMPLES.index(sample)] = 1.0
            return matrix, {
                "model": model,
                "response_model": "qwen3-embedding-8b",
                "requested_provider": provider,
                "response_provider": "Nebius",
                "rows": len(texts),
                "attempts": 1,
                "latency_ms": 1.25,
                "prompt_tokens": len(texts) * 10,
                "total_tokens": len(texts) * 10,
                "cost_usd": "0.000001",
            }

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "canary"
            with (
                mock.patch.object(run, "_call_embeddings", side_effect=fake_call),
                mock.patch.object(run.provider_helpers, "_api_key", return_value="key"),
            ):
                path = asyncio.run(run.qwen_stage0_canary(output, provider="nebius"))

            artifact = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(artifact, sort_keys=True)
            budget = json.loads((output / "budget.json").read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 8)
        self.assertTrue(
            all(
                call["model"] == run.QWEN_STAGE0_MODEL
                and call["dimension"] == 256
                and call["input_type"] is None
                and call["provider"] == "nebius"
                for call in calls
            )
        )
        self.assertIn(list(run.QWEN_STAGE0_SAMPLES), [call["texts"] for call in calls])
        self.assertIn(
            [
                run.QWEN_STAGE0_QUERY_PREFIX + sample
                for sample in run.QWEN_STAGE0_SAMPLES
            ],
            [call["texts"] for call in calls],
        )
        self.assertEqual(artifact["status"], "passed")
        self.assertEqual(artifact["failures"], [])
        self.assertEqual(artifact["contract"]["requested_provider"], "nebius")
        self.assertFalse(artifact["provider_identity_proven"])
        self.assertEqual(artifact["calls_with_recorded_cost"], 8)
        self.assertEqual(artifact["recorded_cost_usd"], "0.000008")
        self.assertTrue(all(record["vector_sha256"] for record in artifact["calls"]))
        self.assertNotIn("vectors", artifact)
        for sample in run.QWEN_STAGE0_SAMPLES:
            self.assertNotIn(sample, serialized)
        self.assertEqual(
            budget["reservations"]["embedding-contract:qwen3-8b-256:nebius"],
            "0.01",
        )

    def test_qwen_local_stage0_truncates_dimensions_before_normalizing(self) -> None:
        pooled = np.zeros((1, run.QWEN_STAGE0_DIMENSION + 1), dtype=np.float32)
        pooled[0, :2] = (3.0, 4.0)
        pooled[0, -1] = 100.0

        matrix = run._qwen_local_stage0_finalize(pooled)

        self.assertEqual(matrix.shape, (1, run.QWEN_STAGE0_DIMENSION))
        np.testing.assert_allclose(matrix[0, :2], [0.6, 0.8])
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0])
        with self.assertRaisesRegex(ValueError, "invalid norm"):
            run._qwen_local_stage0_finalize(
                np.zeros((1, run.QWEN_STAGE0_DIMENSION), dtype=np.float32)
            )

    def test_qwen_local_stage0_load_is_revision_pinned_and_offline(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = mock.Mock()
        fake_torch.cuda.is_available.return_value = False
        tokenizer = mock.Mock(padding_side="left", pad_token_id=0)
        model = mock.Mock()
        model.to.return_value = model
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = mock.Mock()
        fake_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        fake_transformers.AutoModel = mock.Mock()
        fake_transformers.AutoModel.from_pretrained.return_value = model

        with mock.patch.dict(
            "sys.modules",
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            returned_torch, returned_tokenizer, returned_model = (
                run._load_qwen_local_stage0("cpu")
            )
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                run._load_qwen_local_stage0("cuda")

        expected = {
            "revision": run.QWEN_LOCAL_STAGE0_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        fake_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            run.QWEN_LOCAL_STAGE0_MODEL,
            **expected,
            padding_side="left",
        )
        fake_transformers.AutoModel.from_pretrained.assert_called_once_with(
            run.QWEN_LOCAL_STAGE0_MODEL,
            **expected,
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        self.assertIs(returned_torch, fake_torch)
        self.assertIs(returned_tokenizer, tokenizer)
        self.assertIs(returned_model, model)

    def test_dense_index_comparison_accepts_equal_score_tie_replacements(self) -> None:
        expected = [
            {
                "unit_id": "unit",
                "rankings": {
                    "0": [["a", 0.9], ["b", 0.8]],
                    "1": [["x", 0.7], ["y", 0.6]],
                },
            }
        ]
        actual = [
            {
                "unit_id": "unit",
                "rankings": {
                    "0": [["a", 0.9], ["c", 0.8]],
                    "1": [["x", 0.7], ["y", 0.6]],
                },
            }
        ]

        comparison = run._benchmark_ranking_comparison(expected, actual)

        self.assertEqual(comparison["set_matches"], 1)
        self.assertEqual(comparison["tie_aware_score_matches"], 2)
        self.assertEqual(comparison["mean_set_recall_at_20"], 0.75)
        self.assertEqual(comparison["mean_score_regret_at_20"], 0.0)

    def test_dense_index_comparison_reports_selected_packet_parity(self) -> None:
        good = {
            "0": [[f"row-0-{index}", 1.0 - index / 10] for index in range(3)],
            "1": [[f"row-1-{index}", 1.0 - index / 10] for index in range(3)],
        }
        repeated = {
            "0": [["row-0-0", 1.0 - index / 10] for index in range(3)],
            "1": [["row-1-0", 1.0 - index / 10] for index in range(3)],
        }
        expected = [
            {"unit_id": "stable", "rankings": good},
            {"unit_id": "failure", "rankings": good},
        ]
        actual = [
            {"unit_id": "stable", "rankings": good},
            {"unit_id": "failure", "rankings": repeated},
        ]
        bank = _bank()
        try:
            comparison = run._benchmark_ranking_comparison(
                expected,
                actual,
                bank=bank,
                unit_channels={
                    "stable": "direct_user",
                    "failure": "direct_user",
                },
            )
        finally:
            bank.close()

        self.assertEqual(
            comparison["selected_packet_parity"],
            {
                "queries": 2,
                "exact_matches": 1,
                "exact_match_rate": 0.5,
                "numpy_selection_failures": 0,
                "candidate_selection_failures": 1,
                "either_selection_failures": 1,
            },
        )

    def test_hnsw_candidates_are_exactly_rescored_with_stable_ties(self) -> None:
        rankings = run._benchmark_exact_rescore(
            ["b", "a", "c"],
            np.asarray([2, 0, 1]),
            np.asarray([[0.5, 0.0], [0.8, 0.0], [0.8, 0.0]]),
            np.asarray([1.0, 0.0]),
            count=2,
        )

        self.assertEqual([row[0] for row in rankings], ["a", "b"])
        np.testing.assert_allclose([row[1] for row in rankings], [0.8, 0.8])

    def test_hnsw_extension_gates_use_set_recall_and_ignore_packet_parity(self) -> None:
        comparison = {
            "mean_set_recall_at_20": 0.99,
            "mean_tie_aware_score_recall_at_20": 0.5,
            "worst_adequately_sized_slice": {
                "slice": "source:example",
                "rankings": 20,
                "mean_set_recall_at_20": 0.96,
            },
            "selected_packet_parity": {"exact_match_rate": 0.0},
        }
        timing = {
            "workers_1": {"p95_ms": 100.0},
            "workers_4": {"p95_ms": 80.0},
        }
        candidate_timing = {
            "workers_1": {"p95_ms": 40.0},
            "workers_4": {"p95_ms": 32.0},
        }

        decision = run._hnsw_extension_gates(
            comparison,
            numpy_timing=timing,
            hnsw_timing=candidate_timing,
        )

        self.assertEqual(
            run.HNSW_EXTENSION_SETTINGS,
            ((512, 160), (512, 320), (1_024, 160), (1_024, 320)),
        )
        self.assertTrue(decision["advances_to_cascade"])
        self.assertTrue(decision["packet_parity_is_diagnostic"])
        self.assertFalse(decision["promotion_eligible"])

    def test_hnsw_extension_slice_gate_uses_candidate_set_recall(self) -> None:
        expected_rankings = {
            "0": [["a", 0.9], ["b", 0.8]],
            "1": [["x", 0.7], ["y", 0.6]],
        }
        actual_rankings = {
            "0": [["a", 0.9], ["c", 0.8]],
            "1": [["x", 0.7], ["z", 0.6]],
        }
        expected = [
            {"unit_id": f"unit-{index}", "rankings": expected_rankings}
            for index in range(10)
        ]
        actual = [
            {"unit_id": f"unit-{index}", "rankings": actual_rankings}
            for index in range(10)
        ]

        comparison = run._benchmark_ranking_comparison(
            expected,
            actual,
            unit_slices={f"unit-{index}": {"source": "example"} for index in range(10)},
        )

        self.assertEqual(comparison["mean_tie_aware_score_recall_at_20"], 1.0)
        self.assertEqual(
            comparison["worst_adequately_sized_slice"]["mean_set_recall_at_20"],
            0.5,
        )

    def test_hnsw_extension_retains_id_and_score_only_retrieval_evidence(self) -> None:
        rankings = [
            {
                "unit_id": "unit",
                "rankings": {
                    "0": [[f"row-0-{index}", 1.0 - index / 10] for index in range(3)],
                    "1": [[f"row-1-{index}", 1.0 - index / 10] for index in range(3)],
                },
            }
        ]
        bank = _bank()
        try:
            evidence = run._benchmark_retrieval_evidence(
                rankings,
                bank=bank,
                unit_channels={"unit": "direct_user"},
            )
        finally:
            bank.close()

        self.assertEqual(
            evidence[0]["selected_ids"],
            ["row-0-0", "row-1-0", "row-0-1", "row-1-1"],
        )
        self.assertEqual(evidence[0]["candidate_ids"]["0"][0], "row-0-0")
        self.assertEqual(evidence[0]["candidate_scores"]["1"][0], 1.0)
        self.assertNotIn("text", json.dumps(evidence))

    def test_lineage_serving_bundle_is_portable_and_omits_dense_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _lineage_serving_source(root / "source")
            run.build_partitioned_sparse_index(source)
            output = root / "lineage-hybrid-v1"

            manifest_path = run.build_lineage_serving_bundle(
                output,
                source_output=source,
                sparse_source=source,
                faiss_module=_FakeFaiss,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest_path.name, "manifest.json")
            self.assertEqual(manifest["source"]["data_manifest_sha256"], "a" * 64)
            self.assertEqual(manifest["source"]["routing_view_sha256"], "b" * 64)
            self.assertEqual(
                Counter(entry["role"] for entry in manifest["files"]),
                Counter({"bank": 1, "sparse": 1, "index": 4, "row_map": 4}),
            )
            for entry in manifest["files"]:
                payload = output / entry["path"]
                self.assertTrue(payload.is_file())
                self.assertFalse(payload.is_symlink())
            self.assertFalse((output / "dense-pplx-4b-256.sqlite3").exists())
            self.assertTrue(
                run.RetrievalEngine(
                    output,
                    run.file_sha256(manifest_path),
                    faiss_module=_FakeFaiss,
                ).available
            )

    def test_lineage_hybrid_parity_is_bound_sanitized_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            sparse_source = root / "sparse"
            source.mkdir()
            sparse_source.mkdir()
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            unit = {"unit_id": "unit-1", "input_channel": "direct_user"}
            variants = {run.LINEAGE_HNSW_VARIANT: [{"unit_id": unit["unit_id"]}]}
            extension_path = source / "extension.json"
            run._atomic_json(
                extension_path,
                {
                    "retrieval_evidence": {
                        "sha256": run._sha256_text(
                            json.dumps(variants, sort_keys=True, separators=(",", ":"))
                        ),
                        "contains_raw_text_or_query_vectors": False,
                        "variants": variants,
                    }
                },
            )
            sparse_identity = {"sha256": "d" * 64}
            sparse_identity_path = sparse_source / "sparse.json"
            run._atomic_json(sparse_identity_path, sparse_identity)
            selected_ids = ["a"]
            sparse_rankings = {"0": [], "1": []}
            retrieval_path = sparse_source / "validation-retrieval.jsonl"
            run._atomic_jsonl(
                retrieval_path,
                [
                    {
                        "unit_id": unit["unit_id"],
                        "method": run.PARTITIONED_HYBRID_METHOD,
                        "status": "ok",
                        "selected_ids": selected_ids,
                        "branch_candidate_ids": {"sparse": sparse_rankings},
                    }
                ],
            )
            serving_manifest = root / "serving-manifest.json"
            run._atomic_json(
                serving_manifest,
                {
                    "schema_version": 1,
                    "variant": run.LINEAGE_SERVING_VARIANT,
                    "bank": {"sha256": "b" * 64},
                    "sparse": {"sha256": sparse_identity["sha256"]},
                    "source": {
                        "manifest_sha256": run.file_sha256(source / "manifest.json"),
                        "extension_sha256": run.file_sha256(extension_path),
                        "sparse_identity_sha256": run.file_sha256(sparse_identity_path),
                    },
                },
            )
            evidence_path = root / "parity.json"

            with (
                mock.patch.object(
                    run,
                    "_lineage_serving_source_contract",
                    return_value=(
                        {"bank": {"sha256": "b" * 64, "path": "bank.sqlite3"}},
                        {},
                        extension_path,
                        sparse_identity,
                        sparse_identity_path,
                    ),
                ),
                mock.patch.object(
                    run, "_load_local_evidence", return_value=([], [unit])
                ),
                mock.patch.object(
                    run,
                    "_reload_unit_texts",
                    return_value={unit["unit_id"]: ("review", "private query")},
                ),
                mock.patch.object(
                    run,
                    "_open_partitioned_sparse_index",
                    return_value=(mock.Mock(), {}),
                ),
                mock.patch.object(run.sqlite3, "connect", return_value=mock.Mock()),
                mock.patch.object(
                    run,
                    "_partitioned_replay_records",
                    return_value=(
                        {},
                        {
                            "selected_ids": selected_ids,
                            "branch_candidate_ids": {"sparse": sparse_rankings},
                        },
                    ),
                ),
            ):
                evidence = run.write_lineage_hybrid_parity(
                    source,
                    sparse_source=sparse_source,
                    serving_manifest=serving_manifest,
                    evidence_output=evidence_path,
                )
                with self.assertRaises(FileExistsError):
                    run.write_lineage_hybrid_parity(
                        source,
                        sparse_source=sparse_source,
                        serving_manifest=serving_manifest,
                        evidence_output=evidence_path,
                    )

            self.assertEqual(evidence["status"], "passed")
            self.assertFalse(evidence["provider_calls"])
            self.assertEqual(evidence["exact_packet_matches"], 1)
            self.assertEqual(evidence["sparse_branch_differences"], 0)
            self.assertEqual(
                evidence["source"]["serving_manifest_sha256"],
                run.file_sha256(serving_manifest),
            )
            self.assertNotIn("private query", evidence_path.read_text(encoding="utf-8"))

    def test_lineage_hnsw_extension_has_one_fixed_recipe(self) -> None:
        self.assertEqual(run._hnsw_extension_settings("full_lineage"), ((1_024, 160),))
        self.assertEqual(
            run._hnsw_extension_settings("full"), run.HNSW_EXTENSION_SETTINGS
        )

    def test_hnsw_extension_worker_cannot_target_a_historical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaisesRegex(ValueError, "result path is not isolated"):
                run._benchmark_dense_index_worker(
                    path,
                    config_name="pplx-4b",
                    backend="faiss_hnsw",
                    query_path=path / "queries.npy",
                    result_path=path / "validation-dense-indexes-pplx-4b.json",
                    hnsw_extension=True,
                )

    def test_hnsw_cascade_selects_ordered_non_dominated_variants(self) -> None:
        artifact = {
            "variant_decisions": {
                "fast_low_ef": {"advances_to_cascade": True},
                "high_recall": {"advances_to_cascade": True},
                "slower_duplicate": {"advances_to_cascade": True},
                "rejected": {"advances_to_cascade": False},
            },
            "comparisons_to_fresh_numpy_ground_truth": {
                "fast_low_ef": {"mean_set_recall_at_20": 0.98},
                "high_recall": {"mean_set_recall_at_20": 0.995},
                "slower_duplicate": {"mean_set_recall_at_20": 0.995},
                "rejected": {"mean_set_recall_at_20": 1.0},
            },
            "backends": {
                "faiss_hnsw": {
                    "variants": {
                        "fast_low_ef": {
                            "parameters": {"ef_search": 512, "overretrieve": 160},
                            "timing": {"workers_4": {"p95_ms": 5.0}},
                        },
                        "high_recall": {
                            "parameters": {"ef_search": 1_024, "overretrieve": 160},
                            "timing": {"workers_4": {"p95_ms": 7.0}},
                        },
                        "slower_duplicate": {
                            "parameters": {"ef_search": 1_024, "overretrieve": 320},
                            "timing": {"workers_4": {"p95_ms": 9.0}},
                        },
                        "rejected": {
                            "parameters": {"ef_search": 64, "overretrieve": 80},
                            "timing": {"workers_4": {"p95_ms": 1.0}},
                        },
                    }
                }
            },
        }

        self.assertEqual(
            run.select_hnsw_cascade_variants(artifact),
            ["fast_low_ef", "high_recall"],
        )

    def test_hnsw_cascade_source_contract_pins_query_and_index_parameters(
        self,
    ) -> None:
        artifact = {
            "query_matrix": {
                "fresh_for_this_run": True,
                "model": run.DENSE_CONFIGS["pplx-4b"]["query_model"],
                "input_type": None,
                "dtype": "float32",
            },
            "backends": {
                "faiss_hnsw": {
                    "variants": {
                        f"faiss_hnsw_ef{ef_search}_top{overretrieve}": {
                            "parameters": {
                                "m": 32,
                                "ef_construction": 200,
                                "ef_search": ef_search,
                                "overretrieve": overretrieve,
                                "exact_rescore": 20,
                                "exact_rescore_dtype": "float32",
                            }
                        }
                        for ef_search, overretrieve in run.HNSW_EXTENSION_SETTINGS
                    }
                }
            },
        }

        run._validate_hnsw_extension_source_contract(artifact)
        for field, invalid in (
            ("fresh_for_this_run", False),
            ("model", "changed-model"),
            ("input_type", "query"),
            ("dtype", "float64"),
        ):
            changed = json.loads(json.dumps(artifact))
            changed["query_matrix"][field] = invalid
            with (
                self.subTest(query_field=field),
                self.assertRaisesRegex(ValueError, "source contract changed"),
            ):
                run._validate_hnsw_extension_source_contract(changed)
        changed = json.loads(json.dumps(artifact))
        del changed["query_matrix"]["input_type"]
        with self.assertRaisesRegex(ValueError, "source contract changed"):
            run._validate_hnsw_extension_source_contract(changed)
        variant = "faiss_hnsw_ef512_top160"
        for field, invalid in (
            ("m", 16),
            ("ef_construction", 100),
            ("ef_search", 256),
            ("overretrieve", 80),
            ("exact_rescore", 10),
            ("exact_rescore_dtype", "float64"),
        ):
            changed = json.loads(json.dumps(artifact))
            changed["backends"]["faiss_hnsw"]["variants"][variant]["parameters"][
                field
            ] = invalid
            with (
                self.subTest(hnsw_parameter=field),
                self.assertRaisesRegex(ValueError, "source contract changed"),
            ):
                run._validate_hnsw_extension_source_contract(changed)

    def test_hnsw_cascade_copies_only_an_exact_equivalent_numpy_response(self) -> None:
        numpy_job = {
            "job_id": "numpy-job",
            "arm": "numpy-arm",
            "prompt": "system prompt",
            "text": "packet text",
            "prompt_sha256": "prompt",
            "row": {
                "panel_id": "unit",
                "text_sha256": "text",
                "input_channel": "direct_user",
            },
            "selected_ids": ["a", "b", "c", "d"],
            "retrieval_fallback": False,
            "retrieval_latency_ms": 12.0,
        }
        hnsw_job = {
            **numpy_job,
            "job_id": "hnsw-job",
            "arm": "hnsw-arm",
            "retrieval_latency_ms": 1.0,
        }
        numpy_record = {
            "job_id": "numpy-job",
            "arm": "numpy-arm",
            "row_id": "unit",
            "status": "ok",
            "prompt_sha256": "prompt",
            "text_sha256": "text",
            "input_channel": "direct_user",
            "selected_ids": ["a", "b", "c", "d"],
            "retrieval_fallback": False,
            "probability": 0.25,
            "cost_usd": "0.01",
            "client_seconds": 2.0,
            "transport": "strict_logprob",
            "requested_provider": "cloudflare",
            "requested_model": run.MODEL,
        }

        copied = run.copy_equivalent_hnsw_review(
            numpy_job,
            hnsw_job,
            numpy_record,
        )

        self.assertEqual(copied["job_id"], "hnsw-job")
        self.assertEqual(copied["arm"], "hnsw-arm")
        self.assertEqual(copied["probability"], 0.25)
        self.assertEqual(copied["cost_usd"], "0")
        self.assertEqual(copied["client_seconds"], 2.0)
        self.assertEqual(copied["deduplicated_from_job_id"], "numpy-job")
        self.assertTrue(copied["provider_response_reused"])

    def test_hnsw_cascade_refuses_a_non_distinct_recomputed_job_id(self) -> None:
        job = {
            "job_id": "same-job",
            "arm": "numpy-arm",
            "prompt": "system prompt",
            "text": "packet text",
            "prompt_sha256": "prompt",
            "row": {
                "panel_id": "unit",
                "text_sha256": "text",
                "input_channel": "direct_user",
            },
            "selected_ids": [],
            "retrieval_fallback": True,
            "retrieval_latency_ms": 0.0,
        }
        record = {
            "job_id": "same-job",
            "arm": "numpy-arm",
            "row_id": "unit",
            "status": "ok",
            "prompt_sha256": "prompt",
            "text_sha256": "text",
            "input_channel": "direct_user",
            "selected_ids": [],
            "retrieval_fallback": True,
            "transport": "strict_logprob",
            "requested_provider": "cloudflare",
            "requested_model": run.MODEL,
        }

        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            run.copy_equivalent_hnsw_review(job, job, record)

    def test_hnsw_cascade_gates_allow_a_noninferior_direct_comparison(self) -> None:
        gates = run.hnsw_cascade_gates(
            recall_delta=-0.01,
            recall_interval=(-0.03, 0.01),
            fpr_delta=0.0025,
            fpr_interval=(-0.001, 0.0025),
            worst_slice_delta=-0.03,
            numpy_fallbacks=6,
            hnsw_fallbacks=6,
            terminal_failures=0,
        )

        self.assertTrue(all(gates.values()))

    def test_hnsw_hybrid_requires_a_material_quality_gain_for_complexity(self) -> None:
        passing = run._hnsw_hybrid_review_gates(
            recall_delta=0.02,
            recall_interval=(0.005, 0.03),
            fpr_delta=0.0,
            fpr_interval=(-0.001, 0.001),
            worst_slice_delta=-0.01,
            dense_terminal_failures=0,
            hybrid_terminal_failures=0,
        )
        tied = run._hnsw_hybrid_review_gates(
            recall_delta=0.0,
            recall_interval=(-0.01, 0.01),
            fpr_delta=0.0,
            fpr_interval=(0.0, 0.0),
            worst_slice_delta=0.0,
            dense_terminal_failures=0,
            hybrid_terminal_failures=0,
        )

        self.assertTrue(all(passing.values()))
        self.assertFalse(tied["material_quality_gain_for_added_complexity"])
        self.assertTrue(tied["fpr_noninferior"])

    def test_hnsw_analysis_binding_changes_when_retry_history_is_appended(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "manifest.json").write_text("{}\n", encoding="utf-8")
            (output / "validation-retrieval.jsonl").write_text(
                '{"unit_id":"unit"}\n', encoding="utf-8"
            )
            reviews_path = output / "validation-reviews.jsonl"
            reviews_path.write_text(
                '{"job_id":"job","status":"failed"}\n', encoding="utf-8"
            )
            before_records = run._read_jsonl(reviews_path)
            before = run._hnsw_cascade_analysis_binding(output, before_records)
            with reviews_path.open("a", encoding="utf-8") as handle:
                handle.write('{"job_id":"job","status":"ok"}\n')
            after_records = run._read_jsonl(reviews_path)
            after = run._hnsw_cascade_analysis_binding(output, after_records)

        self.assertEqual(before["review_record_count"], 1)
        self.assertEqual(before["latest_job_count"], 1)
        self.assertEqual(after["review_record_count"], 2)
        self.assertEqual(after["latest_job_count"], 1)
        self.assertNotEqual(
            before["validation_reviews_sha256"],
            after["validation_reviews_sha256"],
        )
        self.assertEqual(before["manifest_sha256"], after["manifest_sha256"])
        self.assertEqual(before["retrieval_sha256"], after["retrieval_sha256"])

    def test_budget_reservation_preserves_two_dollar_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run._reserve_budget(output, "first", Decimal("47"))
            run._reserve_budget(output, "first", Decimal("46"))
            run._reserve_budget(output, "second", Decimal("1"))
            with self.assertRaisesRegex(RuntimeError, "reserved research budget"):
                run._reserve_budget(output, "third", Decimal("0.01"))
            state = json.loads((output / "budget.json").read_text(encoding="utf-8"))

        self.assertEqual(state["reservations"], {"first": "47", "second": "1"})

    def test_latest_review_attempt_wins(self) -> None:
        latest = run._latest_job_records(
            [
                {"job_id": "job", "status": "failed"},
                {"job_id": "job", "status": "ok"},
            ]
        )

        self.assertEqual(latest["job"]["status"], "ok")

    def test_review_reservation_number_uses_only_pending_attempts(self) -> None:
        pending = [{"job_id": "representative"}]
        attempts = Counter({"imported-baseline": 1})

        self.assertEqual(run._review_reservation_run_number(pending, attempts), 0)
        attempts["representative"] = 1
        self.assertEqual(run._review_reservation_run_number(pending, attempts), 1)

    def test_analysis_waits_for_available_reviewer_retry(self) -> None:
        first_failure = [{"job_id": "job", "status": "failed"}]
        exhausted = [*first_failure, {"job_id": "job", "status": "failed"}]

        self.assertTrue(run._has_retryable_review_failures(first_failure))
        self.assertFalse(run._has_retryable_review_failures(exhausted))

    def test_embedding_429_backoff_honors_retry_after(self) -> None:
        self.assertEqual(run._embedding_retry_delay(429, "21", 1), 21.0)
        self.assertEqual(run._embedding_retry_delay(429, None, 2), 30.0)
        self.assertEqual(run._embedding_retry_delay(503, None, 2), 2.0)

    def test_full_bank_writer_streams_rows(self) -> None:
        rows = _bank_rows()
        with tempfile.TemporaryDirectory() as temporary:
            path, summary = run._write_bank(
                Path(temporary),
                (row for row in rows),
                {row["source"]: "MIT" for row in rows},
            )
            with sqlite3.connect(path) as connection:
                stored = connection.execute("SELECT COUNT(*) FROM examples").fetchone()[
                    0
                ]

        self.assertEqual(stored, len(rows))
        self.assertEqual(summary["rows"], len(rows))
        self.assertEqual(run._parse_bank_size("full"), "all_rows")
        self.assertEqual(run._parse_bank_size("all-rows"), "all_rows")
        self.assertIsNone(run._parse_bank_size("full-lineage"))

    def test_full_bank_uses_the_downstream_mode_contract(self) -> None:
        rows = _bank_rows()
        licenses = {row["source"]: "MIT" for row in rows}
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(
                run,
                "routing_views",
                return_value={"train": (Path("unused"), {"sha256": "test"})},
            ),
            mock.patch.object(
                run,
                "canonical_rows",
                side_effect=lambda *_args, **_kwargs: iter(rows),
            ),
            mock.patch.object(run, "_bank_candidate", return_value=True),
        ):
            _, summary = run._build_curated_bank(
                Path(temporary),
                bank_size="all_rows",
                references=[],
                licenses=licenses,
            )

        self.assertEqual(summary["mode"], "full")

    def test_full_bank_keeps_one_row_per_labelled_lineage_cell(self) -> None:
        rows = _bank_rows()
        duplicate = {**rows[0], "id": "duplicate", "text": "another variant"}
        other_label = {
            **rows[0],
            "id": "other-label",
            "label": 1,
            "security_tags": ["direct_prompt_injection"],
        }
        selected = run._lineage_representatives([*rows, duplicate, other_label])
        same_cell = min(
            (rows[0], duplicate),
            key=lambda row: run._rank("full-lineage", row["id"]),
        )

        self.assertEqual(len(selected), len(rows) + 1)
        self.assertIn(same_cell["id"], {row["id"] for row in selected})
        self.assertIn(other_label["id"], {row["id"] for row in selected})

    def test_examples_remain_user_level_in_a_json_packet(self) -> None:
        connection = _bank()
        try:
            packet = run._packet(
                connection,
                example_ids=["row-0-0", "row-1-0"],
                text="classify me",
                reverse=False,
            )
        finally:
            connection.close()
        value = json.loads(packet)

        self.assertEqual(value["text_to_classify"], "classify me")
        self.assertEqual([row["label"] for row in value["labeled_examples"]], [0, 1])
        self.assertNotIn("instruction hierarchy example", run.PACKET_PROMPT)

    def test_wmt_assignment_is_balanced_and_keeps_pairs_together(self) -> None:
        sources = {
            variant: [f"{variant} row {index}" for index in range(10)]
            for variant in run.WMT_SOURCE_FILES
        }

        pairs = run._wmt_candidate_pairs(sources)

        self.assertEqual(len(pairs), 10)
        self.assertEqual(
            Counter(clean["subtype"] for clean, _ in pairs),
            Counter({subtype: 2 for subtype in run.WMT_ATTACK_TYPES}),
        )
        self.assertTrue(
            all(
                clean["group_id"] == attack["group_id"]
                and (clean["label"], attack["label"]) == (0, 1)
                for clean, attack in pairs
            )
        )

    def test_group_bootstrap_resamples_complete_pairs(self) -> None:
        rows = [
            {"group_id": "a", "label": 0},
            {"group_id": "a", "label": 1},
            {"group_id": "b", "label": 0},
            {"group_id": "b", "label": 1},
        ]
        incumbent = np.asarray([False, False, False, True])
        candidate = np.asarray([False, True, True, True])

        first = run._paired_group_bootstrap_delta(
            rows, incumbent, candidate, iterations=100, seed=7
        )
        second = run._paired_group_bootstrap_delta(
            rows, incumbent, candidate, iterations=100, seed=7
        )

        self.assertEqual(first, second)
        self.assertEqual(first["groups"], 2)
        self.assertAlmostEqual(first["metrics"]["recall"]["delta"], 0.5)
        self.assertAlmostEqual(first["metrics"]["fpr"]["delta"], 0.5)

    def test_external_bank_gate_accepts_faster_equal_lineage_bank(self) -> None:
        rows = []
        for subtype in run.WMT_ATTACK_TYPES:
            for index in range(20):
                group_id = f"{subtype}-{index}"
                rows.extend(
                    (
                        {"group_id": group_id, "label": 0, "subtype": subtype},
                        {"group_id": group_id, "label": 1, "subtype": subtype},
                    )
                )
        predictions = np.asarray([False, True] * (len(rows) // 2))
        result = run._external_bank_comparison(
            rows,
            {
                run.EXTERNAL_LINEAGE_ARM: {"predictions": predictions},
                run.EXTERNAL_ALL_ROWS_ARM: {"predictions": predictions},
            },
            [
                {
                    "method": run.EXTERNAL_LINEAGE_ARM,
                    "status": "ok",
                    "exact_search_ms": 1.0,
                },
                {
                    "method": run.EXTERNAL_ALL_ROWS_ARM,
                    "status": "ok",
                    "exact_search_ms": 2.0,
                },
            ],
            {
                "analysis_contract": {
                    "recall_noninferiority_margin": 0.01,
                    "fpr_noninferiority_margin": 0.0025,
                    "critical_subtype_recall_margin": 0.03,
                }
            },
        )

        self.assertEqual(result["decision"], "lineage_confirmed")
        self.assertTrue(all(result["gates"].values()))


if __name__ == "__main__":
    unittest.main()
