from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from morgott.models import retrieval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _FakeIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.d = self.vectors.shape[1]
        self.ntotal = self.vectors.shape[0]
        self.metric_type = 0
        self.hnsw = SimpleNamespace(
            efConstruction=200,
            efSearch=0,
            nb_neighbors=lambda level: 64 if level == 0 else 32,
        )
        self.search_counts: list[int] = []
        self.search_thread_ids: list[int] = []

    def search(self, query: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        self.search_counts.append(count)
        self.search_thread_ids.append(threading.get_ident())
        scores = self.vectors @ query[0]
        positions = np.argsort(-scores, kind="stable")[:count]
        return scores[positions][None, :], positions.astype(np.int64)[None, :]

    def reconstruct_batch(self, positions: np.ndarray) -> np.ndarray:
        return self.vectors[positions]


class _FakeFaiss:
    METRIC_INNER_PRODUCT = 0
    IndexHNSWFlat = _FakeIndex

    def __init__(self, indexes: dict[str, _FakeIndex]) -> None:
        self.indexes = indexes
        self._omp_threads = threading.local()
        self.pin_thread_ids: list[int] = []

    def omp_set_num_threads(self, threads: int) -> None:
        self._omp_threads.value = threads
        self.pin_thread_ids.append(threading.get_ident())

    def omp_get_max_threads(self) -> int:
        return getattr(self._omp_threads, "value", 0)

    def read_index(self, path: str) -> _FakeIndex:
        return self.indexes[Path(path).name]


class _EmbeddingResponse:
    status = 200

    def __init__(self, payload=None):
        self.payload = payload or {
            "model": "pplx-embed-v1-4b",
            "data": [{"index": 0, "embedding": [3.0, 4.0] + [0.0] * 254}],
            "usage": {"prompt_tokens": 7, "cost": 0.000001},
        }
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def json(self, *, content_type=None):
        del content_type
        return self.payload

    async def iter_chunked(self, size):
        body = json.dumps(self.payload).encode()
        for start in range(0, len(body), size):
            yield body[start : start + size]


class _EmbeddingSession:
    def __init__(self, response=None):
        self.response = response or _EmbeddingResponse()

    def post(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        return self.response


class OpenRouterEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_uses_the_frozen_contract_and_normalizes_the_vector(self):
        session = _EmbeddingSession()
        embedder = retrieval.OpenRouterEmbedder("secret", session=session)

        result = await embedder.embed("query text")

        np.testing.assert_allclose(result.vector[:2], [0.6, 0.8])
        self.assertEqual(result.input_tokens, 7)
        self.assertEqual(result.cost_usd, 0.000001)
        self.assertEqual(session.url, "https://openrouter.ai/api/v1/embeddings")
        self.assertEqual(
            session.kwargs["json"],
            {
                "model": "perplexity/pplx-embed-v1-4b",
                "input": ["query text"],
                "dimensions": 256,
                "encoding_format": "float",
                "provider": {
                    "order": ["perplexity/int8"],
                    "allow_fallbacks": False,
                },
            },
        )

    async def test_embed_rejects_cost_that_overflows_float(self):
        payload = await _EmbeddingResponse().json()
        payload["usage"]["cost"] = "1e10000"
        embedder = retrieval.OpenRouterEmbedder(
            "secret", session=_EmbeddingSession(_EmbeddingResponse(payload))
        )

        with self.assertRaisesRegex(RuntimeError, "invalid cost"):
            await embedder.embed("query text")

    async def test_embed_rejects_an_oversized_provider_response(self):
        class OversizedResponse(_EmbeddingResponse):
            async def iter_chunked(self, size):
                del size
                yield b"x" * (retrieval._EMBEDDING_RESPONSE_MAX_BYTES + 1)

        embedder = retrieval.OpenRouterEmbedder(
            "secret", session=_EmbeddingSession(OversizedResponse())
        )

        with self.assertRaisesRegex(RuntimeError, "query embedding failed"):
            await embedder.embed("query text")


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.bundle = Path(self.temporary.name)
        self.rows = self._write_bank()
        self._write_sparse()
        self.faiss, self.manifest_sha256 = self._write_dense_and_manifest()
        self.engine = retrieval.RetrievalEngine(
            self.bundle,
            self.manifest_sha256,
            faiss_module=self.faiss,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_bank(self) -> list[dict[str, object]]:
        path = self.bundle / "bank.sqlite3"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE examples (
                rowid INTEGER PRIMARY KEY,
                example_id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                label INTEGER NOT NULL CHECK (label IN (0, 1)),
                input_channel TEXT NOT NULL,
                source TEXT NOT NULL,
                group_id TEXT NOT NULL,
                subtype TEXT NOT NULL,
                license TEXT NOT NULL
            );
            """
        )
        rows = []
        for channel in ("direct_user", "untrusted_content"):
            for label in (0, 1):
                for position in range(3):
                    example_id = f"{channel}-{label}-{position}"
                    label_term = "negativeonly" if label == 0 else "positiveonly"
                    text = (
                        f"alpha instruction sample {channel} {label} {position} "
                        f"{label_term}"
                    )
                    row = {
                        "example_id": example_id,
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "label": label,
                        "input_channel": channel,
                        "source": f"source-{channel}-{label}-{position}",
                        "group_id": f"group-{channel}-{label}-{position}",
                        "subtype": "benign" if label == 0 else "direct_jailbreak",
                        "license": "Apache-2.0",
                    }
                    cursor = connection.execute(
                        """
                        INSERT INTO examples(
                            example_id, text, text_sha256, label, input_channel,
                            source, group_id, subtype, license
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        tuple(row.values()),
                    )
                    rows.append({**row, "rowid": int(cursor.lastrowid)})
        connection.commit()
        connection.close()
        return rows

    def _write_sparse(self) -> None:
        connection = sqlite3.connect(self.bundle / "sparse.sqlite3")
        for channel in ("direct_user", "untrusted_content"):
            for label in (0, 1):
                table = f"fts_{channel}_{label}"
                connection.execute(
                    f"""
                    CREATE VIRTUAL TABLE {table} USING fts5(
                        text,
                        content='',
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
                connection.executemany(
                    f"INSERT INTO {table}(rowid, text) VALUES (?, ?)",
                    (
                        (row["rowid"], row["text"])
                        for row in self.rows
                        if row["input_channel"] == channel and row["label"] == label
                    ),
                )
        connection.commit()
        connection.close()

    def _write_dense_and_manifest(self) -> tuple[_FakeFaiss, str]:
        indexes = {}
        partitions = {}
        files = []
        base_vectors = np.zeros((3, 256), dtype=np.float32)
        base_vectors[0, 0] = 1.0
        base_vectors[1, :2] = (0.9, np.sqrt(0.19))
        base_vectors[2, :2] = (0.8, 0.6)
        for channel in ("direct_user", "untrusted_content"):
            for label in (0, 1):
                name = f"{channel}-{label}"
                index_path = self.bundle / f"hnsw-{name}.faiss"
                rowids_path = self.bundle / f"hnsw-{name}-rowids.npy"
                index_path.write_bytes(f"fake-index:{name}".encode())
                partition_rows = [
                    int(row["rowid"])
                    for row in self.rows
                    if row["input_channel"] == channel and row["label"] == label
                ]
                with rowids_path.open("wb") as handle:
                    np.save(
                        handle,
                        np.asarray(partition_rows, dtype=np.uint32),
                        allow_pickle=False,
                    )
                indexes[index_path.name] = _FakeIndex(base_vectors)
                partitions[name] = {
                    "input_channel": channel,
                    "label": label,
                    "rows": len(partition_rows),
                    "index_path": index_path.name,
                    "index_sha256": _sha256(index_path),
                    "index_bytes": index_path.stat().st_size,
                    "rowids_path": rowids_path.name,
                    "rowids_sha256": _sha256(rowids_path),
                    "rowids_bytes": rowids_path.stat().st_size,
                    "rowids_dtype": "uint32",
                }
                files.extend(
                    (
                        {
                            "path": index_path.name,
                            "sha256": _sha256(index_path),
                            "bytes": index_path.stat().st_size,
                            "role": "index",
                        },
                        {
                            "path": rowids_path.name,
                            "sha256": _sha256(rowids_path),
                            "bytes": rowids_path.stat().st_size,
                            "role": "row_map",
                        },
                    )
                )
        bank_path = self.bundle / "bank.sqlite3"
        sparse_path = self.bundle / "sparse.sqlite3"
        bank_sha256 = _sha256(bank_path)
        files.extend(
            (
                {
                    "path": bank_path.name,
                    "sha256": bank_sha256,
                    "bytes": bank_path.stat().st_size,
                    "role": "bank",
                },
                {
                    "path": sparse_path.name,
                    "sha256": _sha256(sparse_path),
                    "bytes": sparse_path.stat().st_size,
                    "role": "sparse",
                },
            )
        )
        manifest = {
            "schema_version": 1,
            "variant": "lineage_hybrid_v1",
            "parameters": {
                "m": 32,
                "ef_construction": 200,
                "ef_search": 1024,
                "overretrieve": 160,
                "exact_rescore": 20,
                "exact_rescore_dtype": "float32",
            },
            "dense": {
                "model": "perplexity/pplx-embed-v1-4b",
                "dimension": 256,
                "input_type": None,
                "metric": "inner_product",
            },
            "bank": {
                "path": bank_path.name,
                "rows": len(self.rows),
                "mode": "full_lineage",
                "sha256": bank_sha256,
                "bytes": bank_path.stat().st_size,
            },
            "sparse": {
                "path": sparse_path.name,
                "sha256": _sha256(sparse_path),
                "bytes": sparse_path.stat().st_size,
                "bank_sha256": bank_sha256,
                "tokenizer": "unicode61 remove_diacritics 2",
                "contentless": True,
                "maximum_terms": 8,
                "candidates_per_label": 50,
                "timeout_ms": 250.0,
            },
            "source": {
                "data_manifest_sha256": "a" * 64,
                "routing_view_sha256": "b" * 64,
            },
            "provider_egress": {
                "provider_safe": True,
                "license_policy": "public_allowlist_v2",
                "sensitive_text_screen": (
                    "morgott.sources.tasks._sensitive_text_reasons"
                ),
                "max_example_bytes": 1024,
            },
            "partitions": partitions,
            "files": sorted(files, key=lambda value: value["path"]),
        }
        path = self.bundle / "manifest.json"
        path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return _FakeFaiss(indexes), _sha256(path)

    def test_hybrid_retrieval_is_balanced_and_deterministic(self) -> None:
        self.assertTrue(self.engine.available)
        text = "alpha instruction request"
        sparse = self.engine.sparse(text, "direct_user")
        result = self.engine.retrieve(
            text,
            "direct_user",
            np.r_[1.0, np.zeros(255, dtype=np.float32)],
            sparse,
        )

        self.assertEqual(sparse.status, "ok")
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual([example.label for example in result.examples], [0, 1, 0, 1])
        self.assertEqual(
            {example.input_channel for example in result.examples}, {"direct_user"}
        )
        self.assertEqual(len({example.source for example in result.examples}), 4)
        self.assertEqual(len({example.lineage for example in result.examples}), 4)
        self.assertGreaterEqual(result.dense_ms, 0.0)
        self.assertGreaterEqual(result.sparse_ms, 0.0)
        self.assertGreaterEqual(result.fusion_ms, 0.0)
        self.assertNotIn("text", vars(result))
        self.assertNotIn("embedding", vars(result))
        self.assertTrue(
            all(index.hnsw.efSearch == 1024 for index in self.faiss.indexes.values())
        )
        self.assertEqual(self.faiss.omp_get_max_threads(), 1)

    def test_dense_search_pins_faiss_on_the_worker_thread(self) -> None:
        def retrieve() -> retrieval.RetrievalResult:
            return self.engine.retrieve(
                "alpha instruction request",
                "direct_user",
                np.r_[1.0, np.zeros(255, dtype=np.float32)],
                None,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(retrieve).result()

        search_threads = {
            thread_id
            for index in self.faiss.indexes.values()
            for thread_id in index.search_thread_ids
        }
        self.assertEqual(result.status, "dense_fallback")
        self.assertEqual(len(search_threads), 1)
        self.assertTrue(search_threads.issubset(set(self.faiss.pin_thread_ids)))

    def test_manifest_bank_rows_must_be_positive_and_match_partitions(self) -> None:
        manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )

        for invalid_rows in (0, True, len(self.rows) + 1):
            with self.subTest(rows=invalid_rows):
                changed = {
                    **manifest,
                    "bank": {**manifest["bank"], "rows": invalid_rows},
                }
                with self.assertRaises(ValueError):
                    retrieval._validate_manifest(changed)

    def test_manifest_requires_the_provider_egress_contract(self) -> None:
        manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )
        manifest["provider_egress"]["provider_safe"] = False

        with self.assertRaises(ValueError):
            retrieval._validate_manifest(manifest)

    def test_bank_sensitive_text_disables_retrieval(self) -> None:
        text = "api key: sk-this-is-a-provider-token"
        connection = sqlite3.connect(self.bundle / "bank.sqlite3")
        connection.execute(
            "UPDATE examples SET text = ?, text_sha256 = ? WHERE rowid = 1",
            (text, hashlib.sha256(text.encode()).hexdigest()),
        )
        connection.commit()
        connection.close()
        manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )

        with self.assertRaises(ValueError):
            retrieval._validate_bank(
                self.bundle / "bank.sqlite3",
                manifest["partitions"],
                len(self.rows),
            )

    def test_bank_nonpublic_license_disables_retrieval(self) -> None:
        connection = sqlite3.connect(self.bundle / "bank.sqlite3")
        connection.execute("UPDATE examples SET license = 'unknown' WHERE rowid = 1")
        connection.commit()
        connection.close()
        manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )

        with self.assertRaises(ValueError):
            retrieval._validate_bank(
                self.bundle / "bank.sqlite3",
                manifest["partitions"],
                len(self.rows),
            )

    def test_bank_mixed_license_disables_retrieval(self) -> None:
        connection = sqlite3.connect(self.bundle / "bank.sqlite3")
        connection.execute(
            "UPDATE examples SET license = ? WHERE rowid = 1",
            ("MIT attacks; mixed benchmark context licenses",),
        )
        connection.commit()
        connection.close()
        manifest = json.loads(
            (self.bundle / "manifest.json").read_text(encoding="utf-8")
        )

        with self.assertRaises(ValueError):
            retrieval._validate_bank(
                self.bundle / "bank.sqlite3",
                manifest["partitions"],
                len(self.rows),
            )

    def test_rrf_retains_the_reviewed_50_candidate_fusion_depth(self) -> None:
        dense = tuple(
            tuple(f"dense-{label}-{index}" for index in range(20)) for label in (0, 1)
        )
        sparse = tuple(
            tuple(f"sparse-{label}-{index}" for index in range(50)) for label in (0, 1)
        )

        fused = retrieval._rrf(dense, sparse)

        self.assertEqual(tuple(map(len, fused)), (50, 50))

    def test_sparse_failure_preserves_dense_packet(self) -> None:
        text = "a bb"
        embedding = np.r_[1.0, np.zeros(255, dtype=np.float32)]
        unavailable = self.engine.sparse(text, "direct_user")

        result = self.engine.retrieve(text, "direct_user", embedding, unavailable)
        reference = self.engine.retrieve(text, "direct_user", embedding, None)

        self.assertEqual(unavailable.status, "fallback")
        self.assertEqual(result.status, "dense_fallback")
        self.assertEqual(result.examples, reference.examples)
        self.assertEqual(len(result.examples), 4)

    def test_partial_sparse_ranking_still_fuses_with_dense(self) -> None:
        text = "negativeonly"
        sparse = self.engine.sparse(text, "direct_user")

        result = self.engine.retrieve(
            text,
            "direct_user",
            np.r_[1.0, np.zeros(255, dtype=np.float32)],
            sparse,
        )

        self.assertEqual(sparse.status, "ok")
        self.assertTrue(sparse.candidate_ids[0])
        self.assertEqual(sparse.candidate_ids[1], ())
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.examples), 4)

    def test_sparse_sqlite_error_preserves_dense_packet(self) -> None:
        text = "alpha instruction"
        embedding = np.r_[1.0, np.zeros(255, dtype=np.float32)]
        with patch.object(
            retrieval,
            "_read_only_sqlite",
            side_effect=sqlite3.OperationalError("synthetic failure"),
        ):
            unavailable = self.engine.sparse(text, "direct_user")

        result = self.engine.retrieve(text, "direct_user", embedding, unavailable)
        reference = self.engine.retrieve(text, "direct_user", embedding, None)

        self.assertEqual(unavailable.status, "fallback")
        self.assertEqual(unavailable.fallback_reason, "sparse_unavailable")
        self.assertEqual(result.examples, reference.examples)

    def test_valid_invalid_sparse_result_preserves_dense_packet(self) -> None:
        text = "alpha instruction"
        embedding = np.r_[1.0, np.zeros(255, dtype=np.float32)]
        invalid = retrieval.SparseResult(
            status="invalid",
            candidate_ids=((), ()),
            elapsed_ms=1.0,
            bundle_sha256=self.engine.manifest_sha256,
            query_sha256=hashlib.sha256(text.encode()).hexdigest(),
            input_channel="direct_user",
            fallback_reason="sparse_invalid",
        )

        result = self.engine.retrieve(text, "direct_user", embedding, invalid)
        reference = self.engine.retrieve(text, "direct_user", embedding, None)

        self.assertEqual(result.status, "dense_fallback")
        self.assertEqual(result.fallback_reason, "sparse_invalid")
        self.assertEqual(result.examples, reference.examples)

    def test_hybrid_failure_preserves_dense_packet(self) -> None:
        text = "alpha instruction"
        embedding = np.r_[1.0, np.zeros(255, dtype=np.float32)]
        sparse = self.engine.sparse(text, "direct_user")
        reference = self.engine.retrieve(text, "direct_user", embedding, None)

        with patch.object(retrieval, "_rrf", side_effect=ValueError("synthetic")):
            result = self.engine.retrieve(text, "direct_user", embedding, sparse)

        self.assertEqual(result.status, "dense_fallback")
        self.assertEqual(result.fallback_reason, "invalid_balanced_packet")
        self.assertEqual(result.examples, reference.examples)

    def test_mismatched_sparse_result_falls_back_to_dense(self) -> None:
        sparse = self.engine.sparse("alpha instruction", "direct_user")
        embedding = np.r_[1.0, np.zeros(255, dtype=np.float32)]
        result = self.engine.retrieve(
            "different text",
            "direct_user",
            embedding,
            sparse,
        )
        reference = self.engine.retrieve(
            "different text", "direct_user", embedding, None
        )

        self.assertEqual(result.status, "dense_fallback")
        self.assertEqual(result.fallback_reason, "invalid_sparse_result")
        self.assertEqual(result.examples, reference.examples)

    def test_invalid_embedding_returns_no_examples(self) -> None:
        result = self.engine.retrieve(
            "alpha instruction",
            "direct_user",
            np.zeros(256, dtype=np.float32),
            self.engine.sparse("alpha instruction", "direct_user"),
        )

        self.assertEqual(result.status, "no_examples")
        self.assertEqual(result.fallback_reason, "invalid_embedding")
        self.assertEqual(result.examples, ())

    def test_dense_balancing_failure_returns_no_examples(self) -> None:
        index = self.faiss.indexes["hnsw-direct_user-0.faiss"]
        index.vectors = index.vectors[:1]
        index.ntotal = 1
        result = self.engine.retrieve(
            "alpha instruction",
            "direct_user",
            np.r_[1.0, np.zeros(255, dtype=np.float32)],
            None,
        )

        self.assertEqual(result.status, "no_examples")
        self.assertEqual(result.fallback_reason, "dense_failure")
        self.assertEqual(result.examples, ())

    def test_payload_hash_mismatch_disables_retrieval(self) -> None:
        (self.bundle / "hnsw-direct_user-0.faiss").write_bytes(b"changed")
        engine = retrieval.RetrievalEngine(
            self.bundle,
            self.manifest_sha256,
            faiss_module=self.faiss,
        )

        self.assertFalse(engine.available)
        result = engine.retrieve(
            "alpha instruction",
            "direct_user",
            np.r_[1.0, np.zeros(255, dtype=np.float32)],
            None,
        )
        self.assertEqual(result.status, "no_examples")
        self.assertEqual(result.fallback_reason, "bundle_unavailable")

    def test_retrieval_trace_rejects_invalid_telemetry(self) -> None:
        trace = retrieval.RetrievalTrace(
            status="ok",
            total_ms=12.0,
            embedding_ms=8.0,
            dense_ms=2.0,
            sparse_ms=4.0,
            fusion_ms=0.1,
            embedding_input_tokens=42,
            embedding_cost_usd=0.000001,
            selected_example_count=4,
            selected_packet_sha256="a" * 64,
        )
        self.assertEqual(trace.selected_example_count, 4)
        with self.assertRaises(ValueError):
            retrieval.RetrievalTrace(
                status="ok",
                total_ms=-1.0,
                embedding_ms=0.0,
                dense_ms=0.0,
                sparse_ms=0.0,
                fusion_ms=0.0,
                embedding_input_tokens=None,
                embedding_cost_usd=None,
                selected_example_count=0,
                selected_packet_sha256=None,
            )
        with self.assertRaises(ValueError):
            retrieval.RetrievalTrace(
                status="ok",
                total_ms=1.0,
                embedding_ms=1.0,
                dense_ms=0.0,
                sparse_ms=0.0,
                fusion_ms=0.0,
                embedding_input_tokens=1.5,  # type: ignore[arg-type]
                embedding_cost_usd=None,
                selected_example_count=4,
                selected_packet_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
