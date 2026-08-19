from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.retrieval_assisted_reviewer import colbert, run


class _FakeColbert:
    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        *,
        fail: bool = False,
        token_counts: dict[str, int] | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.fail = fail
        self.token_counts = token_counts or {}
        self.preprocess_calls: list[dict[str, object]] = []
        self.prompts = {"query": "[Q] ", "document": "[D] "}
        self._modules = [
            type(
                "InputModule",
                (),
                {
                    "query_length": 8192,
                    "document_length": 8192,
                    "query_expansion": None,
                },
            )(),
            type(
                "MultiVectorMask",
                (),
                {
                    "skiplist_words": [],
                    "skiplist_tasks": ["document"],
                    "keep_only_token_ids": None,
                },
            )(),
        ]

    def __getitem__(self, index: int) -> object:
        return self._modules[index]

    def __iter__(self):
        return iter(self._modules)

    def preprocess(
        self, sentences: list[str], **kwargs: object
    ) -> dict[str, np.ndarray]:
        self.preprocess_calls.append({"sentences": sentences, **kwargs})
        return {
            "input_ids": np.zeros(
                (1, self.token_counts.get(sentences[0], 1)), dtype=np.int64
            )
        }

    def encode_query(self, sentence: str, **_: object) -> object:
        if self.fail:
            raise RuntimeError("model failed")
        return self.embeddings[sentence]

    def encode_document(self, sentences: list[str], **_: object) -> object:
        if self.fail:
            raise RuntimeError("model failed")
        return [self.embeddings[value] for value in sentences]

    def similarity(self, query: object, documents: object) -> np.ndarray:
        return np.asarray(
            [[colbert.maxsim_score(query, document) for document in documents]],
            dtype=np.float32,
        )


def _bank() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    run._create_bank_schema(connection)
    rows = []
    for label in (0, 1):
        for index in range(2):
            rows.append(
                {
                    "id": f"row-{label}-{index}",
                    "text": f"document-{label}-{index}",
                    "label": label,
                    "source": f"source-{label}-{index}",
                    "input_channel": "direct_user",
                    "group_id": f"group-{label}-{index}",
                    "security_tags": (
                        ["benign"] if label == 0 else ["direct_prompt_injection"]
                    ),
                }
            )
    run._insert_bank_rows(
        connection,
        rows,
        {row["source"]: "MIT" for row in rows},
    )
    return connection


def _baseline() -> dict:
    return {
        "unit_id": "unit-1",
        "method": colbert.MLATEON_RRF_METHOD,
        "status": "ok",
        "failure_code": None,
        "selected_ids": ["row-0-0", "row-1-0", "row-0-1", "row-1-1"],
        "candidate_ids": {
            "0": ["row-0-0", "row-0-1"],
            "1": ["row-1-0", "row-1-1"],
        },
        "branch_candidate_ids": {
            "dense": {"0": ["row-0-0"], "1": ["row-1-0"]},
            "sparse": {"0": ["row-0-1"], "1": ["row-1-1"]},
        },
    }


class ColbertRerankTests(unittest.TestCase):
    def test_stage0_writes_hash_only_repeatable_evidence(self) -> None:
        query = np.zeros((1, colbert.MLATEON_DIMENSION), dtype=np.float32)
        query[0, 0] = 1.0
        benign = query.copy()
        attack = np.zeros((1, colbert.MLATEON_DIMENSION), dtype=np.float32)
        attack[0, 1] = 1.0
        embeddings = {
            colbert.MLATEON_STAGE0_SAMPLES[0]: query,
            colbert.MLATEON_STAGE0_SAMPLES[1]: benign,
            colbert.MLATEON_STAGE0_SAMPLES[2]: attack,
            colbert.MLATEON_STAGE0_SAMPLES[3]: benign,
            colbert.MLATEON_STAGE0_SAMPLES[4]: attack,
        }
        model = _FakeColbert(embeddings)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / colbert.MLATEON_REVISION
            snapshot.mkdir()
            output = root / "canary"
            with (
                mock.patch.object(colbert, "_load_mlateon", return_value=model),
                mock.patch.object(
                    colbert,
                    "_snapshot_identity",
                    return_value={"model": colbert.MLATEON_MODEL, "sha256": "frozen"},
                ),
                mock.patch.object(
                    colbert,
                    "_runtime_identity",
                    return_value={"sentence-transformers": "6.0.0"},
                ),
                mock.patch.object(
                    colbert,
                    "MLATEON_STAGE0_REFERENCE_SCORES",
                    (1.0, 0.0, 1.0, 0.0),
                ),
            ):
                colbert.stage0_canary(output, snapshot=snapshot, device="cpu")
            artifact = colbert.run._read_json(output / "mlateon-stage0.json")

        self.assertTrue(artifact["passed"])
        self.assertEqual(
            artifact["repeat_score_hashes"][0], artifact["repeat_score_hashes"][1]
        )
        self.assertNotIn("raw_text", artifact)
        self.assertNotIn("vectors", artifact)

    def test_reranks_the_frozen_union_with_exact_maxsim(self) -> None:
        bank = _bank()
        try:
            first = np.zeros(colbert.MLATEON_DIMENSION, dtype=np.float32)
            first[0] = 1.0
            second = np.zeros(colbert.MLATEON_DIMENSION, dtype=np.float32)
            second[1] = 1.0
            embeddings = {
                "query": np.stack((first, second)),
                "document-0-0": first[None, :],
                "document-0-1": np.stack((first, second)),
                "document-1-0": second[None, :],
                "document-1-1": np.stack((first, second)),
            }
            baseline = {
                "unit_id": "unit-1",
                "method": colbert.MLATEON_RRF_METHOD,
                "status": "ok",
                "failure_code": None,
                "selected_ids": ["row-0-0", "row-1-0", "row-0-1", "row-1-1"],
                "candidate_ids": {
                    "0": ["row-0-0", "row-0-1"],
                    "1": ["row-1-0", "row-1-1"],
                },
                "branch_candidate_ids": {
                    "dense": {
                        "0": ["row-0-0"],
                        "1": ["row-1-0"],
                    },
                    "sparse": {
                        "0": ["row-0-1"],
                        "1": ["row-1-1"],
                    },
                },
            }
            record = colbert.rerank_hybrid_record(
                bank,
                _FakeColbert(embeddings),
                query_text="query",
                input_channel="direct_user",
                baseline=baseline,
                latency_gate_ms=1_000.0,
                document_cache={
                    f"row-{label}-{index}": embeddings[f"document-{label}-{index}"]
                    for label in (0, 1)
                    for index in range(2)
                },
            )
        finally:
            bank.close()

        self.assertFalse(record["colbert_fallback"])
        self.assertEqual(
            record["candidate_ids"],
            {
                "0": ["row-0-1", "row-0-0"],
                "1": ["row-1-1", "row-1-0"],
            },
        )
        self.assertEqual(
            record["selected_ids"],
            ["row-0-1", "row-1-1", "row-0-0", "row-1-0"],
        )

    def test_model_failure_preserves_the_rrf_decision_fields(self) -> None:
        bank = _bank()
        baseline = {
            "unit_id": "unit-1",
            "method": colbert.MLATEON_RRF_METHOD,
            "status": "ok",
            "failure_code": None,
            "selected_ids": ["row-0-0", "row-1-0", "row-0-1", "row-1-1"],
            "candidate_ids": {
                "0": ["row-0-0", "row-0-1"],
                "1": ["row-1-0", "row-1-1"],
            },
            "branch_candidate_ids": {
                "dense": {"0": ["row-0-0"], "1": ["row-1-0"]},
                "sparse": {"0": ["row-0-1"], "1": ["row-1-1"]},
            },
        }
        try:
            record = colbert.rerank_hybrid_record(
                bank,
                _FakeColbert({}, fail=True),
                query_text="query",
                input_channel="direct_user",
                baseline=baseline,
                latency_gate_ms=1_000.0,
            )
        finally:
            bank.close()

        self.assertTrue(record["colbert_fallback"])
        self.assertEqual(record["candidate_ids"], baseline["candidate_ids"])
        self.assertEqual(record["selected_ids"], baseline["selected_ids"])
        self.assertEqual(record["status"], baseline["status"])

    def test_document_cache_miss_preserves_the_rrf_decision_fields(self) -> None:
        bank = _bank()
        baseline = _baseline()
        query = np.zeros((1, colbert.MLATEON_DIMENSION), dtype=np.float32)
        query[0, 0] = 1.0
        try:
            record = colbert.rerank_hybrid_record(
                bank,
                _FakeColbert({"query": query}),
                query_text="query",
                input_channel="direct_user",
                baseline=baseline,
                latency_gate_ms=1_000.0,
                document_cache={},
            )
        finally:
            bank.close()

        self.assertTrue(record["colbert_fallback"])
        self.assertEqual(record["colbert_failure_code"], "KeyError")
        self.assertEqual(record["candidate_ids"], baseline["candidate_ids"])
        self.assertEqual(record["selected_ids"], baseline["selected_ids"])

    def test_overlength_query_falls_back_before_encoding(self) -> None:
        bank = _bank()
        baseline = _baseline()
        model = _FakeColbert({}, token_counts={"query": 8193})
        try:
            record = colbert.rerank_hybrid_record(
                bank,
                model,
                query_text="query",
                input_channel="direct_user",
                baseline=baseline,
                latency_gate_ms=1_000.0,
                document_cache={
                    example_id: np.ones(
                        (1, colbert.MLATEON_DIMENSION), dtype=np.float32
                    )
                    for example_id in baseline["selected_ids"]
                },
            )
        finally:
            bank.close()

        self.assertTrue(record["colbert_fallback"])
        self.assertEqual(record["colbert_failure_code"], "ValueError")
        self.assertEqual(record["selected_ids"], baseline["selected_ids"])
        self.assertEqual(
            model.preprocess_calls,
            [
                {
                    "sentences": ["query"],
                    "prompt": "[Q] ",
                    "task": "query",
                    "processing_kwargs": {
                        "text": {"truncation": False, "padding": False}
                    },
                }
            ],
        )

    def test_rerank_requires_the_stage0_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "canary"
            output.mkdir()
            colbert.run._atomic_json(
                output / "mlateon-stage0.json",
                {
                    "passed": True,
                    "model": {"sha256": "frozen"},
                    "device": "cpu",
                    "runtime": {"sentence-transformers": "5.0.0"},
                },
            )
            with (
                mock.patch.object(
                    colbert,
                    "_snapshot_identity",
                    return_value={"sha256": "frozen"},
                ),
                mock.patch.object(
                    colbert,
                    "_runtime_identity",
                    return_value={"sentence-transformers": "6.0.0"},
                ),
                self.assertRaisesRegex(ValueError, "matching passed"),
            ):
                colbert.rerank_study(
                    output,
                    source=root / "unused",
                    snapshot=root / "unused-snapshot",
                    device="cpu",
                )

    def test_latency_evidence_includes_fallback_cost_and_is_not_a_c4_gate(self) -> None:
        evidence = colbert._latency_evidence(
            [
                {"colbert": {"latency_ms": 1.0, "colbert_fallback": False}},
                {"colbert": {"latency_ms": 200.0, "colbert_fallback": True}},
            ]
        )

        self.assertGreater(evidence["latency_p95_ms"], 100.0)
        self.assertEqual(evidence["success_only_latency_p95_ms"], 1.0)
        self.assertEqual(
            evidence["latency_population"],
            "all attempts including post-execution fallbacks",
        )
        self.assertFalse(evidence["local_sequential_added_p95_under_target"])
        self.assertFalse(evidence["target_added_p95_passed"])


if __name__ == "__main__":
    unittest.main()
