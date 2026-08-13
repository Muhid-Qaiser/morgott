from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.guard_baselines import adapters, run
from morgott.models.mmbert.score_journal import ScoreJournal, ScoreJournalSpec


class PromptGuardBaselineTests(unittest.TestCase):
    def test_current_panel_slug_keeps_historical_output_distinct(self) -> None:
        slug = "prompt-guard-2-86m-current-panel"
        spec = adapters.BASELINES[slug]

        self.assertNotIn("prompt-guard-2-86m-full-mixture", adapters.BASELINES)
        self.assertEqual(spec.repo_id, "meta-llama/Llama-Prompt-Guard-2-86M")
        self.assertEqual(
            spec.revision,
            "a8ded8e697ce7c355e395a0df51f94adb4a2fd27",
        )
        self.assertEqual(spec.adapter, "prompt_guard_2")
        self.assertEqual(spec.dtype, "float16")
        self.assertEqual(spec.max_tokens, 512)
        self.assertEqual(spec.batch_size, 32)
        self.assertEqual(spec.native_threshold, 0.5)
        self.assertEqual(spec.positive_class, "softmax class index 1")

    def test_prompt_guard_uses_pinned_anonymous_class_one(self) -> None:
        spec = adapters.BASELINES["prompt-guard-2-86m-current-panel"]
        guard = adapters.PromptGuard2Encoder(spec, batch_size=32)
        guard.model = SimpleNamespace(config=SimpleNamespace(num_labels=2))
        guard.tokenizer = SimpleNamespace(model_max_length=512)

        self.assertEqual(guard._positive_index(), 1)

        guard.model.config.num_labels = 3
        with self.assertRaisesRegex(adapters.ExtractionUnavailable, "binary head"):
            guard._positive_index()


class GuardPanelCacheTests(unittest.TestCase):
    @staticmethod
    def _panel() -> dict:
        return {
            "panel_sha256": "1" * 64,
            "row_identity_sha256": {},
            "slices": {},
            "redteam": None,
            "population": {},
        }

    def test_panel_cli_uses_shared_verified_prep_cache_by_default(self) -> None:
        with (
            patch.object(sys, "argv", ["run.py", "--panel-only"]),
            patch.object(run, "build_panel", return_value=self._panel()) as build,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run.main(), 0)

        self.assertEqual(
            build.call_args.kwargs["prep_cache"],
            Path("artifacts/mmbert/prep-cache"),
        )

    def test_panel_cli_can_explicitly_disable_prep_cache(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["run.py", "--panel-only", "--no-prep-cache"],
            ),
            patch.object(run, "build_panel", return_value=self._panel()) as build,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run.main(), 0)

        self.assertIsNone(build.call_args.kwargs["prep_cache"])


class _FakeBaseline:
    def __init__(self, scores: dict[str, float]) -> None:
        self.spec = SimpleNamespace(slug="fake-guard", max_tokens=3)
        self._scores = scores
        self.calls: list[list[str]] = []

    def score(self, texts: list[str]) -> tuple[np.ndarray, list[bool]]:
        self.calls.append(list(texts))
        return (
            np.asarray([self._scores[text] for text in texts], dtype=np.float64),
            [text.endswith("overflow") for text in texts],
        )


class _FakeBucketBaseline:
    def __init__(
        self,
        scores: dict[str, float],
        lengths: dict[str, int],
        *,
        bucket_rows: int = 4,
    ) -> None:
        self.spec = SimpleNamespace(slug="fake-bucket-guard", max_tokens=8)
        self._scores = scores
        self._lengths = lengths
        self.bucket_rows = bucket_rows
        self.prepared_calls: list[list[str]] = []
        self.model_batch_lengths: list[list[int]] = []

    def batching(self) -> dict:
        return {
            "strategy": adapters.RENDERED_LENGTH_BATCHING,
            "bucket_rows": self.bucket_rows,
            "sort_key": "exact_rendered_token_count_then_original_row_offset",
            "restore_order_before_journal_append": True,
        }

    def prepare_for_scoring(
        self, texts: list[str]
    ) -> tuple[list[list[int]], list[bool]]:
        self.prepared_calls.append(list(texts))
        ids = []
        for text in texts:
            code = int(round(self._scores[text] * 100))
            ids.append([code, *([0] * (self._lengths[text] - 1))])
        return ids, [text.endswith("overflow") for text in texts]

    def score_prepared(self, prompt_ids: list[list[int]]) -> np.ndarray:
        self.model_batch_lengths.append([len(ids) for ids in prompt_ids])
        return np.asarray([ids[0] / 100 for ids in prompt_ids], dtype=np.float64)


def _rows() -> list[dict]:
    return [
        {
            "id": f"private-id-{index}",
            "text": (
                f"private-prompt-{index}-overflow"
                if index in {1, 5}
                else f"private-prompt-{index}"
            ),
            "label": index % 2,
            "source": "synthetic",
            "input_channel": "direct_user",
            "pair_id": f"pair-{index // 2}",
            "security_tags": ("direct_prompt_injection",) if index % 2 else (),
        }
        for index in range(7)
    ]


class GuardPanelIdentityTests(unittest.TestCase):
    def test_panel_identity_binds_scoring_inputs_but_canonicalizes_tags(self):
        rows = _rows()
        original = run._panel_sha256({"canonical_dev_test": rows})

        changes = (
            {**rows[0], "text": rows[0]["text"] + " changed"},
            {**rows[0], "label": 1 - rows[0]["label"]},
            {**rows[0], "source": "different-source"},
            {**rows[0], "input_channel": "indirect_document"},
            {**rows[0], "pair_id": "different-pair"},
            {**rows[0], "security_tags": ("indirect_prompt_injection",)},
        )
        for changed in changes:
            with self.subTest(changed=changed):
                candidate = [changed, *rows[1:]]
                self.assertNotEqual(
                    original,
                    run._panel_sha256({"canonical_dev_test": candidate}),
                )

        reordered_tags = [dict(rows[0]), *rows[1:]]
        reversed_tags = [dict(rows[0]), *rows[1:]]
        reordered_tags[0]["security_tags"] = ("harmful_intent", "direct_jailbreak")
        reversed_tags[0]["security_tags"] = tuple(
            reversed(reordered_tags[0]["security_tags"])
        )
        self.assertEqual(
            run._panel_sha256({"canonical_dev_test": reordered_tags}),
            run._panel_sha256({"canonical_dev_test": reversed_tags}),
        )


class GuardEvaluationInputIdentityTests(unittest.TestCase):
    def test_missing_reserve_keeps_the_azure_pull_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, run.REDTEAM_PULL):
                run.build_panel(
                    data_dir=root / "data",
                    external_dir=root / "external",
                    pairs=root / "pairs.jsonl.gz",
                    redteam=root / "redteam.jsonl.gz",
                    allow_drift=False,
                )

    def test_changed_input_fails_the_publication_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            external_dir = root / "external"
            data_dir.mkdir()
            external_dir.mkdir()
            paths = (
                data_dir / "manifest.json",
                external_dir / "manifest.json",
                root / "pairs.jsonl.gz",
                root / "redteam.jsonl.gz",
            )
            for path in paths:
                path.write_bytes(b"original")
            inputs = {
                "data_dir": data_dir,
                "external_dir": external_dir,
                "pairs": paths[2],
                "redteam": paths[3],
            }
            expected = run._evaluation_input_sha256(**inputs)

            for path in paths:
                with self.subTest(path=path.name):
                    path.write_bytes(b"changed")
                    with self.assertRaisesRegex(
                        ValueError,
                        "guard-baseline inputs changed during evaluation",
                    ):
                        run._require_unchanged_evaluation_inputs(expected, **inputs)
                    path.write_bytes(b"original")


class GuardScoreJournalTests(unittest.TestCase):
    def test_resume_scores_only_missing_batches_and_persists_no_text(self) -> None:
        rows = _rows()
        scores = {row["text"]: index / 10 for index, row in enumerate(rows)}
        baseline = _FakeBaseline(scores)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            journal = run._open_score_journal(
                root,
                panel_sha256="1" * 64,
                label="canonical_dev_test",
                rows=rows,
                batch_size=2,
                model_sha256="2" * 64,
                scoring_sha256="3" * 64,
            )
            journal.append(
                np.asarray(
                    [
                        [scores[row["text"]], float(row["text"].endswith("overflow"))]
                        for row in rows[:4]
                    ],
                    dtype=np.float64,
                )
            )

            scored = run.score_rows(
                baseline,
                rows,
                batch_size=2,
                label="canonical_dev_test",
                journal=journal,
            )

            np.testing.assert_allclose(
                scored["scores"],
                np.asarray([index / 10 for index in range(7)]),
            )
            np.testing.assert_array_equal(
                scored["overflow"],
                np.asarray([False, True, False, False, False, True, False]),
            )
            self.assertEqual(
                baseline.calls,
                [
                    [rows[4]["text"], rows[5]["text"]],
                    [rows[6]["text"]],
                ],
            )
            self.assertEqual(scored["runtime"]["resumed_rows"], 4)
            self.assertEqual(scored["runtime"]["scored_rows"], 3)
            self.assertEqual(
                scored["runtime"]["batching"]["strategy"],
                adapters.PANEL_ORDER_BATCHING,
            )

            persisted = b"".join(
                path.read_bytes() for path in root.rglob("*") if path.is_file()
            )
            for row in rows:
                self.assertNotIn(row["id"].encode(), persisted)
                self.assertNotIn(row["text"].encode(), persisted)

    def test_length_buckets_restore_score_and_overflow_order(self) -> None:
        rows = _rows()
        scores = {row["text"]: index / 100 for index, row in enumerate(rows)}
        lengths = {
            row["text"]: length
            for row, length in zip(rows, (8, 1, 6, 2, 5, 3, 4), strict=True)
        }
        baseline = _FakeBucketBaseline(scores, lengths)

        scored = run.score_rows(
            baseline,
            rows,
            batch_size=2,
            label="canonical_dev_test",
        )

        np.testing.assert_allclose(
            scored["scores"],
            np.asarray([index / 100 for index in range(len(rows))]),
        )
        np.testing.assert_array_equal(
            scored["overflow"],
            np.asarray([False, True, False, False, False, True, False]),
        )
        self.assertEqual(
            baseline.prepared_calls,
            [
                [row["text"] for row in rows[:4]],
                [row["text"] for row in rows[4:]],
            ],
        )
        self.assertEqual(
            baseline.model_batch_lengths,
            [[1, 2], [6, 8], [3, 4], [5]],
        )
        batching = scored["runtime"]["batching"]
        self.assertEqual(batching["prepared_rows_current_invocation"], 7)
        self.assertEqual(batching["rendered_tokens_current_invocation"], 29)
        self.assertEqual(batching["padded_tokens_current_invocation"], 33)
        self.assertEqual(
            batching["panel_order_padded_tokens_current_invocation"],
            42,
        )
        self.assertAlmostEqual(
            batching["padded_token_reduction_fraction_vs_panel_order"],
            9 / 42,
        )

    def test_length_bucket_resume_keeps_original_order_and_journal_private(
        self,
    ) -> None:
        rows = _rows()
        scores = {row["text"]: index / 100 for index, row in enumerate(rows)}
        lengths = {
            row["text"]: length
            for row, length in zip(rows, (8, 1, 6, 2, 5, 3, 4), strict=True)
        }
        baseline = _FakeBucketBaseline(scores, lengths)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            journal = run._open_score_journal(
                root,
                panel_sha256="1" * 64,
                label="canonical_dev_test",
                rows=rows,
                batch_size=2,
                model_sha256="2" * 64,
                scoring_sha256="3" * 64,
            )
            journal.append(
                np.asarray(
                    [
                        [scores[row["text"]], float(row["text"].endswith("overflow"))]
                        for row in rows[:4]
                    ],
                    dtype=np.float64,
                )
            )

            scored = run.score_rows(
                baseline,
                rows,
                batch_size=2,
                label="canonical_dev_test",
                journal=journal,
            )

            np.testing.assert_allclose(
                scored["scores"],
                np.asarray([index / 100 for index in range(len(rows))]),
            )
            np.testing.assert_array_equal(
                scored["overflow"],
                np.asarray([False, True, False, False, False, True, False]),
            )
            self.assertEqual(
                baseline.prepared_calls,
                [[row["text"] for row in rows[4:]]],
            )
            self.assertEqual(scored["runtime"]["resumed_rows"], 4)
            self.assertEqual(
                scored["runtime"]["batching"]["prepared_rows_current_invocation"],
                3,
            )

            manifest = (root / "canonical_dev_test" / "manifest.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("input_ids", manifest)
            self.assertNotIn("prompt_ids", manifest)
            for shard in (root / "canonical_dev_test" / "shards").glob("*.npz"):
                with np.load(shard, allow_pickle=False) as payload:
                    self.assertEqual(
                        set(payload.files),
                        {"identity_sha256", "scores", "start", "stop"},
                    )
            persisted = b"".join(
                path.read_bytes() for path in root.rglob("*") if path.is_file()
            )
            for row in rows:
                self.assertNotIn(row["id"].encode(), persisted)
                self.assertNotIn(row["text"].encode(), persisted)

    def test_batching_strategy_changes_model_identity_and_rejects_resume(self) -> None:
        spec = adapters.BASELINES["kanana-safeguard-prompt-2.1b"]
        bucketed = adapters.KananaSafeguardGuard(spec, batch_size=spec.batch_size)
        panel_order = adapters.KananaSafeguardGuard(
            replace(
                spec,
                batching_strategy=adapters.PANEL_ORDER_BATCHING,
                length_bucket_rows=None,
                attention_backend=None,
            ),
            batch_size=spec.batch_size,
        )
        old_model = run._journal_model_sha256(panel_order)
        new_model = run._journal_model_sha256(bucketed)
        self.assertNotEqual(old_model, new_model)

        rows = _rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            run._open_score_journal(
                root,
                panel_sha256="1" * 64,
                label="canonical_dev_test",
                rows=rows,
                batch_size=2,
                model_sha256=old_model,
                scoring_sha256="3" * 64,
            )
            with self.assertRaisesRegex(ValueError, "identity or schema mismatch"):
                run._open_score_journal(
                    root,
                    panel_sha256="1" * 64,
                    label="canonical_dev_test",
                    rows=rows,
                    batch_size=2,
                    model_sha256=new_model,
                    scoring_sha256="3" * 64,
                )

    def test_length_bucket_resume_rejects_a_partial_bucket(self) -> None:
        rows = _rows()
        scores = {row["text"]: index / 100 for index, row in enumerate(rows)}
        lengths = {row["text"]: 2 for row in rows}
        baseline = _FakeBucketBaseline(scores, lengths)
        with tempfile.TemporaryDirectory() as temporary:
            journal = run._open_score_journal(
                Path(temporary) / "journal",
                panel_sha256="1" * 64,
                label="canonical_dev_test",
                rows=rows,
                batch_size=2,
                model_sha256="2" * 64,
                scoring_sha256="3" * 64,
            )
            journal.append(np.asarray([[0.0, 0.0], [0.01, 1.0]]))

            with self.assertRaisesRegex(ValueError, "complete 4-row"):
                run.score_rows(
                    baseline,
                    rows,
                    batch_size=2,
                    label="canonical_dev_test",
                    journal=journal,
                )

    def test_metadata_only_change_invalidates_slice_identity(self) -> None:
        rows = _rows()
        original = run._journal_panel_sha256("1" * 64, "sep", rows)
        changed = [dict(row) for row in rows]
        changed[0]["security_tags"] = ("indirect_prompt_injection",)

        self.assertNotEqual(
            original,
            run._journal_panel_sha256("1" * 64, "sep", changed),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            ScoreJournal(
                root,
                ScoreJournalSpec(
                    model_sha256="2" * 64,
                    panel_sha256=original,
                    scoring_sha256="3" * 64,
                    rows=len(rows),
                    batch_size=2,
                    columns=run.JOURNAL_COLUMNS,
                ),
            )
            with self.assertRaisesRegex(ValueError, "identity or schema mismatch"):
                ScoreJournal(
                    root,
                    ScoreJournalSpec(
                        model_sha256="2" * 64,
                        panel_sha256=run._journal_panel_sha256(
                            "1" * 64, "sep", changed
                        ),
                        scoring_sha256="3" * 64,
                        rows=len(rows),
                        batch_size=2,
                        columns=run.JOURNAL_COLUMNS,
                    ),
                )

            for changed_identity in (
                {
                    "model_sha256": "4" * 64,
                    "scoring_sha256": "3" * 64,
                    "batch_size": 2,
                },
                {
                    "model_sha256": "2" * 64,
                    "scoring_sha256": "4" * 64,
                    "batch_size": 2,
                },
                {
                    "model_sha256": "2" * 64,
                    "scoring_sha256": "3" * 64,
                    "batch_size": 1,
                },
            ):
                with self.assertRaisesRegex(ValueError, "identity or schema mismatch"):
                    ScoreJournal(
                        root,
                        ScoreJournalSpec(
                            model_sha256=changed_identity["model_sha256"],
                            panel_sha256=original,
                            scoring_sha256=changed_identity["scoring_sha256"],
                            rows=len(rows),
                            batch_size=changed_identity["batch_size"],
                            columns=run.JOURNAL_COLUMNS,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
