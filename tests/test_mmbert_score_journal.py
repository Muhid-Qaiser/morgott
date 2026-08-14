from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from morgott.models.mmbert import evaluate as mmbert_evaluate
from morgott.models.mmbert import score_journal as score_journal_module
from morgott.models.mmbert.score_journal import ScoreJournal, ScoreJournalSpec


def _spec(**updates) -> ScoreJournalSpec:
    values = {
        "model_sha256": "1" * 64,
        "panel_sha256": "2" * 64,
        "scoring_sha256": "3" * 64,
        "rows": 5,
        "batch_size": 2,
    }
    values.update(updates)
    return ScoreJournalSpec(**values)


class ScoreJournalTests(unittest.TestCase):
    def test_output_and_journal_paths_must_be_disjoint_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            journal = root / "journal"
            score_journal_module.require_disjoint_paths(output, journal)

            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                score_journal_module.require_disjoint_paths(output, output)
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                score_journal_module.require_disjoint_paths(
                    output,
                    output / "nested-journal",
                )
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                score_journal_module.require_disjoint_paths(
                    journal / "nested-output",
                    journal,
                )

    def test_panel_identity_binds_metric_metadata_but_canonicalizes_tags(self):
        row = {
            "id": "row:1",
            "text": "synthetic unit-test value",
            "label": 1,
            "source": "unit_test",
            "input_channel": "direct_user",
            "group_id": "group:1",
            "pair_id": "pair:1",
            "validation_component_id": "component:1",
            "security_tags": ["harmful_intent", "direct_jailbreak"],
        }
        identity = mmbert_evaluate._score_panel_sha256([row])
        reordered = {
            **row,
            "security_tags": list(reversed(row["security_tags"])),
        }
        self.assertEqual(
            mmbert_evaluate._score_panel_sha256([reordered]),
            identity,
        )
        changes = (
            {**row, "pair_id": "pair:2"},
            {**row, "group_id": "group:2"},
            {**row, "validation_component_id": "component:2"},
            {**row, "security_tags": ["direct_jailbreak"]},
        )
        for changed in changes:
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    mmbert_evaluate._score_panel_sha256([changed]),
                    identity,
                )

    def test_current_identity_binds_fixed_panel_order_and_preserves_legacy_digest(self):
        common = {
            "model_sha256": "1" * 64,
            "scoring_sha256": "2" * 64,
            "training_max_tokens": 1024,
            "evaluation_max_tokens": 512,
        }
        legacy_expected = hashlib.sha256(
            json.dumps(
                common,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            mmbert_evaluate._evaluation_identity_sha256(**common),
            legacy_expected,
        )

        ordered = tuple(
            (name, str(index) * 64)
            for index, name in enumerate(
                mmbert_evaluate.EVALUATION_PANEL_ORDER,
                start=3,
            )
        )
        current = mmbert_evaluate._evaluation_identity_sha256(
            **common,
            identity_schema_version=(
                mmbert_evaluate.EVALUATION_IDENTITY_SCHEMA_VERSION
            ),
            ordered_panel_sha256=ordered,
        )
        for index in range(len(ordered)):
            changed = list(ordered)
            changed[index] = (changed[index][0], "9" * 64)
            with self.subTest(panel=ordered[index][0]):
                self.assertNotEqual(
                    current,
                    mmbert_evaluate._evaluation_identity_sha256(
                        **common,
                        identity_schema_version=(
                            mmbert_evaluate.EVALUATION_IDENTITY_SCHEMA_VERSION
                        ),
                        ordered_panel_sha256=tuple(changed),
                    ),
                )

        swapped = list(ordered)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(ValueError, "order"):
            mmbert_evaluate._evaluation_identity_sha256(
                **common,
                identity_schema_version=(
                    mmbert_evaluate.EVALUATION_IDENTITY_SCHEMA_VERSION
                ),
                ordered_panel_sha256=tuple(swapped),
            )
        with self.assertRaisesRegex(ValueError, "schema version"):
            mmbert_evaluate._evaluation_identity_sha256(
                **common,
                identity_schema_version=3,
                ordered_panel_sha256=ordered,
            )

    def test_append_resume_and_assemble_numeric_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            first = ScoreJournal(root, _spec())
            self.assertEqual(first.missing_ranges(2), [(0, 2), (2, 4), (4, 5)])
            self.assertEqual(first.append(np.asarray([0.1, 0.2])), (0, 2))

            resumed = ScoreJournal(root, _spec())
            self.assertEqual(resumed.completed_rows, 2)
            resumed.append(np.asarray([0.3, 0.4, 0.5]), start=2)
            self.assertTrue(resumed.complete)
            np.testing.assert_allclose(
                resumed.scores()[:, 0],
                np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            )
            with sqlite3.connect(resumed.database_path) as database:
                identity = json.loads(
                    database.execute("SELECT identity_json FROM journal").fetchone()[0]
                )
                payload_types = {
                    row[0]
                    for row in database.execute(
                        "SELECT DISTINCT typeof(value) FROM scores"
                    )
                }
            self.assertEqual(identity["columns"], ["score"])
            self.assertEqual(identity["dtype"], "float64")
            self.assertEqual(payload_types, {"real"})
            self.assertNotIn("text", json.dumps(identity).lower())

    def test_stale_implicit_writer_fails_before_relabelling_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            first = ScoreJournal(root, _spec())
            stale = ScoreJournal(root, _spec())

            first.append(np.asarray([0.1, 0.2]))
            with self.assertRaisesRegex(ValueError, "next contiguous"):
                stale.append(np.asarray([0.3]))

            self.assertEqual(stale.completed_rows, 2)
            stale.append(np.asarray([0.3, 0.4, 0.5]))
            np.testing.assert_allclose(
                stale.scores()[:, 0],
                np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            )

    def test_concurrent_same_range_writers_serialize_and_one_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            first = ScoreJournal(root, _spec())
            second = ScoreJournal(root, _spec())
            ready = threading.Barrier(3)
            outcomes: dict[str, object] = {}

            def append(name, journal, values):
                ready.wait()
                try:
                    outcomes[name] = journal.append(values, start=0)
                except Exception as error:
                    outcomes[name] = error

            first_thread = threading.Thread(
                target=append,
                args=("first", first, np.asarray([0.1, 0.2])),
            )
            second_thread = threading.Thread(
                target=append,
                args=("second", second, np.asarray([0.3, 0.4])),
            )
            first_thread.start()
            second_thread.start()
            ready.wait()
            first_thread.join(2)
            second_thread.join(2)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(
                sum(outcome == (0, 2) for outcome in outcomes.values()),
                1,
            )
            self.assertEqual(
                sum(isinstance(outcome, ValueError) for outcome in outcomes.values()),
                1,
            )
            reopened = ScoreJournal(root, _spec())
            self.assertEqual(reopened.completed_rows, 2)
            reopened.append(np.asarray([0.5, 0.6, 0.7]), start=2)
            self.assertEqual(reopened.scores().shape, (5, 1))

    def test_identity_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            ScoreJournal(root, _spec())
            with self.assertRaisesRegex(ValueError, "identity or schema"):
                ScoreJournal(root, _spec(batch_size=1))

    def test_invalid_ranges_and_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = ScoreJournal(Path(temporary) / "journal", _spec())
            with self.assertRaisesRegex(ValueError, "next contiguous"):
                journal.append(np.asarray([0.1]), start=1)
            with self.assertRaisesRegex(ValueError, "score array"):
                journal.append(np.asarray([np.nan]))
            with self.assertRaisesRegex(ValueError, "score array"):
                journal.append(np.ones((1, 2)))

    def test_interrupted_append_rolls_back_the_whole_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            journal = ScoreJournal(root, _spec())
            with sqlite3.connect(journal.database_path) as database:
                database.execute(
                    """
                    CREATE TRIGGER interrupt_append
                    BEFORE INSERT ON scores
                    WHEN NEW.row_index = 1
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated interruption');
                    END
                    """
                )
            with self.assertRaises(sqlite3.IntegrityError):
                journal.append(np.asarray([0.1, 0.2]))

            resumed = ScoreJournal(root, _spec())
            self.assertEqual(resumed.completed_rows, 0)
            with sqlite3.connect(journal.database_path) as database:
                self.assertEqual(
                    database.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 0
                )
                database.execute("DROP TRIGGER interrupt_append")
            resumed.append(np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]))
            self.assertTrue(resumed.complete)

    def test_non_contiguous_database_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "journal"
            journal = ScoreJournal(root, _spec())
            journal.append(np.asarray([0.1, 0.2]))
            with sqlite3.connect(journal.database_path) as database:
                database.execute(
                    "DELETE FROM scores WHERE row_index = 0 AND column_index = 0"
                )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                ScoreJournal(root, _spec())

    def test_evaluator_resumes_at_the_next_outer_batch(self):
        rows = [
            {
                "id": f"row:{index}",
                "text": str(index),
                "label": index % 2,
                "source": "unit_test",
                "input_channel": "direct_user",
            }
            for index in range(513)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            journal = ScoreJournal(
                Path(temporary) / "journal",
                _spec(rows=len(rows), batch_size=8),
            )
            calls = 0

            def interrupted(_encoder, _tokenizer, _head, texts, *, batch_size):
                nonlocal calls
                self.assertEqual(batch_size, 8)
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated scoring interruption")
                return np.asarray([float(text) / 1000 for text in texts])

            with (
                patch.object(mmbert_evaluate, "score_texts", interrupted),
                self.assertRaisesRegex(RuntimeError, "interruption"),
            ):
                mmbert_evaluate._score(
                    rows,
                    object(),
                    object(),
                    object(),
                    batch_size=8,
                    journal=journal,
                )
            self.assertEqual(journal.completed_rows, 512)

            resumed_texts = []

            def resumed(_encoder, _tokenizer, _head, texts, *, batch_size):
                self.assertEqual(batch_size, 8)
                resumed_texts.extend(texts)
                return np.asarray([float(text) / 1000 for text in texts])

            with patch.object(mmbert_evaluate, "score_texts", resumed):
                scored = mmbert_evaluate._score(
                    rows,
                    object(),
                    object(),
                    object(),
                    batch_size=8,
                    journal=ScoreJournal(
                        Path(temporary) / "journal",
                        _spec(rows=513, batch_size=8),
                    ),
                )
            self.assertEqual(resumed_texts, ["512"])
            self.assertEqual(scored["scores"].shape, (513,))
            self.assertAlmostEqual(scored["scores"][-1], 0.512)


if __name__ == "__main__":
    unittest.main()
