import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from audit import (
    FILES,
    NearIndex,
    _active_overlap,
    _metrics,
    _record,
    _within_source,
    load_source,
)


class PromptShieldAuditTests(unittest.TestCase):
    def test_loader_requires_pinned_schema_hash_and_counts(self):
        rows = [{"prompt": "Ordinary request", "label": 0}]
        data = json.dumps(rows).encode()
        expected = {
            "name": "tiny.json",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": 1,
            "positive": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "tiny.json").write_bytes(data)
            with patch.dict(FILES, {"tiny": expected}, clear=True):
                loaded = load_source(Path(directory))
            self.assertEqual(loaded["tiny"][0]["prompt"], "Ordinary request")

            bad = json.dumps([{"prompt": "x", "label": 0, "source": "guess"}]).encode()
            Path(directory, "tiny.json").write_bytes(bad)
            expected.update(bytes=len(bad), sha256=hashlib.sha256(bad).hexdigest())
            with patch.dict(FILES, {"tiny": expected}, clear=True):
                with self.assertRaisesRegex(ValueError, "unexpected fields"):
                    load_source(Path(directory))

    def test_near_index_excludes_exact_and_uses_hamming_limit(self):
        index = NearIndex()
        base = int("123456789abcdef0123456789abcdef0", 16)
        index.add(base, "same", {"train"})
        index.add(base ^ 0b111111, "near", {"test"})
        index.add(base ^ 0b1111111, "far", {"test"})
        self.assertEqual(index.query(base, "same"), [("near", ("test",))])

    def test_cross_split_overlap_reports_conflict_without_prompt_content(self):
        train = [_record("Same prompt", 0, "train", 0)]
        validation = [_record(" same   PROMPT ", 1, "validation", 0)]
        test = [_record("Different prompt entirely", 0, "test", 0)]
        result = _within_source(
            {"train": train, "validation": validation, "test": test}
        )
        pair = result["cross_split"]["train__validation"]
        self.assertEqual(pair["raw_exact_unique_texts"], 0)
        self.assertEqual(pair["normalized_exact_unique_texts"], 1)
        self.assertEqual(pair["normalized_label_conflict_texts"], 1)
        self.assertNotIn("Same prompt", json.dumps(result))

    def test_metrics_keep_exact_confusion_counts(self):
        metrics = _metrics(
            np.asarray([1, 1, 0, 0]), np.asarray([0.9, 0.1, 0.8, 0.2]), 0.5
        )
        self.assertEqual(
            (
                metrics["true_positive"],
                metrics["false_negative"],
                metrics["false_positive"],
                metrics["true_negative"],
            ),
            (1, 1, 1, 1),
        )

    def test_active_fit_overlap_marks_rows_without_copying_text(self):
        row = _record(" same   PROMPT ", 0, "test", 0)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "train.jsonl").write_text(
                json.dumps({"text": "Same prompt", "label": 0}) + "\n"
            )
            result = _active_overlap({"test": [row]}, Path(directory))
        self.assertEqual(
            result["by_promptshield_split"]["test"]["any_fit_overlap_rows"], 1
        )
        self.assertTrue(row["active_fit_overlap"])
        self.assertNotIn("Same prompt", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
