import unittest
from pathlib import Path

from experiments.pipeline_benchmark.reviewer_channel_split_screen import _sample


class RepresentativeScreenTest(unittest.TestCase):
    def test_sample(self) -> None:
        rows = _sample(Path("artifacts/pipeline_benchmark/20260816"))
        self.assertEqual(len(rows), len({row["panel_id"] for row in rows}))
        self.assertEqual(len(rows), 256)
        self.assertEqual(sum(row["source"] == "sep" for row in rows), 128)
        self.assertTrue(
            all(
                row["input_channel"] == "untrusted_content"
                and row["text_chars"] <= 4_096
                for row in rows
            )
        )
