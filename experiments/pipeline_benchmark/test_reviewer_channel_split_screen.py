import unittest
from pathlib import Path

from experiments.pipeline_benchmark.reviewer_channel_split_screen import _sample

_ARTIFACTS = (
    Path(__file__).resolve().parents[2] / "artifacts" / "pipeline_benchmark" / "20260816"
)
_MANIFEST = _ARTIFACTS / "manifest.json"


class RepresentativeScreenTest(unittest.TestCase):
    # In a checkout without LFS content the manifest is a pointer file.
    @unittest.skipUnless(
        _MANIFEST.is_file() and _MANIFEST.read_bytes()[:1] == b"{",
        "requires the retained benchmark artifacts",
    )
    def test_sample(self) -> None:
        rows = _sample(_ARTIFACTS)
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
