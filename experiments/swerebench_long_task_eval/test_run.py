from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run


class SweRebenchLongTaskEvalTests(unittest.TestCase):
    def test_panel_omits_text_and_preserves_lineage(self) -> None:
        source = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "a" * 40,
            "created_at": "2026-01-01",
            "problem_statement": "A" * 4_096,
        }

        panel = run._panel_row(source)

        self.assertEqual(set(panel), run.PANEL_FIELDS)
        self.assertEqual(panel["split_group_id"], "swerebench:repo:owner/repo")
        self.assertEqual(panel["length_bucket"], "4096_to_8191_chars")
        self.assertNotIn("text", panel)

    def test_assessment_validation_is_strict_and_bounded(self) -> None:
        assessment = {
            "token_count": 512,
            "window_count": 2,
            "low_windows": 1,
            "middle_windows": 1,
            "high_windows": 0,
            "max_mmbert_score": 0.5,
            "latency_ms": 10.0,
        }

        self.assertTrue(run._valid_assessment(assessment))
        self.assertFalse(run._valid_assessment(assessment | {"extra": 1}))
        self.assertFalse(
            run._valid_assessment(
                {
                    key: value
                    for key, value in assessment.items()
                    if key != "token_count"
                }
            )
        )
        self.assertFalse(run._valid_assessment(assessment | {"max_mmbert_score": 1.5}))


if __name__ == "__main__":
    unittest.main()
