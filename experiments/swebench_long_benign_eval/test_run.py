from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run


class SwebenchLongBenignEvalTests(unittest.TestCase):
    def test_panel_excludes_text_and_preserves_repository_lineage(self) -> None:
        source = {
            "id": "swebench_verified:example__project-1",
            "text": "A" * 2_048,
            "normalized_text_sha256": "a" * 64,
            "source_instance_id": "example__project-1",
            "source_repository": "example/project",
            "split_group_id": "swebench_verified:repo:example/project",
            "source_role": "dev_test",
            "routing_label": 0,
            "input_channel": "direct_user",
        }

        panel = run._panel_row(source)

        self.assertEqual(set(panel), run.PANEL_FIELDS)
        self.assertEqual(panel["repository"], "example/project")
        self.assertEqual(panel["length_bucket"], "2048_to_4095_chars")
        self.assertNotIn("text", panel)

    def test_decision_stops_on_irreparable_false_positives_or_call_budget(self) -> None:
        self.assertEqual(
            run._decision(492, hard_restricted=4, middle_windows=4_000),
            "eligible_for_bounded_remote_phase",
        )
        self.assertEqual(
            run._decision(492, hard_restricted=5, middle_windows=1),
            "reject_registered_local_gate_long_benign",
        )
        self.assertEqual(
            run._decision(492, hard_restricted=0, middle_windows=4_001),
            "stop_before_remote_window_budget",
        )


if __name__ == "__main__":
    unittest.main()
