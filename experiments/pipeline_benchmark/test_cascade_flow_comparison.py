from __future__ import annotations

import unittest

from experiments.pipeline_benchmark import cascade_flow_comparison as comparison


class CascadeFlowComparisonTests(unittest.TestCase):
    def test_long_untrusted_inputs_skip_full_context_and_review_only_middle_windows(
        self,
    ) -> None:
        rows = [
            {"artifact_id": "all-low", "input_channel": "untrusted_content"},
            {"artifact_id": "middle", "input_channel": "untrusted_content"},
            {"artifact_id": "single", "input_channel": "direct_user"},
        ]
        scores = {
            "all-low": {"window_scores": [0.01, 0.02]},
            "middle": {"window_scores": [0.01, 0.2]},
            "single": {"window_scores": [0.2]},
        }
        selection = {
            "thresholds": {
                "direct_low": 0.1,
                "untrusted_low": 0.1,
                "local_high": 0.99,
                "reviewer": 0.5,
            }
        }

        replay = comparison.on_demand_predictions(
            rows,
            scores,
            {"all-low": 1.0, "middle": 1.0, "single": 0.2},
            {("middle", 1): 0.8},
            selection,
        )

        self.assertEqual(replay["predictions"].tolist(), [False, True, False])
        self.assertEqual(replay["artifact_calls"].tolist(), [0, 0, 1])
        self.assertEqual(replay["window_calls"].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
