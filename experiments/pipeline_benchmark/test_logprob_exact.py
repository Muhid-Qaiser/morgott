from __future__ import annotations

import unittest

from experiments.pipeline_benchmark import logprob_exact


class ExactLogprobTests(unittest.TestCase):
    def test_replay_preserves_full_context_and_batched_windows(self) -> None:
        rows = [
            {"artifact_id": "single", "input_channel": "direct_user"},
            {"artifact_id": "untrusted", "input_channel": "untrusted_content"},
            {"artifact_id": "all-low", "input_channel": "untrusted_content"},
            {"artifact_id": "direct", "input_channel": "direct_user"},
        ]
        scores = {
            "single": {"window_scores": [0.2]},
            "untrusted": {"window_scores": [0.01, 0.2]},
            "all-low": {"window_scores": [0.01, 0.02]},
            "direct": {"window_scores": [0.2, 0.2, 0.2, 0.2, 0.2]},
        }
        artifacts = {
            "single": 0.4,
            "untrusted": 0.4,
            "all-low": 0.4,
            "direct": 1.0,
        }
        windows = {
            ("untrusted", 1): 0.8,
            ("direct", 0): 0.8,
            ("direct", 1): None,
            ("direct", 2): 0.2,
            ("direct", 3): 0.2,
        }
        selection = {
            "thresholds": {
                "direct_low": 0.1,
                "untrusted_low": 0.1,
                "local_high": 0.99,
                "reviewer": 0.5,
            }
        }

        replay = logprob_exact.exact_predictions(
            rows, scores, artifacts, windows, selection
        )

        self.assertEqual(replay["predictions"].tolist(), [False, True, False, True])
        self.assertEqual(replay["artifact_calls"].tolist(), [1, 1, 1, 0])
        self.assertEqual(replay["window_calls"].tolist(), [0, 1, 0, 4])
        self.assertEqual(replay["invalid_reviews"].tolist(), [0, 0, 0, 1])

    def test_required_windows_skip_null_and_flagged_full_context(self) -> None:
        rows = [
            {"artifact_id": "untrusted", "input_channel": "untrusted_content"},
            {"artifact_id": "direct", "input_channel": "direct_user"},
        ]
        scores = {
            "untrusted": {"window_scores": [0.01, 0.2]},
            "direct": {"window_scores": [0.01, 0.2]},
        }
        selected = {
            "thresholds": {
                "direct_low": 0.1,
                "untrusted_low": 0.1,
                "local_high": 0.99,
                "reviewer": 0.5,
            }
        }

        required = logprob_exact.required_window_keys(
            rows,
            scores,
            {"untrusted": 0.8, "direct": 0.8},
            {"conservative": None, "balanced": selected},
        )

        self.assertEqual(required, {("direct", 1)})

    def test_probability_parser_fails_closed(self) -> None:
        values, failures = logprob_exact._probability_values(
            ["valid", "invalid", "failed"],
            [
                {"row_id": "valid", "status": "ok", "probability": 0.2},
                {"row_id": "invalid", "status": "ok", "probability": 1.1},
                {"row_id": "failed", "status": "failed", "probability": None},
            ],
        )

        self.assertEqual(values, {"valid": 0.2, "invalid": None, "failed": None})
        self.assertEqual(failures, 2)


if __name__ == "__main__":
    unittest.main()
