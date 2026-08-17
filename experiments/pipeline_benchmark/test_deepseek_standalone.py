from __future__ import annotations

import unittest

import numpy as np

from experiments.pipeline_benchmark import deepseek_standalone


class DeepSeekStandaloneTests(unittest.TestCase):
    def test_fixed_fpr_threshold_preserves_boundary_ties(self) -> None:
        labels = np.asarray([0, 0, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.9, 0.8], dtype=np.float64)
        invalid = np.zeros(3, dtype=bool)

        threshold = deepseek_standalone.select_threshold(labels, scores, invalid, 0.5)

        self.assertEqual(threshold, 0.8)
        predictions = invalid | (scores >= threshold)
        self.assertEqual(predictions.tolist(), [False, True, True])

    def test_invalid_negative_can_make_fixed_fpr_infeasible(self) -> None:
        labels = np.asarray([0, 0, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.0, 0.8], dtype=np.float64)
        invalid = np.asarray([False, True, False])

        threshold = deepseek_standalone.select_threshold(labels, scores, invalid, 0.0)

        self.assertIsNone(threshold)

    def test_transports_parse_separately_and_fail_closed(self) -> None:
        ids = ["valid", "invalid", "failed"]
        records = [
            {
                "row_id": "valid",
                "status": "ok",
                "verdict": 0,
                "probability": 0.2,
            },
            {
                "row_id": "invalid",
                "status": "ok",
                "verdict": 1,
                "probability": 1.1,
            },
            {
                "row_id": "failed",
                "status": "failed",
                "verdict": None,
                "probability": None,
            },
        ]

        scores, score_invalid = deepseek_standalone._logprob_values(ids, records)
        hard, hard_invalid = deepseek_standalone._hard_values(ids, records)

        self.assertEqual(scores.tolist(), [0.2, 0.0, 0.0])
        self.assertEqual(score_invalid.tolist(), [False, True, True])
        self.assertEqual(hard.tolist(), [False, True, True])
        self.assertEqual(hard_invalid.tolist(), [False, False, True])

    def test_paired_deltas_keep_contract_direction_explicit(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        cloud_hard = np.asarray([False, False, False, True])
        decart = np.asarray([False, True, True, True])

        result = deepseek_standalone._paired(
            labels,
            cloud_hard,
            decart,
            {"0.01": np.asarray([False, False, True, True])},
        )

        comparison = result["decart_true_hard_minus_cloudflare_logprob_hard"]
        self.assertEqual(comparison["direction"], "candidate_minus_incumbent")
        self.assertEqual(comparison["metrics"]["recall"]["delta"], 0.5)
        self.assertIn("cloudflare_fixed_0.01_minus_decart_true_hard", result)


if __name__ == "__main__":
    unittest.main()
