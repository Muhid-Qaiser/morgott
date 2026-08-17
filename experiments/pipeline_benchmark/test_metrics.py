from __future__ import annotations

import math
import unittest

import numpy as np

from experiments.pipeline_benchmark import metrics


def _rows() -> list[dict]:
    return [
        {
            "artifact_id": "clean-direct",
            "label": 0,
            "input_channel": "direct_user",
            "dataset": "canonical",
            "language": "en",
        },
        {
            "artifact_id": "clean-untrusted",
            "label": 0,
            "input_channel": "untrusted_content",
            "dataset": "external",
            "language": "en",
        },
        {
            "artifact_id": "attack-direct",
            "label": 1,
            "input_channel": "direct_user",
            "dataset": "canonical",
            "language": "es",
        },
        {
            "artifact_id": "attack-untrusted",
            "label": 1,
            "input_channel": "untrusted_content",
            "dataset": "external",
            "language": "es",
        },
    ]


class PipelineBenchmarkMetricTests(unittest.TestCase):
    def test_artifacts_metrics_and_slices_are_not_window_weighted(self) -> None:
        windows = [
            {**_rows()[0], "window_index": 0, "local_score": 0.1},
            {**_rows()[0], "window_index": 1, "local_score": 0.4},
            {**_rows()[2], "window_index": 0, "local_score": 0.35},
            {**_rows()[2], "window_index": 1, "local_score": 0.8},
        ]
        artifacts = metrics.aggregate_artifacts(windows)

        self.assertEqual([row["local_score"] for row in artifacts], [0.4, 0.8])
        self.assertEqual([row["window_count"] for row in artifacts], [2, 2])
        summary = metrics.summarize_slices(artifacts, [False, True])
        self.assertEqual(summary["aggregate"]["rows"], 2)
        self.assertEqual(summary["by_input_channel"]["direct_user"]["tp"], 1)

        ranked = metrics.score_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.4)
        self.assertEqual((ranked["tp"], ranked["fp"]), (1, 1))
        self.assertAlmostEqual(ranked["auroc"], 0.75)
        self.assertAlmostEqual(ranked["average_precision"], 5 / 6)
        self.assertEqual(len(ranked["fpr_95"]), 2)

    def test_calibration_threshold_is_transported_unchanged(self) -> None:
        transferred = metrics.fixed_fpr_evaluation(
            [0, 0, 0, 1, 1],
            [0.1, 0.2, 0.9, 0.8, 0.95],
            [0, 0, 1, 1],
            [0.79, 0.81, 0.8, 0.99],
            targets=(1 / 3,),
        )[f"{1 / 3:g}"]

        self.assertEqual(transferred["threshold"], 0.8)
        self.assertEqual(transferred["calibration"]["fp"], 1)
        self.assertEqual(transferred["evaluation"]["threshold"], 0.8)
        self.assertEqual(transferred["evaluation"]["fp"], 1)
        self.assertEqual(transferred["evaluation"]["tp"], 2)

    def test_wilson_prevalence_and_bootstrap_are_deterministic(self) -> None:
        self.assertEqual(metrics.wilson_interval(0, 0), None)
        lower, upper = metrics.wilson_interval(5, 10)
        self.assertAlmostEqual(lower, 0.236593, places=5)
        self.assertAlmostEqual(upper, 0.763407, places=5)

        projected = metrics.prevalence_projections(0.8, 0.01, prevalences=(0.001,))[
            "0.001"
        ]
        self.assertAlmostEqual(projected["expected_review_rate"], 0.01079)
        self.assertAlmostEqual(projected["expected_precision"], 0.8 / 10.79)

        arguments = (
            [0, 0, 0, 1, 1, 1],
            [False, True, False, True, False, False],
            [False, False, False, True, True, False],
        )
        first = metrics.paired_stratified_bootstrap_delta(
            *arguments, iterations=100, seed=7
        )
        second = metrics.paired_stratified_bootstrap_delta(
            *arguments, iterations=100, seed=7
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["metrics"]["recall"]["delta"], 1 / 3)
        self.assertAlmostEqual(first["metrics"]["fpr"]["delta"], -1 / 3)

    def test_cascade_arms_fail_safe_and_grid_sizes_match_contract(self) -> None:
        rows = _rows()
        local_scores = [0.01, 0.15, 0.5, 0.99999]
        logprob, calls, invalid = metrics.cascade_predictions(
            rows,
            local_scores,
            [None, math.nan, 0.7, None],
            direct_low=0.05,
            untrusted_low=0.1,
            local_high=0.99999,
            arm="logprob",
            reviewer_threshold=0.9,
        )
        np.testing.assert_array_equal(logprob, [False, True, False, True])
        np.testing.assert_array_equal(calls, [False, True, True, False])
        np.testing.assert_array_equal(invalid, [False, True, False, False])

        hard, _, _ = metrics.cascade_predictions(
            rows,
            local_scores,
            [None, 0, 1, None],
            direct_low=0.05,
            untrusted_low=0.1,
            local_high=0.99999,
            arm="hard_verdict",
        )
        np.testing.assert_array_equal(hard, [False, False, True, True])

        small_grid = metrics.threshold_grid(
            rows,
            local_scores,
            [0.0, 0.0, 1.0, 0.0],
            arm="logprob",
            direct_lows=(0.05,),
            untrusted_lows=(0.1,),
            local_highs=(0.99, 0.999),
            reviewer_thresholds=(0.5, 0.9),
        )
        self.assertEqual(len(small_grid), 4)
        self.assertTrue(all(candidate["call_rate"] == 0.5 for candidate in small_grid))

    def test_profile_constraints_and_tie_breaking_are_deterministic(self) -> None:
        def candidate(
            name: str, *, fpr: float, channel_fpr: float, recall: float, worst: float
        ) -> dict:
            return {
                "configuration_id": name,
                "metrics": {"fpr": fpr, "recall": recall},
                "max_channel_fpr": channel_fpr,
                "call_rate": 0.1,
                "worst_slice_recall": worst,
            }

        candidates = [
            candidate("b", fpr=0.01, channel_fpr=0.02, recall=0.9, worst=0.8),
            candidate("a", fpr=0.01, channel_fpr=0.02, recall=0.9, worst=0.8),
            candidate("high", fpr=0.04, channel_fpr=0.07, recall=0.95, worst=0.9),
        ]
        selected = metrics.select_profiles(candidates)

        self.assertEqual(selected["conservative"]["configuration_id"], "a")
        self.assertEqual(selected["balanced"]["configuration_id"], "a")
        self.assertEqual(selected["high_recall"]["configuration_id"], "high")

    def test_duplicate_artifacts_are_rejected(self) -> None:
        duplicate = [_rows()[0], _rows()[0]]
        with self.assertRaisesRegex(ValueError, "one row per artifact"):
            metrics.summarize_slices(duplicate, [False, False])
        with self.assertRaisesRegex(ValueError, "labels must"):
            metrics.binary_metrics([0.5], [False])
        with self.assertRaisesRegex(ValueError, "predictions must"):
            metrics.binary_metrics([0], [2])


if __name__ == "__main__":
    unittest.main()
