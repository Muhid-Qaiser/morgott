import unittest
from unittest.mock import patch

import numpy as np

from morgott.models.detector import (
    DIRECT_OPERATING_FPR_BUDGETS,
    DIRECT_PRECISION_FLOORS,
    DIRECT_REVIEW_PRECISION_FLOOR,
    choose_threshold,
    choose_threshold_for_precision,
    scan,
)


class DetectorTests(unittest.TestCase):
    def test_threshold_is_locked_without_false_positives(self):
        threshold = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.0)
        self.assertEqual(threshold, 0.8)

    def test_threshold_falls_back_above_the_maximum_score(self):
        threshold = choose_threshold(
            [0, 0, 1, 1],
            [0.9, 0.8, 0.2, 0.1],
            0.0,
        )

        self.assertEqual(threshold, np.nextafter(0.9, np.inf))

    def test_threshold_prefers_the_lower_equal_recall_candidate(self):
        threshold = choose_threshold([0, 0, 1], [0.9, 0.7, 1.0], 0.5)

        self.assertEqual(threshold, 0.9)

    def test_fast_threshold_matches_the_original_search(self):
        def original(labels, scores, max_fpr):
            labels = np.asarray(labels)
            scores = np.asarray(scores)
            best_threshold = float(np.nextafter(scores.max(), np.inf))
            best_recall = -1.0
            for threshold in np.unique(scores)[::-1]:
                predictions = scores >= threshold
                fpr = predictions[labels == 0].mean()
                recall = predictions[labels == 1].mean()
                if fpr <= max_fpr and (
                    recall > best_recall
                    or (recall == best_recall and threshold < best_threshold)
                ):
                    best_threshold = float(threshold)
                    best_recall = float(recall)
            return best_threshold

        rng = np.random.default_rng(42)
        special = np.asarray(
            [-np.inf, -0.0, 0.0, 0.5, 1.0, np.inf, np.nan],
            dtype=np.float64,
        )
        for case in range(500):
            size = int(rng.integers(2, 14))
            labels = rng.integers(0, 3, size=size)
            labels[0:2] = [0, 1]
            scores = np.where(
                rng.random(size) < 0.7,
                rng.choice(special, size=size),
                rng.normal(size=size),
            )
            budget = float(rng.choice([-0.1, 0.0, 0.1, 0.5, 1.0, 1.1]))
            for label_values, score_values in (
                (labels, scores),
                (labels.tolist(), scores.tolist()),
            ):
                expected = original(label_values, score_values, budget)
                actual = choose_threshold(label_values, score_values, budget)
                with self.subTest(case=case, budget=budget):
                    if np.isnan(expected):
                        self.assertTrue(np.isnan(actual))
                    else:
                        self.assertEqual(
                            np.float64(actual).view(np.uint64),
                            np.float64(expected).view(np.uint64),
                        )

    def test_declared_fpr_diagnostics_remain_available(self):
        self.assertEqual(DIRECT_OPERATING_FPR_BUDGETS, (0.001, 0.005, 0.01, 0.02, 0.05))

        labels = np.asarray([0] * 1_000 + [1] * 20)
        scores = np.concatenate(
            (np.linspace(0.0, 0.99, 1_000), np.linspace(0.4, 1.0, 20))
        )
        thresholds = [
            choose_threshold(labels, scores, budget)
            for budget in DIRECT_OPERATING_FPR_BUDGETS
        ]
        self.assertTrue(all(a >= b for a, b in zip(thresholds, thresholds[1:])))
        for budget, threshold in zip(
            DIRECT_OPERATING_FPR_BUDGETS, thresholds, strict=True
        ):
            observed_fpr = np.mean(scores[:1_000] >= threshold)
            self.assertLessEqual(observed_fpr, budget)

    def test_precision_profiles_and_highest_threshold_tie_break(self):
        self.assertEqual(DIRECT_PRECISION_FLOORS, (0.80, 0.85, 0.90, 0.95))
        self.assertEqual(DIRECT_REVIEW_PRECISION_FLOOR, 0.85)

        threshold = choose_threshold_for_precision([1, 0, 0], [0.9, 0.8, 0.7], 0.5)
        self.assertEqual(threshold, 0.9)

    def test_precision_threshold_rejects_unmet_floor(self):
        with self.assertRaisesRegex(ValueError, "no observed threshold"):
            choose_threshold_for_precision([0, 1], [0.9, 0.8], 1.0)
        with self.assertRaisesRegex(ValueError, "min_precision"):
            choose_threshold_for_precision([0, 1], [0.1, 0.9], 0.0)

    def test_elevated_sensor_signal_does_not_block(self):
        class Model:
            def __init__(self, probability):
                self.probability = probability

            def predict_proba(self, texts):
                return np.asarray(
                    [[1 - self.probability, self.probability] for _ in texts]
                )

        artifact = {
            "operating_mode": "shadow",
            "channels": {
                "untrusted_content": {
                    "target": "indirect injection",
                    "threshold": 0.8,
                    "model": Model(0.1),
                },
                "direct_user": {
                    "target": "direct injection",
                    "threshold": 0.8,
                    "model": Model(0.9),
                },
            },
        }
        with patch("morgott.models.detector.joblib.load", return_value=artifact):
            result = scan("ordinary-looking injected task", channel="untrusted_content")
        self.assertEqual(result["signal"], "elevated")
        self.assertEqual(result["decision"], "allow")
        self.assertTrue(result["review_recommended"])
        self.assertEqual(result["triggered_by"], ["direct_user"])


if __name__ == "__main__":
    unittest.main()
