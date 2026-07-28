from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR / "_archived"))

import eval_phase3  # noqa: E402
from common import fpr_and_recall  # noqa: E402
from encoder_infer import direct_head_probability, route_probability  # noqa: E402
from reproduce_check import direct_route_rows  # noqa: E402


class ExperimentMetricTests(unittest.TestCase):
    def test_phase3_default_discovery_fails_on_an_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "archived_crashed").mkdir()
            with (
                patch.object(eval_phase3, "RUNS_DIR", Path(temporary)),
                self.assertRaisesRegex(FileNotFoundError, "missing run summary"),
            ):
                eval_phase3.discover()

    def test_fpr_budget_is_not_exceeded_at_one_row_boundary(self):
        labels = np.asarray([0] * 1_000 + [1, 1])
        scores = np.asarray([*np.linspace(0.0, 0.999, 1_000), 0.9985, 1.0])

        fpr, recall, threshold = fpr_and_recall(scores, labels, target=0.001)

        self.assertEqual(fpr, 0.001)
        self.assertEqual(recall, 1.0)
        self.assertEqual(threshold, 0.9985)

    def test_zero_fpr_can_select_a_positive_score_above_all_negatives(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])

        fpr, recall, threshold = fpr_and_recall(scores, labels, target=0.0)

        self.assertEqual(fpr, 0.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(threshold, 0.8)

    def test_tied_negative_scores_do_not_break_the_fpr_budget(self):
        labels = np.asarray([0, 0, 0, 1])
        scores = np.asarray([0.9, 0.9, 0.1, 0.95])

        fpr, recall, threshold = fpr_and_recall(scores, labels, target=1 / 3)

        self.assertEqual(fpr, 0.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(threshold, 0.95)

    def test_reproduction_uses_route_labels_instead_of_head_target_availability(self):
        records = [
            {
                "security_label": "benign",
                "targets": {"direct_instruction_subversion": None},
            },
            {
                "security_label": "harmful_non_injection",
                "targets": {"direct_instruction_subversion": None},
            },
            {
                "security_label": "direct_jailbreak",
                "targets": {"direct_instruction_subversion": 1},
            },
            {
                "security_label": "direct_prompt_injection",
                "targets": {"direct_instruction_subversion": 1},
            },
            {
                "security_label": "uncertain",
                "targets": {"direct_instruction_subversion": 1},
            },
        ]

        selected, labels = direct_route_rows(records)

        self.assertEqual(selected.tolist(), [0, 1, 2, 3])
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])

    def test_route_probability_uses_the_direct_or_jailbreak_head(self):
        logits = np.asarray(
            [
                [0.0, -10.0, 2.0, -10.0],
                [3.0, -10.0, -1.0, -10.0],
            ]
        )

        direct = direct_head_probability(logits)
        route = route_probability(logits)

        np.testing.assert_allclose(direct, [0.5, 1 / (1 + np.exp(-3.0))])
        np.testing.assert_allclose(
            route,
            [1 / (1 + np.exp(-2.0)), 1 / (1 + np.exp(-3.0))],
        )


if __name__ == "__main__":
    unittest.main()
