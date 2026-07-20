import unittest

import numpy as np

from run_embeddings import OPERATING_FPR_BUDGETS, operating_points, precision_profiles


class ProfileTests(unittest.TestCase):
    def test_profiles_are_validation_selected_and_explicit_when_unattainable(self):
        rows = [
            {"label": 1, "group_id": "positive"},
            {"label": 0, "group_id": "high-negative"},
            {"label": 0, "group_id": "low-negative"},
        ]
        scores = np.asarray([0.8, 0.9, 0.1])
        points = operating_points(rows, scores, {"same": (rows, scores)})
        self.assertEqual(
            [point["validation_fpr_budget"] for point in points],
            list(OPERATING_FPR_BUDGETS),
        )
        self.assertTrue(
            all(
                point["validation"]["fpr"] <= point["validation_fpr_budget"]
                for point in points
            )
        )
        self.assertTrue(
            all(
                not profile["attained"]
                for profile in precision_profiles(rows, scores, {})
            )
        )


if __name__ == "__main__":
    unittest.main()
