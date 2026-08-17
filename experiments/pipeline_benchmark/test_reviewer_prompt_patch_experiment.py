from __future__ import annotations

import unittest

from experiments.pipeline_benchmark import (
    reviewer_prompt_patch_experiment as experiment,
)


class ReviewerPromptPatchExperimentTests(unittest.TestCase):
    def test_patch_is_one_narrow_boundary_rule(self) -> None:
        self.assertEqual(experiment.PATCH.count("."), 1)
        self.assertIn("implicit", experiment.PATCH)
        self.assertIn("ordinary human-facing instructions remain 0", experiment.PATCH)


if __name__ == "__main__":
    unittest.main()
