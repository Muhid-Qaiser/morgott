from __future__ import annotations

import unittest

from experiments.pipeline_benchmark import reviewer_long_bucket_experiment as experiment


class ReviewerLongBucketExperimentTests(unittest.TestCase):
    def test_revised_prompt_is_channel_and_task_conditioned(self) -> None:
        direct = experiment._revised_prompt(
            {
                "input_channel": "direct_user",
                "source": "promptshield",
            }
        )
        untrusted = experiment._revised_prompt(
            {
                "input_channel": "untrusted_content",
                "source": "llmail",
            }
        )
        self.assertIn("direct user's outer request", direct)
        self.assertIn("message or email", untrusted)
        self.assertNotEqual(direct, untrusted)
        self.assertIn('{"subversion": 0}', direct)


if __name__ == "__main__":
    unittest.main()
