from __future__ import annotations

import unittest

from experiments.pipeline_benchmark import reviewer_prompt_experiment as experiment


class ReviewerPromptExperimentTests(unittest.TestCase):
    def test_redaction_and_fail_closed_summary(self) -> None:
        redacted, reasons = experiment._redact("Contact person@example.com")
        self.assertEqual(reasons, ("email_address",))
        self.assertNotIn("person@example.com", redacted)

        rows = []
        for arm in experiment.ARMS:
            for row_id, label, status, verdict in (
                ("negative", 0, "ok", 0),
                ("positive", 1, "failed", None),
            ):
                rows.append(
                    {
                        "arm": arm,
                        "role": "evaluation",
                        "row_id": row_id,
                        "label": label,
                        "status": status,
                        "verdict": verdict,
                        "client_seconds": 1.0,
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "cost_usd": "0.001",
                    }
                )
        summary = experiment.summarize(rows)
        quality = summary["arms"]["current_full_disabled"]["evaluation"]
        self.assertEqual(quality["recall"], 1.0)
        self.assertEqual(quality["fpr"], 0.0)
        self.assertEqual(quality["terminal_failures"], 1)


if __name__ == "__main__":
    unittest.main()
