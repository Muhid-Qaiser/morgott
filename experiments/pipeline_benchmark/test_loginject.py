import unittest

from experiments.pipeline_benchmark.loginject import summarize


class LogInjectSummaryTests(unittest.TestCase):
    def test_reports_local_routing_without_inventing_remote_outcome(self):
        records = [
            {"artifact_id": "p:clean", "label": 0, "local_score": 0.01},
            {"artifact_id": "p:attack", "label": 1, "local_score": 0.9},
        ]
        selection = {
            "logprob": {
                "balanced": {"thresholds": {"untrusted_low": 0.05, "local_high": 0.8}}
            }
        }
        result = summarize(records, selection)
        self.assertEqual(result["profiles"]["balanced"]["attack_local_high_recall"], 1)
        self.assertEqual(result["remote_cascade_status"], "pending_provider_winner")


if __name__ == "__main__":
    unittest.main()
