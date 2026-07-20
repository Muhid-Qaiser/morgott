import unittest

import numpy as np

from experiments.nemotron_agentic_ipi.audit import (
    _group_counts,
    _score_summary,
    _validate_policy_references,
)


class AuditTests(unittest.TestCase):
    def test_grouped_recall_preserves_every_source_dimension(self):
        rows = [
            {
                "domain": "legal",
                "attack_category": "unauthorized_action",
                "injection_vector": "case_summary",
                "target_tool": "update_case_notes",
            },
            {
                "domain": "hr",
                "attack_category": "exfiltration",
                "injection_vector": "resume_text",
                "target_tool": "send_email",
            },
        ]
        self.assertEqual(_group_counts(rows)["domain"], {"hr": 1, "legal": 1})
        result = _score_summary(rows, np.asarray([0.9, 0.1]), 0.5)
        self.assertEqual(result["elevated"], 1)
        self.assertEqual(result["by_group"]["domain"]["legal"]["recall"], 1.0)
        self.assertEqual(result["by_group"]["domain"]["hr"]["recall"], 0.0)

    def test_policy_reference_must_bind_to_processed_source_metadata(self):
        row = {
            "source_id": "340",
            "domain": "legal",
            "attack_category": "unauthorized_action",
            "injection_vector": "case_summary",
            "target_tool": "update_case_notes",
        }
        scenario = {
            "source_reference": {
                "source_id": "340",
                "domain": "legal",
                "attack_category": "unauthorized_action",
                "injection_vector": "case_summary",
                "target_tool": "update_case_notes",
            }
        }
        _validate_policy_references([row], [scenario])
        scenario["source_reference"]["target_tool"] = "send_email"
        with self.assertRaisesRegex(ValueError, "target_tool"):
            _validate_policy_references([row], [scenario])
        scenario["source_reference"]["source_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing Nemotron source ID"):
            _validate_policy_references([row], [scenario])


if __name__ == "__main__":
    unittest.main()
