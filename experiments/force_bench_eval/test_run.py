from __future__ import annotations

import unittest

from experiments.force_bench_eval import run


def _row(plugin: str, query: str) -> dict:
    row = {
        "query": query,
        "plugin": plugin,
        "timeout": 300,
        "tags": [
            {
                "tag": tag,
                "metric": True,
                "assertions": [{"text": "source rubric", "level": "critical"}],
            }
            for tag in run.REQUIRED_RUBRICS
        ],
    }
    if plugin != "business_brief":
        row["segment"] = "AR" if plugin == "erp_qa" else "Financial Health"
    if plugin == "erp_qa":
        row["scenario"] = "Credit Limit"
    return row


class ForceBenchEvalTest(unittest.TestCase):
    def test_source_validation_and_panel_exclude_query_text(self) -> None:
        rows = [
            _row("erp_qa", "What is the current credit limit?"),
            _row("finance_qa", "Compare the current ratios of two suppliers."),
            _row("business_brief", "Business brief for Contoso."),
        ]

        run._validate_source(rows, expected_rows=3)
        panel = run._panel_rows(rows)

        self.assertEqual(len(panel), 3)
        self.assertEqual({row["plugin"] for row in panel}, set(run.PLUGINS))
        self.assertTrue(all(set(row) == run.PANEL_FIELDS for row in panel))
        self.assertTrue(all(row["input_channel"] == "direct_user" for row in panel))
        self.assertTrue(all("query" not in row and "text" not in row for row in panel))

        duplicate = rows + [_row("business_brief", " BUSINESS  brief for contoso. ")]
        with self.assertRaisesRegex(ValueError, "duplicate normalized query"):
            run._validate_source(duplicate, expected_rows=4)

    def test_decision_is_fixed_at_one_percent(self) -> None:
        self.assertEqual(
            run._decision(251, restricted=2, incomplete=0, provider_failures=0),
            "retain_as_bounded_finance_benign_evidence",
        )
        self.assertEqual(
            run._decision(251, restricted=3, incomplete=0, provider_failures=0),
            "reject_fixed_cascade_finance_false_positive_gate",
        )
        self.assertEqual(
            run._decision(251, restricted=0, incomplete=1, provider_failures=0),
            "inconclusive_incomplete",
        )
        self.assertEqual(
            run._decision(251, restricted=0, incomplete=0, provider_failures=1),
            "inconclusive_incomplete",
        )

    def test_zero_observations_do_not_prove_one_percent_fpr(self) -> None:
        upper = run._wilson_upper(0, 251)
        self.assertGreater(upper, 0.01)
        self.assertLess(upper, 0.02)


if __name__ == "__main__":
    unittest.main()
