import json
import unittest
from unittest.mock import patch

from morgott.sources.finance import (
    _financebench_rows,
    _harper_has_lexical_content,
    _tatqa_sample,
    _tatqa_table_text,
)


class FinanceSourceTests(unittest.TestCase):
    def test_harper_omits_only_marker_only_segments(self):
        self.assertFalse(_harper_has_lexical_content("[noise] <unk> [cough]"))
        self.assertTrue(_harper_has_lexical_content("[noise] check my balance"))

    def test_tatqa_preserves_context_group_and_channel(self):
        question = _tatqa_sample(
            text="What was the 2023 revenue?",
            split="train",
            source_id="question-1",
            context_id="context-1",
            category="financial_question",
            input_channel="direct_user",
            metadata={"source_question_uid": "question-1"},
        )
        paragraph = _tatqa_sample(
            text="Revenue increased in 2023.",
            split="test",
            source_id="paragraph-1",
            context_id="context-1",
            category="financial_report_paragraph",
            input_channel="untrusted_content",
            metadata={"source_paragraph_uid": "paragraph-1"},
        )
        self.assertEqual(question["split_group_id"], paragraph["split_group_id"])
        self.assertEqual(question["source_role"], "candidate")
        self.assertEqual(paragraph["source_role"], "dev_test")
        self.assertEqual(question["input_channel"], "direct_user")
        self.assertEqual(paragraph["input_channel"], "untrusted_content")
        self.assertIn("not_human_safety_annotation", question["label_basis"])
        self.assertEqual(
            _tatqa_table_text([["Metric", "2023"], ["Revenue", "$5"]]),
            "Metric\t2023\nRevenue\t$5",
        )

    def test_financebench_keeps_only_questions_and_evidence_as_dev_test(self):
        source_rows = []
        for index in range(150):
            source_rows.append(
                {
                    "financebench_id": f"id-{index}",
                    "company": "Example Co",
                    "doc_name": f"document-{index % 3}",
                    "question_type": "domain-relevant",
                    "question": f"What is metric {index}?",
                    "answer": "must not persist",
                    "justification": "must not persist",
                    "dataset_subset_label": "OPEN_SOURCE",
                    "evidence": [
                        {
                            "evidence_text": f"Evidence passage {index}.",
                            "doc_name": f"document-{index % 3}",
                            "evidence_page_num": index,
                            "evidence_text_full_page": "must not persist",
                        }
                    ],
                }
            )
        data = b"".join(json.dumps(row).encode() + b"\n" for row in source_rows)
        with patch(
            "morgott.sources.finance._github_raw",
            return_value=(
                data,
                "a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
            ),
        ):
            rows, _, profile, _ = _financebench_rows()
            rows = list(rows)
        self.assertEqual(len(rows), 300)
        self.assertTrue(all(row["source_role"] == "dev_test" for row in rows))
        self.assertEqual(
            {row["input_channel"] for row in rows},
            {"direct_user", "untrusted_content"},
        )
        self.assertNotIn("must not persist", str(rows))
        self.assertEqual(profile["documents"], 3)


if __name__ == "__main__":
    unittest.main()
