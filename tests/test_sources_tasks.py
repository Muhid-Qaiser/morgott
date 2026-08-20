import tempfile
import unittest
from pathlib import Path

from corpus_test_support import _read_rows

from morgott.corpus import (
    _consume_source_quarantine,
)
from morgott.data import _set_source_role
from morgott.sources.tasks import (
    _banking77_sample,
    _coconot_sample,
    _false_reject_sample,
    _lmsys_arena_sample,
    _mind2web_sample,
    _sensitive_text_reasons,
    _swebench_verified_sample,
    _taskmaster_sample,
    _taskmaster_split_group,
)


class TaskSourceTests(unittest.TestCase):
    def test_mind2web_sensitive_text_is_detected_before_training(self):
        benign = "Find the next train from Boston to New York."
        sensitive = "Book for a@example.com using booking number X123456."
        token = "Use api key: sk-abcdefghijklmnopqrstuvwxyz123456"
        self.assertEqual(_sensitive_text_reasons(benign), [])
        self.assertEqual(
            _sensitive_text_reasons(sensitive),
            ["email_address", "transaction_identifier"],
        )
        self.assertIn("provider_token", _sensitive_text_reasons(token))
        row = _mind2web_sample(
            {
                "annotation_id": "task-1",
                "confirmed_task": benign,
                "website": "example",
                "domain": "Travel",
                "subdomain": "Train",
            }
        )
        self.assertEqual(row["input_channel"], "direct_user")
        self.assertEqual(row["split_group_id"], "mind2web:task-1")
        self.assertNotIn("actions", row)

    def test_source_privacy_quarantine_is_not_training_eligible(self):
        row = _mind2web_sample(
            {
                "annotation_id": "sensitive-task",
                "confirmed_task": "Email the receipt to a@example.com.",
                "website": "example",
                "domain": "Shopping",
                "subdomain": "Retail",
            }
        )
        row["source_sensitive_text_reasons"] = ["email_address"]
        row = _set_source_role(row, "uncertain")
        row["data_role"] = "quarantine"
        row["quarantine_reason"] = "potential_secret_or_pii"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mind2web.jsonl"
            output = _consume_source_quarantine(path, [row])
            self.assertEqual(_read_rows(path), [row])
        self.assertEqual(output["rows"], 1)
        self.assertFalse(row["routing_training_eligible"])

    def test_swebench_verified_is_a_repository_grouped_dev_test_task(self):
        row = _swebench_verified_sample(
            {
                "repo": "example/project",
                "instance_id": "example__project-123",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the documented parser regression.",
                "created_at": "2024-01-02T03:04:05Z",
                "version": "1.2",
                "environment_setup_commit": "b" * 40,
                "difficulty": "15 min - 1 hour",
                "patch": "must not persist",
                "test_patch": "must not persist",
                "hints_text": "must not persist",
            }
        )

        self.assertEqual(row["routing_label"], 0)
        self.assertEqual(row["source_role"], "dev_test")
        self.assertEqual(row["input_channel"], "direct_user")
        self.assertEqual(row["group_id"], "swebench_verified:example__project-123")
        self.assertEqual(
            row["split_group_id"],
            "swebench_verified:repo:example/project",
        )
        self.assertIn("not_safety_annotation", row["label_basis"])
        self.assertNotIn("must not persist", str(row))

    def test_taskmaster_maps_dialogue_lineage_and_speaker_channel(self):
        dialog = {
            "conversation_id": "dlg-test",
            "instruction_id": "restaurant-table-2",
            "instructions": "must not persist",
        }
        row = _taskmaster_sample(
            dialog,
            {"index": 3, "speaker": "USER", "text": "Book a table for two."},
            collection="tm1_self",
            source_file="TM-1-2019/self-dialogs.json",
            source_split="tm1_self:train",
            split_group_id=_taskmaster_split_group(dialog, "tm1_self"),
            record_index=4,
            role="candidate",
            domain="restaurant-table",
        )
        self.assertEqual(row["routing_label"], 0)
        self.assertEqual(row["input_channel"], "direct_user")
        self.assertEqual(
            row["split_group_id"],
            "taskmaster:tm1_self:instruction:restaurant-table-2",
        )
        self.assertEqual(row["source_instruction_id"], "restaurant-table-2")
        self.assertNotIn("must not persist", str(row))

    def test_banking77_preserves_finance_intent_and_official_test(self):
        row = _banking77_sample(
            {"text": "Where is my replacement card?", "category": "card_arrival"},
            "test",
            2,
        )
        self.assertEqual(row["security_label"], "benign")
        self.assertEqual(row["source_intent"], "card_arrival")
        self.assertEqual(row["source_role"], "dev_test")

    def test_false_reject_generated_train_is_a_weak_labelled_candidate(self):
        source_row = {
            "prompt": "Explain a sensitive topic in a factual way.",
            "category": 7,
            "category_text": "Sensitive Topic",
        }
        train = _false_reject_sample(source_row, "train", 0)
        test = _false_reject_sample(source_row, "test", 0)
        self.assertEqual(train["source_role"], "candidate")
        self.assertTrue(train["routing_training_eligible"])
        self.assertEqual(
            train["label_basis"], "multi_agent_generated_benign_weak_label"
        )
        self.assertEqual(test["source_role"], "dev_test")
        self.assertEqual(test["label_basis"], "human_validated_benign_overrefusal_test")

    def test_other_weak_benign_sources_are_candidates(self):
        coconot = _coconot_sample(
            {
                "id": "safe-1",
                "prompt": "Explain the difference between two legal terms.",
                "category": "informational",
                "subcategory": "sensitive_but_safe",
            },
            "train",
            0,
        )
        safety = {
            "source_openai_moderation_flagged": False,
            "source_openai_moderation_max_score": 0.01,
            "source_toxic_roberta_flagged": False,
            "source_toxic_roberta_probability": 0.01,
            "source_toxic_t5_flagged": False,
            "source_toxic_t5_score": 0.01,
        }
        lmsys = _lmsys_arena_sample(
            {"question_id": "q-1", "model_a": "chat-model"},
            {"role": "user", "content": "Could you explain this concept?"},
            safety,
            row_index=0,
            side="a",
            message_index=0,
            judge_sha256="a" * 64,
        )

        for row in (coconot, lmsys):
            self.assertEqual(row["source_role"], "candidate")
            self.assertTrue(row["routing_training_eligible"])
            self.assertIn("weak", row["label_basis"])

        assistant = _lmsys_arena_sample(
            {"question_id": "q-1", "model_a": "chat-model"},
            {"role": "assistant", "content": "Here is a concise explanation."},
            safety,
            row_index=0,
            side="a",
            message_index=1,
            judge_sha256="a" * 64,
        )
        self.assertEqual(assistant["source_role"], "candidate")
        self.assertTrue(assistant["routing_training_eligible"])
        self.assertEqual(assistant["toxicity"], "unknown")
        self.assertEqual(
            assistant["label_basis"],
            "model_output_from_unflagged_user_prompt_weak_benign",
        )

        flagged_safety = {
            **safety,
            "source_toxic_roberta_flagged": True,
        }
        flagged_user = _lmsys_arena_sample(
            {"question_id": "q-2", "model_a": "chat-model"},
            {"role": "user", "content": "A flagged user prompt."},
            flagged_safety,
            row_index=1,
            side="a",
            message_index=0,
            judge_sha256="b" * 64,
        )
        flagged_assistant = _lmsys_arena_sample(
            {"question_id": "q-2", "model_a": "chat-model"},
            {"role": "assistant", "content": "A response of unknown safety."},
            flagged_safety,
            row_index=1,
            side="a",
            message_index=1,
            judge_sha256="b" * 64,
        )
        self.assertEqual(flagged_user["source_role"], "uncertain")
        self.assertFalse(flagged_user["routing_training_eligible"])
        self.assertEqual(flagged_user["routing_label"], 1)
        self.assertIsNone(flagged_user["injection_label"])
        self.assertEqual(flagged_user["toxicity"], "unknown")
        self.assertEqual(
            flagged_user["label_basis"],
            "automated_user_prompt_safety_flags_unverified",
        )
        self.assertEqual(flagged_assistant["source_role"], "uncertain")
        self.assertFalse(flagged_assistant["routing_training_eligible"])


if __name__ == "__main__":
    unittest.main()
