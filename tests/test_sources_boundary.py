import unittest

from corpus_test_support import _boundary_row

from morgott.sources.boundary import (
    _boundary_pair_sample,
    _validate_boundary_rows,
)


class BoundarySourceTests(unittest.TestCase):
    def test_boundary_pairs_map_only_instruction_families_to_injection(self):
        direct = _boundary_pair_sample(
            _boundary_row("direct", "pair-direct", 1), "train"
        )
        indirect = _boundary_pair_sample(
            _boundary_row(
                "indirect",
                "pair-indirect",
                1,
                family="rag_context_poisoning",
                source_context="retrieved_document",
            ),
            "train",
        )
        authorization = _boundary_pair_sample(
            _boundary_row(
                "authorization",
                "pair-authorization",
                1,
                family="approval_workflow_bypass",
                source_context="agent_tool_request",
            ),
            "train",
        )
        self.assertEqual(direct["security_label"], "direct_prompt_injection")
        self.assertEqual(indirect["security_label"], "indirect_prompt_injection")
        self.assertEqual(indirect["input_channel"], "untrusted_content")
        self.assertIsNone(authorization["injection_label"])
        self.assertEqual(authorization["security_label"], "uncertain")
        self.assertTrue(
            all(
                not row["routing_training_eligible"]
                for row in (direct, indirect, authorization)
            )
        )

    def test_boundary_validation_enforces_pair_and_scenario_isolation(self):
        train_pair = [
            _boundary_row("train-0", "pair-train", 0),
            _boundary_row("train-1", "pair-train", 1),
        ]
        profile = _validate_boundary_rows({"train": train_pair})
        self.assertEqual(profile["pairs"], 1)
        broken_pair = [{**row} for row in train_pair]
        broken_pair[1]["pair_id"] = "different-pair"
        with self.assertRaisesRegex(ValueError, "aligned binary pair"):
            _validate_boundary_rows({"train": broken_pair})

        validation_pair = [
            _boundary_row(
                "validation-0",
                "pair-validation",
                0,
                scenario_id="scenario-1",
                split="validation",
            ),
            _boundary_row(
                "validation-1",
                "pair-validation",
                1,
                scenario_id="scenario-1",
                split="validation",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "cross official splits"):
            _validate_boundary_rows(
                {"train": train_pair, "validation": validation_pair}
            )


if __name__ == "__main__":
    unittest.main()
