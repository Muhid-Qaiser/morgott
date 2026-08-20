import tempfile
import unittest
from pathlib import Path

from morgott.policy import (
    REFERENCE_POLICY,
    SCENARIOS,
    authorize,
    run_policy_ablation,
)
from morgott.runtime import SourcedValue, enforce


class PolicyTests(unittest.TestCase):
    def test_runtime_lineage_union_is_monotone(self):
        task = SourcedValue.source(
            "Summarize the selected record.",
            source="task.request",
            provenance="user_request",
        )
        record = SourcedValue.source(
            "private record",
            source="record:message-17.text",
            provenance="untrusted_tool_output",
            sensitive=True,
        )

        result = SourcedValue.derived("summary", task, record)

        self.assertEqual(
            result.sources,
            frozenset({"task.request", "record:message-17.text"}),
        )
        self.assertEqual(
            result.provenance,
            frozenset({"user_request", "untrusted_tool_output", "planner_output"}),
        )
        self.assertTrue(result.sensitive)

    def test_runtime_enforces_source_binding_before_effect(self):
        policy = {
            "capabilities": {
                "send_channel_message": {
                    "constrained_arguments": {"channel_id": ["channel-42"]},
                    "free_arguments": ["body"],
                    "allow_sensitive_data": False,
                    "allowed_argument_sources": {"body": ["record:message-17.text"]},
                }
            }
        }
        task = SourcedValue.source(
            "Send the summary to channel 42.",
            source="task.request",
            provenance="user_request",
        )
        allowed_record = SourcedValue.source(
            "allowed",
            source="record:message-17.text",
            provenance="untrusted_tool_output",
        )
        other_record = SourcedValue.source(
            "other",
            source="record:message-99.text",
            provenance="untrusted_tool_output",
        )
        destination = SourcedValue.source(
            "channel-42",
            source="task.destination",
            provenance="user_request",
        )
        committed = []

        allowed = enforce(
            policy,
            "send_channel_message",
            {
                "channel_id": destination,
                "body": SourcedValue.derived("A concise summary.", allowed_record),
            },
            influenced_by=(SourcedValue.derived("tool call", task, allowed_record),),
            effect=lambda tool, arguments: committed.append((tool, arguments)),
        )
        rejected = enforce(
            policy,
            "send_channel_message",
            {
                "channel_id": destination,
                "body": SourcedValue.derived(
                    "laundered summary", allowed_record, other_record
                ),
            },
            influenced_by=(
                SourcedValue.derived("tool call", task, allowed_record, other_record),
            ),
            effect=lambda tool, arguments: committed.append((tool, arguments)),
        )

        self.assertEqual(allowed, (True, "allowed"))
        self.assertEqual(
            rejected,
            (False, "argument_source_not_granted:body"),
        )
        self.assertEqual(
            committed,
            [
                (
                    "send_channel_message",
                    {"channel_id": "channel-42", "body": "A concise summary."},
                )
            ],
        )

    def test_attack_actions_are_denied_and_benign_actions_allowed(self):
        for scenario in SCENARIOS:
            allowed, _ = authorize(
                REFERENCE_POLICY, scenario["action"], scenario["context"]
            )
            self.assertEqual(allowed, scenario["kind"] == "benign", scenario["name"])

    def test_unknown_action_fields_fail_closed(self):
        action = dict(SCENARIOS[-1]["action"], detector_score=0.0)
        self.assertEqual(
            authorize(REFERENCE_POLICY, action, SCENARIOS[-1]["context"])[1],
            "invalid_action_schema",
        )

    def test_missing_provenance_fails_closed(self):
        context = {"contains_sensitive_data": False, "provenance": []}
        self.assertEqual(
            authorize(REFERENCE_POLICY, SCENARIOS[-1]["action"], context)[1],
            "invalid_context_schema",
        )
        context["provenance"] = ["   "]
        self.assertEqual(
            authorize(REFERENCE_POLICY, SCENARIOS[-1]["action"], context)[1],
            "invalid_context_schema",
        )

    def test_malformed_inputs_fail_closed(self):
        self.assertEqual(
            authorize(REFERENCE_POLICY, None, {})[1], "invalid_action_schema"
        )
        self.assertEqual(
            authorize(REFERENCE_POLICY, SCENARIOS[-1]["action"], None)[1],
            "invalid_context_schema",
        )
        self.assertEqual(
            authorize({}, SCENARIOS[-1]["action"], SCENARIOS[-1]["context"])[1],
            "invalid_policy",
        )

    def test_malformed_capability_values_cannot_grant_authority(self):
        action = {"tool": "side_effect", "arguments": {"value": "allowed"}}
        capability = {
            "constrained_arguments": {"value": ["allowed"]},
            "free_arguments": [],
            "allow_sensitive_data": False,
            "requires_trusted_origin": True,
        }
        cases = [
            (
                capability | {"requires_trusted_origin": 0},
                {
                    "contains_sensitive_data": False,
                    "provenance": ["untrusted_document"],
                },
            ),
            (
                capability | {"allow_sensitive_data": "false"},
                {
                    "contains_sensitive_data": True,
                    "provenance": ["user_request"],
                },
            ),
            (
                capability | {"constrained_arguments": {"value": "allowed"}},
                {
                    "contains_sensitive_data": False,
                    "provenance": ["user_request"],
                },
            ),
        ]
        for malformed, context in cases:
            with self.subTest(capability=malformed):
                self.assertEqual(
                    authorize(
                        {"capabilities": {action["tool"]: malformed}},
                        action,
                        context,
                    ),
                    (False, "invalid_policy"),
                )

    def test_provenance_is_load_bearing_not_decorative(self):
        """Flipping ONLY provenance must flip the decision.

        Guards against the earlier state of this policy, where `provenance` was
        validated for shape and then never read, and attacks were really being
        stopped by a two-entry tool allowlist.
        """
        action = {
            "tool": "transfer_funds",
            "arguments": {"to": "treasury-ops", "amount": "100"},
        }
        untrusted = {
            "contains_sensitive_data": False,
            "provenance": ["untrusted_document"],
        }
        trusted = {"contains_sensitive_data": False, "provenance": ["user_request"]}

        allowed_untrusted, reason = authorize(REFERENCE_POLICY, action, untrusted)
        self.assertFalse(allowed_untrusted)
        self.assertEqual(reason, "untrusted_origin_for_capability")

        allowed_trusted, _ = authorize(REFERENCE_POLICY, action, trusted)
        self.assertTrue(allowed_trusted)

    def test_trusted_origin_cannot_expand_task_scoped_grants(self):
        trusted = {"contains_sensitive_data": False, "provenance": ["user_request"]}
        cases = [
            (
                {
                    "tool": "write_memory",
                    "arguments": {"namespace": "global", "value": "outside grant"},
                },
                "argument_not_granted:value",
            ),
            (
                {
                    "tool": "transfer_funds",
                    "arguments": {"to": "attacker-wallet", "amount": "100"},
                },
                "argument_not_granted:to",
            ),
            (
                {
                    "tool": "update_case_notes",
                    "arguments": {"case_id": "OTHER", "notes": "CANARY-SYNTHETIC"},
                },
                "argument_not_granted:case_id",
            ),
            (
                {
                    "tool": "add_warehouse_inventory",
                    "arguments": {
                        "warehouse_id": "SYNTHETIC-WAREHOUSE-001",
                        "sku": "SYNTHETIC-SKU-001",
                        "quantity": "999",
                    },
                },
                "argument_not_granted:quantity",
            ),
            (
                {
                    "tool": "update_order_status",
                    "arguments": {
                        "order_id": "SYNTHETIC-ORDER-001",
                        "status": "fulfilled",
                    },
                },
                "argument_not_granted:status",
            ),
        ]

        for action, reason in cases:
            with self.subTest(tool=action["tool"]):
                self.assertEqual(
                    authorize(REFERENCE_POLICY, action, trusted),
                    (False, reason),
                )

    def test_mixed_provenance_takes_the_lowest_trust(self):
        action = {
            "tool": "write_memory",
            "arguments": {"namespace": "global", "value": "remember this"},
        }
        mixed = {
            "contains_sensitive_data": False,
            "provenance": ["user_request", "untrusted_rag"],
        }
        self.assertEqual(
            authorize(REFERENCE_POLICY, action, mixed)[1],
            "untrusted_origin_for_capability",
        )

    def test_extra_policy_keys_fail_closed(self):
        allowed, reason = authorize(
            {"capabilities": {}, "defaults": {}},
            {"tool": "save_summary", "arguments": {}},
            {"contains_sensitive_data": False, "provenance": ["user_request"]},
        )
        self.assertEqual((allowed, reason), (False, "invalid_policy"))

    def test_unknown_provenance_channel_fails_closed(self):
        action = {
            "tool": "save_summary",
            "arguments": {
                "path": "/workspace/summaries/report.txt",
                "summary": "ok",
            },
        }
        context = {"contains_sensitive_data": False, "provenance": ["some_new_channel"]}
        self.assertEqual(
            authorize(REFERENCE_POLICY, action, context)[1], "unknown_provenance"
        )

    def test_untrusted_origin_cannot_trigger_sensitive_egress(self):
        action = {
            "tool": "send_email",
            "arguments": {
                "to": "analyst@morgott.test",
                "subject": "Requested report",
                "body": "secret",
            },
        }
        context = {
            "contains_sensitive_data": True,
            "provenance": ["untrusted_email"],
        }
        self.assertEqual(
            authorize(REFERENCE_POLICY, action, context)[1],
            "sensitive_data_from_untrusted_origin",
        )

    def test_granted_tools_are_not_denied_merely_by_absence(self):
        """Every tool the attack scenarios use is actually granted.

        If a future edit drops one, its attack would be denied by
        `tool_not_granted` again and the ablation would silently stop testing
        provenance at all.
        """
        used = {
            scenario["action"]["tool"]
            for scenario in SCENARIOS
            if scenario["kind"] == "attack"
        }
        granted = set(REFERENCE_POLICY["capabilities"])
        self.assertEqual(used - granted, set())

    def test_ablation_commits_no_unauthorized_action(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy_ablation(Path(directory))
        self.assertEqual(
            result["without_action_monitor"]["unauthorized_actions_committed"], 9
        )
        self.assertEqual(
            result["reference_monitor"]["unauthorized_actions_committed"], 0
        )
        self.assertEqual(result["reference_monitor"]["benign_actions_committed"], 3)

    def test_agentic_ipi_scenarios_cover_categories_without_source_content(self):
        scenarios = [
            scenario
            for scenario in SCENARIOS
            if scenario.get("source_reference", {}).get("dataset")
            == "nemotron_agentic_ipi"
        ]
        self.assertEqual(
            {scenario["source_reference"]["attack_category"] for scenario in scenarios},
            {
                "unauthorized_action",
                "data_modification",
                "denial_of_service",
                "exfiltration",
            },
        )
        for scenario in scenarios:
            self.assertEqual(
                set(scenario["source_reference"]),
                {
                    "dataset",
                    "source_id",
                    "domain",
                    "attack_category",
                    "injection_vector",
                    "target_tool",
                },
            )
            self.assertEqual(
                scenario["action"]["tool"], scenario["source_reference"]["target_tool"]
            )


if __name__ == "__main__":
    unittest.main()
