import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from morgott.data import (
    deduplicate,
    manifest_output_hashes,
    normalize_text,
    read_verified_jsonl,
)
from morgott.detector import (
    DIRECT_OPERATING_FPR_BUDGETS,
    DIRECT_PRECISION_FLOORS,
    DIRECT_REVIEW_PRECISION_FLOOR,
    choose_threshold,
    choose_threshold_for_precision,
    scan,
    split_fit_validation,
    validation_mask,
)
from morgott.policy import (
    REFERENCE_POLICY,
    SCENARIOS,
    authorize,
    execute,
    run_policy_ablation,
)


class DataTests(unittest.TestCase):
    def test_manifest_hashes_guard_jsonl_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b'{"text":"hello"}\n'
            source = root / "sample.jsonl"
            source.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"outputs": {"sample": {"sha256": digest}}}),
                encoding="utf-8",
            )
            hashes = manifest_output_hashes(manifest)
            self.assertEqual(
                read_verified_jsonl(source, hashes["sample"]), [{"text": "hello"}]
            )
            source.write_bytes(b'{"text":"changed"}\n')
            with self.assertRaises(RuntimeError):
                read_verified_jsonl(source, hashes["sample"])

    def test_normalization_and_deduplication(self):
        self.assertEqual(
            normalize_text("  ＩＧＮＯＲＥ\n previous  "), "ignore previous"
        )
        rows = [
            {"text": "Same text", "label": 1},
            {"text": " same   TEXT ", "label": 1},
            {"text": "conflict", "label": 0},
            {"text": "CONFLICT", "label": 1},
            {"text": "blocked", "label": 0},
        ]
        kept, stats = deduplicate(rows, {normalize_text("blocked")})
        self.assertEqual([row["text"] for row in kept], ["Same text"])
        self.assertEqual(
            stats,
            {"blocked_by_train": 1, "duplicates": 1, "label_conflicts": 2},
        )

    def test_group_split_is_stable(self):
        rows = [
            {"group_id": "same", "label": 0},
            {"group_id": "same", "label": 1},
            {"group_id": "different", "label": 0},
        ]
        fit, validation = split_fit_validation(rows)
        locations = {id(row): "fit" if row in fit else "validation" for row in rows}
        self.assertEqual(locations[id(rows[0])], locations[id(rows[1])])
        self.assertEqual(
            validation_mask(rows).tolist(), [row in validation for row in rows]
        )

    def test_split_group_keeps_shared_context_together(self):
        rows = [
            {"group_id": "clean", "split_group_id": "context", "label": 0},
            {"group_id": "attack", "split_group_id": "context", "label": 1},
        ]
        fit, validation = split_fit_validation(rows)
        self.assertTrue(
            all(row in fit for row in rows) or all(row in validation for row in rows)
        )


class DetectorTests(unittest.TestCase):
    def test_threshold_is_locked_without_false_positives(self):
        threshold = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.0)
        self.assertEqual(threshold, 0.8)

    def test_declared_fpr_diagnostics_remain_available(self):
        self.assertEqual(DIRECT_OPERATING_FPR_BUDGETS, (0.001, 0.005, 0.01, 0.02, 0.05))

        labels = np.asarray([0] * 1_000 + [1] * 20)
        scores = np.concatenate(
            (np.linspace(0.0, 0.99, 1_000), np.linspace(0.4, 1.0, 20))
        )
        thresholds = [
            choose_threshold(labels, scores, budget)
            for budget in DIRECT_OPERATING_FPR_BUDGETS
        ]
        self.assertTrue(all(a >= b for a, b in zip(thresholds, thresholds[1:])))
        for budget, threshold in zip(
            DIRECT_OPERATING_FPR_BUDGETS, thresholds, strict=True
        ):
            observed_fpr = np.mean(scores[:1_000] >= threshold)
            self.assertLessEqual(observed_fpr, budget)

    def test_precision_profiles_and_highest_threshold_tie_break(self):
        self.assertEqual(DIRECT_PRECISION_FLOORS, (0.80, 0.85, 0.90, 0.95))
        self.assertEqual(DIRECT_REVIEW_PRECISION_FLOOR, 0.85)

        threshold = choose_threshold_for_precision([1, 0, 0], [0.9, 0.8, 0.7], 0.5)
        self.assertEqual(threshold, 0.9)

    def test_precision_threshold_rejects_unmet_floor(self):
        with self.assertRaisesRegex(ValueError, "no observed threshold"):
            choose_threshold_for_precision([0, 1], [0.9, 0.8], 1.0)
        with self.assertRaisesRegex(ValueError, "min_precision"):
            choose_threshold_for_precision([0, 1], [0.1, 0.9], 0.0)

    def test_elevated_sensor_signal_does_not_block(self):
        class Model:
            def __init__(self, probability):
                self.probability = probability

            def predict_proba(self, texts):
                return np.asarray(
                    [[1 - self.probability, self.probability] for _ in texts]
                )

        artifact = {
            "operating_mode": "shadow",
            "channels": {
                "untrusted_content": {
                    "target": "indirect injection",
                    "threshold": 0.8,
                    "model": Model(0.1),
                },
                "direct_user": {
                    "target": "direct injection",
                    "threshold": 0.8,
                    "model": Model(0.9),
                },
            },
        }
        with patch("morgott.detector.joblib.load", return_value=artifact):
            result = scan("ordinary-looking injected task", channel="untrusted_content")
        self.assertEqual(result["signal"], "elevated")
        self.assertEqual(result["decision"], "allow")
        self.assertTrue(result["review_recommended"])
        self.assertEqual(result["triggered_by"], ["direct_user"])


class PolicyTests(unittest.TestCase):
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

    def test_executor_commits_only_an_authorized_snapshot(self):
        action = {
            "tool": SCENARIOS[-1]["action"]["tool"],
            "arguments": dict(SCENARIOS[-1]["action"]["arguments"]),
        }
        committed = []
        allowed, _ = execute(
            REFERENCE_POLICY, action, SCENARIOS[-1]["context"], committed
        )
        action["arguments"]["body"] = "changed after authorization"
        self.assertTrue(allowed)
        self.assertEqual(len(committed), 1)
        self.assertEqual(
            committed[0]["arguments"]["body"], "The public report is ready."
        )

    def test_ablation_commits_no_unauthorized_action(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy_ablation(Path(directory))
        self.assertEqual(
            result["input_filter_only"]["unauthorized_actions_committed"], 8
        )
        self.assertEqual(
            result["reference_monitor"]["unauthorized_actions_committed"], 0
        )
        self.assertEqual(result["reference_monitor"]["benign_actions_committed"], 2)

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
