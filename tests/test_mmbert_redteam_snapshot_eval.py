from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.mmbert_redteam_snapshot_eval import run
from morgott.models.mmbert.evaluate import _evaluation_model_sha256
from morgott.models.mmbert.score_journal import ScoreJournal, ScoreJournalSpec

BASE_MODEL = {
    "id": run.MODEL_ID,
    "revision": run.MODEL_REVISION,
    "config_sha256": "0" * 64,
    "pytorch_model_sha256": "a" * 64,
    "special_tokens_map_sha256": "c" * 64,
    "tokenizer_config_sha256": "d" * 64,
    "tokenizer_json_sha256": "b" * 64,
}


def _head_contract() -> dict:
    return {
        "architecture": "legacy_sequential_binary_v1",
        "outputs": 1,
        "columns": {"0": "instruction_subversion"},
        "primary_column": 0,
    }


def _result(*, run_name: str = "single-output-run") -> dict:
    contract = _head_contract()
    return {
        "schema_version": 1,
        "purpose": "maintained full-data advisory mmBERT training",
        "run_name": run_name,
        "model_id": run.MODEL_ID,
        "model_revision": run.MODEL_REVISION,
        "adaptation": "lora",
        "generic_target": "instruction_subversion",
        "head_contract": contract,
        "training_identity": {
            "run_name": run_name,
            "head_contract": contract,
            "length_grouped": False,
        },
    }


class BindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "single-output-run"
        self.run_dir.mkdir()
        self.snapshot = self.root / "update-017000.pt"
        self.snapshot.write_bytes(b"trusted-snapshot")
        self.full_dir = self.run_dir / "evaluation-update-17000"
        self.full_dir.mkdir()
        self.scores = self.full_dir / "scores.npy"
        self.scores.write_bytes(b"numeric-score-artifact")
        self.expected_inputs = {
            "data_manifest_sha256": "1" * 64,
            "external_manifest_sha256": "2" * 64,
            "pair_archive_sha256": "3" * 64,
            "additional_pair_archive_sha256": "4" * 64,
            "routing_views": {},
        }
        result = _result()
        (self.run_dir / "result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        result_sha = hashlib.sha256(
            (self.run_dir / "result.json").read_bytes()
        ).hexdigest()
        snapshot_sha = hashlib.sha256(self.snapshot.read_bytes()).hexdigest()
        threshold = 0.9
        self.report = {
            "schema_version": 1,
            "purpose": run.FULL_EVALUATION_PURPOSE,
            "advisory_only": True,
            "model_id": run.MODEL_ID,
            "model_revision": run.MODEL_REVISION,
            "adaptation": "lora",
            "head_contract": _head_contract(),
            "run_result_sha256": result_sha,
            "evaluation_model_sha256": _evaluation_model_sha256(
                result_sha, snapshot_sha
            ),
            "evaluated_checkpoint": {
                "sha256": snapshot_sha,
                "update": 17000,
                "epoch": 3,
                "role": "pre_registered_comparison",
            },
            "inputs": self.expected_inputs,
            "thresholds": {
                "source": "canonical calibration components only",
                "selected": {run.SHARED_TARGET: threshold},
            },
            "calibration": {
                "row_identity_sha256": "5" * 64,
                "component_thresholds": {
                    run.SHARED_TARGET: {
                        "status": "available",
                        "threshold": threshold,
                    }
                },
                "metrics": {"threshold": threshold, "recall": 0.8},
            },
            "canonical_dev_test": {"metrics": {"threshold": threshold, "recall": 0.7}},
            "runtime": {"batch_size": 24},
            "scores": {
                "path": "scores.npy",
                "sha256": hashlib.sha256(self.scores.read_bytes()).hexdigest(),
                "columns": ["label", *run.SCORE_COLUMNS],
                "slices": {
                    "calibration": [0, 10],
                    "dev_test": [10, 20],
                    "promptshield": [20, 30],
                    "sep": [30, 40],
                },
            },
        }
        self.full = self.full_dir / "evaluation.json"
        self.full.write_text(json.dumps(self.report) + "\n", encoding="utf-8")

    def _write_current_full_evaluation(
        self,
        *,
        training_max_tokens: int,
        evaluation_max_tokens: int,
    ) -> dict:
        microbatch_size = 24
        token_budget = microbatch_size * training_max_tokens
        result = _result()
        result.update(
            {
                "max_tokens": training_max_tokens,
                "token_budget": token_budget,
                "training": {
                    "max_tokens": training_max_tokens,
                    "microbatch_size": microbatch_size,
                    "token_budget": token_budget,
                },
            }
        )
        result["training_identity"].update(
            {
                "schema_version": 5,
                "max_tokens": training_max_tokens,
                "microbatch_size": microbatch_size,
                "token_budget": token_budget,
            }
        )
        result_path = self.run_dir / "result.json"
        result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
        snapshot_sha256 = hashlib.sha256(self.snapshot.read_bytes()).hexdigest()
        model_sha256 = _evaluation_model_sha256(result_sha256, snapshot_sha256)
        scoring_sha256 = "2" * 64
        evaluation_identity_sha256 = run.mmbert_evaluate._evaluation_identity_sha256(
            model_sha256=model_sha256,
            scoring_sha256=scoring_sha256,
            training_max_tokens=training_max_tokens,
            evaluation_max_tokens=evaluation_max_tokens,
        )
        report = copy.deepcopy(self.report)
        report.update(
            {
                "run_result_sha256": result_sha256,
                "evaluation_model_sha256": model_sha256,
                "training_max_tokens": training_max_tokens,
                "evaluation_max_tokens": evaluation_max_tokens,
                "native_context_evaluation": (
                    training_max_tokens == evaluation_max_tokens
                ),
                "evaluation_identity_sha256": evaluation_identity_sha256,
            }
        )
        report["scores"].update(
            {
                "scoring_sha256": scoring_sha256,
                "evaluation_identity_sha256": evaluation_identity_sha256,
                "training_max_tokens": training_max_tokens,
                "evaluation_max_tokens": evaluation_max_tokens,
            }
        )
        report["runtime"].update(
            {
                "training_max_tokens": training_max_tokens,
                "evaluation_max_tokens": evaluation_max_tokens,
                "native_context_evaluation": (
                    training_max_tokens == evaluation_max_tokens
                ),
            }
        )
        self.full.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return report

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_binding_requires_exact_run_snapshot_scores_inputs_and_threshold(self):
        binding = run.validate_binding(
            self.run_dir,
            self.snapshot,
            self.full,
            expected_inputs=self.expected_inputs,
        )
        self.assertEqual(binding.update, 17000)
        self.assertEqual(binding.threshold, 0.9)
        self.assertEqual(binding.batch_size, 24)

        changed = dict(self.expected_inputs)
        changed["pair_archive_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=changed,
            )

        self.scores.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=self.expected_inputs,
            )

    def test_snapshot_filename_and_checkpoint_hash_are_both_bound(self):
        renamed = self.snapshot.with_name("update-017001.pt")
        renamed.write_bytes(self.snapshot.read_bytes())
        with self.assertRaisesRegex(ValueError, "differs from the snapshot"):
            run.validate_binding(
                self.run_dir,
                renamed,
                self.full,
                expected_inputs=self.expected_inputs,
            )

    def test_current_full_evaluation_binds_exact_native_1024_context(self):
        report = self._write_current_full_evaluation(
            training_max_tokens=1024,
            evaluation_max_tokens=1024,
        )
        binding = run.validate_binding(
            self.run_dir,
            self.snapshot,
            self.full,
            expected_inputs=self.expected_inputs,
            evaluation_max_tokens=1024,
        )
        self.assertEqual(binding.training_max_tokens, 1024)
        self.assertEqual(binding.evaluation_max_tokens, 1024)
        self.assertTrue(binding.native_context_evaluation)
        self.assertEqual(
            binding.full_evaluation_context_contract,
            run.FULL_EVALUATION_CONTEXT_CONTRACT,
        )
        self.assertEqual(
            binding.full_evaluation_identity_sha256,
            report["evaluation_identity_sha256"],
        )
        self.assertEqual(
            binding.full_scoring_sha256,
            report["scores"]["scoring_sha256"],
        )

        with self.assertRaisesRegex(ValueError, "explicit --evaluation-max-tokens"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=self.expected_inputs,
            )
        with self.assertRaisesRegex(ValueError, "requires a retained snapshot"):
            run.validate_binding(
                self.run_dir,
                None,
                self.full,
                expected_inputs=self.expected_inputs,
                evaluation_max_tokens=1024,
            )
        with self.assertRaisesRegex(ValueError, "context, or scoring identity"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=self.expected_inputs,
                evaluation_max_tokens=512,
            )

    def test_schema2_full_evaluation_binds_panels_before_reserve_use(self):
        report = self._write_current_full_evaluation(
            training_max_tokens=1024,
            evaluation_max_tokens=1024,
        )
        report["base_model"] = BASE_MODEL
        report["evaluation_model_sha256"] = _evaluation_model_sha256(
            report["run_result_sha256"],
            report["evaluated_checkpoint"]["sha256"],
            base_model=BASE_MODEL,
        )
        ordered = tuple(
            (name, str(index) * 64)
            for index, name in enumerate(
                run.mmbert_evaluate.EVALUATION_PANEL_ORDER,
                start=3,
            )
        )
        for (_, report_name), (_, digest) in zip(
            run.mmbert_evaluate._EVALUATION_REPORT_PANELS,
            ordered,
            strict=True,
        ):
            report.setdefault(report_name, {})["score_panel_sha256"] = digest
        identity_document = run.mmbert_evaluate._evaluation_identity_document(
            model_sha256=report["evaluation_model_sha256"],
            scoring_sha256=report["scores"]["scoring_sha256"],
            training_max_tokens=1024,
            evaluation_max_tokens=1024,
            ordered_panel_sha256=ordered,
        )
        identity_sha256 = run.mmbert_evaluate._evaluation_identity_sha256(
            model_sha256=report["evaluation_model_sha256"],
            scoring_sha256=report["scores"]["scoring_sha256"],
            training_max_tokens=1024,
            evaluation_max_tokens=1024,
            identity_schema_version=(
                run.mmbert_evaluate.EVALUATION_IDENTITY_SCHEMA_VERSION
            ),
            ordered_panel_sha256=ordered,
        )
        report.update(
            {
                "schema_version": run.mmbert_evaluate.EVALUATION_SCHEMA_VERSION,
                "evaluation_identity": identity_document,
                "evaluation_identity_sha256": identity_sha256,
            }
        )
        report["scores"]["evaluation_identity_sha256"] = identity_sha256
        self.full.write_text(json.dumps(report) + "\n", encoding="utf-8")

        binding = run.validate_binding(
            self.run_dir,
            self.snapshot,
            self.full,
            expected_inputs=self.expected_inputs,
            evaluation_max_tokens=1024,
        )
        self.assertEqual(
            binding.full_evaluation_identity_sha256,
            identity_sha256,
        )

        report["evaluation_identity"]["ordered_score_panels"].reverse()
        self.full.write_text(json.dumps(report) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity document mismatch"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=self.expected_inputs,
                evaluation_max_tokens=1024,
            )

    def test_partial_context_metadata_and_explicit_legacy_use_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "current cap-aware"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=self.expected_inputs,
                evaluation_max_tokens=512,
            )

        report = self._write_current_full_evaluation(
            training_max_tokens=1024,
            evaluation_max_tokens=1024,
        )
        del report["runtime"]["evaluation_max_tokens"]
        self.full.write_text(json.dumps(report) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "partial context"):
            run.validate_binding(
                self.run_dir,
                self.snapshot,
                self.full,
                expected_inputs=self.expected_inputs,
                evaluation_max_tokens=1024,
            )

    def test_packaged_selected_one_head_is_strictly_bound(self):
        run_dir = self.root / "packaged-single-output-run"
        run_dir.mkdir()
        head = run_dir / "head.safetensors"
        head.write_bytes(b"one-head-package")
        adapter = run_dir / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_bytes(b"{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        adapter_files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(adapter.iterdir())
        }
        curve_row = {
            "epoch": 2,
            "updates": 17000,
            "selection_loss": 0.2,
            "selection_rule": "source_macro",
            "pre_registered_comparison": True,
            "interim": True,
        }
        selected = {
            "epoch": 2,
            "updates": 17000,
            "selection_role": "secondary",
            "selection_rule": "source_macro",
            "selection_loss": 0.2,
            "validation_point_role": "periodic_validation",
            "pre_registered_comparison": True,
        }
        result = _result(run_name=run_dir.name)
        result.update(
            {
                "training": {
                    "selected_epoch": 2,
                    "selected_updates": 17000,
                    "selected_checkpoint": selected,
                    "curve": [curve_row],
                },
                "artifact": {
                    "weights_provenance": {
                        "source": "training.selected_checkpoint",
                        "epoch": 2,
                        "updates": 17000,
                    },
                    "head": "head.safetensors",
                    "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
                    "adapter": "adapter",
                    "adapter_files": adapter_files,
                },
            }
        )
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
        full_dir = run_dir / "evaluation"
        full_dir.mkdir()
        scores = full_dir / "scores.npy"
        scores.write_bytes(b"one-column-full-scores")
        report = dict(self.report)
        report.update(
            {
                "head_contract": _head_contract(),
                "run_result_sha256": result_sha,
                "scores": {
                    **self.report["scores"],
                    "sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
                    "columns": ["label", *run.SCORE_COLUMNS],
                },
            }
        )
        report.pop("evaluated_checkpoint")
        report.pop("evaluation_model_sha256")
        full = full_dir / "evaluation.json"
        full.write_text(json.dumps(report) + "\n", encoding="utf-8")

        binding = run.validate_binding(
            run_dir,
            None,
            full,
            expected_inputs=self.expected_inputs,
        )
        self.assertEqual(binding.checkpoint_kind, "packaged_selected")
        self.assertEqual(binding.role, run.PACKAGED_CHECKPOINT_ROLE)
        self.assertEqual(binding.evaluation_model_sha256, result_sha)

        (adapter / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "adapter hash mismatch"):
            run.validate_binding(
                run_dir,
                None,
                full,
                expected_inputs=self.expected_inputs,
            )

    def test_snapshot_and_packaged_full_evaluation_forms_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "requires the matching --snapshot"):
            run.validate_binding(
                self.run_dir,
                None,
                self.full,
                expected_inputs=self.expected_inputs,
            )

    def test_two_output_run_is_not_maintained(self):
        result = _result()
        result["head_contract"] = {
            "architecture": "shared_trunk_separate_binary_projections_v1",
            "outputs": 2,
            "columns": {
                "0": "instruction_subversion",
                "1": "harmful_intent",
            },
            "primary_column": 0,
        }
        with self.assertRaisesRegex(ValueError, "single-output"):
            run._validate_run_contract(self.run_dir, result)


def _rows() -> list[dict]:
    return [
        {
            "id": "private-row-one",
            "text": "private prompt one",
            "label": 1,
            "source": "redteam_campaigns",
            "input_channel": "direct_user",
            "group_id": "private-group-one",
            "security_tags": (),
            "prompt_kind": "attack_prompt",
            "attack_mode": "classic",
            "category": "control-category",
            "subversion_basis": "marker_attested",
        },
        {
            "id": "private-row-two",
            "text": "private prompt two",
            "label": 1,
            "source": "redteam_campaigns",
            "input_channel": "direct_user",
            "group_id": "private-group-two",
            "security_tags": (),
            "prompt_kind": "bare_harmful",
            "attack_mode": "memory",
            "category": "control-category",
            "subversion_basis": "None",
        },
    ]


class ReportTests(unittest.TestCase):
    def test_report_and_journal_persist_no_prompt_or_row_identity(self):
        rows = _rows()
        panel_sha = run.guard_run._journal_panel_sha256(
            run.guard_run.REDTEAM_SHA256, "redteam_reserve", rows
        )
        with tempfile.TemporaryDirectory() as temporary:
            journal = ScoreJournal(
                Path(temporary) / "journal",
                ScoreJournalSpec(
                    model_sha256="1" * 64,
                    panel_sha256=panel_sha,
                    scoring_sha256="2" * 64,
                    rows=2,
                    batch_size=1,
                    columns=run.SCORE_COLUMNS,
                ),
            )
            scores = np.asarray([[0.95], [0.1]], dtype=np.float64)
            journal.append(scores)
            binding = run.EvaluationBinding(
                result={
                    "run_name": "single-output-run",
                    "head_contract": _head_contract(),
                },
                full_evaluation={
                    "inputs": {"data_manifest_sha256": "3" * 64},
                    "calibration": {"row_identity_sha256": "4" * 64},
                    "canonical_dev_test": {
                        "metrics": {"threshold": 0.9, "recall": 0.75}
                    },
                },
                run_result_sha256="5" * 64,
                full_evaluation_sha256="6" * 64,
                full_score_sha256="7" * 64,
                checkpoint_sha256="8" * 64,
                checkpoint_kind="retained_update_snapshot",
                evaluation_model_sha256="1" * 64,
                update=17000,
                epoch=3,
                role="pre_registered_comparison",
                threshold=0.9,
                batch_size=1,
            )
            report = run._build_report(
                binding=binding,
                rows=rows,
                head_scores=scores,
                panel_sha256=panel_sha,
                journal=journal,
                journal_model_sha256="1" * 64,
                base_model=BASE_MODEL,
                truncation=np.asarray([False, True]),
                runtime_seconds=1.0,
                resumed_rows=0,
                device="test-device",
                peak_reserved_bytes=123,
                provenance={"sources": {}, "uv_lock_sha256": "9" * 64},
                scoring_sha256="a" * 64,
            )
            serialized = json.dumps(report, sort_keys=True).encode()
            persisted = serialized + b"".join(
                path.read_bytes() for path in journal.root.rglob("*") if path.is_file()
            )

        for row in rows:
            self.assertNotIn(row["id"].encode(), persisted)
            self.assertNotIn(row["text"].encode(), persisted)
        self.assertIsNone(report["instruction_subversion"]["aggregate"]["fpr"])
        self.assertIsNone(report["instruction_subversion"]["aggregate"]["precision"])
        self.assertEqual(
            report["instruction_subversion"]["subversion_attested"]["flag_rate"],
            1.0,
        )
        self.assertEqual(
            report["instruction_subversion"]["subversion_attested"]["recall"],
            1.0,
        )
        self.assertEqual(
            report["instruction_subversion"]["bare_harmful_control"]["flag_rate"],
            0.0,
        )
        self.assertNotIn("harmful_intent", report)
        self.assertEqual(report["purpose"], run.PURPOSE)
        self.assertEqual(report["scores"]["columns"], ["score"])
        self.assertEqual(report["scores"]["shape"], [2, 1])
        self.assertEqual(report["training_max_tokens"], 512)
        self.assertEqual(report["evaluation_max_tokens"], 512)
        self.assertTrue(report["native_context_evaluation"])
        self.assertEqual(
            report["full_panel_evaluation"]["context_contract"],
            run.LEGACY_FULL_EVALUATION_CONTEXT_CONTRACT,
        )

    def test_hash_identities_bind_checkpoint_full_eval_batch_and_sources(self):
        base = run.EvaluationBinding(
            result={"head_contract": _head_contract()},
            full_evaluation={},
            run_result_sha256="1" * 64,
            full_evaluation_sha256="2" * 64,
            full_score_sha256="3" * 64,
            checkpoint_sha256="4" * 64,
            checkpoint_kind="packaged_selected",
            evaluation_model_sha256="1" * 64,
            update=10,
            epoch=1,
            role=run.PACKAGED_CHECKPOINT_ROLE,
            threshold=0.9,
            batch_size=8,
        )
        provenance = {"sources": {"run.py": "5" * 64}}
        scoring = run._scoring_sha256(base, provenance)
        journal_model = run._journal_model_sha256(base, BASE_MODEL)
        identity = run._evaluation_identity_sha256(
            binding=base,
            journal_model_sha256=journal_model,
            panel_sha256="6" * 64,
            reserve_identity_sha256="7" * 64,
            scoring_sha256=scoring,
        )
        variants = [
            replace(base, checkpoint_sha256="8" * 64),
            replace(base, full_evaluation_sha256="9" * 64),
            replace(base, batch_size=16),
            replace(
                base,
                evaluation_max_tokens=1024,
                native_context_evaluation=False,
            ),
        ]
        for variant in variants:
            variant_scoring = run._scoring_sha256(variant, provenance)
            variant_model = run._journal_model_sha256(variant, BASE_MODEL)
            self.assertNotEqual(
                identity,
                run._evaluation_identity_sha256(
                    binding=variant,
                    journal_model_sha256=variant_model,
                    panel_sha256="6" * 64,
                    reserve_identity_sha256="7" * 64,
                    scoring_sha256=variant_scoring,
                ),
            )
        changed_sources = run._scoring_sha256(
            base,
            {"sources": {"run.py": "a" * 64}},
        )
        self.assertNotEqual(scoring, changed_sources)
        changed_base = {**BASE_MODEL, "tokenizer_json_sha256": "c" * 64}
        self.assertNotEqual(
            journal_model,
            run._journal_model_sha256(base, changed_base),
        )

    def test_truncation_uses_the_bound_context_cap(self):
        class Tokenizer:
            def __init__(self):
                self.max_lengths = []

            def __call__(self, texts, **kwargs):
                self.max_lengths.append(kwargs["max_length"])
                return {"length": [min(700, kwargs["max_length"])] * len(texts)}

        tokenizer = Tokenizer()
        rows = _rows()[:1]
        flags_512 = run._truncation_flags(
            tokenizer,
            rows,
            batch_size=1,
            max_tokens=512,
        )
        flags_1024 = run._truncation_flags(
            tokenizer,
            rows,
            batch_size=1,
            max_tokens=1024,
        )
        self.assertEqual(tokenizer.max_lengths, [513, 1025])
        self.assertEqual(flags_512.tolist(), [True])
        self.assertEqual(flags_1024.tolist(), [False])

    def test_slice_metadata_change_invalidates_journal_identity(self):
        rows = _rows()
        original = run.guard_run._journal_panel_sha256(
            run.guard_run.REDTEAM_SHA256, "redteam_reserve", rows
        )
        changed = [dict(row) for row in rows]
        changed[0]["subversion_basis"] = "None"
        self.assertNotEqual(
            original,
            run.guard_run._journal_panel_sha256(
                run.guard_run.REDTEAM_SHA256, "redteam_reserve", changed
            ),
        )

    def test_cli_defaults_are_snapshot_specific_and_noncolliding(self):
        base = Path("artifacts/mmbert/runs/single-output-run")
        snapshot = Path("artifacts/mmbert/runs/.arm6.snapshots/update-023000.pt")
        with (
            patch.object(sys, "argv", ["run", str(base), "--snapshot", str(snapshot)]),
            patch.object(run, "evaluate_reserve", return_value=Path("done")) as call,
            patch("builtins.print"),
        ):
            self.assertEqual(run.main(), 0)
        self.assertEqual(
            call.call_args.kwargs["full_evaluation"],
            base / "evaluation-update-23000/evaluation.json",
        )
        self.assertEqual(
            call.call_args.kwargs["output"],
            base / "redteam-reserve-evaluation-update-23000",
        )
        self.assertEqual(
            call.call_args.kwargs["score_journal"],
            base / ".redteam-reserve-evaluation-update-23000.score-journal",
        )

    def test_cli_defaults_to_packaged_selected_checkpoint(self):
        base = Path("artifacts/mmbert/runs/single-output-run")
        with (
            patch.object(sys, "argv", ["run", str(base)]),
            patch.object(run, "evaluate_reserve", return_value=Path("done")) as call,
            patch("builtins.print"),
        ):
            self.assertEqual(run.main(), 0)
        self.assertIsNone(call.call_args.kwargs["snapshot"])
        self.assertEqual(
            call.call_args.kwargs["full_evaluation"],
            base / "evaluation/evaluation.json",
        )
        self.assertEqual(
            call.call_args.kwargs["output"],
            base / "redteam-reserve-evaluation",
        )
        self.assertEqual(
            call.call_args.kwargs["score_journal"],
            base / ".redteam-reserve-evaluation.score-journal",
        )

    def test_cli_cap_aware_defaults_include_both_contexts(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "single-output-run"
            base.mkdir()
            result = {
                "max_tokens": 1024,
                "token_budget": 24_576,
                "training_identity": {
                    "schema_version": 5,
                    "max_tokens": 1024,
                    "microbatch_size": 24,
                    "token_budget": 24_576,
                },
                "training": {
                    "max_tokens": 1024,
                    "microbatch_size": 24,
                    "token_budget": 24_576,
                },
            }
            (base / "result.json").write_text(
                json.dumps(result) + "\n", encoding="utf-8"
            )
            snapshot = Path(temporary) / "update-017000.pt"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "run",
                        str(base),
                        "--snapshot",
                        str(snapshot),
                        "--evaluation-max-tokens",
                        "1024",
                    ],
                ),
                patch.object(
                    run, "evaluate_reserve", return_value=Path("done")
                ) as call,
                patch("builtins.print"),
            ):
                self.assertEqual(run.main(), 0)

        suffix = "trainctx1024-evalctx1024"
        self.assertEqual(
            call.call_args.kwargs["full_evaluation"],
            base / f"evaluation-update-17000-{suffix}/evaluation.json",
        )
        self.assertEqual(
            call.call_args.kwargs["output"],
            base / f"redteam-reserve-evaluation-update-17000-{suffix}",
        )
        self.assertEqual(
            call.call_args.kwargs["score_journal"],
            base / f".redteam-reserve-evaluation-update-17000-{suffix}.score-journal",
        )
        self.assertEqual(call.call_args.kwargs["evaluation_max_tokens"], 1024)


if __name__ == "__main__":
    unittest.main()
