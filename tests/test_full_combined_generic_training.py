from __future__ import annotations

import builtins
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

import train_combined_generic_head as combined_training  # noqa: E402
import train_full_combined_generic_head as full_training  # noqa: E402
from prepare_combined_generic import strict_hash  # noqa: E402
from train_combined_generic_head import resolve_model_revision  # noqa: E402
from train_full_combined_generic_head import (  # noqa: E402
    BalancedIndexCycle,
    CanonicalFeatureCache,
    PairIndexCycle,
    _bfloat16_to_uint16,
    _pair_indices,
    _stable_json_sha256,
    _uint16_to_bfloat16,
    canonical_feature_cache_spec,
    iter_canonical_batches,
    objective_spec,
    run_directory_name,
    training_objective_loss,
    validate_feature_cache_artifact,
)

from morgott.data import text_hash  # noqa: E402


def target_report(*, full: bool) -> dict:
    if full:
        return {
            "schema_version": 2,
            "purpose": (
                "artifact-only full generic instruction-subversion training recipe"
            ),
            "generic_target": "instruction_subversion",
            "eligibility": {
                "canonical_pool": (
                    "all retained rows after eligibility and leakage filters"
                ),
                "routing_training_eligible": True,
                "input_channel": ["direct_user", "untrusted_content"],
                "label_field": "injection_label",
                "labels": [0, 1],
                "routing_label_used": False,
                "promptshield_subtypes_assigned": False,
                "matched_pairs_are_weak_supervision": True,
            },
        }
    return {
        "schema_version": 2,
        "purpose": (
            "artifact-only update-matched generic instruction-subversion experiment"
        ),
        "generic_target": "instruction_subversion",
        "eligibility": {
            "routing_training_eligible": True,
            "input_channel": ["direct_user", "untrusted_content"],
            "label_field": "injection_label",
            "labels": [0, 1],
            "exclude_security_label": "uncertain",
            "exclude_if_all_origins_are_weak_or_unverified": True,
            "routing_label_used": False,
        },
    }


def current_inputs_and_provenance(*, full: bool) -> tuple[dict, dict]:
    repository = Path(__file__).resolve().parents[1]
    manifest = repository / "data/manifest.json"
    inputs = {
        "manifest": {
            "path": "data/manifest.json",
            "sha256": combined_training.file_sha256(manifest),
        }
    }
    provenance = {
        "runner_sha256": combined_training.file_sha256(
            EXPERIMENTS_DIR
            / (
                "prepare_full_combined_generic.py"
                if full
                else "prepare_combined_generic.py"
            )
        ),
        "strict_normalizer_sha256": combined_training.file_sha256(
            EXPERIMENTS_DIR / "strict_normalize.py"
        ),
        "overlap_module_sha256": combined_training.file_sha256(
            repository / "src/morgott/overlap.py"
        ),
        "canonical_text_helper_sha256": combined_training.file_sha256(
            repository / "src/morgott/data.py"
        ),
    }
    if full:
        provenance.update(
            {
                "base_preparation_runner_sha256": combined_training.file_sha256(
                    EXPERIMENTS_DIR / "prepare_combined_generic.py"
                ),
            }
        )
    return inputs, provenance


def write_population(directory: Path, name: str, records: list[dict]) -> dict:
    path = directory / f"{name}.jsonl"
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))
    labels = Counter(str(record["generic_label"]) for record in records)
    return {
        "path": str(path),
        "sha256": combined_training.file_sha256(path),
        "rows": len(records),
        "labels": dict(sorted(labels.items())),
    }


def valid_update_report(directory: Path) -> dict:
    def record(
        name: str,
        label: int,
        *,
        dataset: str,
        channel: str | None,
        source: str,
        component: int | None,
    ) -> dict:
        value = {
            "id": name,
            "text": f"Unique detector validation text {name}",
            "dataset": dataset,
            "generic_target": "instruction_subversion",
            "generic_label": label,
            "source": source,
            "group_id": name,
            "channel": channel,
            "strict_text_sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        if component is not None:
            value["validation_component_id"] = f"validation-component:{component:064x}"
        if dataset == "promptshield":
            value["subtype_training_eligible"] = False
        return value

    checkpoint = [
        record(
            "checkpoint-negative",
            0,
            dataset="morgott",
            channel="direct_user",
            source="source-a",
            component=1,
        ),
        record(
            "checkpoint-positive",
            1,
            dataset="morgott",
            channel="untrusted_content",
            source="source-b",
            component=2,
        ),
    ]
    calibration = [
        record(
            f"calibration-{index}",
            index % 2,
            dataset="morgott",
            channel=("direct_user" if index in {0, 1, 4, 5} else "untrusted_content"),
            source="source-a" if index < 4 else "source-b",
            component=index + 3,
        )
        for index in range(8)
    ]
    promptshield = [
        record(
            f"promptshield-{label}",
            label,
            dataset="promptshield",
            channel=None,
            source="promptshield",
            component=None,
        )
        for label in (0, 1)
    ]
    outputs = {
        "validation_morgott_selection": write_population(
            directory,
            "validation_morgott_selection",
            checkpoint,
        ),
        "validation_morgott_calibration": write_population(
            directory,
            "validation_morgott_calibration",
            calibration,
        ),
        "validation_promptshield": write_population(
            directory,
            "validation_promptshield",
            promptshield,
        ),
    }
    inputs, provenance = current_inputs_and_provenance(full=False)
    report = target_report(full=False)
    report.update(
        {
            "inputs": inputs,
            "outputs": outputs,
            "provenance": provenance,
            "validation_partition": {
                "target_checkpoint_fraction": 0.2,
                "actual_checkpoint_fraction": 0.2,
                "total_rows": 10,
                "checkpoint_selection_rows": 2,
                "calibration_rows": 8,
                "components": 10,
                "component_basis": [
                    "source+group_id",
                    "conservative_near_overlap",
                ],
                "component_calibration": {
                    "component_id_field": "validation_component_id",
                    "component_id_definition": (
                        "SHA-256 over sorted row id and strict-text SHA-256 pairs"
                    ),
                    "target_unit": (
                        "lineage-and-near validation component within trusted channel"
                    ),
                    "score_aggregation": (
                        "maximum negative score per component within trusted channel"
                    ),
                    "family_confidence": 0.95,
                    "per_channel_confidence": 0.975,
                    "multiplicity_correction": "Bonferroni",
                    "family_scope": (
                        "the two trusted channels, with a separate family for each target"
                    ),
                    "trusted_channels": ["direct_user", "untrusted_content"],
                    "pooled_negative_role": "empirical diagnostic only",
                    "components_by_role": {
                        "checkpoint_selection": 2,
                        "calibration": 8,
                    },
                    "negative_evidence_by_role": {
                        "checkpoint_selection": {
                            "rows_by_channel": {"direct_user": 1},
                            "components_by_channel": {"direct_user": 1},
                            "rows_by_source": {"source-a": 1},
                            "components_by_source": {"source-a": 1},
                        },
                        "calibration": {
                            "rows_by_channel": {
                                "direct_user": 2,
                                "untrusted_content": 2,
                            },
                            "components_by_channel": {
                                "direct_user": 2,
                                "untrusted_content": 2,
                            },
                            "rows_by_source": {"source-a": 2, "source-b": 2},
                            "components_by_source": {
                                "source-a": 2,
                                "source-b": 2,
                            },
                        },
                    },
                    "inference_caveat": (
                        "Components and recurring source families are not IID or "
                        "sampled from a deployment distribution; confidence bounds "
                        "are development evidence, not production guarantees."
                    ),
                },
                "disjointness": {
                    "row": True,
                    "normalized": True,
                    "strict": True,
                    "lineage_group": True,
                    "near": True,
                    "validation_component": True,
                },
                "checkpoint_selection": [
                    "morgott_validation_checkpoint_selection",
                    "promptshield_validation",
                ],
                "threshold_calibration": "morgott_validation_calibration_only",
                "promptshield_used_for_threshold": False,
            },
        }
    )
    return report


def valid_full_report(directory: Path) -> dict:
    base_directory = directory / "base"
    base_directory.mkdir()
    base_report = valid_update_report(base_directory)
    base_report_path = base_directory / "selection_report.json"
    base_report_path.write_text(json.dumps(base_report))
    inputs, provenance = current_inputs_and_provenance(full=True)
    inputs["base_update_matched_selection"] = {
        "path": str(base_report_path),
        "sha256": combined_training.file_sha256(base_report_path),
    }
    report = target_report(full=True)
    report.update(
        {
            "inputs": inputs,
            "provenance": provenance,
            "training_recipe": {
                "pair_atoms_preserved": True,
                "pair_ranking_capable": True,
            },
            "validation": {
                "selection_report": str(base_report_path),
                "morgott": base_report["outputs"]["validation_morgott_selection"],
                "morgott_calibration": base_report["outputs"][
                    "validation_morgott_calibration"
                ],
                "promptshield": base_report["outputs"]["validation_promptshield"],
                "checkpoint_selection_only": ["morgott", "promptshield"],
                "threshold_calibration_only": "morgott_calibration",
                "promptshield_used_for_threshold": False,
                "component_calibration": base_report["validation_partition"][
                    "component_calibration"
                ],
            },
        }
    )
    return report


class FullCombinedGenericTrainingTests(unittest.TestCase):
    def test_update_trainer_rejects_source_code_changed_during_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "runner.py"
            source.write_text("before\n")
            paths = {"runner_sha256": source}
            expected = {
                "runner_sha256": combined_training.file_sha256(source),
            }

            combined_training._verify_source_hashes(paths, expected)
            source.write_text("after\n")

            with self.assertRaisesRegex(ValueError, "source changed during run"):
                combined_training._verify_source_hashes(paths, expected)

    def test_both_trainers_reject_schema_one_before_artifact_or_model_access(self):
        original_import = builtins.__import__

        def reject_model_stack(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"safetensors", "torch", "transformers"}:
                raise AssertionError(f"model stack imported before preflight: {name}")
            return original_import(name, *args, **kwargs)

        cases = (
            (
                combined_training,
                "selection_report.json",
                ["train_combined_generic_head.py"],
            ),
            (
                full_training,
                "full_selection_report.json",
                [
                    "train_full_combined_generic_head.py",
                    "--objective",
                    "canonical_uniform",
                ],
            ),
        )
        for module, report_name, arguments in cases:
            with tempfile.TemporaryDirectory() as temporary:
                selection_dir = Path(temporary)
                (selection_dir / report_name).write_text('{"schema_version": 1}\n')
                with (
                    self.subTest(module=module.__name__),
                    patch.object(
                        sys,
                        "argv",
                        [
                            *arguments,
                            "--selection-dir",
                            str(selection_dir),
                        ],
                    ),
                    patch("builtins.__import__", side_effect=reject_model_stack),
                    self.assertRaisesRegex(ValueError, "schema version 2"),
                ):
                    module.main()

    def test_selection_preflight_requires_exact_target_purpose_and_eligibility(self):
        for report, is_full in (
            (target_report(full=False), False),
            (target_report(full=True), True),
        ):
            for field, value in (
                ("purpose", "stale purpose"),
                ("generic_target", "routing"),
                ("eligibility", {"label_field": "routing_label"}),
            ):
                stale = deepcopy(report)
                stale[field] = value
                with (
                    self.subTest(full=is_full, field=field),
                    self.assertRaisesRegex(ValueError, "target contract"),
                ):
                    combined_training.validate_selection_report(
                        stale,
                        selection_dir=Path("/unused"),
                        full=is_full,
                    )

    def test_selection_preflight_requires_current_manifest_and_preparation_hashes(self):
        for is_full in (False, True):
            inputs, provenance = current_inputs_and_provenance(full=is_full)
            report = target_report(full=is_full)
            report["inputs"] = inputs
            report["provenance"] = provenance
            stale_manifest = deepcopy(report)
            stale_manifest["inputs"]["manifest"]["sha256"] = "0" * 64
            with (
                self.subTest(full=is_full, field="manifest"),
                self.assertRaisesRegex(ValueError, "manifest"),
            ):
                combined_training.validate_selection_report(
                    stale_manifest,
                    selection_dir=Path("/unused"),
                    full=is_full,
                )
            for field in provenance:
                stale = deepcopy(report)
                stale["provenance"][field] = "0" * 64
                with (
                    self.subTest(full=is_full, field=field),
                    self.assertRaisesRegex(ValueError, "provenance"),
                ):
                    combined_training.validate_selection_report(
                        stale,
                        selection_dir=Path("/unused"),
                        full=is_full,
                    )

    def test_update_preflight_verifies_validation_partition_and_populations(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection_dir = Path(temporary)
            report = valid_update_report(selection_dir)
            combined_training.validate_selection_report(
                report,
                selection_dir=selection_dir,
            )

            mutations = []
            stale = deepcopy(report)
            stale["validation_partition"]["target_checkpoint_fraction"] = 0.5
            mutations.append(("checkpoint fraction", stale))
            stale = deepcopy(report)
            stale["validation_partition"]["promptshield_used_for_threshold"] = True
            mutations.append(("PromptShield threshold", stale))
            stale = deepcopy(report)
            stale["validation_partition"]["disjointness"]["near"] = False
            mutations.append(("partition overlap", stale))
            stale = deepcopy(report)
            stale["validation_partition"]["total_rows"] = 11
            mutations.append(("partition rows", stale))
            stale = deepcopy(report)
            del stale["outputs"]["validation_morgott_calibration"]
            mutations.append(("missing calibration", stale))
            stale = deepcopy(report)
            stale["outputs"]["validation_morgott_selection"]["sha256"] = "0" * 64
            mutations.append(("checkpoint hash", stale))
            stale = deepcopy(report)
            stale["outputs"]["validation_promptshield"]["labels"] = {
                "0": 2,
                "1": 0,
            }
            mutations.append(("PromptShield labels", stale))
            stale = deepcopy(report)
            stale["validation_partition"]["component_calibration"][
                "score_aggregation"
            ] = "pooled rows"
            mutations.append(("component calibration", stale))
            stale = deepcopy(report)
            stale["validation_partition"]["component_calibration"]["family_scope"] = (
                "pooled channels"
            )
            mutations.append(("component family scope", stale))
            stale = deepcopy(report)
            stale["validation_partition"]["component_calibration"]["unexpected"] = True
            mutations.append(("unexpected component calibration field", stale))

            for name, stale in mutations:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "validation"),
                ):
                    combined_training.validate_selection_report(
                        stale,
                        selection_dir=selection_dir,
                    )

    def test_full_preflight_recursively_verifies_base_and_validation_contract(self):
        artifacts = Path(__file__).resolve().parents[1] / "artifacts"
        with tempfile.TemporaryDirectory(dir=artifacts) as temporary:
            selection_dir = Path(temporary)
            report = valid_full_report(selection_dir)
            combined_training.validate_selection_report(
                report,
                selection_dir=selection_dir,
                full=True,
            )

            mutations = []
            stale = deepcopy(report)
            stale["inputs"]["base_update_matched_selection"]["sha256"] = "0" * 64
            mutations.append(("base hash", stale))
            stale = deepcopy(report)
            stale["validation"]["morgott_calibration"]["sha256"] = "0" * 64
            mutations.append(("calibration spec", stale))
            stale = deepcopy(report)
            stale["validation"]["threshold_calibration_only"] = "promptshield"
            mutations.append(("threshold role", stale))
            stale = deepcopy(report)
            stale["validation"]["promptshield_used_for_threshold"] = True
            mutations.append(("PromptShield threshold", stale))
            stale = deepcopy(report)
            stale["validation"]["checkpoint_selection_only"] = ["morgott"]
            mutations.append(("checkpoint roles", stale))
            stale = deepcopy(report)
            stale["validation"]["component_calibration"]["family_confidence"] = 0.9
            mutations.append(("component calibration", stale))

            for name, stale in mutations:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "full selection"),
                ):
                    combined_training.validate_selection_report(
                        stale,
                        selection_dir=selection_dir,
                        full=True,
                    )

    def test_objective_specs_separate_data_inclusion_from_weighting(self):
        labels = np.asarray([0] * 12 + [1] * 8)

        canonical = objective_spec(
            "canonical_uniform",
            canonical_rows=60,
            promptshield_labels=labels,
            matched_pair_rows=20,
        )
        full_uniform = objective_spec(
            "full_uniform",
            canonical_rows=60,
            promptshield_labels=labels,
            matched_pair_rows=20,
        )
        full_balanced = objective_spec(
            "full_balanced",
            canonical_rows=60,
            promptshield_labels=labels,
            matched_pair_rows=20,
        )

        self.assertEqual(
            canonical["domain_bce_coefficients"],
            {"morgott": 1.0, "promptshield": 0.0, "matched_pairs": 0.0},
        )
        self.assertEqual(canonical["canonical_weighting"], "uniform_per_row")
        self.assertEqual(
            full_uniform["domain_bce_coefficients"],
            {"morgott": 0.6, "promptshield": 0.2, "matched_pairs": 0.2},
        )
        self.assertEqual(
            full_uniform["promptshield_class_loss_correction"],
            {"0": 1.2, "1": 0.8},
        )
        self.assertEqual(full_uniform["canonical_weighting"], "uniform_per_row")
        self.assertEqual(
            full_balanced["domain_bce_coefficients"],
            {
                "morgott": 1.0 / 3.0,
                "promptshield": 1.0 / 3.0,
                "matched_pairs": 1.0 / 3.0,
            },
        )
        self.assertEqual(
            full_balanced["canonical_weighting"],
            "label_source_group_balanced",
        )
        self.assertEqual(
            full_balanced["promptshield_class_loss_correction"],
            {"0": 1.0, "1": 1.0},
        )
        self.assertNotIn("same_update_schedule_across_objectives", canonical)
        self.assertEqual(
            canonical["cross_objective_schedule_comparability"],
            (
                "requires identical recorded seed, epochs, domain batch sizes, "
                "learning rate, and shuffle buffer"
            ),
        )

    def test_training_loss_applies_each_objective_and_separates_pair_ranking(self):
        import torch

        labels = np.asarray([0] * 12 + [1] * 8)
        arguments = {
            "canonical_logits": torch.tensor([-2.0, 2.0], requires_grad=True),
            "canonical_targets": torch.tensor([0.0, 1.0]),
            "canonical_weights": torch.tensor([2.0, 4.0]),
            "promptshield_logits": torch.tensor([2.0, 2.0], requires_grad=True),
            "promptshield_targets": torch.tensor([0.0, 1.0]),
            "benign_logits": torch.tensor([2.0], requires_grad=True),
            "attack_logits": torch.tensor([-2.0], requires_grad=True),
        }
        specs = {
            name: objective_spec(
                name,
                canonical_rows=60,
                promptshield_labels=labels,
                matched_pair_rows=20,
            )
            for name in (
                "canonical_uniform",
                "full_uniform",
                "full_balanced",
            )
        }

        canonical = training_objective_loss(
            **arguments,
            objective=specs["canonical_uniform"],
            pair_ranking_weight=0.0,
        )
        full_uniform = training_objective_loss(
            **arguments,
            objective=specs["full_uniform"],
            pair_ranking_weight=0.0,
        )
        full_balanced = training_objective_loss(
            **arguments,
            objective=specs["full_balanced"],
            pair_ranking_weight=0.0,
        )
        ranked = training_objective_loss(
            **arguments,
            objective=specs["full_uniform"],
            pair_ranking_weight=0.25,
        )

        self.assertAlmostEqual(canonical["total"].item(), 0.1269280110, places=6)
        self.assertAlmostEqual(
            full_uniform["domain_bce"].item(),
            0.7669280110,
            places=6,
        )
        self.assertAlmostEqual(
            full_balanced["domain_bce"].item(),
            1.2115466851,
            places=6,
        )
        self.assertAlmostEqual(ranked["ranking"].item(), 4.0181499279, places=6)
        self.assertAlmostEqual(ranked["total"].item(), 1.7714654930, places=6)
        logits = (
            arguments["canonical_logits"],
            arguments["promptshield_logits"],
            arguments["benign_logits"],
            arguments["attack_logits"],
        )
        canonical_gradients = torch.autograd.grad(canonical["total"], logits)
        self.assertTrue(torch.count_nonzero(canonical_gradients[0]))
        for gradient in canonical_gradients[1:]:
            self.assertFalse(torch.count_nonzero(gradient))
        full_uniform_gradients = torch.autograd.grad(full_uniform["total"], logits)
        for gradient in full_uniform_gradients:
            self.assertTrue(torch.count_nonzero(gradient))
        benign_gradient, attack_gradient = torch.autograd.grad(
            ranked["ranking"],
            (arguments["benign_logits"], arguments["attack_logits"]),
        )
        self.assertGreater(benign_gradient.item(), 0.0)
        self.assertLess(attack_gradient.item(), 0.0)
        with self.assertRaisesRegex(
            ValueError,
            "canonical_uniform cannot use pair ranking",
        ):
            training_objective_loss(
                **arguments,
                objective=specs["canonical_uniform"],
                pair_ranking_weight=0.25,
            )

    def test_run_directory_name_records_objective_and_pair_ranking(self):
        self.assertEqual(
            run_directory_name(
                "jhu-clsp/mmBERT-base",
                objective="full_uniform",
                pair_ranking_weight=0.25,
                seed=42,
            ),
            "jhu-clsp-mmbert-base_objective-full-uniform_pair-rank-0p25_s42",
        )
        self.assertEqual(
            run_directory_name(
                "jhu-clsp/mmBERT-base",
                objective="full_balanced",
                pair_ranking_weight=0.0,
                seed=43,
            ),
            "jhu-clsp-mmbert-base_objective-full-balanced_pair-rank-0p0_s43",
        )

    def test_cli_requires_an_explicit_objective(self):
        stderr = StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "train_full_combined_generic_head.py",
                    "--selection-dir",
                    "does-not-exist",
                ],
            ),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            full_training.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--objective", stderr.getvalue())
        self.assertIn("required", stderr.getvalue())

    def test_model_revision_resolution_matches_evaluator_contract(self):
        pinned = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
        self.assertEqual(resolve_model_revision("jhu-clsp/mmBERT-base", None), pinned)
        self.assertEqual(
            resolve_model_revision("jhu-clsp/mmBERT-base", pinned),
            pinned,
        )
        self.assertEqual(resolve_model_revision("custom/model", "a" * 40), "a" * 40)
        for model_id, revision in (
            ("jhu-clsp/mmBERT-base", "b" * 40),
            ("custom/model", None),
            ("custom/model", "main"),
            ("custom/model", "A" * 40),
        ):
            with self.subTest(model_id=model_id, revision=revision):
                with self.assertRaises(ValueError):
                    resolve_model_revision(model_id, revision)

    def test_both_trainers_reject_an_alternate_known_model_revision_at_cli(self):
        cases = (
            (
                combined_training,
                ["train_combined_generic_head.py"],
            ),
            (
                full_training,
                [
                    "train_full_combined_generic_head.py",
                    "--objective",
                    "canonical_uniform",
                ],
            ),
        )
        for module, arguments in cases:
            stderr = StringIO()
            with (
                self.subTest(module=module.__name__),
                patch.object(
                    sys,
                    "argv",
                    [
                        *arguments,
                        "--model-revision",
                        "b" * 40,
                        "--selection-dir",
                        "does-not-exist",
                    ],
                ),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                module.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("must use pinned revision", stderr.getvalue())

    def test_feature_cache_identity_pins_extraction_record_chunk(self):
        report_path = Path(__file__).resolve().parents[1] / "artifacts/report.json"
        arguments = {
            "selection_report_path": report_path,
            "selection_report_sha256": "selection",
            "canonical_spec": {
                "path": "artifacts/index.sqlite",
                "sha256": "index",
                "rows": 7,
            },
            "canonical_input_spec": {
                "path": "data/train.jsonl",
                "sha256": "train",
                "rows": 9,
            },
            "model_id": "tiny",
            "model_revision": "revision",
            "hidden_size": 2,
            "max_tokens": 8,
            "token_budget": 16,
            "runner_sha256": "runner",
            "head_helper_sha256": "helper",
            "strict_normalizer_sha256": "normalizer",
            "canonical_projection_sha256": "projection",
            "packages": {"torch": "1"},
        }

        small = canonical_feature_cache_spec(
            **arguments,
            feature_record_chunk=128,
        )
        large = canonical_feature_cache_spec(
            **arguments,
            feature_record_chunk=256,
        )

        self.assertEqual(small["feature_record_chunk"], 128)
        self.assertNotEqual(_stable_json_sha256(small), _stable_json_sha256(large))

    def test_bfloat16_cache_storage_roundtrips_exact_bits(self):
        import torch

        values = torch.tensor(
            [[-3.25, -0.0, 0.125], [1.0, 7.5, float("inf")]],
            dtype=torch.bfloat16,
        )

        stored = _bfloat16_to_uint16(values)
        restored = _uint16_to_bfloat16(stored)

        self.assertEqual(stored.dtype, np.dtype("<u2"))
        self.assertTrue(
            torch.equal(values.view(torch.uint16), restored.view(torch.uint16))
        )

    def test_feature_chunk_casts_float32_encoder_output_to_bfloat16(self):
        from contextlib import nullcontext
        from types import SimpleNamespace

        import torch

        class Tokenizer:
            pad_token_id = 0

            def __call__(self, _texts, **_kwargs):
                return {"input_ids": [[1, 2, 3], [4, 5]]}

        class Encoder:
            config = SimpleNamespace(hidden_size=2)

            def __call__(self, *, input_ids, attention_mask):
                del attention_mask
                hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
                return SimpleNamespace(last_hidden_state=hidden)

        torch_full = torch.full
        torch_tensor = torch.tensor

        def cpu_full(*args, **kwargs):
            kwargs.pop("device", None)
            return torch_full(*args, **kwargs)

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return torch_tensor(*args, **kwargs)

        with (
            patch.object(torch, "autocast", return_value=nullcontext()),
            patch.object(torch, "full", side_effect=cpu_full),
            patch.object(torch, "tensor", side_effect=cpu_tensor),
        ):
            features = combined_training._feature_chunk(
                Encoder(),
                Tokenizer(),
                [{"text": "first"}, {"text": "second"}],
                max_tokens=8,
                token_budget=16,
            )

        expected = torch.tensor(
            [
                [1, 1, 2, 2, 3, 3],
                [4, 4, 4.5, 4.5, 5, 5],
            ],
            dtype=torch.bfloat16,
        )
        self.assertEqual(features.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(features, expected))

    def test_feature_cache_rejects_a_different_expected_spec(self):
        import hashlib
        import json

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = np.arange(12, dtype="<u2").reshape(2, 6)
            data_path = directory / "canonical_features.uint16"
            data.tofile(data_path)
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            report = {
                "schema_version": 1,
                "spec": {"model_id": "expected", "rows": 2, "feature_width": 6},
                "data": {
                    "file": data_path.name,
                    "sha256": digest,
                    "bytes": data_path.stat().st_size,
                    "shape": [2, 6],
                    "storage_dtype": "uint16_le",
                    "logical_dtype": "bfloat16",
                },
            }
            report["spec_sha256"] = _stable_json_sha256(report["spec"])
            (directory / "cache_report.json").write_text(json.dumps(report))

            with self.assertRaisesRegex(ValueError, "cache spec mismatch"):
                validate_feature_cache_artifact(
                    directory,
                    expected_spec={
                        "model_id": "different",
                        "rows": 2,
                        "feature_width": 6,
                    },
                )

    def test_feature_cache_reads_selected_rows_as_bfloat16(self):
        import hashlib
        import json

        import torch

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = torch.tensor(
                [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
                dtype=torch.bfloat16,
            )
            stored = _bfloat16_to_uint16(values)
            data_path = directory / "canonical_features.uint16"
            stored.tofile(data_path)
            spec = {"model_id": "tiny", "rows": 2, "feature_width": 3}
            report = {
                "schema_version": 1,
                "spec": spec,
                "data": {
                    "file": data_path.name,
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "bytes": data_path.stat().st_size,
                    "shape": [2, 3],
                    "storage_dtype": "uint16_le",
                    "logical_dtype": "bfloat16",
                },
            }
            report["spec_sha256"] = _stable_json_sha256(spec)
            (directory / "cache_report.json").write_text(json.dumps(report))

            cache = CanonicalFeatureCache(
                validate_feature_cache_artifact(directory, expected_spec=spec)
            )
            try:
                selected = cache.take(np.asarray([1, 0], dtype=np.int64))
            finally:
                cache.close()

            self.assertTrue(torch.equal(selected, values[[1, 0]]))

    def test_canonical_batches_emit_each_feature_index_once(self):
        import sqlite3

        connection = sqlite3.connect(":memory:")
        connection.execute(
            """
            CREATE TABLE candidates (
                id TEXT,
                generic_label INTEGER,
                byte_offset INTEGER,
                objective_weight REAL
            )
            """
        )
        for index in range(13):
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                (f"row-{index}", index % 2, index * 10, 1.0 + index / 10),
            )

        batches = list(
            iter_canonical_batches(
                connection,
                batch_size=3,
                shuffle_buffer=5,
                seed=29,
            )
        )
        records = [record for batch in batches for record in batch]

        self.assertEqual(
            sorted(row["feature_index"] for row in records), list(range(13))
        )
        self.assertEqual(
            {row["id"] for row in records}, {f"row-{i}" for i in range(13)}
        )
        connection.close()

    def test_feature_cache_resumes_after_a_completed_chunk(self):
        import json
        import sqlite3

        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "train.jsonl"
            rows = []
            offsets = []
            with source_path.open("wb") as source:
                for index in range(3):
                    text = f"example {index}"
                    row = {
                        "id": f"row-{index}",
                        "text": text,
                        "source": "test",
                        "split_group_id": f"group-{index}",
                        "input_channel": "direct_user",
                        "injection_label": index % 2,
                        "routing_training_eligible": True,
                        "security_label": "benign",
                        "label_basis": "source_label",
                        "data_role": "train",
                        "normalized_text_sha256": text_hash(text),
                    }
                    offsets.append(source.tell())
                    source.write((json.dumps(row) + "\n").encode())
                    rows.append(row)

            connection = sqlite3.connect(":memory:")
            connection.execute(
                """
                CREATE TABLE candidates (
                    id TEXT,
                    generic_label INTEGER,
                    normalized_text_sha256 TEXT,
                    strict_text_sha256 TEXT,
                    byte_offset INTEGER
                )
                """
            )
            for row, offset in zip(rows, offsets, strict=True):
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                    (
                        row["id"],
                        row["injection_label"],
                        row["normalized_text_sha256"],
                        strict_hash(row["text"]),
                        offset,
                    ),
                )
            spec = {
                "model_id": "tiny",
                "rows": 3,
                "feature_width": 6,
                "feature_record_chunk": 1,
                "max_tokens": 8,
                "token_budget": 8,
            }
            calls = 0

            def interrupted_extract(_encoder, _tokenizer, records, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return torch.full(
                    (len(records), 6),
                    records[0]["feature_index"] + 1,
                    dtype=torch.bfloat16,
                )

            with (
                patch.object(
                    full_training,
                    "extract_features",
                    side_effect=interrupted_extract,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                full_training.prepare_canonical_feature_cache(
                    root / "cache",
                    spec,
                    connection=connection,
                    canonical_path=source_path,
                    encoder=object(),
                    tokenizer=object(),
                    chunk_rows=1,
                )

            def resumed_extract(_encoder, _tokenizer, records, **_kwargs):
                return torch.full(
                    (len(records), 6),
                    records[0]["feature_index"] + 1,
                    dtype=torch.bfloat16,
                )

            with patch.object(
                full_training,
                "extract_features",
                side_effect=resumed_extract,
            ):
                cache, activity = full_training.prepare_canonical_feature_cache(
                    root / "cache",
                    spec,
                    connection=connection,
                    canonical_path=source_path,
                    encoder=object(),
                    tokenizer=object(),
                    chunk_rows=1,
                )
            try:
                restored = cache.take(np.asarray([0, 1, 2]))
            finally:
                cache.close()
                connection.close()

            self.assertEqual(activity["resumed_from_rows"], 1)
            self.assertEqual(activity["encoded_rows"], 2)
            self.assertTrue(
                torch.equal(
                    restored,
                    torch.tensor([[1] * 6, [2] * 6, [3] * 6], dtype=torch.bfloat16),
                )
            )

    def test_balanced_cycle_is_deterministic_and_balances_every_batch(self):
        labels = np.asarray([0, 0, 0, 1])
        first = BalancedIndexCycle(labels, seed=19)
        second = BalancedIndexCycle(labels, seed=19)

        for _ in range(5):
            first_batch = first.take(4)
            second_batch = second.take(4)
            np.testing.assert_array_equal(first_batch, second_batch)
            self.assertEqual(
                np.bincount(labels[first_batch], minlength=2).tolist(),
                [2, 2],
            )

    def test_pair_cycle_is_deterministic_and_keeps_pair_indices_atomic(self):
        first = PairIndexCycle(3, seed=23)
        second = PairIndexCycle(3, seed=23)

        np.testing.assert_array_equal(first.take(5), second.take(5))
        np.testing.assert_array_equal(first.take(5), second.take(5))

    def test_pair_index_projection_aligns_benign_and_attack_halves(self):
        records = [
            {
                "id": "pair-b:attack",
                "dataset": "matched_pairs",
                "source": "matched_pairs_generated",
                "pair_family": "generated_matched_instruction_subversion",
                "pair_id": "pair-b",
                "pair_role": "attack",
                "pair_label": 1,
                "generic_label": 1,
            },
            {
                "id": "pair-a:benign",
                "dataset": "matched_pairs",
                "source": "matched_pairs_generated",
                "pair_family": "generated_matched_instruction_subversion",
                "pair_id": "pair-a",
                "pair_role": "benign",
                "pair_label": 0,
                "generic_label": 0,
            },
            {
                "id": "pair-b:benign",
                "dataset": "matched_pairs",
                "source": "matched_pairs_generated",
                "pair_family": "generated_matched_instruction_subversion",
                "pair_id": "pair-b",
                "pair_role": "benign",
                "pair_label": 0,
                "generic_label": 0,
            },
            {
                "id": "pair-a:attack",
                "dataset": "matched_pairs",
                "source": "matched_pairs_generated",
                "pair_family": "generated_matched_instruction_subversion",
                "pair_id": "pair-a",
                "pair_role": "attack",
                "pair_label": 1,
                "generic_label": 1,
            },
        ]

        benign, attack = _pair_indices(records)

        self.assertEqual(benign.tolist(), [1, 2])
        self.assertEqual(attack.tolist(), [3, 0])


if __name__ == "__main__":
    unittest.main()
