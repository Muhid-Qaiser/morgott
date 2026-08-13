from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.mmbert_longcode_snapshot_eval import run
from morgott.models.mmbert import evaluate as mmbert_evaluate

BASE_MODEL = {
    "id": mmbert_evaluate.MODEL_ID,
    "revision": mmbert_evaluate.MODEL_REVISION,
    "config_sha256": "0" * 64,
    "pytorch_model_sha256": "a" * 64,
    "special_tokens_map_sha256": "c" * 64,
    "tokenizer_config_sha256": "d" * 64,
    "tokenizer_json_sha256": "b" * 64,
}


def _write_json(path: Path, value: object) -> str:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_gzip_rows(path: Path, rows: list[dict]) -> tuple[str, str]:
    content = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(content)
    return hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(
        content
    ).hexdigest()


def _pair_row(index: int, repository: str) -> dict:
    return {
        "attack": f"synthetic attack {index}",
        "attack_span": "synthetic span",
        "attack_span_start": 3,
        "benign": f"synthetic benign {index}",
        "channel": "direct_user",
        "instance_id": f"synthetic:{index}",
        "repository": repository,
        "source": "nebius/SWE-rebench-V2",
        "source_revision": run.EXPECTED_SOURCE_REVISION,
    }


class FullEvaluationBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_path = self.root / "candidate"
        self.run_path.mkdir()
        self.snapshot = self.root / "update-017000.pt"
        self.snapshot.write_bytes(b"synthetic snapshot bytes")
        self.snapshot_sha256 = hashlib.sha256(self.snapshot.read_bytes()).hexdigest()
        self.result = {
            "max_tokens": 512,
            "token_budget": 12_288,
            "training_identity": {
                "schema_version": 5,
                "max_tokens": 512,
                "microbatch_size": 24,
                "token_budget": 12_288,
            },
            "training": {
                "max_tokens": 512,
                "microbatch_size": 24,
                "token_budget": 12_288,
            },
        }
        self.run_sha256 = _write_json(self.run_path / "result.json", self.result)
        self.full_dir = self.root / "full"
        self.full_dir.mkdir()
        self.score_path = self.full_dir / "scores.npy"
        self.score_path.write_bytes(b"numeric-only score artifact")
        self.score_sha256 = hashlib.sha256(self.score_path.read_bytes()).hexdigest()
        self.model_sha256 = mmbert_evaluate._evaluation_model_sha256(
            self.run_sha256, self.snapshot_sha256
        )
        self.scoring_sha256 = "2" * 64
        self.identity_sha256 = mmbert_evaluate._evaluation_identity_sha256(
            model_sha256=self.model_sha256,
            scoring_sha256=self.scoring_sha256,
            training_max_tokens=512,
            evaluation_max_tokens=512,
        )
        self.additional_sha256 = "8" * 64
        self.evaluation = {
            "schema_version": 1,
            "purpose": "advisory mmBERT development evaluation",
            "advisory_only": True,
            "run_result_sha256": self.run_sha256,
            "evaluated_checkpoint": {
                "sha256": self.snapshot_sha256,
                "update": 17_000,
                "epoch": 3,
                "role": "pre_registered_comparison",
            },
            "training_max_tokens": 512,
            "evaluation_max_tokens": 512,
            "native_context_evaluation": True,
            "evaluation_model_sha256": self.model_sha256,
            "evaluation_identity_sha256": self.identity_sha256,
            "scores": {
                "path": "scores.npy",
                "sha256": self.score_sha256,
                "scoring_sha256": self.scoring_sha256,
                "evaluation_identity_sha256": self.identity_sha256,
                "training_max_tokens": 512,
                "evaluation_max_tokens": 512,
            },
            "inputs": {
                "additional_pair_archive_sha256": self.additional_sha256,
            },
            "thresholds": {
                "source": "canonical calibration components only",
                "selected": {run.DESCRIPTIVE_FPR_BUDGET: 0.75},
            },
            "calibration": {
                "component_thresholds": {
                    run.DESCRIPTIVE_FPR_BUDGET: {
                        "status": "available",
                        "threshold": 0.75,
                    }
                },
                "metrics": {"threshold": 0.75},
            },
            "canonical_dev_test": {"metrics": {"threshold": 0.75}},
        }
        self.full_evaluation = self.full_dir / "evaluation.json"
        _write_json(self.full_evaluation, self.evaluation)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_snapshot_cap_and_threshold_are_bound(self):
        binding = run._bind_full_evaluation(
            self.run_path,
            self.snapshot,
            self.full_evaluation,
            evaluation_max_tokens=512,
            require_update=17_000,
            require_additional_pairs_sha256=self.additional_sha256,
        )
        self.assertEqual(binding.snapshot_sha256, self.snapshot_sha256)
        self.assertEqual(binding.evaluation_identity_sha256, self.identity_sha256)
        self.assertEqual(binding.threshold, 0.75)

    def test_schema2_binds_ordered_panels_and_rejects_tampering(self):
        current_model_sha256 = mmbert_evaluate._evaluation_model_sha256(
            self.run_sha256,
            self.snapshot_sha256,
            base_model=BASE_MODEL,
        )
        self.evaluation.update(
            {
                "base_model": BASE_MODEL,
                "evaluation_model_sha256": current_model_sha256,
            }
        )
        ordered = tuple(
            (name, str(index) * 64)
            for index, name in enumerate(
                mmbert_evaluate.EVALUATION_PANEL_ORDER,
                start=3,
            )
        )
        for (_, report_name), (_, digest) in zip(
            mmbert_evaluate._EVALUATION_REPORT_PANELS,
            ordered,
            strict=True,
        ):
            self.evaluation.setdefault(report_name, {})["score_panel_sha256"] = digest
        identity_document = mmbert_evaluate._evaluation_identity_document(
            model_sha256=current_model_sha256,
            scoring_sha256=self.scoring_sha256,
            training_max_tokens=512,
            evaluation_max_tokens=512,
            ordered_panel_sha256=ordered,
        )
        identity_sha256 = mmbert_evaluate._evaluation_identity_sha256(
            model_sha256=current_model_sha256,
            scoring_sha256=self.scoring_sha256,
            training_max_tokens=512,
            evaluation_max_tokens=512,
            identity_schema_version=(
                mmbert_evaluate.EVALUATION_IDENTITY_SCHEMA_VERSION
            ),
            ordered_panel_sha256=ordered,
        )
        self.evaluation.update(
            {
                "schema_version": mmbert_evaluate.EVALUATION_SCHEMA_VERSION,
                "evaluation_identity": identity_document,
                "evaluation_identity_sha256": identity_sha256,
            }
        )
        self.evaluation["scores"]["evaluation_identity_sha256"] = identity_sha256
        _write_json(self.full_evaluation, self.evaluation)

        binding = run._bind_full_evaluation(
            self.run_path,
            self.snapshot,
            self.full_evaluation,
            evaluation_max_tokens=512,
        )
        self.assertEqual(binding.evaluation_identity_sha256, identity_sha256)

        self.evaluation["promptshield_test"]["score_panel_sha256"] = "9" * 64
        _write_json(self.full_evaluation, self.evaluation)
        with self.assertRaisesRegex(ValueError, "identity document mismatch"):
            run._bind_full_evaluation(
                self.run_path,
                self.snapshot,
                self.full_evaluation,
                evaluation_max_tokens=512,
            )

    def test_mismatched_cap_update_or_numeric_score_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requested snapshot and cap"):
            run._bind_full_evaluation(
                self.run_path,
                self.snapshot,
                self.full_evaluation,
                evaluation_max_tokens=1024,
            )
        with self.assertRaisesRegex(ValueError, "required fixed update"):
            run._bind_full_evaluation(
                self.run_path,
                self.snapshot,
                self.full_evaluation,
                evaluation_max_tokens=512,
                require_update=16_500,
            )
        self.score_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "score artifact hash mismatch"):
            run._bind_full_evaluation(
                self.run_path,
                self.snapshot,
                self.full_evaluation,
                evaluation_max_tokens=512,
            )

    def test_longcode_identity_binds_full_evaluation_threshold_and_batch(self):
        binding = run._bind_full_evaluation(
            self.run_path,
            self.snapshot,
            self.full_evaluation,
            evaluation_max_tokens=512,
        )
        common = {
            "journal_model_sha256": "2" * 64,
            "pair_manifest_sha256": "3" * 64,
            "scoring_sha256": "4" * 64,
            "batch_size": 8,
        }
        baseline = run._evaluation_identity_sha256(binding=binding, **common)

        for field, value in (
            ("evaluation_identity_sha256", "5" * 64),
            ("full_evaluation_sha256", "6" * 64),
            ("full_score_sha256", "7" * 64),
            ("full_scoring_sha256", "8" * 64),
            ("threshold", 0.8),
        ):
            with self.subTest(field=field):
                changed = replace(binding, **{field: value})
                self.assertNotEqual(
                    run._evaluation_identity_sha256(binding=changed, **common),
                    baseline,
                )
        self.assertNotEqual(
            run._evaluation_identity_sha256(
                binding=binding,
                **{**common, "batch_size": 16},
            ),
            baseline,
        )
        self.assertNotEqual(
            run._evaluation_identity_sha256(
                binding=binding,
                **{**common, "journal_model_sha256": "9" * 64},
            ),
            baseline,
        )

    def test_longcode_scoring_identity_binds_uv_lock(self):
        with patch.object(
            run,
            "source_provenance",
            return_value={"sources": {}, "uv_lock_sha256": "1" * 64},
        ) as provenance:
            baseline = run._longcode_scoring_sha256(512)
            provenance.return_value = {
                "sources": {},
                "uv_lock_sha256": "2" * 64,
            }
            changed = run._longcode_scoring_sha256(512)

        self.assertNotEqual(baseline, changed)


class FrozenPairInputTests(unittest.TestCase):
    def test_pair_rows_reject_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.jsonl.gz"
            row = _pair_row(1, "synthetic/repository")
            serialized = json.dumps(row, sort_keys=True)
            duplicated = serialized[:-1] + ',"repository":"changed"}\n'
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(duplicated)

            with self.assertRaisesRegex(ValueError, "duplicate key"):
                run._pair_rows(path)

    def test_manifest_archives_and_repository_boundary_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {}
            for split, repository in (
                ("validation", "synthetic/validation"),
                ("dev_test", "synthetic/dev-test"),
            ):
                path = root / split / "pairs.jsonl.gz"
                archive_sha256, content_sha256 = _write_gzip_rows(
                    path, [_pair_row(1 if split == "validation" else 2, repository)]
                )
                outputs[split] = {
                    "path": "pairs.jsonl.gz",
                    "sha256": archive_sha256,
                    "content_sha256": content_sha256,
                    "pairs": 1,
                    "repositories": 1,
                }
            outputs["train"] = {
                "path": "pairs.jsonl.gz",
                "sha256": "8" * 64,
                "content_sha256": "9" * 64,
                "pairs": 1,
                "repositories": 1,
            }
            manifest = {
                "schema_version": 1,
                "source": {"revision": run.EXPECTED_SOURCE_REVISION},
                "split_unit": "repository",
                "outputs": outputs,
            }
            manifest_sha256 = _write_json(root / "manifest.json", manifest)
            with patch.object(run, "EXPECTED_PAIR_MANIFEST_SHA256", manifest_sha256):
                _, observed_sha256, splits = run._load_pair_inputs(root)
            self.assertEqual(observed_sha256, manifest_sha256)
            self.assertEqual(set(splits), {"validation", "dev_test"})

            dev_rows = [_pair_row(2, "synthetic/validation")]
            archive_sha256, content_sha256 = _write_gzip_rows(
                root / "dev_test" / "pairs.jsonl.gz", dev_rows
            )
            manifest["outputs"]["dev_test"].update(
                sha256=archive_sha256,
                content_sha256=content_sha256,
            )
            manifest_sha256 = _write_json(root / "manifest.json", manifest)
            with (
                patch.object(run, "EXPECTED_PAIR_MANIFEST_SHA256", manifest_sha256),
                self.assertRaisesRegex(ValueError, "repositories cross"),
            ):
                run._load_pair_inputs(root)

    def test_aggregate_metrics_do_not_persist_text_or_repository_identity(self):
        rows = (
            {
                "benign": "private synthetic benign value",
                "attack": "private synthetic attack value",
                "repository": "private/repository-name",
            },
        )
        metrics = run._metrics(
            rows,
            np.asarray([0.1]),
            np.asarray([0.9]),
            np.asarray([1]),
            np.asarray([2]),
            0.5,
        )
        serialized = json.dumps(metrics)
        self.assertNotIn("private synthetic", serialized)
        self.assertNotIn("private/repository-name", serialized)
        self.assertEqual(metrics["overall"]["both_correct"], 1.0)


if __name__ == "__main__":
    unittest.main()
