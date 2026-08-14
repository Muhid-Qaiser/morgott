from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from morgott.models.mmbert import evaluate as mmbert_evaluate


class SnapshotEvaluationTests(unittest.TestCase):
    @staticmethod
    def _metrics(*, update: int, interim: bool = True, comparison: bool = False):
        result = {
            "epoch": 1,
            "updates": update,
            "selection_loss": 0.25,
            "pre_registered_comparison": comparison,
        }
        if interim:
            result["interim"] = True
        return result

    @staticmethod
    def _write_snapshot(
        path: Path,
        *,
        identity: dict,
        metrics: dict,
        head: dict,
        adapter=None,
        encoder=None,
    ) -> None:
        import torch

        torch.save(
            {
                "loss": metrics["selection_loss"],
                "epoch": metrics["epoch"],
                "updates": metrics["updates"],
                "head": head,
                "adapter": adapter,
                "encoder": encoder,
                "training_identity": identity,
                "metrics": metrics,
            },
            path,
        )

    def test_frozen_snapshot_requires_identity_and_restores_exact_state(self):
        import torch

        identity = {"schema_version": 4, "run_name": "arm-6"}
        metrics = self._metrics(update=500, comparison=True)
        result = {
            "adaptation": "frozen",
            "training_identity": identity,
            "training": {"curve": [metrics]},
        }
        head = torch.nn.Linear(2, 1)
        selected = {
            name: torch.full_like(value, 0.75)
            for name, value in head.state_dict().items()
        }
        encoder = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "update-000500.pt"
            self._write_snapshot(
                snapshot,
                identity=identity,
                metrics=metrics,
                head=selected,
            )
            checkpoint = mmbert_evaluate._load_snapshot(
                snapshot,
                result=result,
                encoder=encoder,
                head=head,
            )

            self.assertEqual(checkpoint["update"], 500)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["role"], "pre_registered_comparison")
            self.assertEqual(
                checkpoint["sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest()
            )
            for name, value in head.state_dict().items():
                self.assertTrue(torch.equal(value, selected[name]))
            self.assertNotEqual(
                mmbert_evaluate._evaluation_model_sha256(
                    "a" * 64,
                    checkpoint["sha256"],
                ),
                "a" * 64,
            )

            mismatched = snapshot.with_name("update-000501.pt")
            changed_identity = {**identity, "run_name": "another-run"}
            changed_metrics = self._metrics(update=501)
            self._write_snapshot(
                mismatched,
                identity=changed_identity,
                metrics=changed_metrics,
                head=selected,
            )
            with self.assertRaisesRegex(ValueError, "training identity"):
                mmbert_evaluate._load_snapshot(
                    mismatched,
                    result=result,
                    encoder=encoder,
                    head=head,
                )

    def test_lora_snapshot_checks_and_restores_every_adapter_tensor(self):
        import torch

        class Encoder:
            def __init__(self):
                self.adapter = {"layer.lora_A.weight": torch.zeros(2, 2)}

            def eval(self):
                return self

        peft = types.ModuleType("peft")
        peft.get_peft_model_state_dict = lambda encoder: {
            name: value.detach().clone() for name, value in encoder.adapter.items()
        }

        def restore(encoder, state):
            encoder.adapter = {
                name: value.detach().clone() for name, value in state.items()
            }
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        peft.set_peft_model_state_dict = restore
        identity = {"schema_version": 4, "run_name": "lora"}
        metrics = self._metrics(update=1500)
        head = torch.nn.Linear(2, 1)
        selected_head = {
            name: value.detach().clone() for name, value in head.state_dict().items()
        }
        selected_adapter = {"layer.lora_A.weight": torch.ones(2, 2)}
        encoder = Encoder()
        result = {
            "adaptation": "lora",
            "training_identity": identity,
            "training": {"curve": [metrics]},
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(sys.modules, {"peft": peft}),
        ):
            snapshot = Path(temporary) / "update-001500.pt"
            self._write_snapshot(
                snapshot,
                identity=identity,
                metrics=metrics,
                head=selected_head,
                adapter=selected_adapter,
            )
            mmbert_evaluate._load_snapshot(
                snapshot,
                result=result,
                encoder=encoder,
                head=head,
            )

        self.assertTrue(
            torch.equal(
                encoder.adapter["layer.lora_A.weight"],
                selected_adapter["layer.lora_A.weight"],
            )
        )

    def test_snapshot_cli_uses_a_distinct_update_output_by_default(self):
        run = Path("runs/arm-6")
        snapshot = Path("runs/.arm-6.snapshots/update-005000.pt")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "evaluate",
                    str(run),
                    "--snapshot",
                    str(snapshot),
                    "--tokenizer-workers",
                    "5",
                ],
            ),
            patch.object(
                mmbert_evaluate, "evaluate", return_value=Path("done")
            ) as run_eval,
            patch.object(
                mmbert_evaluate,
                "_read_run_result",
                return_value={"max_tokens": 512},
            ),
            patch("builtins.print"),
        ):
            self.assertEqual(mmbert_evaluate.main(), 0)

        self.assertEqual(run_eval.call_args.kwargs["snapshot"], snapshot)
        self.assertEqual(
            run_eval.call_args.kwargs["output"],
            run / "evaluation-update-5000",
        )
        self.assertEqual(
            run_eval.call_args.kwargs["score_journal"],
            run / ".evaluation-update-5000.score-journal",
        )
        self.assertEqual(run_eval.call_args.kwargs["tokenizer_workers"], 5)


class TokenizerExecutionTests(unittest.TestCase):
    def test_auto_default_overrides_stale_serial_state_with_cgroup_budget(self):
        with (
            patch.dict(
                os.environ,
                {"TOKENIZERS_PARALLELISM": "false", "RAYON_NUM_THREADS": "128"},
            ),
            patch.object(mmbert_evaluate, "_usable_cpus", return_value=13),
        ):
            value = mmbert_evaluate._configure_tokenizer_execution()
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "true")
            self.assertEqual(os.environ["RAYON_NUM_THREADS"], "13")

        self.assertEqual(
            value,
            {
                "parallelism": True,
                "rayon_threads": 13,
                "selection": "automatic_cgroup_budget",
            },
        )

    def test_one_worker_is_the_explicit_serial_opt_out(self):
        with patch.dict(os.environ, {}, clear=True):
            value = mmbert_evaluate._configure_tokenizer_execution(1)
            self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")
            self.assertEqual(os.environ["RAYON_NUM_THREADS"], "1")
        self.assertFalse(value["parallelism"])
        self.assertEqual(value["selection"], "explicit")

    def test_invalid_worker_counts_fail_before_model_loading(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mmbert_evaluate._configure_tokenizer_execution(value)

    def test_execution_setting_does_not_change_score_journal_identity(self):
        with patch.dict(os.environ, {}, clear=True):
            before = mmbert_evaluate._scoring_sha256()
            mmbert_evaluate._configure_tokenizer_execution(1)
            serial = mmbert_evaluate._scoring_sha256()
            mmbert_evaluate._configure_tokenizer_execution(8)
            parallel = mmbert_evaluate._scoring_sha256()
        self.assertEqual(before, serial)
        self.assertEqual(serial, parallel)

    def test_uv_lock_changes_score_journal_identity(self):
        lock = Path(mmbert_evaluate.__file__).resolve().parents[4] / "uv.lock"
        lock_sha256 = ["0" * 64]

        def digest(path):
            return lock_sha256[0] if Path(path).resolve() == lock else "1" * 64

        with patch.object(mmbert_evaluate, "file_sha256", side_effect=digest):
            before = mmbert_evaluate._scoring_sha256()
            lock_sha256[0] = "2" * 64
            after = mmbert_evaluate._scoring_sha256()

        self.assertNotEqual(before, after)


class BaseModelIdentityTests(unittest.TestCase):
    def test_verified_base_files_bind_current_but_not_legacy_model_identity(self):
        contents = {
            "config.json": b"reviewed config bytes",
            "pytorch_model.bin": b"reviewed encoder bytes",
            "special_tokens_map.json": b"reviewed special-token bytes",
            "tokenizer_config.json": b"reviewed tokenizer config bytes",
            "tokenizer.json": b"reviewed tokenizer bytes",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name, value in contents.items():
                path = root / name
                path.write_bytes(value)
                paths[name] = path
            base_model = {
                "id": mmbert_evaluate.MODEL_ID,
                "revision": mmbert_evaluate.MODEL_REVISION,
                "config_sha256": hashlib.sha256(contents["config.json"]).hexdigest(),
                "pytorch_model_sha256": hashlib.sha256(
                    contents["pytorch_model.bin"]
                ).hexdigest(),
                "special_tokens_map_sha256": hashlib.sha256(
                    contents["special_tokens_map.json"]
                ).hexdigest(),
                "tokenizer_config_sha256": hashlib.sha256(
                    contents["tokenizer_config.json"]
                ).hexdigest(),
                "tokenizer_json_sha256": hashlib.sha256(
                    contents["tokenizer.json"]
                ).hexdigest(),
            }
            registry = root / "model-artifacts.json"
            registry.write_text(
                json.dumps({"schema_version": 2, "base_model": base_model}),
                encoding="utf-8",
            )

            def download(repo_id, filename, *, revision):
                self.assertEqual(repo_id, mmbert_evaluate.MODEL_ID)
                self.assertEqual(revision, mmbert_evaluate.MODEL_REVISION)
                return str(paths[filename])

            with (
                patch.object(mmbert_evaluate, "_MODEL_REGISTRY", registry),
                patch("huggingface_hub.hf_hub_download", side_effect=download),
            ):
                verified = mmbert_evaluate._verified_base_model_identity()
                self.assertEqual(verified, base_model)

                legacy = mmbert_evaluate._evaluation_model_sha256(
                    "1" * 64,
                    "2" * 64,
                )
                current = mmbert_evaluate._evaluation_model_sha256(
                    "1" * 64,
                    "2" * 64,
                    base_model=verified,
                )
                self.assertEqual(
                    legacy,
                    "b484f050389469857e7773f0766a002226a2e3203925056e9a08b66a389047c9",
                )
                self.assertNotEqual(current, legacy)
                for field in (
                    "config_sha256",
                    "pytorch_model_sha256",
                    "special_tokens_map_sha256",
                    "tokenizer_config_sha256",
                    "tokenizer_json_sha256",
                ):
                    with self.subTest(identity_field=field):
                        changed = {**verified, field: "f" * 64}
                        self.assertNotEqual(
                            current,
                            mmbert_evaluate._evaluation_model_sha256(
                                "1" * 64,
                                "2" * 64,
                                base_model=changed,
                            ),
                        )

                for name, path in paths.items():
                    with self.subTest(cache_file=name):
                        path.write_bytes(contents[name] + b" changed")
                        with self.assertRaisesRegex(
                            ValueError,
                            "base model hash mismatch",
                        ):
                            mmbert_evaluate._verified_base_model_identity()
                        path.write_bytes(contents[name])


class EvaluationInputIdentityTests(unittest.TestCase):
    def test_input_identity_brackets_panel_assembly(self):
        source = inspect.getsource(mmbert_evaluate.evaluate)
        capture = source.index("input_sha256 = _evaluation_input_sha256")
        first_read = source.index("views = routing_views")
        panel = source.index("population_rows = {")
        recheck = source.index("_require_unchanged_evaluation_inputs", panel)

        self.assertLess(capture, first_read)
        self.assertLess(panel, recheck)

    def test_changed_input_fails_the_publication_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            external_dir = root / "external"
            data_dir.mkdir()
            external_dir.mkdir()
            paths = (
                data_dir / "manifest.json",
                external_dir / "manifest.json",
                root / "pairs.jsonl.gz",
                root / "additional-pairs.jsonl.gz",
            )
            for path in paths:
                path.write_bytes(b"original")
            inputs = {
                "data_dir": data_dir,
                "external_dir": external_dir,
                "pairs": paths[2],
                "additional_pairs": paths[3],
            }
            expected = mmbert_evaluate._evaluation_input_sha256(**inputs)

            for path in paths:
                with self.subTest(path=path.name):
                    path.write_bytes(b"changed")
                    with self.assertRaisesRegex(
                        ValueError,
                        "evaluation inputs changed during evaluation",
                    ):
                        mmbert_evaluate._require_unchanged_evaluation_inputs(
                            expected,
                            **inputs,
                        )
                    path.write_bytes(b"original")


if __name__ == "__main__":
    unittest.main()
