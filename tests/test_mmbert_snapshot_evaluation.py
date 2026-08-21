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


class JournalScoringIdentityTests(unittest.TestCase):
    """The journal key must move with scoring sources and with nothing else.

    ``_scoring_sha256`` stays the published whole-file evaluation identity;
    the journal key narrows to the score-producing call chain so a report or
    CLI fix between a crash and a resume cannot brick hours of GPU scoring.
    """

    def _tree(self) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        module = root / "src" / "morgott" / "models" / "mmbert"
        module.mkdir(parents=True)
        real = Path(mmbert_evaluate.__file__).resolve()
        for name in ("evaluate.py", "core.py", "data.py", "score_journal.py"):
            (module / name).write_text(
                real.with_name(name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        (root / "src" / "morgott" / "normalization.py").write_text(
            (real.parents[2] / "normalization.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        pins = dict.fromkeys(mmbert_evaluate._JOURNAL_SCORING_PINS, "9.9.9")
        self._write_lock(root, {**pins, "ruff": "1.0.0"})
        return root, module

    @staticmethod
    def _write_lock(root: Path, versions: dict) -> None:
        entries = "\n".join(
            f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
            for name, pinned in sorted(versions.items())
            for version in ([pinned] if isinstance(pinned, str) else pinned)
        )
        (root / "uv.lock").write_text(f"version = 1\n\n{entries}", encoding="utf-8")

    @staticmethod
    def _edit(path: Path, old: str, new: str) -> None:
        source = path.read_text(encoding="utf-8")
        assert source.count(old) == 1, old
        path.write_text(source.replace(old, new), encoding="utf-8")

    def _digests(self, module: Path) -> tuple[str, str]:
        with patch.object(mmbert_evaluate, "__file__", str(module / "evaluate.py")):
            return (
                mmbert_evaluate._journal_scoring_sha256(),
                mmbert_evaluate._scoring_sha256(),
            )

    def test_real_tree_digest_is_computable_and_binds_the_context_cap(self):
        # Fails closed right here if the symbol list goes stale in this repo.
        narrow = mmbert_evaluate._journal_scoring_sha256(512)
        self.assertRegex(narrow, r"\A[0-9a-f]{64}\Z")
        self.assertNotEqual(narrow, mmbert_evaluate._journal_scoring_sha256(1024))

    def test_listed_symbols_cover_the_scoring_call_chain(self):
        listed = {
            f"{file_name}:{name}"
            for file_name, names in mmbert_evaluate._JOURNAL_SCORING_SOURCES
            for name in names
        }
        self.assertLessEqual(
            {
                "core.py:batch_logits",
                "core.py:file_sha256",
                "core.py:pool",
                "core.py:score_logits",
                "core.py:score_texts",
                "data.py:batches",
                "evaluate.py:_assert_restored_state",
                "evaluate.py:_load_snapshot",
                "evaluate.py:_padded_token_groups",
                "evaluate.py:_restore_snapshot_state",
                "evaluate.py:_score",
                "evaluate.py:_score_single_texts",
                "evaluate.py:_sigmoid",
                "evaluate.py:_validate_state",
                "evaluate.py:_verified_base_model_identity",
            },
            listed,
        )

    def test_unrelated_edit_keeps_the_journal_key_but_moves_the_full_hash(self):
        _, module = self._tree()
        journal_before, full_before = self._digests(module)
        self._edit(
            module / "evaluate.py",
            "without promoting it into authorization",
            "with an unrelated docstring edit",
        )
        journal_after, full_after = self._digests(module)
        self.assertEqual(journal_before, journal_after)
        self.assertNotEqual(full_before, full_after)

    def test_editing_a_listed_function_body_moves_the_journal_key(self):
        for file_name, old, new in (
            ("evaluate.py", "positive = values >= 0", "positive = values > 0.0"),
            ("core.py", "values = batch_logits(", "values = 2 * batch_logits("),
        ):
            with self.subTest(file_name=file_name):
                _, module = self._tree()
                before, _ = self._digests(module)
                self._edit(module / file_name, old, new)
                after, _ = self._digests(module)
                self.assertNotEqual(before, after)

    def test_editing_the_journal_module_moves_the_journal_key(self):
        # journal.scores() is the returned panel, so score_journal.py's
        # read/write semantics bind the key whole-file.
        _, module = self._tree()
        before, _ = self._digests(module)
        self._edit(
            module / "score_journal.py",
            "not np.isfinite(values).all()",
            "not np.isfinite(values).any()",
        )
        after, _ = self._digests(module)
        self.assertNotEqual(before, after)

    def test_journal_spec_binds_the_narrow_scoring_key(self):
        # The one-line wiring the whole change hangs on: reverting the spec
        # to the whole-file identity must fail here.
        spec = mmbert_evaluate._journal_spec(
            model_sha256="0" * 64,
            panel_sha256="1" * 64,
            rows=3,
            batch_size=2,
            evaluation_max_tokens=512,
        )
        self.assertEqual(
            spec.scoring_sha256, mmbert_evaluate._journal_scoring_sha256(512)
        )
        self.assertNotEqual(spec.scoring_sha256, mmbert_evaluate._scoring_sha256(512))
        self.assertEqual(spec.columns, ("score",))

    def test_decorating_a_listed_function_moves_the_journal_key(self):
        _, module = self._tree()
        before, _ = self._digests(module)
        self._edit(
            module / "evaluate.py",
            "def _sigmoid(",
            "@some_decorator\ndef _sigmoid(",
        )
        after, _ = self._digests(module)
        self.assertNotEqual(before, after)

    def test_a_missing_listed_symbol_fails_closed(self):
        _, module = self._tree()
        self._edit(module / "evaluate.py", "def _sigmoid(", "def _renamed_sigmoid(")
        with patch.object(mmbert_evaluate, "__file__", str(module / "evaluate.py")):
            with self.assertRaisesRegex(ValueError, "_sigmoid"):
                mmbert_evaluate._journal_scoring_sha256()

    def test_only_scoring_stack_pins_move_the_journal_key(self):
        root, module = self._tree()
        before, _ = self._digests(module)
        pins = dict.fromkeys(mmbert_evaluate._JOURNAL_SCORING_PINS, "9.9.9")

        self._write_lock(root, {**pins, "ruff": "2.0.0"})
        unrelated, _ = self._digests(module)
        self.assertEqual(before, unrelated)

        self._write_lock(root, {**pins, "torch": "8.8.8", "ruff": "1.0.0"})
        bumped, _ = self._digests(module)
        self.assertNotEqual(before, bumped)

        del pins["torch"]
        self._write_lock(root, {**pins, "ruff": "1.0.0"})
        with patch.object(mmbert_evaluate, "__file__", str(module / "evaluate.py")):
            with self.assertRaisesRegex(ValueError, "uv.lock"):
                mmbert_evaluate._journal_scoring_sha256()

    def test_native_numeric_stack_pins_bind_when_present(self):
        # triton and nvidia-* lock entries move bf16 kernel behavior without
        # touching the named pins; absent entries (CPU-only locks) are fine.
        root, module = self._tree()
        pins = dict.fromkeys(mmbert_evaluate._JOURNAL_SCORING_PINS, "9.9.9")
        absent, _ = self._digests(module)

        native = {"nvidia-cublas": "1.0.0", "triton": "3.0.0", "ruff": "1.0.0"}
        self._write_lock(root, {**pins, **native})
        present, _ = self._digests(module)
        self.assertNotEqual(absent, present)

        self._write_lock(root, {**pins, **native, "triton": "3.0.1"})
        bumped, _ = self._digests(module)
        self.assertNotEqual(present, bumped)

    def test_every_duplicate_pin_entry_binds_the_journal_key(self):
        # Forked marker resolution can leave several uv.lock entries for one
        # pinned package; bumping any of them must move the key.
        root, module = self._tree()
        pins = dict.fromkeys(mmbert_evaluate._JOURNAL_SCORING_PINS, "9.9.9")

        self._write_lock(root, {**pins, "torch": ["1.0.0", "9.9.9"]})
        forked, _ = self._digests(module)

        self._write_lock(root, {**pins, "torch": ["2.0.0", "9.9.9"]})
        bumped, _ = self._digests(module)
        self.assertNotEqual(forked, bumped)


class BlockTokenizationEquivalenceTests(unittest.TestCase):
    """Block tokenization must be bit-identical to per-minibatch calls."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        from morgott.models.mmbert.core import MODEL_ID, MODEL_REVISION

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, local_files_only=True
            )
        except OSError:
            # Cache miss: fetch the pinned revision so CI still exercises the
            # real tokenizer; skip only when the Hub is unreachable (offline).
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION
                )
            except OSError as error:
                if os.environ.get("CI"):
                    # A silent skip would let CI go green without ever
                    # verifying the bit-identity claim; fail loudly there.
                    raise
                raise unittest.SkipTest(
                    f"pinned tokenizer unavailable offline: {error}"
                ) from error

    def test_padded_token_groups_match_per_minibatch_padded_tokenization(self):
        import random

        import torch

        from morgott.normalization import strict_normalize

        rng = random.Random(20260820)
        words = [
            "ignore",
            "previous",
            "instructions",
            "reveal",
            "the",
            "system",
            "prompt",
            "Σοφός",
            "текст",
            "短い",
            "finance",
            "x" * 40,
        ]
        texts = []
        for index in range(203):
            if index % 29 == 0:
                texts.append("")
            elif index % 13 == 0:
                texts.append("long " * rng.randint(400, 700))
            else:
                texts.append(
                    " ".join(rng.choice(words) for _ in range(rng.randint(1, 120)))
                )

        for batch_size, max_tokens in ((8, 512), (24, 1024), (5, 1024)):
            with self.subTest(batch_size=batch_size, max_tokens=max_tokens):
                groups = list(
                    mmbert_evaluate._padded_token_groups(
                        self.tokenizer,
                        texts,
                        batch_size=batch_size,
                        max_tokens=max_tokens,
                    )
                )
                starts = range(0, len(texts), batch_size)
                self.assertEqual(len(groups), len(starts))
                for group, start in zip(groups, starts, strict=True):
                    expected = self.tokenizer(
                        [
                            strict_normalize(text)
                            for text in texts[start : start + batch_size]
                        ],
                        add_special_tokens=True,
                        max_length=max_tokens,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                    )
                    self.assertEqual(sorted(group), sorted(expected))
                    for key in expected:
                        self.assertTrue(
                            torch.equal(group[key], expected[key]),
                            msg=f"{key} differs at row {start}",
                        )


if __name__ == "__main__":
    unittest.main()
