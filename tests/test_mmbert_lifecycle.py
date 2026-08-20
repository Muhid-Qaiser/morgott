from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mmbert_test_support import _training_data

from morgott.models.mmbert import train as mmbert_train


class SnapshotRetentionTests(unittest.TestCase):
    """Unconditional per-validation weights, so selection stays reversible.

    `best` and `alternates` keep three points out of roughly fifty. That is
    only safe when the selection signal is stable, and on 2026-08-06 arm 1 it
    was not -- adjacent validations ranged from 0.27 to 0.96.
    """

    @staticmethod
    def _row(updates: int, loss: float, *, interim: bool = False) -> dict:
        row = {
            "epoch": 1,
            "updates": updates,
            "selection_loss": loss,
            "validation_morgott_source_macro_bce": loss + 0.1,
            "validation_worst_source": "tensor_trust_raw",
            "validation_worst_source_bce": 2.6,
            "validation_morgott_by_source": {"tensor_trust_raw": 2.6},
        }
        if interim:
            row["interim"] = True
        return row

    def _save(self, directory: Path, updates: int, loss: float) -> dict:
        import torch

        row = self._row(updates, loss, interim=True)
        mmbert_train._save_snapshot(
            directory,
            row,
            training_identity={"schema_version": 4, "run_name": "test-run"},
            mode="frozen",
            head=torch.nn.Linear(2, 1),
            encoder=None,
            epoch=1,
            updates=updates,
        )
        return row

    def test_directory_is_a_sibling_so_a_relaunch_does_not_skip_the_arm(self):
        # A completed run directory is the durable "already trained" marker.
        self.assertEqual(
            mmbert_train._snapshot_dir(Path("runs"), "arm-4"),
            Path("runs/.arm-4.snapshots"),
        )

    def test_a_point_the_selection_rule_rejects_is_still_retained(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self._save(directory, 3_000, 0.273)
            self._save(directory, 5_500, 0.854)
            self.assertEqual(
                sorted(path.name for path in directory.glob("update-*.pt")),
                ["update-003000.pt", "update-005500.pt"],
            )

    def test_metrics_travel_with_the_weights(self):
        import torch

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self._save(directory, 500, 0.38)
            payload = torch.load(
                directory / "update-000500.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["updates"], 500)
            self.assertAlmostEqual(payload["loss"], 0.38)
            self.assertEqual(
                payload["metrics"]["validation_worst_source"], "tensor_trust_raw"
            )
            self.assertEqual(payload["training_identity"]["run_name"], "test-run")
            self.assertTrue(payload["head"])

    def test_a_replayed_update_overwrites_rather_than_duplicates(self):
        import torch

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self._save(directory, 500, 0.5)
            self._save(directory, 500, 0.4)
            self.assertEqual(len(list(directory.glob("update-*.pt"))), 1)
            payload = torch.load(
                directory / "update-000500.pt", map_location="cpu", weights_only=False
            )
            self.assertAlmostEqual(payload["loss"], 0.4)

    def test_no_temporary_survives_a_completed_write(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self._save(directory, 500, 0.5)
            self.assertEqual(
                [path.name for path in directory.iterdir()], ["update-000500.pt"]
            )

    def test_index_lists_only_points_whose_weights_exist(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            curve = [self._save(directory, 500, 0.38), self._row(1_000, 0.54)]
            mmbert_train._write_snapshot_index(directory, curve)
            index = json.loads((directory / "index.json").read_text())
            self.assertFalse(index["registered"])
            self.assertEqual([point["updates"] for point in index["points"]], [500])

    def test_index_supports_reselection_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            curve = [
                self._save(directory, 500, 0.38),
                self._save(directory, 3_000, 0.27),
                self._save(directory, 8_361, 0.10),
            ]
            mmbert_train._write_snapshot_index(directory, curve)
            index = json.loads((directory / "index.json").read_text())
            best = min(index["points"], key=lambda point: point["selection_loss"])
            self.assertEqual(best["updates"], 8_361)
            self.assertTrue((directory / best["file"]).is_file())

    def test_index_marks_the_pre_registered_comparison_point(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            row = self._save(directory, 17_000, 0.31)
            row["pre_registered_comparison"] = True
            mmbert_train._write_snapshot_index(directory, [row])
            point = json.loads((directory / "index.json").read_text())["points"][0]
            self.assertEqual(point["updates"], 17_000)
            self.assertTrue(point["pre_registered_comparison"])

    def test_fixed_comparison_is_retained_without_periodic_snapshots(self):
        args = mmbert_train._parser().parse_args(
            ["--validation-interval", "500", "--comparison-update", "17000"]
        )
        self.assertEqual(args.snapshot_every, 0)
        self.assertFalse(mmbert_train._should_snapshot(args, 16_500))
        self.assertTrue(mmbert_train._should_snapshot(args, 17_000))
        self.assertFalse(mmbert_train._should_snapshot(args, 17_500))

    def test_index_is_a_noop_when_snapshots_are_disabled(self):
        with tempfile.TemporaryDirectory() as raw:
            absent = Path(raw) / "absent"
            mmbert_train._write_snapshot_index(absent, [self._row(500, 0.3)])
            self.assertFalse(absent.exists())

    def test_default_is_off_so_the_other_arms_are_unchanged(self):
        args = mmbert_train._parser().parse_args(["--mode", "lora"])
        self.assertEqual(args.snapshot_every, 0)
        self.assertEqual(args.comparison_update, 0)

    def test_interval_must_be_a_multiple_of_the_validation_cadence(self):
        # Otherwise it would silently never fire: a snapshot only exists where
        # a validation row does.
        for argv in (
            ["--snapshot-every", "750", "--validation-interval", "500"],
            ["--snapshot-every", "500"],
            ["--snapshot-every", "-1", "--validation-interval", "500"],
            ["--comparison-update", "17000"],
            ["--comparison-update", "17001", "--validation-interval", "500"],
        ):
            with self.subTest(argv=argv):
                with patch.object(sys, "argv", ["train", *argv]):
                    with patch.object(
                        mmbert_train, "prepare_training_data"
                    ) as prepared:
                        with self.assertRaises(ValueError):
                            mmbert_train.main()
                        prepared.assert_not_called()

    def test_the_arm4_setting_is_accepted(self):
        args = mmbert_train._parser().parse_args(
            ["--validation-interval", "500", "--snapshot-every", "500"]
        )
        self.assertEqual(args.snapshot_every, 500)


class AtomicRunPublicationTests(unittest.TestCase):
    """One directory rename publishes the selected and alternate checkpoints."""

    @staticmethod
    def _alternates() -> dict:
        import torch

        return {
            "source_macro_only": {
                "loss": 0.0757,
                "epoch": 3,
                "updates": 23_000,
                "head": {"weight": torch.zeros(1)},
                "adapter": None,
                "encoder": None,
            }
        }

    @staticmethod
    def _data() -> mmbert_train.TrainingData:
        return _training_data(
            data_manifest_sha256="data",
            external_manifest_sha256="external",
        )

    def _publish(self, output: Path) -> Path:
        import torch

        pairs = output / "pairs.jsonl.gz"
        pairs.write_bytes(b"test pairs")
        args = mmbert_train._parser().parse_args(
            [
                "--mode",
                "frozen",
                "--output",
                str(output),
                "--pairs",
                str(pairs),
                "--run-name",
                "arm",
            ]
        )
        curve = [
            {
                "epoch": 3,
                "updates": 23_000,
                "selection_rule": "micro",
                "selection_loss": 0.0757,
                "pre_registered_comparison": False,
                "interim": True,
            }
        ]
        with (
            patch.object(
                mmbert_train,
                "_training_identity",
                return_value={"schema_version": 5},
            ),
            patch.object(mmbert_train, "source_provenance", return_value={}),
            patch.object(mmbert_train, "version", return_value="test"),
            patch.object(torch.cuda, "get_device_name", return_value="test-gpu"),
            patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            patch.object(torch.cuda, "max_memory_reserved", return_value=0),
        ):
            return mmbert_train._save_run(
                output,
                mode="frozen",
                seed=42,
                head=torch.nn.Linear(1, 1),
                encoder=torch.nn.Linear(1, 1),
                report={"canonical_rows": 128},
                curve=curve,
                alternates=self._alternates(),
                selected_epoch=3,
                selected_updates=23_000,
                args=args,
                data=self._data(),
                seconds=1.0,
                run_name="arm",
                optimizer_fused=False,
            )

    def test_success_publishes_one_complete_package(self):
        import torch

        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            destination = self._publish(output)
            alternate_directory = destination / "alternate-selections"

            self.assertEqual(destination, output / "arm")
            self.assertTrue((destination / "head.safetensors").is_file())
            self.assertTrue((destination / "result.json").is_file())
            self.assertTrue((alternate_directory / "source_macro_only.pt").is_file())
            index = json.loads(
                (alternate_directory / "index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["mode"], "frozen")
            self.assertEqual(index["rules"]["source_macro_only"]["updates"], 23_000)
            saved = torch.load(
                alternate_directory / "source_macro_only.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(saved["updates"], 23_000)
            self.assertFalse(list(output.glob(".arm-*")))

    def test_alternate_write_failure_leaves_no_completed_run(self):
        import torch

        def fail_after_partial_write(_candidate, path) -> None:
            Path(path).write_bytes(b"partial")
            raise OSError("forced alternate write failure")

        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            with (
                patch.object(torch, "save", side_effect=fail_after_partial_write),
                self.assertRaisesRegex(OSError, "forced alternate write failure"),
            ):
                self._publish(output)

            self.assertFalse((output / "arm").exists())
            self.assertFalse(list(output.glob(".arm-*")))

    def test_result_distinguishes_training_and_serving_attention(self):
        source = inspect.getsource(mmbert_train._save_run)
        self.assertIn(
            '"attention_implementation": ATTENTION_IMPLEMENTATION',
            source,
        )
        self.assertIn(
            '"serving_attention_implementation": ATTENTION_IMPLEMENTATION',
            source,
        )
        self.assertIn(
            '"training_attention_implementation": args.attention',
            source,
        )


class SelectedCheckpointProvenanceTests(unittest.TestCase):
    """Packaged tensors must name the exact validation state they came from."""

    @staticmethod
    def _row(*, epoch=3, updates=17_000, interim=True, comparison=True) -> dict:
        row = {
            "epoch": epoch,
            "updates": updates,
            "selection_rule": "source_macro",
            "selection_loss": 0.0615,
            "pre_registered_comparison": comparison,
        }
        if interim is not None:
            row["interim"] = interim
        return row

    def test_periodic_selection_records_update_and_independent_roles(self):
        value = mmbert_train._selected_checkpoint_provenance(
            [self._row()],
            selected_epoch=3,
            selected_updates=17_000,
            selection_rule="source_macro",
        )
        self.assertEqual(value["epoch"], 3)
        self.assertEqual(value["updates"], 17_000)
        self.assertEqual(value["selection_role"], "secondary")
        self.assertEqual(value["validation_point_role"], "periodic_validation")
        self.assertTrue(value["pre_registered_comparison"])

    def test_epoch_final_selection_has_an_unambiguous_point_role(self):
        value = mmbert_train._selected_checkpoint_provenance(
            [self._row(updates=25_083, interim=None, comparison=False)],
            selected_epoch=3,
            selected_updates=25_083,
            selection_rule="source_macro",
        )
        self.assertEqual(value["validation_point_role"], "epoch_final")
        self.assertFalse(value["pre_registered_comparison"])

    def test_missing_duplicate_or_mismatched_validation_point_fails_closed(self):
        row = self._row()
        cases = (
            ([], 17_000, "source_macro"),
            ([row, dict(row)], 17_000, "source_macro"),
            ([row], 17_000, "micro"),
            ([{**row, "pre_registered_comparison": 1}], 17_000, "source_macro"),
            ([row], 0, "source_macro"),
        )
        for curve, updates, rule in cases:
            with self.subTest(curve=curve, updates=updates, rule=rule):
                with self.assertRaises(ValueError):
                    mmbert_train._selected_checkpoint_provenance(
                        curve,
                        selected_epoch=3,
                        selected_updates=updates,
                        selection_rule=rule,
                    )

    def test_result_publication_and_call_site_carry_the_exact_update(self):
        save_source = inspect.getsource(mmbert_train._save_run)
        self.assertIn('"selected_updates": selected_updates', save_source)
        self.assertIn('"selected_checkpoint": selected_checkpoint', save_source)
        self.assertIn('"weights_provenance": {', save_source)
        train_source = inspect.getsource(mmbert_train.train)
        self.assertIn('selected_updates=best["updates"]', train_source)


class ResumeContractTests(unittest.TestCase):
    """Interim validation rows must not invalidate a mid-epoch resume."""

    UPDATES_PER_EPOCH = 8_361

    def test_resume_progress_contract(self):
        interim = [{"interim": True} for _ in range(3)]
        cases = (
            ("mid_epoch", 0, 1_500, interim, None, True),
            ("epoch_boundary", 1, 500, [{"interim": True}, {}], {}, True),
            ("missing_boundary", 1, 500, interim, {}, False),
        )
        for name, next_epoch, epoch_updates, curve, best, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    mmbert_train._resume_progress_valid(
                        next_epoch=next_epoch,
                        epoch_updates=epoch_updates,
                        epoch_canonical_seen=0,
                        epoch_loss_sum=0.0,
                        epoch_loss_count=epoch_updates,
                        updates=next_epoch * self.UPDATES_PER_EPOCH + epoch_updates,
                        curve=curve,
                        best=best,
                        epochs=3,
                        updates_per_epoch=self.UPDATES_PER_EPOCH,
                    ),
                    expected,
                )


class TrackioRunNameTests(unittest.TestCase):
    """One explicit identity must name every durable output of a run.

    The derived run name encodes the capacity arm only, so an ablation that
    changes the recipe but not the arm -- `--no-length-grouped` against arm 1 --
    produces an identical name. Trackio's `resume="never"` creates a fresh run
    id under that same name rather than uniquifying it, and the dashboard keys
    on the name, so the two curves merge into one unreadable series. That is
    why an ablation needs `--run-name`, not a Trackio-only alias.
    """

    def test_defaults_to_the_derived_name(self):
        args = mmbert_train._parser().parse_args([])
        self.assertIsNone(args.run_name)
        self.assertEqual(mmbert_train._resolved_run_name(args), "mmbert-lora-full-s42")

    def test_override_is_parsed(self):
        args = mmbert_train._parser().parse_args(
            ["--run-name", "mmbert-lora-full-s42-mb24-nolengthgroup"]
        )
        self.assertEqual(args.run_name, "mmbert-lora-full-s42-mb24-nolengthgroup")

    def test_path_like_or_ambiguous_names_are_rejected(self):
        for name in ("../escape", "/absolute", ".", "has space"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                args = mmbert_train._parser().parse_args(["--run-name", name])
                mmbert_train._resolved_run_name(args)

    def test_the_ablation_and_its_baseline_would_otherwise_collide(self):
        base = mmbert_train._parser().parse_args(["--mode", "lora"])
        abl = mmbert_train._parser().parse_args(
            ["--mode", "lora", "--no-length-grouped"]
        )

        def name(a):
            return mmbert_train._run_name(
                a.mode,
                a.seed,
                microbatch_size=a.microbatch_size,
            )

        self.assertEqual(name(base), name(abl))  # the collision this flag fixes

    def test_the_resolved_name_reaches_every_training_sink(self):
        source = Path(mmbert_train.__file__).read_text(encoding="utf-8")
        body = source[source.index("\ndef train(") :]
        self.assertIn("run_name=run_name", body)
        self.assertIn("checkpoint_name = run_name", body)
        self.assertIn("name=run_name,", source)
        self.assertNotIn("args.trackio_name", source)


@patch.dict(
    sys.modules,
    {"trackio": MagicMock(), "trackio.sqlite_storage": MagicMock()},
)
class TrackioRunClashTests(unittest.TestCase):
    """Starting over must not silently overlay a dead curve on a live one."""

    def _args(self, **overrides):
        args = mmbert_train._parser().parse_args([])
        args.trackio = True
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_existing_name_without_resume_is_refused(self):
        with patch(
            "trackio.sqlite_storage.SQLiteStorage.get_latest_run_record_by_name",
            return_value={"run_id": "abc", "run_name": "taken"},
        ):
            with self.assertRaises(ValueError) as caught:
                mmbert_train._RunTracker(
                    self._args(resume=False), run_name="taken", config={}
                )
        self.assertIn("--resume", str(caught.exception))
        self.assertIn("--run-name", str(caught.exception))

    def test_a_free_name_is_allowed_through(self):
        with patch(
            "trackio.sqlite_storage.SQLiteStorage.get_latest_run_record_by_name",
            return_value=None,
        ) as lookup:
            with patch("trackio.init", return_value=None) as init:
                mmbert_train._RunTracker(
                    self._args(resume=False), run_name="fresh", config={}
                )
        lookup.assert_called_once_with(mmbert_train.TRACKIO_PROJECT, "fresh")
        init.assert_called_once()
        self.assertEqual(init.call_args.kwargs["project"], mmbert_train.TRACKIO_PROJECT)

    def test_resume_skips_the_check(self):
        with patch(
            "trackio.sqlite_storage.SQLiteStorage.get_latest_run_record_by_name",
            side_effect=AssertionError("must not be consulted when resuming"),
        ):
            with patch("trackio.init", return_value=None) as init:
                mmbert_train._RunTracker(
                    self._args(resume=True), run_name="taken", config={}
                )
        init.assert_called_once()
        self.assertEqual(init.call_args.kwargs["resume"], "must")

    def test_a_tracking_lookup_failure_refuses_to_risk_a_duplicate(self):
        with patch(
            "trackio.sqlite_storage.SQLiteStorage.get_latest_run_record_by_name",
            side_effect=RuntimeError("trackio changed"),
        ):
            with patch("trackio.init", return_value=None) as init:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "cannot prove the Trackio run name is unused",
                ):
                    mmbert_train._RunTracker(
                        self._args(resume=False),
                        run_name="whatever",
                        config={},
                    )
        init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
