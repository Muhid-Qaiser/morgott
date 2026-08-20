from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mmbert_test_support import _training_data

from morgott.models.mmbert import data as mmbert_data
from morgott.models.mmbert import train as mmbert_train
from morgott.models.mmbert.train import (
    BalancedIndexCycle,
    PairIndexCycle,
    _bce_from_logits,
    _LengthGroupedCycle,
    _lr_multiplier,
    length_grouped_batches,
)


class TrainingExecutionGuardTests(unittest.TestCase):
    def tearDown(self):
        mmbert_train._verified_fa2_variant.cache_clear()

    def test_metric_window_has_one_host_transfer_per_drain_contract(self):
        import torch

        window = mmbert_train._MetricWindow()
        window.add(
            {"total_loss": torch.tensor(2.0), "primary": torch.tensor(0.5)},
            examples=12,
        )
        window.add(
            {"total_loss": torch.tensor(4.0), "primary": torch.tensor(1.5)},
            examples=10,
        )
        totals, latest, updates, examples = window.drain()
        self.assertEqual(totals, {"primary": 2.0, "total_loss": 6.0})
        self.assertEqual(latest, {"primary": 1.5, "total_loss": 4.0})
        self.assertEqual((updates, examples), (2, 22))
        self.assertEqual(window.updates, 0)

    def test_compiled_backward_autocast_is_forced_off(self):
        from torch._functorch import config

        original = config.backward_pass_autocast
        try:
            config.backward_pass_autocast = "same_as_forward"
            mmbert_train._configure_compiled_backward_autocast()
            self.assertEqual(config.backward_pass_autocast, "off")
        finally:
            config.backward_pass_autocast = original

    def test_spawn_context_is_used_for_parallel_cache_warm(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[len(text)] for text in texts]}

        class Pool:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def imap(self, function, values, chunksize):
                return map(function, values)

        class Context:
            def Pool(self, workers):
                self.workers = workers
                return Pool()

        context = Context()
        cache = mmbert_train._EncodingCache(Tokenizer())
        with patch("multiprocessing.get_context", return_value=context) as get_context:
            cache.warm((f"synthetic-{index}" for index in range(4097)), workers=2)
        get_context.assert_called_once_with("spawn")
        self.assertEqual(context.workers, 2)
        self.assertEqual(len(cache), 4097)

    def test_output_collisions_are_rejected_without_touching_cuda(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            args = mmbert_train._parser().parse_args(
                ["--output", str(output), "--run-name", "explicit-run"]
            )
            (output / "explicit-run").mkdir()
            with self.assertRaisesRegex(FileExistsError, "existing output"):
                mmbert_train._preflight_execution(
                    args, mmbert_train._resolved_run_name(args), check_cuda=False
                )

    def test_resume_requires_the_explicit_names_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            args = mmbert_train._parser().parse_args(
                [
                    "--output",
                    raw,
                    "--run-name",
                    "explicit-run",
                    "--resume",
                ]
            )
            with self.assertRaisesRegex(FileNotFoundError, "explicit-run"):
                mmbert_train._preflight_execution(
                    args, mmbert_train._resolved_run_name(args), check_cuda=False
                )

    def test_fa2_variant_is_local_only_and_digest_pinned(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw)
            variant = (
                cache
                / "kernels--kernels-community--flash-attn2"
                / "snapshots"
                / "239bb21bd566f598d7e2228eab9788b0a9239b2d"
                / Path(mmbert_train.FA2_KERNEL_BINARY).parent
            )
            variant.mkdir(parents=True)
            executable = variant / Path(mmbert_train.FA2_KERNEL_BINARY).name
            executable.write_bytes(b"pinned")
            with (
                patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)),
                patch.object(
                    mmbert_train,
                    "file_sha256",
                    return_value=mmbert_train.FA2_KERNEL_SHA256,
                ),
            ):
                self.assertEqual(mmbert_train._verified_fa2_variant(), variant)

    def test_fa2_variant_rejects_an_unrecorded_executable(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw)
            executable = (
                cache
                / "kernels--kernels-community--flash-attn2"
                / "snapshots"
                / "239bb21bd566f598d7e2228eab9788b0a9239b2d"
                / mmbert_train.FA2_KERNEL_BINARY
            )
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"different")
            with (
                patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)),
                patch.object(mmbert_train, "file_sha256", return_value="0" * 64),
                self.assertRaisesRegex(RuntimeError, "digest mismatch"),
            ):
                mmbert_train._verified_fa2_variant()

    def test_head_contract_is_json_stable(self):
        head = mmbert_train._head_contract()
        self.assertEqual(
            head,
            {
                "outputs": 1,
                "columns": {"0": "instruction_subversion"},
                "primary_column": 0,
                "architecture": "legacy_sequential_binary_v1",
            },
        )
        json.dumps({"head_contract": head})

    def test_partial_pass_is_not_published(self):
        def factory():
            for index in range(5):
                yield {"id": str(index)}

        stream = mmbert_train._EpochStream(factory, expected_rows=5)
        for row in stream:
            if row["id"] == "2":
                break
        self.assertFalse(stream.cached)
        self.assertEqual([row["id"] for row in stream], ["0", "1", "2", "3", "4"])

    def test_population_drift_is_rejected(self):
        def factory():
            for index in range(4):
                yield {"id": str(index)}

        stream = mmbert_train._EpochStream(factory, expected_rows=5)
        with self.assertRaisesRegex(ValueError, "canonical epoch stream changed"):
            list(stream)


class LearningRateScheduleTests(unittest.TestCase):
    TOTAL = 25_083
    WARMUP = 1_254

    def _multiplier(self, step: int) -> float:
        return _lr_multiplier(
            step, total_updates=self.TOTAL, warmup_updates=self.WARMUP
        )

    def test_warmup_rises_from_near_zero_to_exactly_one(self):
        self.assertGreater(self._multiplier(0), 0.0)
        self.assertLess(self._multiplier(0), 0.01)
        self.assertAlmostEqual(self._multiplier(self.WARMUP - 1), 1.0, places=9)

    def test_warmup_is_monotonic(self):
        values = [self._multiplier(step) for step in range(0, self.WARMUP, 37)]
        self.assertEqual(values, sorted(values))

    def test_decay_is_monotonic_and_reaches_zero(self):
        values = [
            self._multiplier(step) for step in range(self.WARMUP, self.TOTAL, 401)
        ]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertAlmostEqual(self._multiplier(self.TOTAL), 0.0, places=9)

    def test_steps_past_the_horizon_clamp_rather_than_turn_negative(self):
        """A resumed run can overshoot `expected_updates` on a short final epoch."""
        self.assertGreaterEqual(self._multiplier(self.TOTAL * 2), 0.0)

    def test_zero_warmup_still_decays(self):
        first = _lr_multiplier(0, total_updates=100, warmup_updates=0)
        last = _lr_multiplier(99, total_updates=100, warmup_updates=0)
        self.assertAlmostEqual(first, 1.0, places=9)
        self.assertLess(last, 0.01)


class LengthGroupingTests(unittest.TestCase):
    def _rows(self, count: int = 4_000) -> list[dict]:
        rng = np.random.default_rng(11)
        return [{"text": "x" * int(n)} for n in rng.integers(5, 4_000, size=count)]

    def _spans(self, batches: list[list[dict]]) -> list[int]:
        return [
            max(len(row["text"]) for row in batch)
            - min(len(row["text"]) for row in batch)
            for batch in batches
        ]

    def test_batches_conserve_every_row_and_the_batch_count(self):
        rows = self._rows()
        plain = list(mmbert_data.batches(iter(rows), 128))
        grouped = list(
            length_grouped_batches(
                iter(rows),
                128,
                key=lambda row: len(row["text"]),
                factor=8,
                rng=np.random.default_rng(3),
            )
        )
        self.assertEqual(len(grouped), len(plain))
        self.assertEqual(
            sorted(id(row) for batch in grouped for row in batch),
            sorted(id(row) for batch in plain for row in batch),
        )

    def test_grouping_collapses_the_within_batch_length_span(self):
        rows = self._rows()
        plain = self._spans(list(mmbert_data.batches(iter(rows), 128)))
        grouped = self._spans(
            list(
                length_grouped_batches(
                    iter(rows),
                    128,
                    key=lambda row: len(row["text"]),
                    factor=8,
                    rng=np.random.default_rng(3),
                )
            )
        )
        # Padding is charged at the longest row in a microbatch, so the span is
        # what the 27% token saving actually comes from.
        self.assertLess(np.median(grouped), np.median(plain) / 5)

    def test_batch_order_is_shuffled_not_short_to_long(self):
        rows = self._rows()
        longest = [
            max(len(row["text"]) for row in batch)
            for batch in length_grouped_batches(
                iter(rows),
                64,
                key=lambda row: len(row["text"]),
                factor=16,
                rng=np.random.default_rng(3),
            )
        ]
        # Ascending order would mean batch content correlates with training
        # time, and in this corpus length tracks source.
        self.assertNotEqual(longest, sorted(longest))

    def test_cycle_wrapper_dispenses_exact_batches_within_range(self):
        rows = self._rows(2_000)
        labels = np.asarray([index % 2 for index in range(2_000)], dtype=np.int64)
        cycle = _LengthGroupedCycle(
            BalancedIndexCycle(labels, seed=1),
            rows,
            key=lambda row: len(row["text"]),
            factor=8,
            seed=5,
        )
        for _ in range(24):
            batch = cycle.take(32)
            self.assertEqual(len(batch), 32)
            self.assertTrue(all(0 <= index < 2_000 for index in batch))

    def test_cycle_wrapper_preserves_class_balance_when_length_tracks_label(self):
        labels = np.asarray([0] * 1_000 + [1] * 1_000, dtype=np.int64)
        rows = [
            {"text": "x" * (index + 1), "label": int(label)}
            for index, label in enumerate(labels)
        ]
        cycle = _LengthGroupedCycle(
            BalancedIndexCycle(labels, seed=1),
            rows,
            key=lambda row: len(row["text"]),
            group=lambda row: row["label"],
            factor=8,
            seed=5,
        )

        for _ in range(24):
            batch = cycle.take(32)
            self.assertEqual(Counter(labels[batch]), Counter({0: 16, 1: 16}))

    def test_cycle_wrapper_state_round_trips_mid_megabatch(self):
        rows = self._rows(2_000)
        labels = np.asarray([index % 2 for index in range(2_000)], dtype=np.int64)
        for row, label in zip(rows, labels, strict=True):
            row["label"] = int(label)

        def build() -> _LengthGroupedCycle:
            return _LengthGroupedCycle(
                BalancedIndexCycle(labels, seed=1),
                rows,
                key=lambda row: len(row["text"]),
                group=lambda row: row["label"],
                factor=8,
                seed=5,
            )

        original = build()
        # Stop part-way through a megabatch: the undispensed queue is the part
        # a naive state_dict would drop.
        for _ in range(3):
            original.take(32)
        state = original.state_dict()
        expected = [original.take(32) for _ in range(9)]

        restored = build()
        restored.load_state_dict(state)
        self.assertEqual([restored.take(32) for _ in range(9)], expected)

    def test_cycle_wrapper_rejects_foreign_state(self):
        rows = self._rows(200)
        labels = np.asarray([index % 2 for index in range(200)], dtype=np.int64)
        cycle = _LengthGroupedCycle(
            BalancedIndexCycle(labels, seed=1),
            rows,
            key=lambda row: len(row["text"]),
            factor=4,
            seed=5,
        )
        with self.assertRaisesRegex(ValueError, "length-grouped cycle state"):
            cycle.load_state_dict({"schema_version": 99})

    def test_pair_wrapper_costs_a_pair_at_its_longer_side(self):
        rng = np.random.default_rng(2)
        pairs = [
            ({"text": "a" * int(a)}, {"text": "b" * int(b)})
            for a, b in rng.integers(5, 3_000, size=(600, 2))
        ]
        cycle = _LengthGroupedCycle(
            PairIndexCycle(len(pairs), seed=7),
            pairs,
            key=lambda pair: max(len(pair[0]["text"]), len(pair[1]["text"])),
            factor=8,
            seed=9,
        )
        spans = []
        for _ in range(10):
            batch = cycle.take(16)
            costs = [
                max(len(pairs[i][0]["text"]), len(pairs[i][1]["text"])) for i in batch
            ]
            spans.append(max(costs) - min(costs))
        self.assertLess(np.median(spans), 600)


class ValidationDirectionalSummaryTests(unittest.TestCase):
    @staticmethod
    def _rows(negative_count: int, positive_logits: list[float]):
        rows = [
            {
                "text": f"negative-{index}",
                "label": 0,
                "source": "banking77" if index < 3 else "mixed",
            }
            for index in range(negative_count)
        ]
        rows.extend(
            {
                "text": f"positive-{index}",
                "label": 1,
                "source": "mixed",
            }
            for index in range(len(positive_logits))
        )
        return rows

    def test_mixed_label_source_has_both_directional_bces(self):
        rows = [
            {"text": "clean-mixed", "label": 0, "source": "mixed"},
            {"text": "attack-mixed", "label": 1, "source": "mixed"},
            {"text": "attack-only", "label": 1, "source": "positive_only"},
            {"text": "finance-clean", "label": 0, "source": "banking77"},
        ]
        logits = np.asarray([-2.0, -1.0, 2.0, 3.0])
        summary = mmbert_train._primary_validation_summary(
            rows,
            logits,
            checkpoint_diagnostics=True,
        )
        self.assertEqual(summary["by_source_label"]["mixed"]["negative"]["rows"], 1)
        self.assertEqual(summary["by_source_label"]["mixed"]["positive"]["rows"], 1)
        self.assertIsNone(
            summary["by_source_label"]["positive_only"]["negative"]["bce"]
        )
        expected_negative = np.mean(
            [
                _bce_from_logits(np.asarray([0]), np.asarray([-2.0])),
                _bce_from_logits(np.asarray([0]), np.asarray([3.0])),
            ]
        )
        expected_positive = np.mean(
            [
                _bce_from_logits(np.asarray([1]), np.asarray([-1.0])),
                _bce_from_logits(np.asarray([1]), np.asarray([2.0])),
            ]
        )
        self.assertAlmostEqual(
            summary["negative_source_label_macro_bce"],
            expected_negative,
        )
        self.assertAlmostEqual(
            summary["positive_source_label_macro_bce"],
            expected_positive,
        )

    def test_threshold_is_invariant_to_positive_logits(self):
        negative_logits = np.linspace(-5.0, 5.0, 100)
        rows = self._rows(100, [-10.0, 10.0])
        first = mmbert_train._checkpoint_empirical_operating_point(
            rows,
            np.concatenate([negative_logits, [-10.0, 10.0]]),
        )
        second = mmbert_train._checkpoint_empirical_operating_point(
            rows,
            np.concatenate([negative_logits, [100.0, 101.0]]),
        )
        self.assertEqual(first["threshold_logit"], second["threshold_logit"])
        self.assertEqual(first["false_positives"], second["false_positives"])
        self.assertEqual(first["false_positive_budget"], 1)
        self.assertNotEqual(first["positive_recall"], second["positive_recall"])

    def test_boundary_ties_never_exceed_the_false_positive_budget(self):
        negative_logits = np.asarray([10.0, 9.0, 9.0, *([-2.0] * 97)])
        rows = self._rows(100, [10.0])
        result = mmbert_train._checkpoint_empirical_operating_point(
            rows,
            np.concatenate([negative_logits, [10.0]]),
        )
        self.assertEqual(result["false_positive_budget"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertLessEqual(result["observed_row_fpr"], 0.01)
        self.assertGreater(result["threshold_logit"], 9.0)

    def test_validation_scores_no_calibration_rows_and_selector_is_unchanged(self):
        checkpoint = [
            {"text": "m-clean", "label": 0, "source": "mixed"},
            {"text": "m-attack", "label": 1, "source": "mixed"},
            {"text": "f-clean", "label": 0, "source": "banking77"},
            {"text": "p-attack", "label": 1, "source": "positive_only"},
        ]
        promptshield = [
            {"text": "ps-clean", "label": 0, "source": "promptshield"},
            {"text": "ps-attack", "label": 1, "source": "promptshield"},
        ]
        data = _training_data(
            promptshield_validation=promptshield,
            checkpoint=checkpoint,
            calibration=[
                {"text": "must-not-be-scored", "label": 0, "source": "calibration"}
            ],
        )

        class Module:
            def __init__(self) -> None:
                self.training = True

            def eval(self) -> None:
                self.training = False

            def train(self, mode: bool = True) -> None:
                self.training = mode

        checkpoint_logits = np.asarray([-2.0, 1.0, 0.5, -0.25])
        promptshield_logits = np.asarray([-1.0, 2.0])
        args = Namespace(
            max_tokens=512,
            microbatch_size=2,
            selection_rule="source_macro",
            comparison_update=0,
        )
        with patch.object(
            mmbert_train,
            "_validation_logits",
            side_effect=[checkpoint_logits, promptshield_logits],
        ) as score:
            row = mmbert_train._validation_row(
                Module(),
                object(),
                Module(),
                data,
                args,
                epoch=1,
                updates=500,
                training_loss=0.3,
                canonical_seen=128,
            )

        self.assertEqual(score.call_count, 2)
        scored_texts = [text for call in score.call_args_list for text in call.args[3]]
        self.assertNotIn("must-not-be-scored", scored_texts)
        expected_morgott = _bce_from_logits(
            np.asarray([0, 1, 0, 1]),
            checkpoint_logits,
        )
        expected_promptshield = _bce_from_logits(
            np.asarray([0, 1]),
            promptshield_logits,
        )
        expected = mmbert_train._selection_loss(
            {"morgott": expected_morgott, "promptshield": expected_promptshield},
            {"morgott": row["validation_morgott_by_source"]},
            "source_macro",
        )
        self.assertAlmostEqual(row["selection_loss"], expected)
        self.assertIn("validation_checkpoint_operating_point", row)


class ValidationMetricFilterTests(unittest.TestCase):
    ROW = {
        "epoch": 2,
        "updates": 10_000,
        "canonical_rows_seen": 209_792,
        "training_loss": 0.144,
        "selection_rule": "source_macro",
        "selection_loss": 0.176,
        "validation_macro_bce": 0.170,
        "validation_morgott_source_macro_bce": 0.340,
        "validation_morgott_negative_source_label_macro_bce": 0.210,
        "validation_morgott_positive_source_label_macro_bce": 0.430,
        "validation_worst_source_bce": 2.763,
        "validation_morgott_by_source": {"mixed": 0.4},
        "validation_morgott_by_source_label": {
            "banking77": {
                "negative": {"rows": 3, "bce": 0.11},
                "positive": {"rows": 2, "bce": 0.91},
            },
            "harper_valley_bank": {
                "negative": {"rows": 3, "bce": 0.12},
                "positive": {"rows": 2, "bce": 0.92},
            },
            "mixed": {
                "negative": {"rows": 3, "bce": 0.2},
                "positive": {"rows": 2, "bce": 0.5},
            },
            "tatqa": {
                "negative": {"rows": 3, "bce": 0.13},
                "positive": {"rows": 2, "bce": 0.93},
            },
        },
        "validation_checkpoint_operating_point": {
            "threshold_logit": 1.25,
            "negative_rows": 100,
            "positive_rows": 50,
            "positive_recall": 0.72,
            "finance_false_positives": 1,
        },
    }

    def _tracker(self):
        class _Fake:
            def __init__(self) -> None:
                self.sent: dict = {}

            def log(self, metrics: dict, step: int | None = None) -> None:
                self.sent = metrics

        tracker = mmbert_train._RunTracker.__new__(mmbert_train._RunTracker)
        tracker._run = _Fake()
        return tracker

    def _log(self) -> dict:
        tracker = self._tracker()
        tracker.log_validation(dict(self.ROW), step=10_000)
        return tracker._run.sent

    def test_exact_compact_validation_keys(self):
        sent = self._log()
        self.assertEqual(
            sorted(sent),
            [
                "checkpoint_diagnostics/finance_false_positives_at_empirical_1pct_row_fpr",
                "checkpoint_diagnostics/positive_recall_at_empirical_1pct_row_fpr",
                "selection_rules/ACTIVE_source_macro_blend",
                "selection_rules/alt_source_macro_only",
                "selection_rules/alt_worst_source",
                "validation/validation_morgott_negative_source_label_macro_bce",
                "validation/validation_morgott_positive_source_label_macro_bce",
            ],
        )
        self.assertTrue(all(isinstance(value, float) for value in sent.values()))

    def test_every_selection_rule_is_logged_with_the_active_one_named(self):
        sent = self._log()
        self.assertEqual(
            sent["selection_rules/ACTIVE_source_macro_blend"],
            self.ROW["selection_loss"],
        )
        self.assertEqual(
            sent["selection_rules/alt_worst_source"],
            self.ROW["validation_worst_source_bce"],
        )

    def test_active_key_names_the_micro_rule_without_a_legacy_alias(self):
        tracker = self._tracker()
        row = dict(self.ROW, selection_rule="micro")
        tracker.log_validation(row, step=10_000)
        self.assertIn("selection_rules/ACTIVE_micro_blend", tracker._run.sent)
        self.assertNotIn("selection_rules/ACTIVE_registered_blend", tracker._run.sent)

    def test_unknown_active_rule_fails_closed(self):
        tracker = self._tracker()
        row = dict(self.ROW, selection_rule="unknown")
        with self.assertRaisesRegex(ValueError, "supported selection rule"):
            tracker.log_validation(row, step=10_000)

    def test_only_three_finance_negative_bces_are_logged(self):
        tracker = self._tracker()
        tracker.log_finance_false_flag_bces(
            self.ROW["validation_morgott_by_source_label"],
            step=10_000,
        )
        self.assertEqual(
            tracker._run.sent,
            {
                "val_bce_false_flags/banking77": 0.11,
                "val_bce_false_flags/harper_valley_bank": 0.12,
                "val_bce_false_flags/tatqa": 0.13,
            },
        )


class TrainingTrackioMetricTests(unittest.TestCase):
    LATEST = {"primary_loss": 0.19}
    AVERAGED = {
        "primary_loss": 0.20,
        "canonical_primary_loss": 0.10,
        "promptshield_loss": 0.06,
        "pair_loss": 0.04,
        "pre_clip_gradient_norm": 0.80,
        "gradient_clipped": 0.25,
    }

    def _metrics(self, *, adapter_lr: float | None):
        return mmbert_train._training_trackio_metrics(
            self.LATEST,
            self.AVERAGED,
            peak_vram_gib=21.5,
            head_lr=3e-4,
            adapter_lr=adapter_lr,
            optimizer_updates_per_second=1.7,
            examples_per_second=217.6,
        )

    def test_run_keeps_only_core_training_curves(self):
        metrics = self._metrics(adapter_lr=1e-4)
        self.assertEqual(
            set(metrics),
            {
                "performance/examples_per_second",
                "performance/optimizer_updates_per_second",
                "train/adapter_lr",
                "train/canonical_primary_loss",
                "train/clip_fraction",
                "train/head_lr",
                "train/loss",
                "train/pair_loss",
                "train/peak_vram_gib",
                "train/pre_clip_gradient_norm",
                "train/promptshield_loss",
            },
        )
        self.assertEqual(metrics["train/loss"], self.LATEST["primary_loss"])

    def test_frozen_run_omits_adapter_rate(self):
        metrics = self._metrics(adapter_lr=None)
        self.assertNotIn("train/adapter_lr", metrics)


class AlternateSelectionTests(unittest.TestCase):
    @staticmethod
    def _row(source_macro: float, worst: float) -> dict:
        return {
            "validation_morgott_source_macro_bce": source_macro,
            "validation_worst_source_bce": worst,
        }

    def _run(self, rows: list[tuple[float, float]]) -> dict:
        import torch

        head = torch.nn.Linear(2, 1)
        alternates: dict = {}
        # State extraction is exercised elsewhere; this covers which checkpoint
        # each rule keeps.
        with patch.object(mmbert_train, "_adapter_state", return_value={"w": 1}):
            for index, (source_macro, worst) in enumerate(rows):
                mmbert_train._update_alternates(
                    alternates,
                    self._row(source_macro, worst),
                    mode="lora",
                    head=head,
                    encoder=None,
                    epoch=1,
                    updates=(index + 1) * 1_000,
                )
        return alternates

    def test_each_rule_keeps_its_own_minimum(self):
        # Mirrors the baseline disagreement: the second point is better on
        # source-macro, the first is better on worst-source.
        alternates = self._run([(0.20, 0.86), (0.17, 1.56)])
        self.assertEqual(alternates["source_macro_only"]["updates"], 2_000)
        self.assertEqual(alternates["worst_source"]["updates"], 1_000)
        self.assertAlmostEqual(alternates["source_macro_only"]["loss"], 0.17)
        self.assertAlmostEqual(alternates["worst_source"]["loss"], 0.86)

    def test_weights_are_retained_for_each_candidate(self):
        alternates = self._run([(0.5, 0.5)])
        for candidate in alternates.values():
            self.assertTrue(candidate["head"])
            self.assertIn("updates", candidate)

    def test_a_worse_point_does_not_displace_the_incumbent(self):
        alternates = self._run([(0.10, 0.10), (0.90, 0.90)])
        for candidate in alternates.values():
            self.assertEqual(candidate["updates"], 1_000)


if __name__ == "__main__":
    unittest.main()
