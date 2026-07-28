from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from train_combined_generic_lora import (  # noqa: E402
    combined_lora_schedule,
    domain_microbatch_weights,
    lora_run_directory_name,
    publish_if_unchanged,
    quantize_training_features,
    select_checkpoint_epoch,
    stable_validation_bces,
)


class CombinedGenericLoraTests(unittest.TestCase):
    def test_validation_bce_does_not_clip_extreme_wrong_logits(self):
        import numpy as np

        bces = stable_validation_bces(
            {
                "morgott": np.asarray([0]),
                "promptshield": np.asarray([1]),
            },
            {
                "morgott": np.asarray([40.0]),
                "promptshield": np.asarray([-40.0]),
            },
        )

        self.assertAlmostEqual(bces["morgott"], 40.0)
        self.assertAlmostEqual(bces["promptshield"], 40.0)
        self.assertAlmostEqual(bces["macro"], 40.0)

    def test_every_validation_call_uses_model_tokenizer_head_and_records(self):
        source = (EXPERIMENTS_DIR / "train_combined_generic_lora.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_validation_logits"
        ]

        self.assertTrue(calls)
        self.assertTrue(all(len(call.args) == 4 for call in calls))

    def test_fixed_gate_schedule_is_update_matched(self):
        self.assertEqual(
            combined_lora_schedule(
                rows_per_half=18_197,
                epochs=3,
                microbatch_size=16,
                effective_batch_size=256,
            ),
            {
                "half_batch_size": 128,
                "updates_per_epoch": 143,
                "updates": 429,
                "forward_backward_microsteps": 6_828,
            },
        )

    def test_partial_update_keeps_each_domain_at_half_the_loss(self):
        weights = domain_microbatch_weights(21, 16)

        self.assertEqual(weights, [8 / 21, 5 / 42])
        self.assertAlmostEqual(sum(weights), 0.5)

    def test_checkpoint_selection_uses_the_minimum_macro_bce_epoch(self):
        curve = [
            {"epoch": 1, "validation_macro_bce": 0.31},
            {"epoch": 2, "validation_macro_bce": 0.20},
            {"epoch": 3, "validation_macro_bce": 0.24},
        ]

        self.assertEqual(select_checkpoint_epoch(curve), 2)

    def test_lora_run_identity_cannot_collide_with_the_frozen_run(self):
        self.assertEqual(
            lora_run_directory_name("jhu-clsp/mmBERT-base", 42),
            "jhu-clsp-mmbert-base_combined_lora-r8_s42",
        )
        self.assertNotEqual(
            lora_run_directory_name("jhu-clsp/mmBERT-base", 42),
            "jhu-clsp-mmbert-base_combined_s42",
        )

    def test_training_features_match_bfloat16_scoring_without_detaching_gradients(self):
        import torch

        pooled = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)

        quantized = quantize_training_features(pooled)
        quantized.float().sum().backward()

        self.assertEqual(quantized.dtype, torch.bfloat16)
        self.assertIsNotNone(pooled.grad)

    def test_publication_refuses_an_input_changed_during_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "m1.jsonl"
            source.write_text("before\n")
            expected = {"m1": hashlib.sha256(b"before\n").hexdigest()}
            source.write_text("after\n")
            pending = directory / "pending"
            pending.mkdir()
            output = directory / "published"

            with patch("os.replace") as replace:
                with self.assertRaisesRegex(ValueError, "source changed"):
                    publish_if_unchanged(
                        pending,
                        output,
                        {"m1": source},
                        expected,
                    )

            replace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
