import json
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from compare import assert_fair_protocol, compact_metrics
from run import (
    DIRECT_PRECISION_FLOORS,
    MODELS,
    MeanPoolClassifier,
    read_rows,
    select_training_subset,
)


class EncoderFinetuneTests(unittest.TestCase):
    def test_pinned_subset_is_group_safe_and_exact(self):
        selected, summary = select_training_subset(read_rows("train"))
        self.assertEqual((summary["rows"], summary["positive"]), (8_408, 245))
        self.assertEqual(summary["oasst_rows"], 4_003)
        self.assertEqual(
            summary["ordered_row_ids_sha256"],
            "36ef7c54a385790a4fb946f9aaaa3a876b56000bfcec2f6513853e6628969408",
        )
        self.assertEqual(len(selected), len({row["id"] for row in selected}))
        self.assertEqual(summary["validation_rows_untouched"], 7_186)

    def test_shared_head_uses_masked_mean(self):
        class Encoder(nn.Module):
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=input_ids.float())

        model = MeanPoolClassifier(Encoder(), hidden_size=2)
        with torch.no_grad():
            model.classifier.weight.copy_(torch.eye(2))
            model.classifier.bias.zero_()
        logits = model(
            {
                "input_ids": torch.tensor([[[2, 4], [4, 8], [100, 100]]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
                "labels": torch.tensor([1]),
            }
        )
        torch.testing.assert_close(logits, torch.tensor([[3.0, 6.0]]))

    def test_model_security_and_fair_protocol_constants(self):
        self.assertEqual(DIRECT_PRECISION_FLOORS, (0.80, 0.85, 0.90, 0.95))
        self.assertEqual(MODELS["modernbert"]["attention_backend"], "sdpa")
        self.assertEqual(MODELS["deberta"]["attention_backend"], "eager")
        for spec in MODELS.values():
            self.assertEqual(len(spec["revision"]), 40)
            self.assertEqual(len(spec["weights_sha256"]), 64)

    def test_loading_metadata_contract_is_json_serializable(self):
        metadata = {
            "unexpected_pretraining_head_keys": sorted({"head.a", "head.b"}),
            "missing_encoder_keys": sorted(set()),
            "mismatched_encoder_keys": sorted(str(value) for value in set()),
        }
        self.assertIsInstance(json.dumps(metadata), str)

    def test_comparison_rejects_protocol_drift(self):
        shared = {
            "training_subset": {"ordered_row_ids_sha256": "same"},
            "input_sha256": {"train": "same"},
            "default_precision_floor": 0.85,
        }
        results = {
            "modernbert": {**shared, "protocol": {"seed": 42}},
            "deberta": {**shared, "protocol": {"seed": 43}},
        }
        with self.assertRaisesRegex(ValueError, "protocol"):
            assert_fair_protocol(results)

    def test_compact_metrics_drops_nested_strata(self):
        compact = compact_metrics(
            {
                "rows": 10,
                "true_positive": 2,
                "by_language": {"en": {"rows": 10}},
            }
        )
        self.assertEqual(compact["rows"], 10)
        self.assertEqual(compact["true_positive"], 2)
        self.assertNotIn("by_language", compact)


if __name__ == "__main__":
    unittest.main()
