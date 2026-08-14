from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from morgott.models.mmbert import evaluate as mmbert_evaluate
from morgott.models.mmbert.core import (
    ATTENTION_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
)
from morgott.models.mmbert.train import _head_contract


class BaseModelLoadingTests(unittest.TestCase):
    def test_loader_forces_the_verified_pytorch_weights(self) -> None:
        class Encoder:
            def __init__(self, weights: str) -> None:
                self.weights = weights

            def to(self, device: str):
                self.device = device
                return self

        class AutoModel:
            @staticmethod
            def from_pretrained(_model_id: str, **kwargs):
                weights = (
                    "pytorch_model.bin"
                    if kwargs.get("use_safetensors") is False
                    else "model.safetensors"
                )
                return Encoder(weights)

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(_model_id: str, **_kwargs):
                return SimpleNamespace(pad_token_id=0)

        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        torch.cuda = SimpleNamespace(is_available=lambda: True)
        transformers = types.ModuleType("transformers")
        transformers.AutoModel = AutoModel
        transformers.AutoTokenizer = AutoTokenizer

        with patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            encoder, _ = mmbert_evaluate._load_pytorch_base_model()

        self.assertEqual(encoder.weights, "pytorch_model.bin")


class HeadContractTests(unittest.TestCase):
    def test_absent_or_explicit_single_output_contract_is_supported(self):
        expected = _head_contract()
        self.assertEqual(mmbert_evaluate._single_output_head_contract({}), expected)
        self.assertEqual(
            mmbert_evaluate._single_output_head_contract({"head_contract": expected}),
            expected,
        )

    def test_non_single_output_contract_is_rejected(self):
        contract = {
            "architecture": "shared_trunk_separate_binary_projections_v1",
            "outputs": 2,
            "columns": {
                "0": "instruction_subversion",
                "1": "harmful_intent",
            },
            "primary_column": 0,
        }
        with self.assertRaisesRegex(ValueError, "single-output"):
            mmbert_evaluate._single_output_head_contract({"head_contract": contract})

    def test_evaluator_builds_the_maintained_head(self):
        class Encoder:
            config = SimpleNamespace(hidden_size=8)

            def eval(self):
                return self

            def parameters(self):
                return ()

        class Head:
            def __init__(self):
                self.loaded = None

            def to(self, device):
                self.device = device
                return self

            def load_state_dict(self, state, *, strict):
                self.loaded = (state, strict)

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            head_path = run / "head.safetensors"
            head_path.write_bytes(b"unit-test-head")
            result = {
                "purpose": "maintained full-data advisory mmBERT training",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "attention_implementation": ATTENTION_IMPLEMENTATION,
                "normalization": "strict",
                "adaptation": "frozen",
                "seed": 42,
                "head_contract": _head_contract(),
                "artifact": {
                    "head": head_path.name,
                    "head_sha256": file_sha256(head_path),
                },
            }
            (run / "result.json").write_text(json.dumps(result), encoding="utf-8")

            safetensors = types.ModuleType("safetensors")
            safetensors_torch = types.ModuleType("safetensors.torch")
            safetensors_torch.load_file = lambda _path: {"state": "loaded"}
            safetensors.torch = safetensors_torch
            head = Head()
            base_model = {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "config_sha256": "0" * 64,
                "pytorch_model_sha256": "1" * 64,
                "special_tokens_map_sha256": "2" * 64,
                "tokenizer_config_sha256": "3" * 64,
                "tokenizer_json_sha256": "2" * 64,
            }
            base_model_checks = [base_model, base_model]
            with (
                patch.dict(
                    sys.modules,
                    {
                        "safetensors": safetensors,
                        "safetensors.torch": safetensors_torch,
                    },
                ),
                patch.object(
                    mmbert_evaluate,
                    "_load_pytorch_base_model",
                    return_value=(Encoder(), object()),
                ),
                patch.object(mmbert_evaluate, "new_head", return_value=head) as build,
                patch.object(
                    mmbert_evaluate,
                    "_verified_base_model_identity",
                    side_effect=lambda: base_model_checks.pop(0),
                ),
            ):
                loaded_result, _, _, loaded_head, loaded_base = (
                    mmbert_evaluate._load_run(run)
                )
                base_model_checks[:] = [
                    base_model,
                    {**base_model, "pytorch_model_sha256": "3" * 64},
                ]
                with self.assertRaisesRegex(
                    ValueError,
                    "base model changed during loading",
                ):
                    mmbert_evaluate._load_run(run)

            build.assert_called_once_with(8, 42)
            self.assertEqual(loaded_result, result)
            self.assertIs(loaded_head, head)
            self.assertEqual(loaded_base, base_model)
            self.assertEqual(head.device, "cuda")
            self.assertEqual(head.loaded, ({"state": "loaded"}, True))


if __name__ == "__main__":
    unittest.main()
