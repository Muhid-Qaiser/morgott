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
    new_head,
)
from morgott.models.mmbert.head_contract import (
    HeadContract,
    new_head_for_result,
    new_multitask_head,
    resolve_head_contract,
)


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
    def test_absent_contract_preserves_the_historical_head_exactly(self):
        import torch

        expected = new_head(8, 42)
        actual = new_head_for_result(8, 42, {})
        self.assertEqual(
            resolve_head_contract({}),
            HeadContract(
                "legacy_sequential_binary_v1",
                1,
                ("instruction_subversion",),
                0,
            ),
        )
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)

    def test_explicit_single_output_contract_is_supported(self):
        result = {
            "head_contract": {
                "architecture": "legacy_sequential_binary_v1",
                "outputs": 1,
                "columns": {"0": "instruction_subversion"},
                "primary_column": 0,
            }
        }
        head = new_head_for_result(8, 7, result)
        self.assertEqual(head[-1].out_features, 1)

    def test_multitask_contract_builds_two_outputs_with_primary_at_zero(self):
        import torch

        result = {
            "head_contract": {
                "architecture": "shared_trunk_separate_binary_projections_v1",
                "outputs": 2,
                "columns": {
                    "0": "instruction_subversion",
                    "1": "harmful_intent",
                },
                "primary_column": 0,
            }
        }
        contract = resolve_head_contract(result)
        head = new_head_for_result(8, 42, result).eval()
        logits = head(torch.zeros(3, 24))
        self.assertEqual(
            contract.columns[contract.primary_column], "instruction_subversion"
        )
        self.assertEqual(logits.shape, (3, 2))
        self.assertEqual(logits[:, contract.primary_column].shape, (3,))

    def test_multitask_primary_path_initializes_identically_to_legacy(self):
        import torch

        legacy = new_head(8, 42)
        expected_rng = torch.random.get_rng_state()
        multitask = new_multitask_head(8, 42)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        self.assertEqual(
            set(multitask.state_dict()),
            {
                "trunk.0.weight",
                "trunk.0.bias",
                "trunk.1.weight",
                "trunk.1.bias",
                "primary.weight",
                "primary.bias",
                "auxiliary.weight",
                "auxiliary.bias",
            },
        )
        for index in (0, 1):
            for name, value in legacy[index].state_dict().items():
                self.assertTrue(
                    torch.equal(value, multitask.trunk[index].state_dict()[name]),
                    f"trunk.{index}.{name}",
                )
        for name, value in legacy[4].state_dict().items():
            self.assertTrue(
                torch.equal(value, multitask.primary.state_dict()[name]),
                f"primary.{name}",
            )

    def test_malformed_or_relabelled_contracts_fail_closed(self):
        valid = {
            "architecture": "shared_trunk_separate_binary_projections_v1",
            "outputs": 2,
            "columns": {
                "0": "instruction_subversion",
                "1": "harmful_intent",
            },
            "primary_column": 0,
        }
        invalid = (
            {**valid, "outputs": 3},
            {**valid, "outputs": True},
            {**valid, "primary_column": 1},
            {**valid, "architecture": "legacy_sequential_binary_v1"},
            {
                **valid,
                "columns": {"0": "harmful_intent", "1": "instruction_subversion"},
            },
            {**valid, "unexpected": "field"},
        )
        for contract in invalid:
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                resolve_head_contract({"head_contract": contract})

    def test_strict_state_loading_rejects_a_width_mismatch(self):
        legacy = new_head_for_result(8, 42, {})
        multitask = new_head_for_result(
            8,
            42,
            {
                "head_contract": {
                    "architecture": "shared_trunk_separate_binary_projections_v1",
                    "outputs": 2,
                    "columns": {
                        "0": "instruction_subversion",
                        "1": "harmful_intent",
                    },
                    "primary_column": 0,
                }
            },
        )
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(multitask.state_dict(), strict=True)

    def test_evaluator_builds_the_head_from_the_recorded_contract(self):
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
                "head_contract": {
                    "architecture": "shared_trunk_separate_binary_projections_v1",
                    "outputs": 2,
                    "columns": {
                        "0": "instruction_subversion",
                        "1": "harmful_intent",
                    },
                    "primary_column": 0,
                },
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
                patch.object(
                    mmbert_evaluate,
                    "new_head_for_result",
                    return_value=head,
                ) as build_head,
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

            build_head.assert_called_once_with(8, 42, loaded_result)
            self.assertIs(loaded_head, head)
            self.assertEqual(loaded_base, base_model)
            self.assertEqual(head.device, "cuda")
            self.assertEqual(head.loaded, ({"state": "loaded"}, True))


if __name__ == "__main__":
    unittest.main()
