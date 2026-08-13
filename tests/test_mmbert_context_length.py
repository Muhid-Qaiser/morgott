from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from morgott.models.mmbert import evaluate as mmbert_evaluate
from morgott.models.mmbert import train as mmbert_train


class _Tokenizer:
    def __init__(self):
        self.max_lengths = []

    def __call__(self, texts, **kwargs):
        self.max_lengths.append(kwargs["max_length"])
        return {
            "input_ids": [
                list(range(min(int(text), kwargs["max_length"]))) for text in texts
            ]
        }


class TrainingContextContractTests(unittest.TestCase):
    def test_parser_default_preserves_512_name_and_1024_derives_distinct_name(self):
        baseline = mmbert_train._parser().parse_args([])
        candidate = mmbert_train._parser().parse_args(["--max-tokens", "1024"])
        self.assertEqual(baseline.max_tokens, 512)
        self.assertEqual(
            mmbert_train._resolved_run_name(baseline), "mmbert-lora-full-s42"
        )
        self.assertEqual(
            mmbert_train._resolved_run_name(candidate),
            "mmbert-lora-full-s42-ctx1024",
        )

    def test_nondefault_explicit_name_must_bind_the_context_cap(self):
        args = mmbert_train._parser().parse_args(
            ["--max-tokens", "1024", "--run-name", "ambiguous"]
        )
        with self.assertRaisesRegex(ValueError, "ctx1024"):
            mmbert_train._resolved_run_name(args)

    def test_encoding_cache_is_cap_bound(self):
        tokenizer = _Tokenizer()
        cache = mmbert_train._EncodingCache(tokenizer, max_tokens=1024)
        self.assertEqual(len(cache.encode(["1200"])[0]), 1024)
        self.assertEqual(tokenizer.max_lengths, [1024])
        with self.assertRaisesRegex(ValueError, "differs"):
            mmbert_train._cached_batch_logits(
                object(),
                tokenizer,
                object(),
                ["1200"],
                train_encoder=False,
                cache=cache,
                max_tokens=512,
            )

    def test_single_head_validation_caps_select_the_intended_path(self):
        with patch.object(
            mmbert_train, "score_logits", return_value=np.asarray([1.0])
        ) as historical:
            observed = mmbert_train._validation_logits(
                object(), object(), object(), ["x"], batch_size=1
            )
        historical.assert_called_once()
        np.testing.assert_array_equal(observed, [1.0])

        class Value:
            def float(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.asarray([2.0])

        grad_enabled = []

        def capped_logits(*args, **kwargs):
            import torch

            grad_enabled.append(torch.is_grad_enabled())
            return Value()

        with patch.object(
            mmbert_train, "_cached_batch_logits", side_effect=capped_logits
        ) as capped:
            observed = mmbert_train._validation_logits(
                SimpleNamespace(eval=lambda: None),
                object(),
                SimpleNamespace(eval=lambda: None),
                ["x"],
                batch_size=1,
                max_tokens=1024,
            )
        self.assertEqual(capped.call_args.kwargs["max_tokens"], 1024)
        self.assertEqual(grad_enabled, [False])
        np.testing.assert_array_equal(observed, [2.0])

    def test_training_identity_binds_cap_and_token_budget(self):
        args = mmbert_train._parser().parse_args(
            ["--max-tokens", "1024", "--run-name", "unit-ctx1024"]
        )
        data = SimpleNamespace(
            data_manifest_sha256="1" * 64,
            external_manifest_sha256="2" * 64,
            views={
                split: (Path(split), {"sha256": str(index) * 64})
                for index, split in enumerate(
                    ("train", "validation", "dev_test"), start=3
                )
            },
        )
        with (
            patch.object(mmbert_train, "file_sha256", return_value="6" * 64),
            patch.object(mmbert_train, "source_provenance", return_value={}),
            patch.object(mmbert_train, "_report", return_value={}),
        ):
            identity = mmbert_train._training_identity(args, data)
        self.assertEqual(identity["schema_version"], 5)
        self.assertEqual(identity["max_tokens"], 1024)
        self.assertEqual(identity["token_budget"], 8 * 1024)


class EvaluationContextContractTests(unittest.TestCase):
    @staticmethod
    def _result(cap: int) -> dict:
        return {
            "max_tokens": cap,
            "token_budget": 24 * cap,
            "training_identity": {
                "schema_version": 5,
                "microbatch_size": 24,
                "max_tokens": cap,
                "token_budget": 24 * cap,
            },
            "training": {
                "microbatch_size": 24,
                "max_tokens": cap,
                "token_budget": 24 * cap,
            },
        }

    def test_native_cap_and_cross_cap_names_are_distinct(self):
        self.assertEqual(mmbert_evaluate._training_max_tokens({}), 512)
        self.assertEqual(mmbert_evaluate._training_max_tokens(self._result(1024)), 1024)
        self.assertEqual(
            mmbert_evaluate._evaluation_output_name(
                snapshot_update=17000,
                training_max_tokens=512,
                evaluation_max_tokens=512,
            ),
            "evaluation-update-17000",
        )
        self.assertEqual(
            mmbert_evaluate._evaluation_output_name(
                snapshot_update=17000,
                training_max_tokens=1024,
                evaluation_max_tokens=512,
            ),
            "evaluation-update-17000-trainctx1024-evalctx512",
        )

    def test_malformed_cap_or_budget_fails_closed(self):
        malformed = self._result(1024)
        malformed["training_identity"]["token_budget"] -= 1
        with self.assertRaisesRegex(ValueError, "context contract"):
            mmbert_evaluate._training_max_tokens(malformed)
        with self.assertRaises(ValueError):
            mmbert_evaluate._training_max_tokens({"max_tokens": 2048})

    def test_unknown_or_malformed_training_identity_schema_fails_closed(self):
        for schema_version in (6, True, "5", -1):
            with self.subTest(schema_version=schema_version):
                result = self._result(1024)
                result["training_identity"]["schema_version"] = schema_version
                with self.assertRaisesRegex(ValueError, "schema"):
                    mmbert_evaluate._training_max_tokens(result)

        result = self._result(1024)
        result["training_identity"] = "schema-5"
        with self.assertRaisesRegex(ValueError, "must be an object"):
            mmbert_evaluate._training_max_tokens(result)

    def test_journal_scoring_and_evaluation_identity_bind_both_caps(self):
        self.assertNotEqual(
            mmbert_evaluate._scoring_sha256(512),
            mmbert_evaluate._scoring_sha256(1024),
        )
        common = {
            "model_sha256": "1" * 64,
            "scoring_sha256": "2" * 64,
            "training_max_tokens": 1024,
        }
        self.assertNotEqual(
            mmbert_evaluate._evaluation_identity_sha256(
                **common, evaluation_max_tokens=512
            ),
            mmbert_evaluate._evaluation_identity_sha256(
                **common, evaluation_max_tokens=1024
            ),
        )

    def test_single_head_default_uses_core_but_cross_cap_uses_local_scorer(self):
        rows = [
            {
                "id": "unit",
                "text": "synthetic",
                "label": 0,
                "source": "unit",
                "input_channel": "direct_user",
            }
        ]
        with (
            patch.object(mmbert_evaluate, "score_texts", return_value=[0.1]) as core,
            patch.object(
                mmbert_evaluate, "_score_single_texts", return_value=np.asarray([0.2])
            ) as capped,
        ):
            baseline = mmbert_evaluate._score(
                rows, object(), object(), object(), batch_size=1
            )
            candidate = mmbert_evaluate._score(
                rows,
                object(),
                object(),
                object(),
                batch_size=1,
                max_tokens=1024,
            )
        core.assert_called_once()
        capped.assert_called_once()
        self.assertEqual(capped.call_args.kwargs["max_tokens"], 1024)
        np.testing.assert_array_equal(baseline["scores"], [0.1])
        np.testing.assert_array_equal(candidate["scores"], [0.2])


if __name__ == "__main__":
    unittest.main()
