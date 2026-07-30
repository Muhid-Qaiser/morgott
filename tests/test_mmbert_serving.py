import copy
import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from morgott.models.mmbert.export_onnx import (
    _cascade_metrics,
    _deepseek_evidence,
    _quality_gate,
    benchmark,
    verify_panel,
)
from morgott.models.mmbert.serving import MmbertRuntime, PreparedText, Window


class _Encoding:
    def __init__(self, ids, offsets, overflowing=()):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.offsets = offsets
        self.overflowing = list(overflowing)


class _Tokenizer:
    def __init__(self, encoding):
        self.encoding = encoding
        self.normalized = None
        self.truncation = None

    def enable_truncation(self, *, max_length, stride):
        self.truncation = (max_length, stride)

    def encode(self, text):
        self.normalized = text
        return self.encoding


class _Session:
    def __call__(self, inputs):
        token_id = int(inputs["input_ids"][0, 1])
        return {"logit": np.array([[0.0 if token_id == 1 else 2.1972246]])}


class _BenchmarkRuntime:
    identity = type(
        "Identity",
        (),
        {
            "runtime": "openvino-test-cpu-bf16",
            "onnx_sha256": "c" * 64,
            "compile_seconds": 1.25,
            "openvino": "test",
            "reported_inference_precision": "bf16",
            "threads": 2,
            "cpu_capabilities": ("BF16", "FP32"),
        },
    )()

    def prepare(self, text):
        return PreparedText(
            normalized_text=text,
            token_count=510,
            windows=(
                Window(
                    index=0,
                    char_start=0,
                    char_end=len(text),
                    input_ids=tuple(range(512)),
                    attention_mask=(1,) * 512,
                ),
            ),
        )

    def score(self, windows):
        del windows
        return (0.55,)


def _write_deepseek_evidence(root):
    panel_sha256 = "a" * 64
    configuration_sha256 = "b" * 64
    ledger = (
        json.dumps(
            {
                "panel_id": "row-1",
                "configuration": "deepseek_coreweave",
                "configuration_sha256": configuration_sha256,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    with gzip.open(root / "followup_results.jsonl.gz", "wb") as handle:
        handle.write(ledger)
    followup = {
        "panel_sha256": panel_sha256,
        "split": {"calibration_panel_ids": ["row-1"]},
    }
    followup_path = root / "followup_manifest.json"
    followup_path.write_text(
        json.dumps(followup, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "panel_sha256": panel_sha256,
        "result_ledger_sha256": hashlib.sha256(ledger).hexdigest(),
        "followup_manifest_sha256": hashlib.sha256(
            followup_path.read_bytes()
        ).hexdigest(),
        "configuration_sha256s": {"deepseek_coreweave": configuration_sha256},
    }
    (root / "followup_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return {"panel": {"sha256": panel_sha256}}, followup_path


class MmbertServingTests(unittest.TestCase):
    def test_verification_does_not_overwrite_registered_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            evidence = output / "verification.json"
            evidence.write_text("registered\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "verification evidence"):
                verify_panel(
                    manifest_path=output / "missing-manifest.json",
                    panel_dir=output,
                    output=output,
                )

            self.assertEqual(evidence.read_text(encoding="utf-8"), "registered\n")

    def test_deployment_benchmark_does_not_overwrite_registered_evidence(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch(
                "morgott.models.mmbert.export_onnx._candidate_runtime",
                return_value=_BenchmarkRuntime(),
            ),
        ):
            output = Path(temporary)
            (output / "model.onnx").write_bytes(b"model")

            result = benchmark(output=output, warmup=1, requests=1)

            self.assertFalse((output / "benchmark.json").exists())
            self.assertEqual(
                set(result),
                {
                    "compile_seconds",
                    "cpu_capabilities",
                    "format",
                    "measured_requests",
                    "openvino",
                    "p50_ms",
                    "p95_below_500_ms",
                    "p95_ms",
                    "qps",
                    "reported_inference_precision",
                    "representative_logit",
                    "requested_inference_precision",
                    "source_onnx_bytes",
                    "source_onnx_sha256",
                    "sustains_5_qps",
                    "threads",
                    "warmup_requests",
                },
            )
            self.assertTrue(result["p95_below_500_ms"])

    def test_verification_reads_versioned_compressed_provider_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel_manifest, _ = _write_deepseek_evidence(root)

            records, calibration_ids = _deepseek_evidence(
                root,
                panel_manifest,
            )

        self.assertEqual(set(records), {"row-1"})
        self.assertEqual(calibration_ids, {"row-1"})

    def test_verification_rejects_a_changed_calibration_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel_manifest, followup_path = _write_deepseek_evidence(root)
            followup = json.loads(followup_path.read_text(encoding="utf-8"))
            followup["split"]["calibration_panel_ids"] = ["row-2"]
            followup_path.write_text(
                json.dumps(followup, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "split changed"):
                _deepseek_evidence(root, panel_manifest)

    def test_serving_quality_gate_allows_one_calibration_false_positive(self):
        reference = {
            "calibration": {"false_positive": 68},
            "evaluation": {
                "recall": 0.6679,
                "fpr": 0.0181,
                "precision": 0.9650,
                "provider_call_rate": 0.2217,
            },
            "evaluation_slices": {
                name: {"recall": 0.7, "fpr": 0.02}
                for name in ("canonical", "promptshield", "sep")
            },
        }
        candidate = copy.deepcopy(reference)
        candidate["calibration"]["false_positive"] += 1

        self.assertTrue(_quality_gate(candidate, reference)["passed"])

        candidate["evaluation"]["recall"] -= 0.01
        self.assertFalse(_quality_gate(candidate, reference)["passed"])

    def test_precision_study_uses_the_frozen_fail_safe_cascade(self):
        panel = [
            {"panel_id": "a", "label": 0, "dataset": "canonical"},
            {"panel_id": "b", "label": 1, "dataset": "promptshield"},
            {"panel_id": "c", "label": 0, "dataset": "sep"},
            {"panel_id": "d", "label": 1, "dataset": "canonical"},
        ]
        records = {
            "a": {"status": "ok", "p_subversion": 0.95},
            "b": {"status": "ok", "p_subversion": 0.01},
            "c": {"status": "ok", "p_subversion": 0.1},
            "d": {"status": "timeout"},
        }

        metrics = _cascade_metrics(
            panel,
            np.asarray([0.1, 0.99999, 0.5, 0.5]),
            records,
            {"a", "b"},
        )

        self.assertEqual(metrics["calibration"]["true_positive"], 1)
        self.assertEqual(metrics["evaluation"]["true_positive"], 1)
        self.assertEqual(metrics["evaluation"]["false_positive"], 0)
        self.assertEqual(metrics["evaluation"]["provider_call_rows"], 2)
        self.assertEqual(metrics["evaluation"]["provider_failures"], 1)

    def test_registered_artifact_hash_mismatch_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.onnx").write_bytes(b"wrong model")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            manifest = {
                "schema_version": 2,
                "advisory_only": True,
                "base_model": {"tokenizer_json_sha256": "0" * 64},
                "models": {
                    "mmbert-lora-full-s42": {
                        "serving": {
                            "format": "onnx-openvino-bf16-v1",
                            "inference_precision": "bf16",
                            "onnx": {
                                "path": "model.onnx",
                                "sha256": "0" * 64,
                            },
                            "tokenizer": {
                                "path": "tokenizer.json",
                                "sha256": "0" * 64,
                            },
                        }
                    }
                },
            }
            path = root / "model-artifacts.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "ONNX model hash mismatch"):
                MmbertRuntime.from_artifacts(path)

    def test_prepared_windows_cover_the_normalized_input_without_truncation(self):
        overflow = _Encoding(
            [101, 3, 4, 102],
            [(0, 0), (4, 5), (6, 7), (0, 0)],
        )
        tokenizer = _Tokenizer(
            _Encoding(
                [101, 1, 2, 3, 102],
                [(0, 0), (0, 1), (2, 3), (4, 5), (0, 0)],
                [overflow],
            )
        )
        runtime = MmbertRuntime(tokenizer=tokenizer, session=None)

        prepared = runtime.prepare("A B C D")

        self.assertEqual(tokenizer.normalized, "a b c d")
        self.assertEqual(tokenizer.truncation, (512, 128))
        self.assertEqual(prepared.token_count, 4)
        self.assertEqual(
            [(window.char_start, window.char_end) for window in prepared.windows],
            [(0, 5), (4, 7)],
        )
        self.assertEqual(
            [
                prepared.normalized_text[window.char_start : window.char_end]
                for window in prepared.windows
            ],
            ["a b c", "c d"],
        )

    def test_scores_every_window_with_a_stable_sigmoid(self):
        overflow = _Encoding(
            [101, 3, 4, 102],
            [(0, 0), (4, 5), (6, 7), (0, 0)],
        )
        tokenizer = _Tokenizer(
            _Encoding(
                [101, 1, 2, 3, 102],
                [(0, 0), (0, 1), (2, 3), (4, 5), (0, 0)],
                [overflow],
            )
        )
        runtime = MmbertRuntime(tokenizer=tokenizer, session=_Session())

        prepared = runtime.prepare("A B C D")

        scores = runtime.score(prepared.windows)

        self.assertEqual(scores[0], 0.5)
        self.assertAlmostEqual(scores[1], 0.9)

    @unittest.skipUnless(
        importlib.util.find_spec("tokenizers"),
        "cascade tokenizer dependency is not installed",
    )
    def test_real_tokenizer_covers_long_input_and_boundary_spanning_text(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing

        tokens = [f"t{index}" for index in range(900)]
        tokens[507:515] = [
            "attack",
            "crosses",
            "the",
            "window",
            "boundary",
            "without",
            "being",
            "cut",
        ]
        vocabulary = {
            token: index + 3 for index, token in enumerate(dict.fromkeys(tokens))
        }
        vocabulary.update({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2})
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        runtime = MmbertRuntime(tokenizer=tokenizer, session=None)
        text = " ".join(tokens)

        prepared = runtime.prepare(text)

        self.assertEqual(prepared.token_count, 900)
        self.assertGreater(len(prepared.windows), 1)
        self.assertEqual(prepared.windows[0].char_start, 0)
        self.assertEqual(prepared.windows[-1].char_end, len(text))
        self.assertTrue(
            any(
                "attack crosses the window boundary without being cut"
                in prepared.normalized_text[window.char_start : window.char_end]
                for window in prepared.windows
            )
        )


if __name__ == "__main__":
    unittest.main()
