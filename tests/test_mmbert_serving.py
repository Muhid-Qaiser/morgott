import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from morgott.models.deepseek_nooa import (
    EVALUATION_REQUEST_SHA256,
    MAX_ATTEMPTS,
    PROMPT_SHA256,
)
from morgott.models.deepseek_nooa import (
    MODEL as DEEPSEEK_MODEL,
)
from morgott.models.deepseek_nooa import (
    PROVIDER as DEEPSEEK_PROVIDER,
)
from morgott.models.mmbert.export_onnx import (
    _cascade_metrics,
    _deepseek_evidence,
    _parity_metrics,
    _quality_gate,
    benchmark,
    verify_panel,
)
from morgott.models.mmbert.serving import (
    MmbertRuntime,
    PreparedText,
    Window,
    _select_inference_precision,
)
from morgott.normalization import strict_normalize


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


class _BatchSession:
    def __call__(self, inputs):
        return {
            "logit": np.asarray(
                [[float(row[1])] for row in inputs["input_ids"]],
                dtype=np.float32,
            )
        }


class _BenchmarkRuntime:
    max_tokens = 1024
    identity = type(
        "Identity",
        (),
        {
            "runtime": "openvino-test-cpu-bf16",
            "onnx_sha256": "c" * 64,
            "compile_seconds": 1.25,
            "loaded_from_cache": False,
            "openvino": "test",
            "requested_inference_precision": "bf16",
            "inference_precision": "bf16",
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
                    input_ids=tuple(range(1024)),
                    attention_mask=(1,) * 1024,
                ),
            ),
        )

    def score(self, windows):
        del windows
        return (0.55,)


class MmbertServingTests(unittest.TestCase):
    def test_export_parity_fails_outside_serving_tolerance(self):
        reference = (np.asarray([[0.0]], dtype=np.float32),)

        self.assertTrue(_parity_metrics(reference, reference, label="test")["passed"])
        with self.assertRaisesRegex(ValueError, "test representative parity"):
            _parity_metrics(
                reference,
                (np.asarray([[0.001]], dtype=np.float32),),
                label="test",
            )

    def test_auto_precision_prefers_bf16_and_falls_back_to_fp32(self):
        self.assertEqual(_select_inference_precision("auto", ("FP32", "BF16")), "bf16")
        self.assertEqual(_select_inference_precision("auto", ("FP32",)), "fp32")
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            _select_inference_precision("bf16", ("FP32",))

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
                    deepseek_evidence_path=output / "unused.jsonl",
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
            self.assertEqual(result["measured_requests"], 1)
            self.assertGreater(result["qps"], 0)
            self.assertGreaterEqual(result["p95_ms"], 0)

    def test_runtime_readiness_does_not_load_benchmark_evidence(self):
        manifest_path = Path(__file__).parents[1] / "model-artifacts.json"
        model_key = "mmbert-lora-full-ctx1024-u17000-s42"
        artifact_path = mock.Mock(
            side_effect=lambda root, spec, **_: root / spec["path"]
        )
        with (
            mock.patch(
                "morgott.models.mmbert.serving.verified_artifact_path",
                artifact_path,
            ),
            mock.patch.object(MmbertRuntime, "_from_verified_files", return_value=None),
        ):
            MmbertRuntime.from_artifacts(manifest_path, model_key=model_key)
        self.assertEqual(
            [call.kwargs["name"] for call in artifact_path.call_args_list],
            ["ONNX model", "tokenizer", "serving export", "serving verification"],
        )

    def test_verification_accepts_typed_current_prompt_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel_id = "row-1"
            panel_manifest = {"panel": {"sha256": "a" * 64}}
            (root / "followup_manifest.json").write_text(
                json.dumps(
                    {
                        "panel_sha256": panel_manifest["panel"]["sha256"],
                        "split": {"calibration_panel_ids": [panel_id]},
                    }
                ),
                encoding="utf-8",
            )
            evidence_path = root / "current-results.jsonl"
            evidence_path.write_text(
                json.dumps(
                    {
                        "attempts": 1,
                        "client_seconds": 0.1,
                        "dataset": "canonical",
                        "failure_code": None,
                        "input_channel": "direct_user",
                        "input_tokens": 10,
                        "job_id": hashlib.sha256(
                            f"{PROMPT_SHA256}\0{EVALUATION_REQUEST_SHA256}\0{panel_id}".encode()
                        ).hexdigest(),
                        "log_odds_subversion": 1.0,
                        "model": DEEPSEEK_MODEL,
                        "output_tokens": 5,
                        "p_subversion": 1 / (1 + math.exp(-1)),
                        "panel_id": panel_id,
                        "prompt_sha256": PROMPT_SHA256,
                        "provider": DEEPSEEK_PROVIDER,
                        "request_sha256": EVALUATION_REQUEST_SHA256,
                        "status": "ok",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            records, calibration_ids = _deepseek_evidence(
                root,
                panel_manifest,
                evidence_path=evidence_path,
                panel=[
                    {
                        "panel_id": panel_id,
                        "dataset": "canonical",
                        "input_channel": "direct_user",
                    }
                ],
            )
            invalid = json.loads(evidence_path.read_text(encoding="utf-8"))
            invalid["attempts"] = MAX_ATTEMPTS + 1
            evidence_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            panel = [
                {
                    "panel_id": panel_id,
                    "dataset": "canonical",
                    "input_channel": "direct_user",
                }
            ]
            with self.assertRaisesRegex(ValueError, "identity is invalid"):
                _deepseek_evidence(
                    root,
                    panel_manifest,
                    evidence_path=evidence_path,
                    panel=panel,
                )
            valid_line = json.dumps({**invalid, "attempts": 1}, sort_keys=True) + "\n"
            evidence_path.write_text(valid_line * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate DeepSeek evidence"):
                _deepseek_evidence(
                    root,
                    panel_manifest,
                    evidence_path=evidence_path,
                    panel=panel,
                )
            evidence_path.write_text(valid_line, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match the frozen panel"):
                _deepseek_evidence(
                    root,
                    panel_manifest,
                    evidence_path=evidence_path,
                    panel=[{**panel[0], "dataset": "promptshield"}],
                )
            (root / "followup_manifest.json").write_text(
                json.dumps(
                    {
                        "panel_sha256": "b" * 64,
                        "split": {"calibration_panel_ids": [panel_id]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "retained DeepSeek split changed"):
                _deepseek_evidence(
                    root,
                    panel_manifest,
                    evidence_path=evidence_path,
                    panel=panel,
                )
            (root / "followup_manifest.json").write_text(
                json.dumps(
                    {
                        "panel_sha256": panel_manifest["panel"]["sha256"],
                        "split": {"calibration_panel_ids": ["row-2"]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "retained DeepSeek split changed"):
                _deepseek_evidence(
                    root,
                    panel_manifest,
                    evidence_path=evidence_path,
                    panel=panel,
                )

        self.assertEqual(set(records), {panel_id})
        self.assertEqual(calibration_ids, {panel_id})

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
            {
                "panel_id": "a",
                "label": 0,
                "dataset": "canonical",
                "input_channel": "direct_user",
            },
            {
                "panel_id": "b",
                "label": 1,
                "dataset": "promptshield",
                "input_channel": "direct_user",
            },
            {
                "panel_id": "c",
                "label": 0,
                "dataset": "sep",
                "input_channel": "untrusted_content",
            },
            {
                "panel_id": "d",
                "label": 1,
                "dataset": "canonical",
                "input_channel": "direct_user",
            },
            # Middle zone only under the 0.025 untrusted floor: a regression to a
            # single 0.2 floor would pass this row locally with no provider call.
            {
                "panel_id": "e",
                "label": 0,
                "dataset": "sep",
                "input_channel": "untrusted_content",
            },
        ]
        records = {
            "a": {"status": "ok", "p_subversion": 0.95},
            "b": {"status": "ok", "p_subversion": 0.01},
            "c": {"status": "ok", "p_subversion": 0.1},
            "d": {"status": "timeout"},
            "e": {"status": "ok", "p_subversion": 0.01},
        }

        metrics = _cascade_metrics(
            panel,
            np.asarray([0.1, 0.99999, 0.5, 0.5, 0.15]),
            records,
            {"a", "b"},
        )

        self.assertEqual(metrics["calibration"]["true_positive"], 1)
        self.assertEqual(metrics["evaluation"]["true_positive"], 1)
        self.assertEqual(metrics["evaluation"]["false_positive"], 0)
        self.assertEqual(metrics["evaluation"]["provider_call_rows"], 3)
        self.assertEqual(metrics["evaluation"]["provider_failures"], 1)

        with self.assertRaisesRegex(ValueError, "probability"):
            _cascade_metrics(
                [
                    {
                        "panel_id": "invalid",
                        "label": 1,
                        "dataset": "canonical",
                        "input_channel": "direct_user",
                    }
                ],
                np.asarray([0.5]),
                {"invalid": {"status": "ok", "p_subversion": np.nan}},
                set(),
            )

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
                    "mmbert-lora-full-ctx1024-u17000-s42": {
                        "serving": {
                            "format": "onnx-openvino-v1",
                            "inference_precision": "bf16",
                            "max_tokens": 1024,
                            "window_overlap": 128,
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
                MmbertRuntime.from_artifacts(
                    path,
                    model_key="mmbert-lora-full-ctx1024-u17000-s42",
                )

    def test_runtime_uses_the_registered_context_length(self):
        tokenizer = _Tokenizer(_Encoding([101, 1, 102], [(0, 0), (0, 1), (0, 0)]))

        runtime = MmbertRuntime(
            tokenizer=tokenizer,
            session=None,
            max_tokens=1024,
            window_overlap=128,
        )

        self.assertEqual(runtime.max_tokens, 1024)
        self.assertEqual(tokenizer.truncation, (1024, 128))
        with self.assertRaisesRegex(ValueError, "context contract"):
            MmbertRuntime(tokenizer=tokenizer, session=None, max_tokens=512)

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
        self.assertEqual(tokenizer.truncation, (1024, 128))
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

    def test_preparation_keeps_registered_strict_normalization(self):
        text = "Привет, как ваши дела?"
        tokenizer = _Tokenizer(
            _Encoding([101, 1, 102], [(0, 0), (0, len(text)), (0, 0)])
        )

        prepared = MmbertRuntime(tokenizer=tokenizer, session=None).prepare(text)

        self.assertEqual(prepared.normalized_text, strict_normalize(text))

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

    def test_batch_scoring_preserves_window_order(self):
        tokenizer = _Tokenizer(_Encoding([101, 1, 102], [(0, 0), (0, 1), (0, 0)]))
        runtime = MmbertRuntime(tokenizer=tokenizer, session=_BatchSession())
        windows = (
            Window(0, 0, 2, (101, 2, 3, 102), (1, 1, 1, 1)),
            Window(1, 0, 1, (101, 0, 102), (1, 1, 1)),
        )

        scores = runtime.score_batch(windows, batch_size=2)

        self.assertAlmostEqual(scores[0], 0.880797, places=6)
        self.assertEqual(scores[1], 0.5)

    def _fake_serving_modules(self, compile_calls, loaded_from_cache=False):
        class _CompiledPort:
            def __init__(self, name):
                self.name = name

            def get_any_name(self):
                return self.name

        class _CompiledModel:
            inputs = (_CompiledPort("input_ids"), _CompiledPort("attention_mask"))
            outputs = ("logit",)

            def get_property(self, name):
                return {
                    "INFERENCE_PRECISION_HINT": "bf16",
                    "INFERENCE_NUM_THREADS": 2,
                    "LOADED_FROM_CACHE": loaded_from_cache,
                }[name]

        class _Core:
            def get_property(self, device, name):
                return ("BF16",)

            def compile_model(self, path, device, properties):
                compile_calls.append((path, device, properties))
                return _CompiledModel()

        fake_openvino = types.SimpleNamespace(
            __version__="2026.3.0-releases/2026/3",
            Core=_Core,
            Type=types.SimpleNamespace(bf16="bf16", f32="f32"),
        )
        fake_tokenizers = types.SimpleNamespace(
            Tokenizer=types.SimpleNamespace(
                from_file=lambda path: _Tokenizer(
                    _Encoding([101, 1, 102], [(0, 0), (0, 1), (0, 0)])
                )
            )
        )
        return {"openvino": fake_openvino, "tokenizers": fake_tokenizers}

    def _from_verified_files_with_fakes(
        self, temporary, compile_calls, environ, loaded_from_cache=False
    ):
        with (
            mock.patch.dict(
                "sys.modules",
                self._fake_serving_modules(compile_calls, loaded_from_cache),
            ),
            mock.patch.dict("os.environ", environ),
        ):
            return MmbertRuntime._from_verified_files(
                Path(temporary) / "model.onnx",
                Path(temporary) / "tokenizer.json",
                onnx_sha256="a" * 64,
                tokenizer_sha256="b" * 64,
                model_key="mmbert-lora-full-ctx1024-u17000-s42",
                max_tokens=1024,
                window_overlap=128,
            )

    def test_compile_uses_a_digest_keyed_openvino_cache_dir(self):
        compile_calls = []
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._from_verified_files_with_fakes(
                temporary,
                compile_calls,
                {"XDG_CACHE_HOME": temporary, "MORGOTT_NO_COMPILE_CACHE": ""},
            )

            # b55902d6 is sha256("BF16")[:8], the fake CPU capability tuple.
            expected = str(
                Path(temporary)
                / "morgott"
                / "openvino"
                / f"{'a' * 64}-2026.3.0-releases-2026-3-bf16-b55902d6"
            )
            self.assertEqual(len(compile_calls), 1)
            _, device, properties = compile_calls[0]
            self.assertEqual(device, "CPU")
            self.assertEqual(properties["CACHE_DIR"], expected)
            self.assertEqual(properties["PERFORMANCE_HINT"], "LATENCY")
            self.assertEqual(properties["INFERENCE_PRECISION_HINT"], "bf16")
            self.assertTrue(Path(expected).is_dir())
            self.assertEqual(runtime.identity.onnx_sha256, "a" * 64)
            self.assertIs(runtime.identity.loaded_from_cache, False)

    def test_identity_records_a_cache_import_reported_by_openvino(self):
        compile_calls = []
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._from_verified_files_with_fakes(
                temporary,
                compile_calls,
                {"XDG_CACHE_HOME": temporary, "MORGOTT_NO_COMPILE_CACHE": ""},
                loaded_from_cache=True,
            )

            self.assertIs(runtime.identity.loaded_from_cache, True)

    def test_compile_cache_degrades_or_disables_without_failing_the_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            blocking_file = Path(temporary) / "not-a-directory"
            blocking_file.write_text("")

            # An unusable cache location degrades to an uncached compile.
            compile_calls = []
            self._from_verified_files_with_fakes(
                temporary,
                compile_calls,
                {
                    "XDG_CACHE_HOME": str(blocking_file),
                    "MORGOTT_NO_COMPILE_CACHE": "",
                },
            )
            self.assertNotIn("CACHE_DIR", compile_calls[0][2])

            # The opt-out keeps the verified-bytes-only compile path.
            compile_calls = []
            self._from_verified_files_with_fakes(
                temporary,
                compile_calls,
                {
                    "XDG_CACHE_HOME": temporary,
                    "MORGOTT_NO_COMPILE_CACHE": "1",
                },
            )
            self.assertNotIn("CACHE_DIR", compile_calls[0][2])

            # A relative XDG_CACHE_HOME is ignored per the XDG spec, falling
            # back to HOME/.cache instead of a cwd-relative directory.
            compile_calls = []
            self._from_verified_files_with_fakes(
                temporary,
                compile_calls,
                {
                    "XDG_CACHE_HOME": "relative-cache",
                    "HOME": temporary,
                    "MORGOTT_NO_COMPILE_CACHE": "",
                },
            )
            self.assertTrue(
                compile_calls[0][2]["CACHE_DIR"].startswith(
                    str(Path(temporary) / ".cache" / "morgott" / "openvino")
                )
            )

    @unittest.skipUnless(
        importlib.util.find_spec("tokenizers"),
        "cascade tokenizer dependency is not installed",
    )
    def test_real_tokenizer_covers_long_input_and_boundary_spanning_text(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing

        tokens = [f"t{index}" for index in range(1400)]
        tokens[1018:1026] = [
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

        self.assertEqual(prepared.token_count, 1400)
        self.assertGreater(len(prepared.windows), 1)
        self.assertEqual(prepared.windows[0].char_start, 0)
        self.assertEqual(prepared.windows[-1].char_end, len(prepared.normalized_text))
        self.assertTrue(
            any(
                "attack crosses the window boundary without being cut"
                in prepared.normalized_text[window.char_start : window.char_end]
                for window in prepared.windows
            )
        )


if __name__ == "__main__":
    unittest.main()
