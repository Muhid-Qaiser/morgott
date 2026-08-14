from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.guard_baselines import adapters


class _ChoiceTokenizer:
    def __init__(self, token_ids: dict[str, list[int]]) -> None:
        self.token_ids = token_ids

    def __call__(self, value, *, add_special_tokens: bool = False) -> dict:
        del add_special_tokens
        if isinstance(value, list):
            return {"input_ids": [self.token_ids[item] for item in value]}
        return {"input_ids": self.token_ids.get(value, [900, 901])}

    @staticmethod
    def decode(ids: list[int], **kwargs) -> str:
        del kwargs
        return ":".join(str(value) for value in ids)


class _GraniteChoiceTokenizer(_ChoiceTokenizer):
    def __init__(self, token_ids: dict[str, list[int]]) -> None:
        super().__init__(token_ids)
        self.decoded = {
            values[0]: token for token, values in token_ids.items() if len(values) == 1
        }

    def __len__(self) -> int:
        return max(self.decoded) + 1

    def decode(self, ids: list[int], **kwargs) -> str:
        del kwargs
        if len(ids) == 1:
            return self.decoded.get(ids[0], "")
        return super().decode(ids)


class _BoundaryMergeTokenizer:
    """Rendered nonempty content costs one more token than empty overhead."""

    raw = {"short": [1, 2], "edge": [1, 2, 3]}

    def __call__(self, value, *, add_special_tokens: bool = False) -> dict:
        del add_special_tokens
        values = value if isinstance(value, list) else [value]
        encoded = [self._encode(item) for item in values]
        return {"input_ids": encoded if isinstance(value, list) else encoded[0]}

    def _encode(self, value: str) -> list[int]:
        if not value.startswith("template:"):
            return self.raw[value]
        content = value.removeprefix("template:")
        if not content:
            return [90, 91]
        content_ids = self.raw.get(content)
        if content_ids is None:
            content_ids = [int(part) for part in content.split(":")]
        return [90, *content_ids, 91, 92]

    @staticmethod
    def decode(ids: list[int], **kwargs) -> str:
        del kwargs
        return ":".join(str(value) for value in ids)


class _StreamBoundaryMergeTokenizer(_BoundaryMergeTokenizer):
    def _encode(self, value: str) -> list[int]:
        if value == "template:":
            return [90, 92]
        return super()._encode(value)

    @staticmethod
    def apply_chat_template(messages, **kwargs) -> str:
        del kwargs
        return f"template:{messages[0]['content']}"


class RuntimeSnapshotIdentityTests(unittest.TestCase):
    CASES = {
        "granite-guardian-3.2-3b-a800m": 2,
        "qwen3guard-stream-4b": 2,
        "aprielguard": 4,
    }

    @staticmethod
    def _snapshot(root: Path, shards: int) -> tuple[Path, list[str]]:
        root.mkdir()
        names = [
            f"model-{index:05d}-of-{shards:05d}.safetensors"
            for index in range(1, shards + 1)
        ]
        for index, name in enumerate(names):
            (root / name).write_bytes(f"weights-{index}".encode())
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        f"tensor.{index}": name for index, name in enumerate(names)
                    }
                }
            ),
            encoding="utf-8",
        )
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (root / "README.md").write_text("not runtime identity\n", encoding="utf-8")
        return root, names

    def test_sharded_granite_qwen_and_apriel_have_exact_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for slug, shard_count in self.CASES.items():
                with self.subTest(slug=slug):
                    spec = adapters.BASELINES[slug]
                    snapshot, names = self._snapshot(base / slug, shard_count)
                    if spec.trust_remote_code:
                        (snapshot / "modeling_guard.py").write_text(
                            "VALUE = 1\n", encoding="utf-8"
                        )

                    identity = adapters._snapshot_identity(snapshot, spec)

                    self.assertEqual(identity["weight_files"], names)
                    self.assertEqual(
                        identity["weight_index"], "model.safetensors.index.json"
                    )
                    for name in names:
                        self.assertEqual(
                            identity["files"][name]["sha256"],
                            hashlib.sha256((snapshot / name).read_bytes()).hexdigest(),
                        )
                    self.assertNotIn("README.md", identity["files"])
                    self.assertEqual(
                        "modeling_guard.py" in identity["files"],
                        spec.trust_remote_code,
                    )

    def test_same_size_weight_mutation_changes_the_authoritative_identity(self) -> None:
        spec = adapters.BASELINES["granite-guardian-3.2-3b-a800m"]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, names = self._snapshot(Path(temporary) / "snapshot", 2)
            before = adapters._snapshot_identity(snapshot, spec)
            path = snapshot / names[0]
            path.write_bytes(b"x" * path.stat().st_size)
            after = adapters._snapshot_identity(snapshot, spec)

        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertNotEqual(
            before["files"][names[0]]["sha256"],
            after["files"][names[0]]["sha256"],
        )

    def test_sharded_index_fails_closed_on_a_missing_or_unsafe_shard(self) -> None:
        spec = adapters.BASELINES["aprielguard"]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, names = self._snapshot(Path(temporary) / "snapshot", 2)
            (snapshot / names[-1]).unlink()
            with self.assertRaisesRegex(adapters.ExtractionUnavailable, "missing"):
                adapters._snapshot_identity(snapshot, spec)

            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"tensor": "../escape.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(adapters.ExtractionUnavailable, "unsafe"):
                adapters._snapshot_identity(snapshot, spec)

    def test_baseline_constructor_uses_each_registered_batch_default(self) -> None:
        for slug in self.CASES:
            with self.subTest(slug=slug):
                baseline = adapters.build_baseline(slug)
                self.assertEqual(
                    baseline.batch_size, adapters.BASELINES[slug].batch_size
                )


class ProtectAIV2AdapterTests(unittest.TestCase):
    def test_spec_pins_model_label_context_dtype_and_native_cutoff(self) -> None:
        spec = adapters.BASELINES["protectai-deberta-v3-prompt-injection-v2"]

        self.assertEqual(
            spec.repo_id,
            "protectai/deberta-v3-base-prompt-injection-v2",
        )
        self.assertEqual(
            spec.revision,
            "90c9989b1a342275dd0d1a95aad283c04e075671",
        )
        self.assertEqual(spec.adapter, "encoder")
        self.assertEqual(spec.max_tokens, 512)
        self.assertEqual(spec.dtype, "bfloat16")
        self.assertEqual(spec.native_threshold, 0.5)
        self.assertEqual(spec.positive_class, "softmax over config label INJECTION")
        self.assertFalse(spec.trust_remote_code)
        self.assertEqual(
            spec.required_snapshot_files,
            (
                "added_tokens.json",
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "spm.model",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        )

    def test_minimized_snapshot_requires_every_allowlisted_file(self) -> None:
        spec = adapters.BASELINES["protectai-deberta-v3-prompt-injection-v2"]
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            snapshot = (
                cache
                / "models--protectai--deberta-v3-base-prompt-injection-v2"
                / "snapshots"
                / spec.revision
            )
            snapshot.mkdir(parents=True)
            for name in spec.required_snapshot_files:
                (snapshot / name).write_bytes(b"pinned")

            with (
                patch(
                    "huggingface_hub.snapshot_download",
                    side_effect=RuntimeError("intentionally minimized"),
                ),
                patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)),
            ):
                self.assertEqual(adapters._snapshot(spec), snapshot)

            (snapshot / "model.safetensors").unlink()
            with (
                patch(
                    "huggingface_hub.snapshot_download",
                    side_effect=RuntimeError("intentionally minimized"),
                ),
                patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)),
                self.assertRaisesRegex(
                    adapters.ExtractionUnavailable,
                    "model.safetensors",
                ),
            ):
                adapters._snapshot(spec)

    def test_generic_encoder_resolves_injection_polarity_by_config_name(self) -> None:
        spec = adapters.BASELINES["protectai-deberta-v3-prompt-injection-v2"]
        guard = adapters.EncoderGuard(spec, batch_size=32)
        guard.model = SimpleNamespace(
            config=SimpleNamespace(
                num_labels=2,
                id2label={0: "SAFE", 1: "INJECTION"},
            )
        )

        self.assertEqual(guard._positive_index(), 1)

        guard.model.config.id2label = {0: "INJECTION", 1: "SAFE"}
        self.assertEqual(guard._positive_index(), 0)

        guard.model.config.id2label = {0: "SAFE", 1: "SAFE"}
        with self.assertRaisesRegex(
            adapters.ExtractionUnavailable,
            "exactly one unsafe class",
        ):
            guard._positive_index()

    def test_load_binds_exact_snapshot_identity_without_remote_code(self) -> None:
        spec = adapters.BASELINES["protectai-deberta-v3-prompt-injection-v2"]

        class _Model:
            config = SimpleNamespace(
                num_labels=2,
                id2label={0: "SAFE", 1: "INJECTION"},
                max_position_embeddings=512,
            )

            def to(self, device: str):
                self.device = device
                return self

            def eval(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"exact weights")
            guard = adapters.EncoderGuard(spec, batch_size=32)
            tokenizer = SimpleNamespace()
            with (
                patch.object(guard, "_smoke"),
                patch.object(adapters, "_snapshot", return_value=snapshot),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ) as load_tokenizer,
                patch(
                    "transformers.AutoModelForSequenceClassification.from_pretrained",
                    return_value=_Model(),
                ) as load_model,
            ):
                guard.load()

        for call in (load_tokenizer.call_args, load_model.call_args):
            self.assertEqual(call.args, (spec.repo_id,))
            self.assertEqual(call.kwargs["revision"], spec.revision)
            self.assertTrue(call.kwargs["local_files_only"])
            self.assertFalse(call.kwargs["trust_remote_code"])
        identity = guard.describe()["model_identity"]
        self.assertEqual(identity["positive_index"], 1)
        self.assertEqual(identity["id2label"], {"0": "SAFE", "1": "INJECTION"})
        self.assertEqual(identity["max_position_embeddings"], 512)
        self.assertEqual(identity["files"]["config.json"]["bytes"], 3)
        self.assertEqual(
            identity["files"]["model.safetensors"]["sha256"],
            hashlib.sha256(b"exact weights").hexdigest(),
        )
        self.assertEqual(
            identity["runtime_snapshot"]["weight_files"], ["model.safetensors"]
        )


class StreamHeadAdapterTests(unittest.TestCase):
    def test_rendered_cap_handles_boundary_merge_overhead(self) -> None:
        spec = adapters.BaselineSpec(
            slug="stream-overflow-test",
            repo_id="local/mock",
            revision="1" * 40,
            max_tokens=5,
            batch_size=2,
            role="test",
            positive_class="P(Unsafe)",
            adapter="stream_head",
        )
        guard = adapters.StreamHeadGuard(spec, batch_size=2)
        guard._overhead = 2
        guard._turn_end_id = 92
        guard.tokenizer = _StreamBoundaryMergeTokenizer()
        self.assertEqual(
            len(guard.tokenizer.raw["edge"]),
            spec.max_tokens - guard._overhead,
        )

        turns, overflow = guard._prepare_turns(["short", "edge"])

        self.assertEqual(turns, [[90, 1, 2, 91, 92], [90, 1, 2, 91, 92]])
        self.assertEqual(overflow, [False, True])


class KananaSafeguardAdapterTests(unittest.TestCase):
    def test_spec_pins_three_token_first_position_scalar(self) -> None:
        spec = adapters.BASELINES["kanana-safeguard-prompt-2.1b"]

        self.assertEqual(spec.repo_id, "kakaocorp/kanana-safeguard-prompt-2.1b")
        self.assertEqual(
            spec.revision,
            "167d74d4706b236580b0e48318337c7ac6ba7848",
        )
        self.assertEqual(spec.adapter, "kanana_safeguard")
        self.assertEqual(spec.max_tokens, 8192)
        self.assertEqual(spec.architectural_max_tokens, 8192)
        self.assertEqual(
            spec.batching_strategy,
            adapters.RENDERED_LENGTH_BATCHING,
        )
        self.assertEqual(spec.length_bucket_rows, 512)
        self.assertEqual(spec.attention_backend, "sdpa")
        self.assertEqual(spec.unsafe_tokens, ("<UNSAFE-A1>", "<UNSAFE-A2>"))
        self.assertEqual(spec.safe_tokens, ("<SAFE>",))
        self.assertEqual(
            dict(spec.expected_token_ids),
            {
                "<UNSAFE-A1>": 128256,
                "<UNSAFE-A2>": 128258,
                "<SAFE>": 128257,
            },
        )
        self.assertIsNone(spec.native_threshold)

    def test_labels_must_be_distinct_single_tokens(self) -> None:
        spec = adapters.BASELINES["kanana-safeguard-prompt-2.1b"]
        guard = adapters.KananaSafeguardGuard(spec, batch_size=8)
        guard.tokenizer = _ChoiceTokenizer(
            {
                "<SAFE>": [128257],
                "<UNSAFE-A1>": [128256],
                "<UNSAFE-A2>": [128258],
            }
        )

        self.assertEqual(
            guard._resolve_choice_ids(),
            ([128256, 128258], [128257]),
        )

        guard.tokenizer = _ChoiceTokenizer(
            {
                "<SAFE>": [128257],
                "<UNSAFE-A1>": [42],
                "<UNSAFE-A2>": [128258],
            }
        )
        with self.assertRaisesRegex(adapters.ExtractionUnavailable, "identity changed"):
            guard._resolve_choice_ids()

        guard.tokenizer = _ChoiceTokenizer(
            {
                "<SAFE>": [128257],
                "<UNSAFE-A1>": [1, 2],
                " <UNSAFE-A1>": [3],
                "<UNSAFE-A2>": [128258],
            }
        )
        with self.assertRaisesRegex(adapters.ExtractionUnavailable, "single token"):
            guard._resolve_choice_ids()

        guard.tokenizer = _ChoiceTokenizer(
            {
                "<SAFE>": [128257],
                "<UNSAFE-A1>": [1],
                "<UNSAFE-A2>": [1],
            }
        )
        with self.assertRaisesRegex(adapters.ExtractionUnavailable, "distinct"):
            guard._resolve_choice_ids()

    def test_primary_score_is_one_minus_safe_over_three_labels(self) -> None:
        spec = adapters.BASELINES["kanana-safeguard-prompt-2.1b"]
        guard = adapters.KananaSafeguardGuard(spec, batch_size=8)
        guard._unsafe_ids = [1, 2]
        guard._safe_ids = [0]
        fake_torch = SimpleNamespace(
            logsumexp=lambda values, dim: np.log(np.exp(values).sum(axis=dim)),
            sigmoid=lambda values: 1.0 / (1.0 + np.exp(-values)),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            score = guard._risk_probabilities(
                np.asarray([[0.0, np.log(2.0), np.log(3.0)]])
            )

        np.testing.assert_allclose(score, np.asarray([5.0 / 6.0]))

    def test_load_forces_and_verifies_sdpa(self) -> None:
        spec = adapters.BASELINES["kanana-safeguard-prompt-2.1b"]
        tokenizer = _ChoiceTokenizer(
            {
                "<SAFE>": [128257],
                "<UNSAFE-A1>": [128256],
                "<UNSAFE-A2>": [128258],
            }
        )
        tokenizer.pad_token_id = 0
        tokenizer.apply_chat_template = lambda messages, **kwargs: "template"

        class _Model:
            def __init__(self, resolved: str) -> None:
                self.config = SimpleNamespace(
                    max_position_embeddings=8192,
                    _attn_implementation=resolved,
                )

            def to(self, device: str):
                self.device = device
                return self

            def eval(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "model.safetensors").write_bytes(b"dummy weights")
            (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            guard = adapters.KananaSafeguardGuard(spec, batch_size=8)
            model = _Model("sdpa")
            with (
                patch.object(guard, "_prepare_transformers"),
                patch.object(guard, "_smoke"),
                patch.object(adapters, "_snapshot", return_value=Path(temporary)),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=model,
                ) as load_model,
            ):
                guard.load()

            self.assertEqual(load_model.call_args.kwargs["attn_implementation"], "sdpa")
            self.assertEqual(
                guard.describe()["model_identity"]["attention_backend"],
                {"requested": "sdpa", "resolved": "sdpa"},
            )

            guard = adapters.KananaSafeguardGuard(spec, batch_size=8)
            with (
                patch.object(guard, "_prepare_transformers"),
                patch.object(guard, "_smoke"),
                patch.object(adapters, "_snapshot", return_value=Path(temporary)),
                patch(
                    "transformers.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    return_value=_Model("eager"),
                ),
                self.assertRaisesRegex(
                    adapters.ExtractionUnavailable,
                    "requested attention backend 'sdpa'.*resolved 'eager'",
                ),
            ):
                guard.load()


class GraniteGuardian32AdapterTests(unittest.TestCase):
    def test_spec_is_isolated_from_granite_41_and_caps_at_8192(self) -> None:
        spec = adapters.BASELINES["granite-guardian-3.2-3b-a800m"]
        old = adapters.BASELINES["granite-guardian-4.1-8b"]

        self.assertEqual(
            spec.repo_id,
            "ibm-granite/granite-guardian-3.2-3b-a800m",
        )
        self.assertEqual(
            spec.revision,
            "3de033d89b499a18d9a573b5192bf3b967ef48c5",
        )
        self.assertEqual(spec.max_tokens, 8192)
        self.assertEqual(spec.architectural_max_tokens, 131072)
        self.assertEqual(
            spec.batching_strategy,
            adapters.RENDERED_LENGTH_BATCHING,
        )
        self.assertEqual(spec.length_bucket_rows, 512)
        self.assertEqual(spec.attention_backend, "sdpa")
        self.assertEqual(
            spec.template_kwargs,
            {"guardian_config": {"risk_name": "jailbreak"}},
        )
        self.assertEqual(
            spec.unsafe_tokens,
            ("yes", " yes", " Yes", "Yes", "YES", " YES"),
        )
        self.assertEqual(
            spec.safe_tokens,
            (" no", "no", "No", "NO", " No", " NO"),
        )
        self.assertEqual(
            dict(spec.expected_token_ids),
            {
                "yes": 7134,
                " yes": 9155,
                " Yes": 10100,
                "Yes": 10922,
                "YES": 19354,
                " YES": 24065,
                " no": 1289,
                "no": 1347,
                "No": 2023,
                "NO": 2576,
                " No": 3139,
                " NO": 4435,
            },
        )
        self.assertEqual(old.revision, "69820a3f3c8f265e2fe61b5a8fcea2146c2fcb16")
        self.assertNotEqual(spec.slug, old.slug)
        self.assertEqual(old.batching_strategy, adapters.PANEL_ORDER_BATCHING)
        self.assertIsNone(old.length_bucket_rows)
        self.assertIsNone(old.attention_backend)

    def test_guardian_config_must_change_the_pinned_template(self) -> None:
        spec = adapters.BASELINES["granite-guardian-3.2-3b-a800m"]
        guard = adapters.GraniteGuardianGuard(spec, batch_size=4)

        class _TemplateTokenizer:
            @staticmethod
            def apply_chat_template(messages, **kwargs) -> str:
                del messages
                return "configured" if "guardian_config" in kwargs else "plain"

        guard.tokenizer = _TemplateTokenizer()
        guard._verify_template()

        guard.tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: "same"
        )
        with self.assertRaisesRegex(adapters.ExtractionUnavailable, "ignores"):
            guard._verify_template()

    def test_choice_ids_match_every_vendor_trimmed_casefold_variant(self) -> None:
        spec = adapters.BASELINES["granite-guardian-3.2-3b-a800m"]
        guard = adapters.GraniteGuardianGuard(spec, batch_size=4)
        guard.tokenizer = _GraniteChoiceTokenizer(
            {token: [token_id] for token, token_id in spec.expected_token_ids}
        )

        unsafe_ids, safe_ids = guard._resolve_choice_ids()

        self.assertEqual(set(unsafe_ids), {7134, 9155, 10100, 10922, 19354, 24065})
        self.assertEqual(set(safe_ids), {1289, 1347, 2023, 2576, 3139, 4435})

        guard.tokenizer = _GraniteChoiceTokenizer(
            {
                **{token: [token_id] for token, token_id in spec.expected_token_ids},
                " yEs ": [24066],
            }
        )
        with self.assertRaisesRegex(adapters.ExtractionUnavailable, "unsafe token set"):
            guard._resolve_choice_ids()

    def test_rendered_cap_handles_boundary_merge_overhead(self) -> None:
        spec = adapters.BaselineSpec(
            slug="granite-overflow-test",
            repo_id="local/mock",
            revision="1" * 40,
            max_tokens=5,
            batch_size=2,
            role="test",
            positive_class="P(Yes)",
            adapter="granite",
            unsafe_tokens=("Yes",),
            safe_tokens=("No",),
        )
        guard = adapters.GraniteGuardianGuard(spec, batch_size=2)
        guard._overhead = 2
        guard.tokenizer = _BoundaryMergeTokenizer()
        guard._render = lambda text: f"template:{text}"
        self.assertEqual(
            len(guard.tokenizer.raw["edge"]),
            spec.max_tokens - guard._overhead,
        )

        prompts, prompt_ids, overflow = guard._prepare_prompts(["short", "edge"])

        self.assertEqual(prompts, ["template:short", "template:1:2"])
        self.assertEqual([len(ids) for ids in prompt_ids], [5, 5])
        self.assertEqual(overflow, [False, True])


if __name__ == "__main__":
    unittest.main()
