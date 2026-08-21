"""CPU serving primitives for the retained advisory mmBERT cascade."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...normalization import strict_normalize
from ..downstream import subversion_probability
from .core import ATTENTION_IMPLEMENTATION
from .inference import verified_artifact_path

DEFAULT_MODEL_KEY = "mmbert-lora-full-ctx1024-u17000-s42"
MODEL_MAX_TOKENS = 1024
WINDOW_OVERLAP = 128
SERVING_FORMAT = "onnx-openvino-v1"
EXPORT_FORMAT = "onnx-fp32-v1"
VERIFICATION_FORMAT = "openvino-cpu-bf16-panel-study-v2"


def _select_inference_precision(
    requested: Literal["auto", "bf16", "fp32"],
    cpu_capabilities: tuple[str, ...],
) -> Literal["bf16", "fp32"]:
    if requested == "auto":
        return "bf16" if "BF16" in cpu_capabilities else "fp32"
    if requested == "bf16" and "BF16" not in cpu_capabilities:
        raise RuntimeError("the deployment CPU does not support OpenVINO BF16")
    return requested


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    model_key: str
    runtime: str
    onnx_sha256: str
    tokenizer_sha256: str
    compile_seconds: float
    openvino: str
    requested_inference_precision: str
    inference_precision: str
    reported_inference_precision: str
    threads: int
    cpu_capabilities: tuple[str, ...]
    max_tokens: int
    window_overlap: int
    loaded_from_cache: bool


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    char_start: int
    char_end: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedText:
    normalized_text: str
    token_count: int
    windows: tuple[Window, ...]


def _score_windows(
    windows: tuple[Window, ...],
    *,
    batch_size: int,
    infer: Callable[[dict[str, np.ndarray]], object],
) -> tuple[float, ...]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores = [0.0] * len(windows)
    order = sorted(range(len(windows)), key=lambda index: len(windows[index].input_ids))
    for start in range(0, len(order), batch_size):
        indexes = order[start : start + batch_size]
        batch = [windows[index] for index in indexes]
        length = max(len(window.input_ids) for window in batch)
        input_ids = np.zeros((len(batch), length), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for row, window in enumerate(batch):
            size = len(window.input_ids)
            input_ids[row, :size] = window.input_ids
            attention_mask[row, :size] = window.attention_mask
        values = np.asarray(
            infer({"input_ids": input_ids, "attention_mask": attention_mask})
        ).reshape(-1)
        if values.size != len(batch) or not np.all(np.isfinite(values)):
            raise ValueError("model returned an invalid logit")
        for index, value in zip(indexes, values, strict=True):
            scores[index] = subversion_probability(0.0, float(value))
    return tuple(scores)


class MmbertRuntime:
    """Prepare complete inputs and score every registered context window."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        session: Any,
        identity: RuntimeIdentity | None = None,
        max_tokens: int = MODEL_MAX_TOKENS,
        window_overlap: int = WINDOW_OVERLAP,
    ) -> None:
        if (
            type(max_tokens) is not int
            or max_tokens != MODEL_MAX_TOKENS
            or type(window_overlap) is not int
            or not 0 <= window_overlap < max_tokens
        ):
            raise ValueError("invalid serving context contract")
        self._tokenizer = tokenizer
        self._session = session
        self.identity = identity
        self.max_tokens = max_tokens
        self.window_overlap = window_overlap
        tokenizer.enable_truncation(
            max_length=max_tokens,
            stride=window_overlap,
        )

    @classmethod
    def from_artifacts(
        cls,
        manifest_path: Path,
        *,
        model_key: str = DEFAULT_MODEL_KEY,
        inference_precision: Literal["auto", "bf16", "fp32"] = "bf16",
    ) -> MmbertRuntime:
        if inference_precision not in {"auto", "bf16", "fp32"}:
            raise ValueError("inference precision must be auto, bf16, or fp32")
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 2
            or manifest.get("advisory_only") is not True
        ):
            raise ValueError("model registry contract failed")
        entry = manifest.get("models", {}).get(model_key)
        serving = entry.get("serving") if isinstance(entry, dict) else None
        if (
            not isinstance(serving, dict)
            or serving.get("format") != SERVING_FORMAT
            or serving.get("inference_precision") != "bf16"
            or serving.get("max_tokens") != MODEL_MAX_TOKENS
            or type(serving.get("window_overlap")) is not int
            or not 0 <= serving["window_overlap"] < serving["max_tokens"]
        ):
            raise ValueError(f"{model_key} has no verified ONNX serving artifact")

        root = manifest_path.parent
        onnx_path = verified_artifact_path(
            root,
            serving.get("onnx"),
            name="ONNX model",
        )
        tokenizer_path = verified_artifact_path(
            root,
            serving.get("tokenizer"),
            name="tokenizer",
        )
        tokenizer_sha256 = serving["tokenizer"]["sha256"]
        if tokenizer_sha256 != manifest.get("base_model", {}).get(
            "tokenizer_json_sha256"
        ):
            raise ValueError("serving tokenizer differs from the registered base model")
        evidence = {
            name: json.loads(
                verified_artifact_path(
                    root,
                    serving.get(name),
                    name=f"serving {name}",
                ).read_text(encoding="utf-8")
            )
            for name in ("export", "verification")
        }
        exported = evidence["export"]
        verification = evidence["verification"]
        parity = exported.get("representative_parity", {})
        sequence_lengths = parity.get("sequence_lengths", ())
        comparisons = parity.get("comparisons", {})
        if (
            exported.get("format") != EXPORT_FORMAT
            or exported.get("model_key") != model_key
            or exported.get("source_attention_implementation")
            != ATTENTION_IMPLEMENTATION
            or exported.get("export_attention_implementation") != "eager"
            or exported.get("source_result_sha256")
            != entry.get("result", {}).get("sha256")
            or exported.get("onnx", {}).get("sha256") != serving["onnx"]["sha256"]
            or exported.get("tokenizer", {}).get("sha256") != tokenizer_sha256
            or exported.get("max_tokens", serving["max_tokens"])
            != serving["max_tokens"]
            or exported.get("window_overlap", serving["window_overlap"])
            != serving["window_overlap"]
            or serving["max_tokens"] not in sequence_lengths
            or not any(
                0 < length < serving["max_tokens"] for length in sequence_lengths
            )
            or parity.get("rtol") != 1e-4
            or parity.get("atol") != 1e-4
            or set(comparisons) != {"sdpa_to_eager", "sdpa_to_onnx", "eager_to_onnx"}
            or any(
                not isinstance(value, dict) or value.get("passed") is not True
                for value in comparisons.values()
            )
            or verification.get("format") != VERIFICATION_FORMAT
            or verification.get("model_key") != model_key
            or verification.get("source_onnx_sha256") != serving["onnx"]["sha256"]
            or verification.get("tokenizer_sha256") != tokenizer_sha256
            or verification.get("quality_gate", {}).get("passed") is not True
        ):
            raise ValueError("registered serving evidence contract failed")
        return cls._from_verified_files(
            onnx_path,
            tokenizer_path,
            onnx_sha256=serving["onnx"]["sha256"],
            tokenizer_sha256=tokenizer_sha256,
            model_key=model_key,
            max_tokens=serving["max_tokens"],
            window_overlap=serving["window_overlap"],
            inference_precision=inference_precision,
        )

    @classmethod
    def _from_verified_files(
        cls,
        onnx_path: Path,
        tokenizer_path: Path,
        *,
        onnx_sha256: str,
        tokenizer_sha256: str,
        model_key: str,
        max_tokens: int,
        window_overlap: int,
        inference_precision: Literal["auto", "bf16", "fp32"] = "bf16",
    ) -> MmbertRuntime:
        try:
            import openvino as ov
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("install the cascade extra to serve mmBERT") from error

        core = ov.Core()
        cpu_capabilities = tuple(
            str(value)
            for value in core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
        )
        selected_precision = _select_inference_precision(
            inference_precision,
            cpu_capabilities,
        )
        properties: dict[str, Any] = {
            "INFERENCE_PRECISION_HINT": (
                ov.Type.bf16 if selected_precision == "bf16" else ov.Type.f32
            ),
            "PERFORMANCE_HINT": "LATENCY",
        }
        # The compiled-blob cache key derives from the caller's pinned-digest
        # verification of model.onnx, so a cached blob can never be reused for
        # a different model, OpenVINO version, precision, or CPU feature set
        # (guarding shared cache homes across machines). The blob itself is
        # imported without digest verification and is trusted local runtime
        # state (docs/threat-model.md); MORGOTT_NO_COMPILE_CACHE=1 keeps the
        # verified-bytes-only compile. A relative XDG_CACHE_HOME is ignored
        # per the XDG base directory specification.
        # ponytail: about 1.2GB of blob per cache key with no eviction; add a
        # cleanup policy if disk pressure ever matters.
        if not os.environ.get("MORGOTT_NO_COMPILE_CACHE"):
            cpu_key = hashlib.sha256("|".join(cpu_capabilities).encode()).hexdigest()
            cache_key = (
                f"{onnx_sha256}-{ov.__version__}-{selected_precision}-{cpu_key[:8]}"
            )
            try:
                base = Path(os.environ.get("XDG_CACHE_HOME") or "")
                if not base.is_absolute():
                    base = Path.home() / ".cache"
                cache_dir = base / "morgott" / "openvino" / cache_key.replace("/", "-")
                cache_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, RuntimeError):
                pass
            else:
                properties["CACHE_DIR"] = str(cache_dir)
        compile_started = time.perf_counter()
        compiled_model = core.compile_model(onnx_path, "CPU", properties)
        compile_seconds = time.perf_counter() - compile_started
        if {value.get_any_name() for value in compiled_model.inputs} != {
            "input_ids",
            "attention_mask",
        } or len(compiled_model.outputs) != 1:
            raise ValueError("ONNX model interface does not match the serving contract")
        return cls(
            tokenizer=Tokenizer.from_file(str(tokenizer_path)),
            session=compiled_model,
            identity=RuntimeIdentity(
                model_key=model_key,
                runtime=f"openvino-{ov.__version__}-cpu-{selected_precision}",
                onnx_sha256=onnx_sha256,
                tokenizer_sha256=tokenizer_sha256,
                compile_seconds=compile_seconds,
                openvino=ov.__version__,
                requested_inference_precision=inference_precision,
                inference_precision=selected_precision,
                reported_inference_precision=str(
                    compiled_model.get_property("INFERENCE_PRECISION_HINT")
                ),
                threads=int(compiled_model.get_property("INFERENCE_NUM_THREADS")),
                cpu_capabilities=cpu_capabilities,
                max_tokens=max_tokens,
                window_overlap=window_overlap,
                loaded_from_cache=bool(
                    compiled_model.get_property("LOADED_FROM_CACHE")
                ),
            ),
            max_tokens=max_tokens,
            window_overlap=window_overlap,
        )

    def prepare(self, text: str) -> PreparedText:
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")

        # ponytail: normalization is O(N) memory; make it stateful only if measured
        # artifact sizes require it.
        normalized = strict_normalize(text)
        if not normalized:
            raise ValueError("text is empty after strict normalization")

        first = self._tokenizer.encode(normalized)
        encodings = (first, *first.overflowing)
        # Overflow windows overlap, so unique token-and-offset pairs prevent
        # token_count from charging the overlap more than once.
        seen_tokens: set[tuple[int, tuple[int, int]]] = set()
        previous_tokens = None
        windows = []
        for index, encoding in enumerate(encodings):
            content = [
                (token_id, offset)
                for token_id, offset in zip(
                    encoding.ids,
                    encoding.offsets,
                    strict=True,
                )
                if offset[1] > offset[0]
            ]
            if not content:
                raise ValueError("tokenizer produced an empty content window")
            content_tokens = set(content)
            char_start = min(offset[0] for _, offset in content)
            char_end = max(offset[1] for _, offset in content)
            if previous_tokens is not None and (
                not previous_tokens & content_tokens or char_end <= windows[-1].char_end
            ):
                raise ValueError(
                    "tokenizer overflow windows are not ordered and overlapping"
                )
            seen_tokens.update(content_tokens)
            windows.append(
                Window(
                    index=index,
                    char_start=char_start,
                    char_end=char_end,
                    input_ids=tuple(encoding.ids),
                    attention_mask=tuple(encoding.attention_mask),
                )
            )
            previous_tokens = content_tokens

        if windows[0].char_start != 0 or windows[-1].char_end != len(normalized):
            raise ValueError("tokenizer windows do not cover the complete input")

        return PreparedText(
            normalized_text=normalized,
            token_count=len(seen_tokens),
            windows=tuple(windows),
        )

    def score(self, windows: tuple[Window, ...]) -> tuple[float, ...]:
        return self.score_batch(windows, batch_size=1)

    def score_batch(
        self,
        windows: tuple[Window, ...],
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        if self._session is None:
            raise RuntimeError("model session is not configured")

        def infer(inputs: dict[str, np.ndarray]):
            outputs = self._session(inputs)
            if len(outputs) != 1:
                raise ValueError("model must return one logit output")
            return next(iter(outputs.values()))

        return _score_windows(windows, batch_size=batch_size, infer=infer)
