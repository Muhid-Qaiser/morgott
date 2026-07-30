"""CPU serving primitives for the retained advisory mmBERT cascade."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...normalization import strict_normalize
from .core import MAX_TOKENS
from .inference import verified_artifact_path

DEFAULT_MODEL_KEY = "mmbert-lora-full-s42"
WINDOW_OVERLAP = 128
SERVING_FORMAT = "onnx-openvino-bf16-v1"


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    runtime: str
    onnx_sha256: str
    tokenizer_sha256: str
    compile_seconds: float
    openvino: str
    reported_inference_precision: str
    threads: int
    cpu_capabilities: tuple[str, ...]


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


class MmbertRuntime:
    """Prepare complete inputs and score every 512-token window."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        session: Any,
        identity: RuntimeIdentity | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._session = session
        self.identity = identity
        tokenizer.enable_truncation(
            max_length=MAX_TOKENS,
            stride=WINDOW_OVERLAP,
        )

    @classmethod
    def from_artifacts(
        cls,
        manifest_path: Path,
    ) -> MmbertRuntime:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 2
            or manifest.get("advisory_only") is not True
        ):
            raise ValueError("model registry contract failed")
        entry = manifest.get("models", {}).get(DEFAULT_MODEL_KEY)
        serving = entry.get("serving") if isinstance(entry, dict) else None
        if (
            not isinstance(serving, dict)
            or serving.get("format") != SERVING_FORMAT
            or serving.get("inference_precision") != "bf16"
        ):
            raise ValueError(
                f"{DEFAULT_MODEL_KEY} has no verified ONNX serving artifact"
            )

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
        return cls._from_verified_files(
            onnx_path,
            tokenizer_path,
            onnx_sha256=serving["onnx"]["sha256"],
            tokenizer_sha256=tokenizer_sha256,
        )

    @classmethod
    def _from_verified_files(
        cls,
        onnx_path: Path,
        tokenizer_path: Path,
        *,
        onnx_sha256: str,
        tokenizer_sha256: str,
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
        if "BF16" not in cpu_capabilities:
            raise RuntimeError("the deployment CPU does not support OpenVINO BF16")
        compile_started = time.perf_counter()
        compiled_model = core.compile_model(
            onnx_path,
            "CPU",
            {
                "INFERENCE_PRECISION_HINT": "bf16",
                "PERFORMANCE_HINT": "LATENCY",
            },
        )
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
                runtime=f"openvino-{ov.__version__}-cpu-bf16",
                onnx_sha256=onnx_sha256,
                tokenizer_sha256=tokenizer_sha256,
                compile_seconds=compile_seconds,
                openvino=ov.__version__,
                reported_inference_precision=str(
                    compiled_model.get_property("INFERENCE_PRECISION_HINT")
                ),
                threads=int(compiled_model.get_property("INFERENCE_NUM_THREADS")),
                cpu_capabilities=cpu_capabilities,
            ),
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
        if self._session is None:
            raise RuntimeError("model session is not configured")
        scores = []
        for window in windows:
            outputs = self._session(
                {
                    "input_ids": np.asarray([window.input_ids], dtype=np.int64),
                    "attention_mask": np.asarray(
                        [window.attention_mask],
                        dtype=np.int64,
                    ),
                },
            )
            if len(outputs) != 1:
                raise ValueError("model must return one logit output")
            values = np.asarray(next(iter(outputs.values()))).reshape(-1)
            if values.size != 1 or not math.isfinite(float(values[0])):
                raise ValueError("model returned an invalid logit")
            logit = float(values[0])
            if logit >= 0:
                probability = 1.0 / (1.0 + math.exp(-logit))
            else:
                exponent = math.exp(logit)
                probability = exponent / (1.0 + exponent)
            scores.append(probability)
        return tuple(scores)
