"""Local scoring and parity helpers for the pipeline benchmark."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from morgott.models.mmbert.core import pool
from morgott.models.mmbert.data import external_rows
from morgott.models.mmbert.evaluate import _load_run
from morgott.models.mmbert.serving import MmbertRuntime
from morgott.normalization import strict_normalize

ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "mmbert-lora-full-ctx1024-u17000-s42"
MODEL_DIR = ROOT / "artifacts" / "models" / MODEL_KEY
MODEL_REGISTRY = ROOT / "model-artifacts.json"
PANEL_PATH = ROOT / "artifacts" / "openrouter_downstream_eval" / "panel.jsonl.gz"
MAX_TOKENS = 1024
WINDOW_OVERLAP = 128
PROMPT_GUARD_MODEL = "meta-llama/Llama-Prompt-Guard-2-86M"
PROMPT_GUARD_REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def load_frozen_panel(path: Path = PANEL_PATH) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) != 20_000 or any("text" in row for row in rows):
        raise ValueError("frozen panel contract failed")
    panel_ids = [row.get("panel_id") for row in rows]
    if len(set(panel_ids)) != len(rows):
        raise ValueError("frozen panel identities are not unique")
    return rows


def load_frozen_texts(
    panel: list[dict[str, Any]],
    *,
    root: Path = ROOT,
) -> dict[str, str]:
    """Reload exact frozen texts from canonical sources without persisting them."""

    return {
        panel_id: row["text"]
        for panel_id, row in load_frozen_source_rows(panel, root=root).items()
    }


def load_frozen_source_rows(
    panel: list[dict[str, Any]],
    *,
    root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Reload and verify complete source rows for an in-memory benchmark run."""

    canonical: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    external: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in panel:
        target = canonical if row["dataset"] == "canonical" else external
        target[row["source"]][row["row_id"]] = row

    rows: dict[str, dict[str, Any]] = {}
    for source, needed in canonical.items():
        path = root / "data" / "sources" / f"{source}.jsonl"
        if not path.is_file():
            raise ValueError(f"canonical source is unavailable: {source}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                source_row = json.loads(line)
                frozen = needed.get(source_row.get("id"))
                if frozen is not None:
                    _accept_source_row(rows, frozen, source_row)

    external_data, _ = external_rows(root / "artifacts" / "mmbert" / "data")
    source_names = {"promptshield": "promptshield_test", "sep": "sep"}
    for dataset, source in external.items():
        source_name = source_names.get(dataset)
        if source_name is None:
            raise ValueError(f"unsupported external panel dataset: {dataset}")
        for source_row in external_data[source_name]:
            frozen = source.get(source_row.get("id"))
            if frozen is not None:
                _accept_source_row(rows, frozen, source_row)

    if len(rows) != len(panel):
        missing = sorted(
            row["panel_id"] for row in panel if row["panel_id"] not in rows
        )
        raise ValueError(f"could not reload {len(missing)} frozen panel texts")
    return rows


def _accept_source_row(
    rows: dict[str, dict[str, Any]],
    frozen: dict[str, Any],
    source_row: dict[str, Any],
) -> None:
    text = source_row.get("text")
    if (
        not isinstance(text, str)
        or not text
        or source_row.get("id") != frozen.get("row_id")
        or hashlib.sha256(text.encode()).hexdigest() != frozen.get("text_sha256")
    ):
        raise ValueError(f"frozen row changed: {frozen.get('panel_id')}")
    panel_id = frozen["panel_id"]
    if panel_id in rows:
        raise ValueError(f"duplicate frozen row: {panel_id}")
    rows[panel_id] = source_row


def score_cuda(
    panel: list[dict[str, Any]],
    texts: dict[str, str],
    *,
    batch_size: int = 24,
    artifact_batch_size: int = 256,
    model_dir: Path = MODEL_DIR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score complete artifacts as ordered 1,024/128 windows on CUDA BF16."""

    if batch_size < 1 or artifact_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", "10")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local benchmark")
    loaded_started = time.perf_counter()
    result, encoder, tokenizer, head, _ = _load_run(model_dir)
    load_seconds = time.perf_counter() - loaded_started
    encoder.eval()
    head.eval()
    torch.cuda.reset_peak_memory_stats()

    records: list[dict[str, Any]] = []
    scoring_started = time.perf_counter()
    selected_batch_size = batch_size
    total_tokens = 0
    total_windows = 0
    with torch.inference_mode():
        for start in range(0, len(panel), artifact_batch_size):
            block = panel[start : start + artifact_batch_size]
            normalized = [strict_normalize(texts[row["panel_id"]]) for row in block]
            encoded = tokenizer(
                normalized,
                add_special_tokens=True,
                max_length=MAX_TOKENS,
                padding=True,
                truncation=True,
                stride=WINDOW_OVERLAP,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            mapping = encoded.pop("overflow_to_sample_mapping").numpy()
            inputs = {
                key: value
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            logits, selected_batch_size = _cuda_logits(
                encoder,
                head,
                inputs,
                batch_size=selected_batch_size,
            )
            window_scores = _sigmoid(logits)
            for block_index, row in enumerate(block):
                indices = np.flatnonzero(mapping == block_index)
                if not len(indices):
                    raise ValueError("tokenizer produced no window for an artifact")
                scores = [float(window_scores[index]) for index in indices]
                token_count = len(
                    tokenizer.encode(normalized[block_index], add_special_tokens=False)
                )
                total_tokens += token_count
                total_windows += len(indices)
                records.append(
                    {
                        "artifact_id": row["panel_id"],
                        "dataset": row["dataset"],
                        "source": row["source"],
                        "input_channel": row["input_channel"],
                        "label": int(row["label"]),
                        "text_sha256": row["text_sha256"],
                        "token_count": token_count,
                        "window_count": len(indices),
                        "window_scores": scores,
                        "local_score": max(scores),
                    }
                )

    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - scoring_started
    identity = {
        "model_key": MODEL_KEY,
        "result_sha256": file_sha256(model_dir / "result.json"),
        "head_sha256": file_sha256(model_dir / "head.safetensors"),
        "max_tokens": MAX_TOKENS,
        "window_overlap": WINDOW_OVERLAP,
        "attention_implementation": result["attention_implementation"],
        "runtime": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
            "requested_batch_size": batch_size,
            "selected_batch_size": selected_batch_size,
            "artifact_batch_size": artifact_batch_size,
            "tokenizer_workers": int(os.environ["RAYON_NUM_THREADS"]),
            "load_seconds": load_seconds,
            "score_seconds": score_seconds,
            "artifacts_per_second": len(panel) / score_seconds,
            "input_tokens_per_second": total_tokens / score_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "artifacts": len(panel),
            "windows": total_windows,
            "input_tokens": total_tokens,
        },
    }
    return records, identity


def _cuda_logits(
    encoder,
    head,
    inputs: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    import torch

    selected = batch_size
    while True:
        try:
            values = []
            for start in range(0, len(inputs["input_ids"]), selected):
                batch = {
                    key: value[start : start + selected].to("cuda")
                    for key, value in inputs.items()
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = encoder(**batch).last_hidden_state
                    logits = head(pool(hidden, batch["attention_mask"]))[:, 0]
                values.append(logits.float().cpu().numpy())
            return np.concatenate(values).astype(np.float64), selected
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if selected == 1:
                raise
            selected = max(1, selected // 2)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    scores = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    scores[~positive] = exponent / (1.0 + exponent)
    return scores


def score_prompt_guard(
    panel: list[dict[str, Any]],
    texts: dict[str, str],
    *,
    batch_size: int = 32,
    artifact_batch_size: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score Prompt Guard 2 with 512-token windows and 64-token overlap."""

    if batch_size < 1 or artifact_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Prompt Guard benchmark")
    loaded_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        PROMPT_GUARD_MODEL,
        revision=PROMPT_GUARD_REVISION,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        PROMPT_GUARD_MODEL,
        revision=PROMPT_GUARD_REVISION,
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - loaded_started
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    records = []
    total_tokens = 0
    total_windows = 0
    with torch.inference_mode():
        for start in range(0, len(panel), artifact_batch_size):
            block = panel[start : start + artifact_batch_size]
            block_texts = [texts[row["panel_id"]] for row in block]
            encoded = tokenizer(
                block_texts,
                add_special_tokens=True,
                max_length=512,
                padding=True,
                truncation=True,
                stride=64,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            mapping = encoded.pop("overflow_to_sample_mapping").numpy()
            window_scores = []
            for offset in range(0, len(encoded["input_ids"]), batch_size):
                inputs = {
                    key: value[offset : offset + batch_size].to("cuda")
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask"}
                }
                probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1)
                window_scores.extend(probabilities[:, 1].cpu().numpy())
            for block_index, row in enumerate(block):
                indices = np.flatnonzero(mapping == block_index)
                scores = [float(window_scores[index]) for index in indices]
                token_count = len(
                    tokenizer.encode(block_texts[block_index], add_special_tokens=False)
                )
                total_tokens += token_count
                total_windows += len(indices)
                records.append(
                    {
                        "artifact_id": row["panel_id"],
                        "dataset": row["dataset"],
                        "source": row["source"],
                        "input_channel": row["input_channel"],
                        "label": int(row["label"]),
                        "text_sha256": row["text_sha256"],
                        "token_count": token_count,
                        "window_count": len(indices),
                        "window_scores": scores,
                        "local_score": max(scores),
                    }
                )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return records, {
        "model": PROMPT_GUARD_MODEL,
        "revision": PROMPT_GUARD_REVISION,
        "max_tokens": 512,
        "window_overlap": 64,
        "runtime": {
            "device": torch.cuda.get_device_name(),
            "dtype": "float16",
            "batch_size": batch_size,
            "load_seconds": load_seconds,
            "score_seconds": seconds,
            "artifacts_per_second": len(panel) / seconds,
            "input_tokens_per_second": total_tokens / seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "artifacts": len(panel),
            "windows": total_windows,
            "input_tokens": total_tokens,
        },
    }


def openvino_parity(
    panel: list[dict[str, Any]],
    texts: dict[str, str],
    cuda_records: list[dict[str, Any]],
    *,
    sample_ids: list[str],
    batch_size: int = 24,
) -> dict[str, Any]:
    """Compare registered CUDA decisions with the maintained OpenVINO runtime."""

    if len(sample_ids) != len(set(sample_ids)) or not sample_ids:
        raise ValueError("parity sample IDs must be unique and non-empty")
    by_id = {row["panel_id"]: row for row in panel}
    cuda = {record["artifact_id"]: record for record in cuda_records}
    runtime = MmbertRuntime.from_artifacts(
        MODEL_REGISTRY,
        model_key=MODEL_KEY,
        inference_precision="auto",
    )
    started = time.perf_counter()
    differences = []
    decisions = {"0.1": 0, "0.2": 0, "0.99999": 0}
    token_count = 0
    for panel_id in sample_ids:
        if panel_id not in by_id or panel_id not in cuda:
            raise ValueError(f"missing parity input: {panel_id}")
        prepared = runtime.prepare(texts[panel_id])
        scores = runtime.score_batch(prepared.windows, batch_size=batch_size)
        openvino_score = max(scores)
        cuda_score = float(cuda[panel_id]["local_score"])
        differences.append(abs(cuda_score - openvino_score))
        token_count += prepared.token_count
        for threshold in decisions:
            decisions[threshold] += (cuda_score >= float(threshold)) != (
                openvino_score >= float(threshold)
            )
    seconds = time.perf_counter() - started
    return {
        "rows": len(sample_ids),
        "runtime": asdict(runtime.identity) if runtime.identity is not None else None,
        "seconds": seconds,
        "artifacts_per_second": len(sample_ids) / seconds,
        "input_tokens_per_second": token_count / seconds,
        "absolute_score_delta": {
            "mean": float(np.mean(differences)),
            "p95": float(np.quantile(differences, 0.95)),
            "maximum": float(np.max(differences)),
        },
        "decision_disagreements": decisions,
        "decision_disagreement_rate": {
            threshold: count / len(sample_ids) for threshold, count in decisions.items()
        },
    }
