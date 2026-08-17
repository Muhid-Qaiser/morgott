"""Full registered OpenVINO replay when CUDA parity exceeds the split trigger."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.pipeline_benchmark import local
from morgott.models.mmbert.serving import MmbertRuntime

DEFAULT_OUTPUT = local.ROOT / "artifacts" / "pipeline_benchmark" / "20260816"


def _split_scores(lengths: list[int], scores: list[float]) -> list[list[float]]:
    if sum(lengths) != len(scores) or any(length < 1 for length in lengths):
        raise ValueError("OpenVINO window score count mismatch")
    result = []
    offset = 0
    for length in lengths:
        result.append(scores[offset : offset + length])
        offset += length
    return result


def score(
    panel: list[dict[str, Any]], texts: dict[str, str], *, artifact_batch_size: int = 64
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = MmbertRuntime.from_artifacts(
        local.MODEL_REGISTRY,
        model_key=local.MODEL_KEY,
        inference_precision="auto",
    )
    started = time.perf_counter()
    records = []
    total_tokens = 0
    total_windows = 0
    for start in range(0, len(panel), artifact_batch_size):
        block = panel[start : start + artifact_batch_size]
        prepared = [runtime.prepare(texts[row["panel_id"]]) for row in block]
        lengths = [len(value.windows) for value in prepared]
        window_scores = runtime.score_batch(
            [window for value in prepared for window in value.windows], batch_size=24
        )
        for row, value, scores in zip(
            block, prepared, _split_scores(lengths, window_scores), strict=True
        ):
            total_tokens += value.token_count
            total_windows += len(scores)
            records.append(
                {
                    "artifact_id": row["panel_id"],
                    "dataset": row["dataset"],
                    "source": row["source"],
                    "input_channel": row["input_channel"],
                    "label": int(row["label"]),
                    "text_sha256": row["text_sha256"],
                    "token_count": value.token_count,
                    "window_count": len(scores),
                    "window_scores": scores,
                    "local_score": max(scores),
                }
            )
    seconds = time.perf_counter() - started
    return records, {
        "identity": asdict(runtime.identity) if runtime.identity is not None else None,
        "artifacts": len(records),
        "windows": total_windows,
        "input_tokens": total_tokens,
        "score_seconds": seconds,
        "artifacts_per_second": len(records) / seconds,
        "input_tokens_per_second": total_tokens / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result_path = args.output / "morgott_1024_openvino_scores.jsonl.gz"
    if result_path.exists():
        raise FileExistsError("OpenVINO quality replay is write-once")
    panel = local.load_frozen_panel()
    records, runtime = score(panel, local.load_frozen_texts(panel))
    with gzip.open(result_path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    runtime["result_sha256"] = local.file_sha256(result_path)
    (args.output / "morgott_1024_openvino_runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()
