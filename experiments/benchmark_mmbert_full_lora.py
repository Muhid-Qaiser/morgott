"""Benchmark safe BF16 microbatch/checkpointing choices for the full LoRA run."""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import time
from pathlib import Path

from morgott.models.mmbert.core import (
    ATTENTION_IMPLEMENTATION,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    add_lora,
    load_base_model,
    new_head,
)
from morgott.models.mmbert.data import (
    canonical_rows,
    external_rows,
    matched_pairs,
    routing_views,
)
from morgott.models.mmbert.train import (
    DOMAIN_WEIGHT,
    _classification_backward,
    _pair_backward,
)

COMBINATIONS = (
    (16, False),
    (16, True),
    (12, False),
    (12, True),
    (8, False),
    (8, True),
)
MINIMUM_HEADROOM_BYTES = int(0.9 * (1 << 30))


def _longest(rows, count: int, *, text) -> list:
    return heapq.nlargest(count, rows, key=lambda row: len(text(row)))


def _representative_rows(
    data_dir: Path,
    external_dir: Path,
    pairs_path: Path,
) -> tuple[list[dict], list[dict], list[tuple[dict, dict]]]:
    views = routing_views(data_dir)
    train_path, train_spec = views["train"]
    canonical = _longest(
        canonical_rows(train_path, train_spec, split="train"),
        128,
        text=lambda row: row["text"],
    )
    canonical = [{**row, "weight": 1.0} for row in canonical]
    external, _ = external_rows(external_dir)
    promptshield = _longest(
        external["promptshield_train"],
        64,
        text=lambda row: row["text"],
    )
    pairs = _longest(
        matched_pairs(pairs_path),
        32,
        text=lambda pair: pair[0]["text"] + pair[1]["text"],
    )
    if (len(canonical), len(promptshield), len(pairs)) != (128, 64, 32):
        raise ValueError("representative benchmark populations are incomplete")
    return canonical, promptshield, pairs


def _benchmark(
    canonical: list[dict],
    promptshield: list[dict],
    pairs: list[tuple[dict, dict]],
    *,
    microbatch_size: int,
    gradient_checkpointing: bool,
) -> dict:
    import torch

    encoder = tokenizer = head = optimizer = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        encoder, tokenizer = load_base_model()
        if gradient_checkpointing:
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            encoder.enable_input_require_grads()
        encoder = add_lora(encoder)
        head = new_head(encoder.config.hidden_size, 42).to("cuda")
        optimizer = torch.optim.AdamW(
            [
                {"params": head.parameters(), "lr": 3e-4},
                {
                    "params": [
                        parameter
                        for parameter in encoder.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": 1e-4,
                },
            ]
        )
        torch.cuda.reset_peak_memory_stats()

        timings = []
        for step in range(3):
            encoder.train()
            head.train()
            if step:
                torch.cuda.synchronize()
                started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _classification_backward(
                encoder,
                tokenizer,
                head,
                canonical,
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=microbatch_size,
                train_encoder=True,
            )
            _classification_backward(
                encoder,
                tokenizer,
                head,
                promptshield,
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=microbatch_size,
                train_encoder=True,
            )
            _pair_backward(
                encoder,
                tokenizer,
                head,
                pairs,
                ranking_weight=0.25,
                microbatch_size=microbatch_size,
                train_encoder=True,
            )
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ],
                1.0,
            )
            optimizer.step()
            torch.cuda.synchronize()
            if step:
                timings.append(time.perf_counter() - started)

        total = torch.cuda.get_device_properties(0).total_memory
        peak_reserved = torch.cuda.max_memory_reserved()
        return {
            "status": "ok",
            "microbatch_size": microbatch_size,
            "gradient_checkpointing": gradient_checkpointing,
            "mean_full_update_seconds": sum(timings) / len(timings),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": peak_reserved,
            "device_total_bytes": total,
            "headroom_bytes": total - peak_reserved,
            "eligible": total - peak_reserved >= MINIMUM_HEADROOM_BYTES,
        }
    except torch.OutOfMemoryError:
        return {
            "status": "oom",
            "microbatch_size": microbatch_size,
            "gradient_checkpointing": gradient_checkpointing,
            "eligible": False,
        }
    finally:
        del optimizer, head, tokenizer, encoder
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("artifacts/mmbert/data"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data-archive/matched_pairs_20260726.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mmbert/full-lora-runtime-benchmark.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace benchmark: {args.output}")

    canonical, promptshield, pairs = _representative_rows(
        args.data_dir,
        args.external_dir,
        args.pairs,
    )
    results = []
    for microbatch_size, gradient_checkpointing in COMBINATIONS:
        print(
            f"benchmarking microbatch={microbatch_size} "
            f"checkpointing={gradient_checkpointing}",
            flush=True,
        )
        result = _benchmark(
            canonical,
            promptshield,
            pairs,
            microbatch_size=microbatch_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    eligible = [result for result in results if result["eligible"]]
    if not eligible:
        raise RuntimeError("no benchmark configuration retained safe VRAM headroom")
    selected = min(eligible, key=lambda result: result["mean_full_update_seconds"])
    report = {
        "schema_version": 1,
        "purpose": "full-mixture rank-8 mmBERT LoRA runtime selection",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "precision": "BF16 mixed precision with FP32 optimizer state",
        "max_tokens": MAX_TOKENS,
        "minimum_headroom_bytes": MINIMUM_HEADROOM_BYTES,
        "representative_rows": {
            "canonical": len(canonical),
            "promptshield": len(promptshield),
            "matched_pairs": len(pairs),
            "selection": "largest raw text lengths from each fitting population",
        },
        "results": results,
        "selected": {
            "microbatch_size": selected["microbatch_size"],
            "gradient_checkpointing": selected["gradient_checkpointing"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
