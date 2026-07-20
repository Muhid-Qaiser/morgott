"""Fair one-seed end-to-end ModernBERT/DeBERTa direct-sensor pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    __version__ as transformers_version,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)

from morgott.data import manifest_output_hashes, read_verified_jsonl
from morgott.detector import (
    DIRECT_OPERATING_FPR_BUDGETS,
    DIRECT_PRECISION_FLOORS,
    DIRECT_REVIEW_PRECISION_FLOOR,
    _rates,
    choose_threshold,
    choose_threshold_for_precision,
    split_fit_validation,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
HERE = Path(__file__).resolve().parent

SEED = 42
MAX_LENGTH = 512
EPOCHS = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EFFECTIVE_BATCH_SIZE = 32
INITIAL_PHYSICAL_BATCH_SIZE = 2
INITIAL_INFERENCE_BATCH_SIZE = 16
OASST_TARGET_ROWS = 4_000

MODELS = {
    "modernbert": {
        "id": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "license": "Apache-2.0",
        "model_type": "modernbert",
        "hidden_size": 768,
        "max_position_embeddings": 8192,
        "attention_backend": "sdpa",
        "weights_sha256": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
        "weights_bytes": 598_635_032,
        "revision_note": "published repository commit",
    },
    "deberta": {
        "id": "microsoft/deberta-v3-base",
        "revision": "de19fe7db5162df5f3d8f0b41321c0267288fd74",
        "license": "MIT",
        "model_type": "deberta-v2",
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "attention_backend": "eager",
        "weights_sha256": "57cbd0cad054ba5be8d4c6965b836e132f029edbbe3ed9c5bc9ef4fe1c40c34e",
        "weights_bytes": 371_101_258,
        "revision_note": (
            "pinned safetensors conversion commit; main revision has only "
            "pytorch_model.bin"
        ),
    },
}

INPUT_SHA256 = manifest_output_hashes(ROOT / "reports/data_manifest.json")
INPUT_SHA256.pop("indirect_train")
INPUT_SHA256.pop("nemotron_agentic_ipi")

EVALUATION_DATASETS = tuple(name for name in INPUT_SHA256 if name != "train")
HARD_NEGATIVE_DATASETS = (
    "oasst1_chat",
    "oasst1_position_stress",
    "xstest",
    "harmbench",
    "do_not_answer",
    "notinject",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(name: str) -> list[dict]:
    return read_verified_jsonl(DATA / f"{name}.jsonl", INPUT_SHA256[name])


def _rank(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()


def select_training_subset(
    rows: list[dict], oasst_target_rows: int = OASST_TARGET_ROWS, seed: int = SEED
) -> tuple[list[dict], dict]:
    fit, validation = split_fit_validation(rows)
    positives = [row for row in fit if row["label"] == 1]
    same_source_negatives = [
        row
        for row in fit
        if row["label"] == 0 and row["source"] in {"toxic_chat", "prompt_injections"}
    ]
    oasst_groups: dict[str, list[dict]] = defaultdict(list)
    for row in fit:
        if row["label"] == 0 and row["source"] == "oasst1":
            oasst_groups[row["split_group_id"]].append(row)

    chosen_oasst = []
    chosen_groups = []
    for group in sorted(
        oasst_groups, key=lambda value: _rank(seed, "oasst-group", value)
    ):
        if len(chosen_oasst) >= oasst_target_rows:
            break
        chosen_groups.append(group)
        chosen_oasst.extend(oasst_groups[group])

    selected = positives + same_source_negatives + chosen_oasst
    selected.sort(key=lambda row: _rank(seed, "training-row", row["id"]))
    selected_groups = {row.get("split_group_id", row["group_id"]) for row in selected}
    validation_groups = {
        row.get("split_group_id", row["group_id"]) for row in validation
    }
    if selected_groups & validation_groups:
        raise RuntimeError("training subset intersects grouped validation")

    row_ids = [row["id"] for row in selected]
    summary = {
        "selection": (
            "all fit positives + all ToxicChat/deepset fit negatives + complete "
            "OASST1 fit groups in SHA256(seed, group) order until at least target rows"
        ),
        "seed": seed,
        "oasst_target_rows": oasst_target_rows,
        "rows": len(selected),
        "positive": sum(row["label"] for row in selected),
        "negative": sum(row["label"] == 0 for row in selected),
        "groups": len(selected_groups),
        "oasst_rows": len(chosen_oasst),
        "oasst_groups": len(chosen_groups),
        "by_source_label": {
            f"{source}:{label}": count
            for (source, label), count in sorted(
                Counter((row["source"], row["label"]) for row in selected).items()
            )
        },
        "ordered_row_ids_sha256": hashlib.sha256(
            "\n".join(row_ids).encode()
        ).hexdigest(),
        "oasst_group_ids_sha256": hashlib.sha256(
            "\n".join(chosen_groups).encode()
        ).hexdigest(),
        "validation_rows_untouched": len(validation),
        "validation_groups_untouched": len(validation_groups),
    }
    return selected, summary


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DynamicCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [example["text"] for example in examples],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [example["label"] for example in examples], dtype=torch.long
        )
        return batch


class MeanPoolClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, 2)
        torch.manual_seed(SEED)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        attention_mask = batch["attention_mask"]
        encoder_inputs = {key: value for key, value in batch.items() if key != "labels"}
        hidden = self.encoder(**encoder_inputs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


def load_model(model_name: str) -> tuple[MeanPoolClassifier, object, dict]:
    spec = MODELS[model_name]
    weights_path = Path(
        hf_hub_download(spec["id"], "model.safetensors", revision=spec["revision"])
    )
    weights_digest = sha256_file(weights_path)
    if (
        weights_digest != spec["weights_sha256"]
        or weights_path.stat().st_size != spec["weights_bytes"]
    ):
        raise RuntimeError(f"{model_name} safetensors integrity check failed")

    config = AutoConfig.from_pretrained(
        spec["id"], revision=spec["revision"], trust_remote_code=False
    )
    if (
        config.model_type != spec["model_type"]
        or config.hidden_size != spec["hidden_size"]
        or config.max_position_embeddings != spec["max_position_embeddings"]
        or config.max_position_embeddings < MAX_LENGTH
    ):
        raise RuntimeError(f"unexpected {model_name} config: {config}")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        trust_remote_code=False,
        use_fast=True,
    )
    encoder, loading = AutoModel.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation=spec["attention_backend"],
        output_loading_info=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise RuntimeError(f"incomplete {model_name} encoder load: {loading}")
    encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False
    model = MeanPoolClassifier(encoder, spec["hidden_size"])
    return (
        model,
        tokenizer,
        {
            "weights_sha256": weights_digest,
            "weights_bytes": weights_path.stat().st_size,
            "unexpected_pretraining_head_keys": sorted(loading["unexpected_keys"]),
            "missing_encoder_keys": sorted(loading["missing_keys"]),
            "mismatched_encoder_keys": sorted(
                str(value) for value in loading["mismatched_keys"]
            ),
        },
    )


def _to_device(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def _divide_gradients(model: nn.Module, denominator: float) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def memory_preflight(
    model_name: str, rows: list[dict], positive_weight: float
) -> tuple[int, dict]:
    attempts = []
    batch_size = INITIAL_PHYSICAL_BATCH_SIZE
    longest = sorted(rows, key=lambda row: len(row["text"]), reverse=True)
    while batch_size >= 1:
        model = tokenizer = optimizer = scaler = batch = weights = logits = loss = None
        try:
            set_seed()
            model, tokenizer, _ = load_model(model_name)
            model.cuda().train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scaler = torch.amp.GradScaler("cuda")
            batch = DynamicCollator(tokenizer)(longest[:batch_size])
            batch = _to_device(batch)
            weights = torch.tensor([1.0, positive_weight], device="cuda")
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(batch)
                loss = F.cross_entropy(
                    logits.float(), batch["labels"], weight=weights, reduction="sum"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _divide_gradients(model, float(weights[batch["labels"]].sum()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            attempts.append(
                {
                    "physical_batch_size": batch_size,
                    "status": "fit",
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "longest_batch_tokens": int(batch["attention_mask"].sum()),
                }
            )
            return batch_size, {"attempts": attempts}
        except torch.OutOfMemoryError:
            attempts.append({"physical_batch_size": batch_size, "status": "cuda_oom"})
            batch_size //= 2
        finally:
            del model, tokenizer, optimizer, scaler, batch, weights, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"{model_name} does not fit at physical batch size 1")


def train_model(
    model: MeanPoolClassifier,
    tokenizer,
    rows: list[dict],
    physical_batch_size: int,
    positive_weight: float,
) -> dict:
    set_seed()
    loader = DataLoader(
        rows,
        batch_size=physical_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=DynamicCollator(tokenizer),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda")
    weights = torch.tensor([1.0, positive_weight], device="cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_examples = 0
    accumulated_weight = 0.0
    loss_sum = 0.0
    loss_weight = 0.0
    optimizer_steps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for batch_index, batch in enumerate(loader, start=1):
        batch = _to_device(batch)
        batch_weight = float(weights[batch["labels"]].sum())
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch)
            loss = F.cross_entropy(
                logits.float(), batch["labels"], weight=weights, reduction="sum"
            )
        scaler.scale(loss).backward()
        accumulated_examples += len(batch["labels"])
        accumulated_weight += batch_weight
        loss_sum += float(loss.detach())
        loss_weight += batch_weight
        final_batch = batch_index == len(loader)
        if accumulated_examples >= EFFECTIVE_BATCH_SIZE or final_batch:
            scaler.unscale_(optimizer)
            _divide_gradients(model, accumulated_weight)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulated_examples = 0
            accumulated_weight = 0.0
            if optimizer_steps % 25 == 0:
                print(
                    f"{datetime.now(UTC).isoformat(timespec='seconds')} "
                    f"optimizer_step={optimizer_steps}/{math.ceil(len(rows) / EFFECTIVE_BATCH_SIZE)}",
                    flush=True,
                )

    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "optimizer_steps": optimizer_steps,
        "weighted_mean_loss": loss_sum / loss_weight,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


@torch.inference_mode()
def score_texts(
    model: MeanPoolClassifier, tokenizer, texts: list[str], batch_size: int
) -> tuple[np.ndarray, float]:
    model.eval()
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    scores = np.empty(len(texts), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        examples = [{"text": texts[index], "label": 0} for index in indices]
        batch = _to_device(DynamicCollator(tokenizer)(examples))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch).float()
        scores[indices] = logits.softmax(-1)[:, 1].cpu().numpy()
    torch.cuda.synchronize()
    return scores, time.perf_counter() - started


def inference_preflight(model: MeanPoolClassifier, tokenizer, rows: list[dict]) -> int:
    longest = sorted(rows, key=lambda row: len(row["text"]), reverse=True)
    batch_size = INITIAL_INFERENCE_BATCH_SIZE
    while batch_size >= 1:
        try:
            score_texts(
                model,
                tokenizer,
                [row["text"] for row in longest[:batch_size]],
                batch_size,
            )
            return batch_size
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size //= 2
    raise RuntimeError("inference does not fit at batch size 1")


@torch.inference_mode()
def measured_latency(model: MeanPoolClassifier, tokenizer, texts: list[str]) -> dict:
    sample = texts[:100]
    for text in sample[:5]:
        score_texts(model, tokenizer, [text], 1)
    timings = []
    for text in sample:
        started = time.perf_counter()
        score_texts(model, tokenizer, [text], 1)
        timings.append((time.perf_counter() - started) * 1_000)
    return {
        "sample_rows": len(sample),
        "batch_size": 1,
        "p50_ms": statistics.median(timings),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def profile_metrics(
    validation_rows: list[dict],
    validation_scores: np.ndarray,
    datasets: dict[str, list[dict]],
    scores: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    labels = [row["label"] for row in validation_rows]
    precision_profiles = []
    for floor in DIRECT_PRECISION_FLOORS:
        try:
            threshold = choose_threshold_for_precision(labels, validation_scores, floor)
        except ValueError:
            precision_profiles.append(
                {
                    "validation_precision_floor": floor,
                    "attained": False,
                    "threshold": None,
                    "validation": None,
                    "sets": {},
                }
            )
            continue
        precision_profiles.append(
            {
                "validation_precision_floor": floor,
                "attained": True,
                "threshold": threshold,
                "validation": _rates(validation_rows, validation_scores, threshold),
                "sets": {
                    name: _rates(datasets[name], scores[name], threshold)
                    for name in datasets
                },
            }
        )

    fpr_points = []
    for budget in DIRECT_OPERATING_FPR_BUDGETS:
        threshold = choose_threshold(labels, validation_scores, budget)
        fpr_points.append(
            {
                "validation_fpr_budget": budget,
                "threshold": threshold,
                "validation": _rates(validation_rows, validation_scores, threshold),
                "sets": {
                    name: _rates(datasets[name], scores[name], threshold)
                    for name in datasets
                },
            }
        )
    return precision_profiles, fpr_points


def run(model_name: str, output: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    train_rows = read_rows("train")
    fit_rows, validation_rows = split_fit_validation(train_rows)
    selected_rows, subset = select_training_subset(train_rows)
    if (
        subset["positive"] != 245
        or subset["by_source_label"].get("toxic_chat:0", 0)
        + subset["by_source_label"].get("prompt_injections:0", 0)
        != 4_160
        or len(validation_rows) != 7_186
    ):
        raise RuntimeError(f"unexpected training subset: {subset}")
    positive_weight = subset["negative"] / subset["positive"]
    physical_batch_size, preflight = memory_preflight(
        model_name, selected_rows, positive_weight
    )

    set_seed()
    model, tokenizer, loading = load_model(model_name)
    model.cuda()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_result = train_model(
        model, tokenizer, selected_rows, physical_batch_size, positive_weight
    )

    inference_batch_size = inference_preflight(model, tokenizer, selected_rows)
    datasets = {name: read_rows(name) for name in EVALUATION_DATASETS}
    all_rows_for_preflight = [
        row for name in EVALUATION_DATASETS for row in datasets[name]
    ]
    inference_batch_size = min(
        inference_batch_size,
        inference_preflight(model, tokenizer, all_rows_for_preflight),
    )
    validation_scores, validation_seconds = score_texts(
        model,
        tokenizer,
        [row["text"] for row in validation_rows],
        inference_batch_size,
    )
    print(
        f"{datetime.now(UTC).isoformat(timespec='seconds')} "
        f"evaluated validation rows={len(validation_rows)}",
        flush=True,
    )
    dataset_scores = {}
    evaluation_timing = {}
    torch.cuda.reset_peak_memory_stats()
    for name, rows in datasets.items():
        dataset_scores[name], seconds = score_texts(
            model,
            tokenizer,
            [row["text"] for row in rows],
            inference_batch_size,
        )
        evaluation_timing[name] = {
            "seconds": seconds,
            "ms_per_text": seconds * 1_000 / len(rows),
        }
        print(
            f"{datetime.now(UTC).isoformat(timespec='seconds')} "
            f"evaluated {name} rows={len(rows)}",
            flush=True,
        )

    hard_rows = [row for name in HARD_NEGATIVE_DATASETS for row in datasets[name]]
    hard_scores = np.concatenate(
        [dataset_scores[name] for name in HARD_NEGATIVE_DATASETS]
    )
    datasets["external_hard_negatives"] = hard_rows
    dataset_scores["external_hard_negatives"] = hard_scores
    precision_profiles, fpr_points = profile_metrics(
        validation_rows, validation_scores, datasets, dataset_scores
    )
    default_profile = next(
        profile
        for profile in precision_profiles
        if profile["validation_precision_floor"] == DIRECT_REVIEW_PRECISION_FLOOR
    )
    default_sets = default_profile["sets"] if default_profile["attained"] else {}
    latency = measured_latency(
        model, tokenizer, [row["text"] for row in validation_rows]
    )

    spec = MODELS[model_name]
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pilot_only": True,
        "model": {
            **spec,
            "parameter_count_with_classifier": parameter_count,
            "weights_format": "safetensors",
            "trust_remote_code": False,
            "pooling": "shared masked mean",
            "classifier": "shared randomly initialized 768x2 linear head",
            "loading": loading,
        },
        "protocol": {
            "seed": SEED,
            "epochs": EPOCHS,
            "max_length": MAX_LENGTH,
            "optimizer": "torch.optim.AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "class-weighted cross entropy, summed then normalized by class mass per effective batch",
            "negative_class_weight": 1.0,
            "positive_class_weight": positive_weight,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "physical_batch_size": physical_batch_size,
            "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // physical_batch_size,
            "gradient_checkpointing": True,
            "dynamic_padding": True,
            "pad_to_multiple_of": 8,
            "mixed_precision": "FP16 CUDA autocast; FP32 parameters, classifier, loss, and optimizer",
            "gradient_clip_norm": 1.0,
        },
        "training_subset": subset,
        "full_fit_rows_available": len(fit_rows),
        "input_sha256": INPUT_SHA256,
        "memory_preflight": preflight,
        "training": train_result,
        "inference": {
            "batch_size": inference_batch_size,
            "validation_seconds": validation_seconds,
            "evaluation_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "evaluation_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "latency": latency,
            "timing": evaluation_timing,
        },
        "default_precision_floor": DIRECT_REVIEW_PRECISION_FLOOR,
        "default_profile_attained": default_profile["attained"],
        "default_validation": default_profile["validation"],
        "default_sets": default_sets,
        "direct_precision_profiles": precision_profiles,
        "direct_fpr_diagnostics": fpr_points,
        "evaluation_note": (
            "direct-user head only; BIPIA/Tensor context scores are direct-sensor "
            "stress checks, not a trained indirect head"
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
        },
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return result


def self_check() -> None:
    assert set(MODELS) == {"modernbert", "deberta"}
    assert all(len(spec["revision"]) == 40 for spec in MODELS.values())
    assert all(len(spec["weights_sha256"]) == 64 for spec in MODELS.values())
    assert MODELS["modernbert"]["attention_backend"] == "sdpa"
    assert MODELS["deberta"]["attention_backend"] == "eager"
    assert DIRECT_REVIEW_PRECISION_FLOOR in DIRECT_PRECISION_FLOORS
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe-subset", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_check()
        return
    if args.describe_subset:
        _, summary = select_training_subset(read_rows("train"))
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if not args.model:
        parser.error("--model is required")
    output = args.output or HERE / f"{args.model}_pilot.json"
    result = run(args.model, output)
    print(
        json.dumps(
            {
                "model": args.model,
                "output": str(output),
                "threshold": next(
                    profile["threshold"]
                    for profile in result["direct_precision_profiles"]
                    if profile["validation_precision_floor"]
                    == DIRECT_REVIEW_PRECISION_FLOOR
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
