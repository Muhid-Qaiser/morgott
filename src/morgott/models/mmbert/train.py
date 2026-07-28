"""Train the maintained full-data frozen-head or rank-8 LoRA mmBERT recipe."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version
from itertools import chain
from pathlib import Path

import numpy as np

from .core import (
    ATTENTION_IMPLEMENTATION,
    INSTRUCTION_SUBVERSION_TAGS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGETS,
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    add_lora,
    batch_logits,
    file_sha256,
    load_base_model,
    new_head,
    score_texts,
    source_provenance,
)
from .data import (
    OverlapGuard,
    batches,
    canonical_rows,
    checkpoint_rows,
    external_rows,
    filter_small_training_sets,
    matched_pairs,
    profile_canonical,
    routing_views,
    shuffled,
    training_rows,
)

DOMAIN_WEIGHT = 1.0 / 3.0


@dataclass
class TrainingData:
    views: dict
    data_manifest_sha256: str
    external_manifest_sha256: str
    promptshield: list[dict]
    promptshield_validation: list[dict]
    pairs: list[tuple[dict, dict]]
    checkpoint: list[dict]
    canonical_counts: dict
    canonical_owners: dict
    removed: dict


def _references(views: dict, external: dict):
    for split in ("validation", "dev_test"):
        path, spec = views[split]
        yield from canonical_rows(path, spec, split=split, eligible_only=False)
    for row in external["promptshield_validation"]:
        yield {**row, "_candidate_dataset": "promptshield_validation"}
    yield from external["promptshield_test"]
    yield from external["sep"]


def prepare_training_data(
    data_dir: Path,
    external_dir: Path,
    pair_archive: Path,
) -> TrainingData:
    views = routing_views(data_dir)
    external, _ = external_rows(external_dir)
    original_pairs = matched_pairs(pair_archive)
    candidates = {
        "promptshield": external["promptshield_train"],
        "promptshield_validation": external["promptshield_validation"],
        "pairs": [row for pair in original_pairs for row in pair],
    }
    kept, small_removed = filter_small_training_sets(
        candidates,
        _references(views, external),
    )
    guard = OverlapGuard(
        chain(
            kept["promptshield"],
            kept["promptshield_validation"],
            external["promptshield_test"],
            external["sep"],
        )
    )
    for split in ("validation", "dev_test"):
        path, spec = views[split]
        guard.add_exact(canonical_rows(path, spec, split=split, eligible_only=False))
    train_path, train_spec = views["train"]
    (
        counts,
        canonical_removed,
        owners,
        pair_rows,
        pair_train_removed,
    ) = profile_canonical(
        canonical_rows(train_path, train_spec, split="train"),
        guard,
        {"pairs": kept["pairs"]},
    )
    kept_pair_ids = {row["id"] for row in pair_rows["pairs"]}
    pairs = [
        pair
        for pair in original_pairs
        if pair[0]["id"] in kept_pair_ids and pair[1]["id"] in kept_pair_ids
    ]
    validation_path, validation_spec = views["validation"]
    checkpoint = checkpoint_rows(
        canonical_rows(validation_path, validation_spec, split="validation")
    )
    if not kept["promptshield"] or not pairs:
        raise ValueError("external training populations became empty")

    return TrainingData(
        views=views,
        data_manifest_sha256=file_sha256(data_dir / "manifest.json"),
        external_manifest_sha256=file_sha256(external_dir / "manifest.json"),
        promptshield=kept["promptshield"],
        promptshield_validation=kept["promptshield_validation"],
        pairs=pairs,
        checkpoint=checkpoint,
        canonical_counts=dict(counts),
        canonical_owners=owners,
        removed={
            "canonical": canonical_removed,
            **small_removed,
            "pairs_against_canonical_train": pair_train_removed["pairs"],
            "pair_atoms": len(original_pairs) - len(pairs),
        },
    )


def _report(data: TrainingData) -> dict:
    return {
        "canonical_rows": sum(data.canonical_counts.values()),
        "canonical_strata": {
            f"{source}:{label}": count
            for (source, label), count in sorted(data.canonical_counts.items())
        },
        "promptshield_rows": len(data.promptshield),
        "matched_pairs": len(data.pairs),
        "checkpoint_rows": len(data.checkpoint),
        "promptshield_validation_rows": len(data.promptshield_validation),
        "removed_for_overlap": data.removed,
    }


def _cycle(rows: list, *, seed: int):
    randomizer = random.Random(seed)
    order = list(range(len(rows)))
    while True:
        randomizer.shuffle(order)
        for index in order:
            yield rows[index]


def _take(iterator, count: int) -> list:
    return [next(iterator) for _ in range(count)]


def _classification_backward(
    encoder,
    tokenizer,
    head,
    rows: list[dict],
    *,
    coefficient: float,
    microbatch_size: int,
    train_encoder: bool,
) -> float:
    import torch

    total = 0.0
    for batch in batches(rows, microbatch_size):
        logits = batch_logits(
            encoder,
            tokenizer,
            head,
            [row["text"] for row in batch],
            train_encoder=train_encoder,
        )
        targets = torch.tensor(
            [row["label"] for row in batch],
            dtype=torch.float32,
            device="cuda",
        )
        weights = torch.tensor(
            [row.get("weight", 1.0) for row in batch],
            dtype=torch.float32,
            device="cuda",
        )
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(),
            targets,
            reduction="none",
        )
        loss = coefficient * (losses * weights).sum() / len(rows)
        loss.backward()
        total += float(loss.detach())
    return total


def _pair_backward(
    encoder,
    tokenizer,
    head,
    pairs: list[tuple[dict, dict]],
    *,
    ranking_weight: float,
    microbatch_size: int,
    train_encoder: bool,
) -> float:
    import torch

    total = 0.0
    pair_microbatch = max(1, microbatch_size // 2)
    for batch in batches(pairs, pair_microbatch):
        benign = [pair[0] for pair in batch]
        attack = [pair[1] for pair in batch]
        logits = batch_logits(
            encoder,
            tokenizer,
            head,
            [row["text"] for row in [*benign, *attack]],
            train_encoder=train_encoder,
        ).float()
        benign_logits, attack_logits = logits.split(len(batch))
        pair_bce = 0.5 * (
            torch.nn.functional.softplus(benign_logits).mean()
            + torch.nn.functional.softplus(-attack_logits).mean()
        )
        ranking = torch.nn.functional.softplus(-(attack_logits - benign_logits)).mean()
        scale = len(batch) / len(pairs)
        loss = scale * (DOMAIN_WEIGHT * pair_bce + ranking_weight * ranking)
        loss.backward()
        total += float(loss.detach())
    return total


def _validation_bce(
    encoder,
    tokenizer,
    head,
    rows: list[dict],
    *,
    batch_size: int,
) -> float:
    scores = score_texts(
        encoder,
        tokenizer,
        head,
        [row["text"] for row in rows],
        batch_size=batch_size,
    )
    labels = np.asarray([row["label"] for row in rows])
    probabilities = np.clip(scores, 1e-12, 1 - 1e-12)
    return float(
        -np.mean(
            labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)
        )
    )


def _cpu_state(module) -> dict:
    return {
        name: value.detach().contiguous().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _adapter_state(encoder) -> dict:
    from peft import get_peft_model_state_dict

    return {
        name: value.detach().contiguous().cpu().clone()
        for name, value in get_peft_model_state_dict(encoder).items()
    }


def _save_run(
    output: Path,
    *,
    mode: str,
    seed: int,
    head,
    encoder,
    report: dict,
    curve: list[dict],
    selected_epoch: int,
    args: argparse.Namespace,
    data: TrainingData,
    seconds: float,
) -> Path:
    from safetensors.torch import save_file

    name = f"mmbert-base-full-{mode}-s{seed}"
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing output: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output, prefix=f".{name}-"))
    try:
        head_path = temporary / "head.safetensors"
        save_file(_cpu_state(head), str(head_path))
        adapter_files = None
        if mode == "lora":
            adapter = temporary / "adapter"
            encoder.save_pretrained(adapter, safe_serialization=True)
            adapter_files = {
                path.name: file_sha256(path)
                for path in sorted(adapter.iterdir())
                if path.is_file()
            }
        targeted_modules = (
            sorted(
                name
                for name, module in encoder.named_modules()
                if hasattr(module, "lora_A")
            )
            if mode == "lora"
            else None
        )
        result = {
            "schema_version": 1,
            "purpose": "maintained full-data advisory mmBERT training",
            "adaptation": mode,
            "generic_target": "instruction_subversion",
            "positive_classes": list(INSTRUCTION_SUBVERSION_TAGS),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "normalization": "strict",
            "max_tokens": MAX_TOKENS,
            "token_budget": args.microbatch_size * MAX_TOKENS,
            "seed": seed,
            "lora": (
                {
                    "rank": LORA_RANK,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "bias": "none",
                    "target_modules_regex": LORA_TARGETS,
                    "targeted_modules": targeted_modules,
                    "adapter_parameters": sum(
                        parameter.numel()
                        for parameter in encoder.parameters()
                        if parameter.requires_grad
                    ),
                }
                if mode == "lora"
                else None
            ),
            "objective": {
                "domains": {
                    "morgott": DOMAIN_WEIGHT,
                    "promptshield": DOMAIN_WEIGHT,
                    "matched_pairs": DOMAIN_WEIGHT,
                },
                "canonical_weighting": "source_label_balanced",
                "pair_ranking_weight": args.pair_ranking_weight,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "microbatch_size": args.microbatch_size,
                "head_learning_rate": args.head_learning_rate,
                "adapter_learning_rate": (
                    args.adapter_learning_rate if mode == "lora" else None
                ),
                "shuffle_buffer": args.shuffle_buffer,
                "selected_epoch": selected_epoch,
                "checkpoint_selection": (
                    "minimum equal-domain mean of Morgott and PromptShield "
                    "validation BCE"
                ),
                "curve": curve,
            },
            "populations": report,
            "runtime_seconds": seconds,
            "packages": {
                package: version(package)
                for package in (
                    "numpy",
                    "peft",
                    "safetensors",
                    "torch",
                    "transformers",
                )
            },
            "artifact": {
                "head": "head.safetensors",
                "head_sha256": file_sha256(head_path),
                "adapter": "adapter" if adapter_files else None,
                "adapter_files": adapter_files,
            },
            "provenance": {
                "routing_views": {
                    split: {
                        "path": spec["path"],
                        "sha256": spec["sha256"],
                        "rows": spec["rows"],
                    }
                    for split, (_, spec) in data.views.items()
                },
                "data_manifest_sha256": data.data_manifest_sha256,
                "external_manifest_sha256": data.external_manifest_sha256,
                "pair_archive_sha256": file_sha256(args.pairs),
                **source_provenance(
                    Path(__file__),
                    Path(__file__).with_name("core.py"),
                    Path(__file__).with_name("data.py"),
                    Path(__file__).with_name("external_data.py"),
                    Path(__file__).resolve().parents[2] / "data.py",
                    Path(__file__).resolve().parents[2] / "normalization.py",
                    Path(__file__).resolve().parents[2] / "overlap.py",
                ),
            },
            "limitations": [
                "This is development evidence, not a production calibration.",
                "PromptShield and SEP are already-open development benchmarks.",
                "Inputs are truncated to the first 512 normalized tokens.",
                "The score is advisory and is not approved for blocking.",
            ],
        }
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def train(args: argparse.Namespace, data: TrainingData) -> Path:
    import torch

    torch.manual_seed(args.seed)
    encoder, tokenizer = load_base_model()
    train_encoder = args.mode == "lora"
    if train_encoder:
        encoder = add_lora(encoder)
        encoder.gradient_checkpointing_enable()
    else:
        encoder.gradient_checkpointing_disable()
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    head = new_head(encoder.config.hidden_size, args.seed).to("cuda")
    head_parameters = list(head.parameters())
    trainable = list(head_parameters)
    parameters = [{"params": head_parameters, "lr": args.head_learning_rate}]
    if train_encoder:
        adapter_parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        trainable.extend(adapter_parameters)
        parameters.append(
            {
                "params": adapter_parameters,
                "lr": args.adapter_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameters)
    promptshield = _cycle(data.promptshield, seed=args.seed + 1)
    pairs = _cycle(data.pairs, seed=args.seed + 2)
    best = None
    curve = []
    started = time.perf_counter()
    train_path, train_spec = data.views["train"]

    for epoch in range(1, args.epochs + 1):
        encoder.train(train_encoder)
        head.train()
        losses = []
        stream = training_rows(
            canonical_rows(train_path, train_spec, split="train"),
            data.canonical_counts,
            data.canonical_owners,
        )
        stream = shuffled(
            stream,
            seed=args.seed + epoch,
            buffer_size=args.shuffle_buffer,
        )
        for morgott in batches(stream, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = _classification_backward(
                encoder,
                tokenizer,
                head,
                morgott,
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
            )
            loss += _classification_backward(
                encoder,
                tokenizer,
                head,
                _take(promptshield, max(1, args.batch_size // 2)),
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
            )
            loss += _pair_backward(
                encoder,
                tokenizer,
                head,
                _take(pairs, max(1, args.batch_size // 4)),
                ranking_weight=args.pair_ranking_weight,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in parameters for parameter in group["params"]],
                1.0,
            )
            optimizer.step()
            losses.append(loss)

        bces = {
            "morgott": _validation_bce(
                encoder,
                tokenizer,
                head,
                data.checkpoint,
                batch_size=args.microbatch_size,
            ),
            "promptshield": _validation_bce(
                encoder,
                tokenizer,
                head,
                data.promptshield_validation,
                batch_size=args.microbatch_size,
            ),
        }
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            **{f"validation_{name}_bce": value for name, value in bces.items()},
            "validation_macro_bce": 0.5 * sum(bces.values()),
        }
        curve.append(row)
        if best is None or row["validation_macro_bce"] < best["loss"]:
            best = {
                "loss": row["validation_macro_bce"],
                "epoch": epoch,
                "head": _cpu_state(head),
                "adapter": _adapter_state(encoder) if train_encoder else None,
            }

    head.load_state_dict(best["head"], strict=True)
    if train_encoder:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(encoder, best["adapter"])
    return _save_run(
        args.output,
        mode=args.mode,
        seed=args.seed,
        head=head,
        encoder=encoder,
        report=_report(data),
        curve=curve,
        selected_epoch=best["epoch"],
        args=args,
        data=data,
        seconds=time.perf_counter() - started,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("frozen", "lora"), default="lora")
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/mmbert/runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--pair-ranking-weight", type=float, default=0.25)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    numeric = (
        args.epochs,
        args.batch_size,
        args.microbatch_size,
        args.shuffle_buffer,
        args.head_learning_rate,
        args.adapter_learning_rate,
    )
    if args.seed < 0 or any(
        not math.isfinite(value) or value <= 0 for value in numeric
    ):
        raise ValueError("training parameters must be finite and positive")
    if not math.isfinite(args.pair_ranking_weight) or args.pair_ranking_weight < 0:
        raise ValueError("pair ranking weight must be finite and non-negative")
    data = prepare_training_data(args.data_dir, args.external_dir, args.pairs)
    print(json.dumps(_report(data), indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    destination = train(args, data)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
