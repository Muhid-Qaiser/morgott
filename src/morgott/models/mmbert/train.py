"""Train the maintained full-data frozen-head or rank-8 LoRA mmBERT recipe."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
    score_logits,
    source_provenance,
)
from .data import (
    OverlapGuard,
    _strict_hash,
    batches,
    canonical_rows,
    external_rows,
    filter_small_training_sets,
    matched_pairs,
    partition_validation_records,
    profile_canonical,
    routing_views,
    shuffled,
    training_rows,
)

DOMAIN_WEIGHT = 1.0 / 3.0
FULL_POPULATION = {
    "canonical_rows": 1_069_607,
    "promptshield_rows": 18_197,
    "matched_pairs": 11_041,
    "checkpoint_rows": 29_293,
    "calibration_rows": 116_488,
    "validation_components": 36_695,
    "promptshield_validation_rows": 985,
}


def _run_name(mode: str, seed: int) -> str:
    if mode == "lora":
        return f"mmbert-lora-full-s{seed}"
    return f"mmbert-base-full-{mode}-s{seed}"


def _save_checkpoint(path: Path, *, identity: dict, state: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(
                {
                    "schema_version": 1,
                    "identity": identity,
                    "state": state,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path, *, identity: dict) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("identity") != identity
        or not isinstance(payload.get("state"), dict)
    ):
        raise ValueError("checkpoint identity or schema mismatch")
    return payload["state"]


def _training_identity(args: argparse.Namespace, data: TrainingData) -> dict:
    return {
        "schema_version": 1,
        "mode": args.mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "shuffle_buffer": args.shuffle_buffer,
        "head_learning_rate": args.head_learning_rate,
        "adapter_learning_rate": args.adapter_learning_rate,
        "pair_ranking_weight": args.pair_ranking_weight,
        "gradient_checkpointing": (
            args.mode == "lora" and not args.no_gradient_checkpointing
        ),
        "data": {
            "manifest_sha256": data.data_manifest_sha256,
            "external_manifest_sha256": data.external_manifest_sha256,
            "pair_archive_sha256": file_sha256(args.pairs),
            "routing_views": {
                split: spec["sha256"] for split, (_, spec) in data.views.items()
            },
            "populations": _report(data),
        },
        "sources": source_provenance(
            Path(__file__),
            Path(__file__).with_name("core.py"),
            Path(__file__).with_name("data.py"),
        ),
    }


@dataclass
class TrainingData:
    views: dict
    data_manifest_sha256: str
    external_manifest_sha256: str
    promptshield: list[dict]
    promptshield_validation: list[dict]
    pairs: list[tuple[dict, dict]]
    checkpoint: list[dict]
    calibration: list[dict]
    validation_partition: dict
    canonical_counts: dict
    canonical_group_counts: dict
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
    *,
    seed: int = 42,
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
        guard.add(canonical_rows(path, spec, split=split, eligible_only=False))
    train_path, train_spec = views["train"]
    (
        counts,
        group_counts,
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
    dev_path, dev_spec = views["dev_test"]
    validation_guard = OverlapGuard(
        chain(
            canonical_rows(
                dev_path,
                dev_spec,
                split="dev_test",
                eligible_only=False,
            ),
            kept["promptshield_validation"],
            external["promptshield_test"],
            external["sep"],
        )
    )
    validation_path, validation_spec = views["validation"]
    validation_candidates = list(
        canonical_rows(validation_path, validation_spec, split="validation")
    )
    (
        _,
        _,
        validation_removed,
        validation_owners,
        _,
        _,
    ) = profile_canonical(validation_candidates, validation_guard, {})
    validation_rows = [
        row
        for row in validation_candidates
        if (
            (owner := validation_owners.get(_strict_hash(row["text"]))) is not None
            and owner[0] == row["id"]
        )
    ]
    validation_roles, validation_partition = partition_validation_records(
        validation_rows,
        seed=seed + 1,
    )
    checkpoint = validation_roles["checkpoint_selection"]
    calibration = validation_roles["calibration"]
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
        calibration=calibration,
        validation_partition=validation_partition,
        canonical_counts=dict(counts),
        canonical_group_counts=dict(group_counts),
        canonical_owners=owners,
        removed={
            "canonical": canonical_removed,
            "validation": validation_removed,
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
        "calibration_rows": len(data.calibration),
        "validation_components": data.validation_partition["components"],
        "promptshield_validation_rows": len(data.promptshield_validation),
        "removed_for_overlap": data.removed,
    }


def _validate_full_recipe(args: argparse.Namespace, report: dict) -> None:
    population = {key: report.get(key) for key in FULL_POPULATION}
    if population != FULL_POPULATION:
        raise ValueError(f"full-mixture population contract failed: {population!r}")
    configuration = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "gradient_checkpointing": (
            args.mode == "lora" and not args.no_gradient_checkpointing
        ),
        "shuffle_buffer": args.shuffle_buffer,
        "head_learning_rate": args.head_learning_rate,
        "adapter_learning_rate": args.adapter_learning_rate,
        "pair_ranking_weight": args.pair_ranking_weight,
    }
    expected = {
        "seed": 42,
        "epochs": 3,
        "batch_size": 128,
        "microbatch_size": 8,
        "gradient_checkpointing": False,
        "shuffle_buffer": 8192,
        "head_learning_rate": 3e-4,
        "adapter_learning_rate": 1e-4,
        "pair_ranking_weight": 0.25,
    }
    if (
        configuration != expected
        or (args.mode != "lora" and args.no_gradient_checkpointing)
        or (args.resume and args.preflight_only)
    ):
        raise ValueError(
            f"full-mixture configuration contract failed: {configuration!r}"
        )


class BalancedIndexCycle:
    """Deterministic class-balanced cycling for PromptShield batches."""

    def __init__(self, labels: np.ndarray, *, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._pools = {
            label: np.flatnonzero(labels == label).astype(np.int64) for label in (0, 1)
        }
        if any(len(pool) == 0 for pool in self._pools.values()):
            raise ValueError("balanced cycle requires both labels")
        self._orders = {
            label: self._rng.permutation(pool) for label, pool in self._pools.items()
        }
        self._positions = {0: 0, 1: 0}

    def _take(self, label: int, count: int) -> list[int]:
        selected = []
        while len(selected) < count:
            order = self._orders[label]
            position = self._positions[label]
            available = min(count - len(selected), len(order) - position)
            selected.extend(order[position : position + available].tolist())
            position += available
            if position == len(order):
                order = self._rng.permutation(self._pools[label])
                position = 0
            self._orders[label] = order
            self._positions[label] = position
        return selected

    def take(self, count: int) -> np.ndarray:
        if count < 2 or count % 2:
            raise ValueError("class-balanced batch size must be positive and even")
        half = count // 2
        selected = self._take(0, half) + self._take(1, half)
        self._rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "pool_sizes": {label: len(pool) for label, pool in self._pools.items()},
            "orders": {label: order.tolist() for label, order in self._orders.items()},
            "positions": dict(self._positions),
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1 or state.get("pool_sizes") != {
            label: len(pool) for label, pool in self._pools.items()
        }:
            raise ValueError("balanced cycle state contract failed")
        for label, pool in self._pools.items():
            order = np.asarray(state.get("orders", {}).get(label), dtype=np.int64)
            position = state.get("positions", {}).get(label)
            if (
                order.shape != pool.shape
                or set(order.tolist()) != set(pool.tolist())
                or type(position) is not int
                or not 0 <= position < len(order)
            ):
                raise ValueError("balanced cycle state contract failed")
            self._orders[label] = order
            self._positions[label] = position
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])


class PairIndexCycle:
    """Deterministic cycling over complete matched-pair atoms."""

    def __init__(self, pairs: int, *, seed: int) -> None:
        if pairs < 1:
            raise ValueError("pair cycle requires at least one pair")
        self._rng = np.random.default_rng(seed)
        self._pool = np.arange(pairs, dtype=np.int64)
        self._order = self._rng.permutation(self._pool)
        self._position = 0

    def take(self, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("pair batch must be positive")
        selected = []
        while len(selected) < count:
            available = min(count - len(selected), len(self._order) - self._position)
            selected.extend(
                self._order[self._position : self._position + available].tolist()
            )
            self._position += available
            if self._position == len(self._order):
                self._order = self._rng.permutation(self._pool)
                self._position = 0
        return np.asarray(selected, dtype=np.int64)

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "pairs": len(self._pool),
            "order": self._order.tolist(),
            "position": self._position,
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        order = np.asarray(state.get("order"), dtype=np.int64)
        position = state.get("position")
        if (
            state.get("schema_version") != 1
            or state.get("pairs") != len(self._pool)
            or order.shape != self._pool.shape
            or set(order.tolist()) != set(self._pool.tolist())
            or type(position) is not int
            or not 0 <= position < len(order)
        ):
            raise ValueError("pair cycle state contract failed")
        self._order = order
        self._position = position
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])


def _classification_backward(
    encoder,
    tokenizer,
    head,
    rows: list[dict],
    *,
    coefficient: float,
    microbatch_size: int,
    train_encoder: bool,
) -> object:
    import torch

    total = None
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
        total = loss.detach() if total is None else total + loss.detach()
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
) -> object:
    import torch

    total = None
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
        total = loss.detach() if total is None else total + loss.detach()
    return total


def _bce_from_logits(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if (
        labels.ndim != 1
        or logits.ndim != 1
        or labels.shape != logits.shape
        or not len(labels)
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("validation BCE requires finite aligned binary rows")
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def _validation_bce(
    encoder,
    tokenizer,
    head,
    rows: list[dict],
    *,
    batch_size: int,
) -> float:
    logits = score_logits(
        encoder,
        tokenizer,
        head,
        [row["text"] for row in rows],
        batch_size=batch_size,
    )
    labels = np.asarray([row["label"] for row in rows])
    return _bce_from_logits(labels, logits)


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
    import torch
    from safetensors.torch import save_file

    name = _run_name(mode, seed)
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
                    "task_type": "FEATURE_EXTRACTION",
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
                "canonical_weighting": "label_source_group_balanced",
                "promptshield_sampling": "class_balanced_cycle",
                "matched_pair_sampling": "complete_pair_cycle",
                "pair_ranking_weight": args.pair_ranking_weight,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "promptshield_batch_size": args.batch_size // 2,
                "pair_batch_pairs": args.batch_size // 4,
                "microbatch_size": args.microbatch_size,
                "mixed_precision": "bfloat16",
                "gradient_checkpointing": (
                    mode == "lora" and not args.no_gradient_checkpointing
                ),
                "head_learning_rate": args.head_learning_rate,
                "adapter_learning_rate": (
                    args.adapter_learning_rate if mode == "lora" else None
                ),
                "shuffle_buffer": args.shuffle_buffer,
                "within_batch_order": "raw_character_length_ascending",
                "updates": math.ceil(report["canonical_rows"] / args.batch_size)
                * args.epochs,
                "selected_epoch": selected_epoch,
                "checkpoint_selection": (
                    "minimum equal-domain mean of Morgott and PromptShield "
                    "validation BCE"
                ),
                "curve": curve,
            },
            "populations": report,
            "runtime_seconds": seconds,
            "runtime": {
                "seconds": seconds,
                "device": torch.cuda.get_device_name(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
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
        if not args.no_gradient_checkpointing:
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            encoder.enable_input_require_grads()
        encoder = add_lora(encoder)
    else:
        encoder.gradient_checkpointing_disable()
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    head = new_head(encoder.config.hidden_size, args.seed).to("cuda")
    head_parameters = list(head.parameters())
    parameters = [{"params": head_parameters, "lr": args.head_learning_rate}]
    if train_encoder:
        adapter_parameters = [
            parameter for parameter in encoder.parameters() if parameter.requires_grad
        ]
        parameters.append(
            {
                "params": adapter_parameters,
                "lr": args.adapter_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameters)
    promptshield_cycle = BalancedIndexCycle(
        np.asarray([row["label"] for row in data.promptshield], dtype=np.int64),
        seed=args.seed + 10_001,
    )
    pair_cycle = PairIndexCycle(len(data.pairs), seed=args.seed + 20_003)
    best = None
    curve = []
    updates = 0
    promptshield_draws = 0
    pair_draws = 0
    next_epoch = 0
    prior_seconds = 0.0
    started = time.perf_counter()
    train_path, train_spec = data.views["train"]
    updates_per_epoch = math.ceil(sum(data.canonical_counts.values()) / args.batch_size)
    expected_updates = updates_per_epoch * args.epochs
    checkpoint = args.output / f".{_run_name(args.mode, args.seed)}.checkpoint.pt"
    identity = _training_identity(args, data)

    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        state = _load_checkpoint(checkpoint, identity=identity)
        next_epoch = state["next_epoch"]
        updates = state["updates"]
        promptshield_draws = state["promptshield_draws"]
        pair_draws = state["pair_draws"]
        prior_seconds = state["runtime_seconds"]
        curve = state["curve"]
        best = state["best"]
        head.load_state_dict(state["head"], strict=True)
        if train_encoder:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(encoder, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        promptshield_cycle.load_state_dict(state["promptshield_cycle"])
        pair_cycle.load_state_dict(state["pair_cycle"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        if (
            type(next_epoch) is not int
            or not 0 < next_epoch <= args.epochs
            or updates != next_epoch * updates_per_epoch
            or len(curve) != next_epoch
            or best is None
        ):
            raise ValueError("resume checkpoint progress contract failed")
        print(
            f"resuming at epoch {next_epoch + 1}/{args.epochs}, "
            f"update {updates}/{expected_updates}",
            flush=True,
        )
    elif checkpoint.exists():
        raise FileExistsError(
            f"checkpoint already exists; pass --resume to use it: {checkpoint}"
        )

    for epoch_index in range(next_epoch, args.epochs):
        epoch = epoch_index + 1
        encoder.train(train_encoder)
        head.train()
        losses = []
        canonical_seen = 0
        stream = training_rows(
            canonical_rows(train_path, train_spec, split="train"),
            data.canonical_counts,
            data.canonical_group_counts,
            data.canonical_owners,
        )
        stream = shuffled(
            stream,
            seed=args.seed + epoch_index,
            buffer_size=args.shuffle_buffer,
        )
        for morgott in batches(stream, args.batch_size):
            morgott.sort(key=lambda row: len(row["text"]))
            canonical_seen += len(morgott)
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
            promptshield_indices = promptshield_cycle.take(args.batch_size // 2)
            promptshield_draws += len(promptshield_indices)
            promptshield_batch = sorted(
                [data.promptshield[int(index)] for index in promptshield_indices],
                key=lambda row: len(row["text"]),
            )
            loss += _classification_backward(
                encoder,
                tokenizer,
                head,
                promptshield_batch,
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
            )
            pair_indices = pair_cycle.take(args.batch_size // 4)
            pair_draws += len(pair_indices)
            pair_batch = sorted(
                [data.pairs[int(index)] for index in pair_indices],
                key=lambda pair: max(
                    len(pair[0]["text"]),
                    len(pair[1]["text"]),
                ),
            )
            loss += _pair_backward(
                encoder,
                tokenizer,
                head,
                pair_batch,
                ranking_weight=args.pair_ranking_weight,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in parameters for parameter in group["params"]],
                1.0,
            )
            optimizer.step()
            losses.append(float(loss))
            updates += 1
            if updates % 100 == 0 or updates == expected_updates:
                elapsed = prior_seconds + time.perf_counter() - started
                eta = elapsed / updates * (expected_updates - updates)
                print(
                    f"epoch {epoch}/{args.epochs} update "
                    f"{updates}/{expected_updates} "
                    f"loss={float(np.mean(losses[-100:])):.5f} "
                    f"elapsed={elapsed / 3600:.2f}h "
                    f"eta={eta / 3600:.2f}h "
                    f"peak_vram={torch.cuda.max_memory_reserved() / (1 << 30):.2f}GiB",
                    flush=True,
                )

        if canonical_seen != sum(data.canonical_counts.values()):
            raise ValueError(
                f"epoch {epoch} saw {canonical_seen} canonical rows, "
                f"expected {sum(data.canonical_counts.values())}"
            )

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
            "canonical_rows_seen": canonical_seen,
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
        elapsed = prior_seconds + time.perf_counter() - started
        _save_checkpoint(
            checkpoint,
            identity=identity,
            state={
                "next_epoch": epoch,
                "updates": updates,
                "promptshield_draws": promptshield_draws,
                "pair_draws": pair_draws,
                "runtime_seconds": elapsed,
                "curve": curve,
                "best": best,
                "head": _cpu_state(head),
                "adapter": _adapter_state(encoder) if train_encoder else None,
                "optimizer": optimizer.state_dict(),
                "promptshield_cycle": promptshield_cycle.state_dict(),
                "pair_cycle": pair_cycle.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
            },
        )
        print(
            f"saved epoch {epoch}/{args.epochs} checkpoint: {checkpoint}",
            flush=True,
        )

    if (
        updates != expected_updates
        or promptshield_draws != expected_updates * (args.batch_size // 2)
        or pair_draws != expected_updates * (args.batch_size // 4)
    ):
        raise ValueError("training update or auxiliary draw contract failed")
    head.load_state_dict(best["head"], strict=True)
    if train_encoder:
        from peft import (
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )

        set_peft_model_state_dict(encoder, best["adapter"])
        restored = get_peft_model_state_dict(encoder)
        if restored.keys() != best["adapter"].keys() or any(
            not torch.equal(restored[name].cpu(), best["adapter"][name])
            for name in restored
        ):
            raise ValueError("restored adapter differs from the selected epoch")
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
        seconds=prior_seconds + time.perf_counter() - started,
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
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    if args.batch_size < 4 or args.batch_size % 4 or args.microbatch_size % 2:
        raise ValueError("batch size must be divisible by four and microbatch by two")
    data = prepare_training_data(
        args.data_dir,
        args.external_dir,
        args.pairs,
        seed=args.seed,
    )
    report = _report(data)
    _validate_full_recipe(args, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    destination = train(args, data)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
