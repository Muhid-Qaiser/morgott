"""Train the maintained frozen-head or rank-8 LoRA mmBERT recipe."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from functools import lru_cache
from importlib.metadata import version
from itertools import chain, islice
from pathlib import Path

import numpy as np

from ...normalization import strict_normalize
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
    source_provenance,
)
from .core import (
    pool as _pool_features,
)
from .data import (
    EXTERNAL_DATA_SCHEMA_VERSION,
    OverlapGuard,
    TrainingData,
    _overlap_values,
    _strict_hash,
    additional_matched_pairs,
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
    "canonical_rows": 1_070_137,
    "promptshield_rows": 18_202,
    "matched_pairs": 11_041,
    "checkpoint_rows": 28_953,
    "calibration_rows": 116_138,
    "validation_components": 36_722,
    "promptshield_validation_rows": 984,
}
ADDITIONAL_PAIR_ARCHIVE_SHA256 = (
    "84a3b1e185755739afca5165ef9aaadb55ce248695bb4c426351f94126ebbbba"
)
ADDITIONAL_PAIR_POPULATION = {**FULL_POPULATION, "matched_pairs": 33_757}
CHECKPOINT_UPDATE_INTERVAL = 500
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
# Pinned FlashAttention-2 kernel. reports/attention-kernel-audit.md recorded
# this revision and the CUDA 13 x86-64 binary SHA-256 below; the loader still
# reaches the Hub even with a warm cache, so the digest is verified on load.
FA2_KERNEL = "kernels-community/flash-attn2@239bb21bd566f598d7e2228eab9788b0a9239b2d"
FA2_KERNEL_BINARY = (
    "build/torch-stable-abi210-cu130-x86_64-linux/_flash_attn2_cuda_2d6ecf3.abi3.so"
)
FA2_KERNEL_SHA256 = "1433d3fe1187211c5ce622a7373d8c0487227384a2b8c74064e5ed6dfc820727"

# The full-mixture contract is three separate things, and conflating them
# blocked every multi-seed and capacity comparison.
#
# PINNED_RECIPE changes what is optimised and stays fixed.
#
# Execution changes only how that optimisation is realised. Gradient
# accumulation is exact here -- `_classification_backward` normalises by the
# whole optimiser batch and `_pair_backward` scales each microbatch by its
# share of the pair list -- so `microbatch_size` cannot move the summed
# gradient. `GradientAccumulationTests` asserts that rather than assuming it.
# It is still not bitwise-neutral: dropout draw shapes and padded lengths
# depend on the partition, so runs at different microbatch sizes are
# statistically equivalent, not reproducible against each other.
#
# The arm records which maintained adaptation recipe is used.
PINNED_RECIPE = {
    "epochs": 3,
    "batch_size": 128,
    "shuffle_buffer": 8192,
    "pair_ranking_weight": 0.25,
    "gradient_checkpointing": False,
}
BASELINE_EXECUTION = {"seed": 42, "microbatch_size": 8}
BASELINE_SEED = BASELINE_EXECUTION["seed"]
SUPPORTED_MAX_TOKENS = (MAX_TOKENS, 1024)


def _runtime_max_tokens(args: argparse.Namespace) -> int:
    value = getattr(args, "max_tokens", MAX_TOKENS)
    if type(value) is not int or value not in SUPPORTED_MAX_TOKENS:
        raise ValueError(f"max tokens must be one of {SUPPORTED_MAX_TOKENS}")
    return value


# `partition_validation_records` is seeded with `seed + 1`, so these two keys
# move with the seed while the other five do not.
PINNED_POPULATION_KEYS = (
    "canonical_rows",
    "promptshield_rows",
    "matched_pairs",
    "validation_components",
    "promptshield_validation_rows",
)
VALIDATION_PARTITION_ROWS = (
    FULL_POPULATION["checkpoint_rows"] + FULL_POPULATION["calibration_rows"]
)
CHECKPOINT_FRACTION_BOUNDS = (0.19, 0.21)

# Learning rates are pinned to the maintained recipes.
BASELINE_ARM_RATES = {
    "frozen": (3e-4, 3e-4),
    "lora": (3e-4, 1e-4),
}


def _arm(args: argparse.Namespace) -> dict:
    """The maintained adaptation recipe used by a run."""
    return {
        "mode": args.mode,
        "lora_rank": LORA_RANK if args.mode == "lora" else None,
    }


def _run_name(
    mode: str,
    seed: int,
    *,
    microbatch_size: int | None = None,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """A different execution recipe needs a different run identity.

    Baseline arm at baseline microbatch reproduces the historical names
    exactly, so retained artifacts keep their paths.
    """
    stem = (
        f"mmbert-lora-full-s{seed}"
        if mode == "lora"
        else f"mmbert-base-full-{mode}-s{seed}"
    )
    suffix = ""
    if (
        microbatch_size is not None
        and microbatch_size != BASELINE_EXECUTION["microbatch_size"]
    ):
        suffix += f"-mb{microbatch_size}"
    if max_tokens != MAX_TOKENS:
        suffix += f"-ctx{max_tokens}"
    return stem + suffix


_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _resolved_run_name(args: argparse.Namespace) -> str:
    """Resolve one run identity for artifacts, checkpoints, and telemetry."""
    max_tokens = _runtime_max_tokens(args)
    name = getattr(args, "run_name", None) or _run_name(
        args.mode,
        args.seed,
        microbatch_size=args.microbatch_size,
        max_tokens=max_tokens,
    )
    if not _RUN_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "--run-name must be 1-128 ASCII letters, digits, dots, underscores, "
            "or hyphens; it must start with a letter or digit"
        )
    if max_tokens != MAX_TOKENS and f"ctx{max_tokens}" not in name:
        raise ValueError(f"non-default context --run-name must include ctx{max_tokens}")
    return name


def _preflight_execution(
    args: argparse.Namespace,
    run_name: str,
    *,
    check_cuda: bool = True,
) -> None:
    """Reject a doomed run before the prepared-corpus build starts."""
    output = args.output
    destination = output / run_name
    checkpoint = output / f".{run_name}.checkpoint.pt"
    snapshots = _snapshot_dir(output, run_name)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"training output is not a directory: {output}")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing output: {destination}")
    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    else:
        if checkpoint.exists():
            raise FileExistsError(
                f"checkpoint already exists; pass --resume to use it: {checkpoint}"
            )
        if snapshots.exists():
            raise FileExistsError(
                f"snapshot directory already exists for this run: {snapshots}"
            )
    if not check_cuda:
        return
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("mmBERT training requires a visible CUDA device")
    # Availability can be true while driver initialisation or allocation still
    # fails. Force both now rather than after a multi-minute corpus preparation.
    try:
        torch.cuda.init()
        torch.empty(1, device="cuda")
    except Exception as error:
        raise RuntimeError("CUDA initialisation failed before training") from error
    if args.attention == "fa2":
        _verified_fa2_variant()
    if args.trackio:
        try:
            from trackio.gpu import gpu_available
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "--trackio requires the tracking extra: uv sync --extra tracking"
            ) from error
        if not gpu_available():
            raise RuntimeError(
                "--trackio requires working NVML GPU telemetry; install the "
                "tracking extra and verify the NVIDIA driver"
            )


_TRACKIO_VALIDATION_METRICS = (
    "validation_morgott_negative_source_label_macro_bce",
    "validation_morgott_positive_source_label_macro_bce",
)
TRACKIO_PROJECT = "morgott"
CHECKPOINT_SELECTION_RULES = ("micro", "source_macro")
_TRACKIO_FINANCE_NEGATIVE_SOURCES = (
    "banking77",
    "harper_valley_bank",
    "tatqa",
)


# Selection is otherwise irreversible: `best` holds weights in a single slot, so
# the checkpoint a rule rejects is gone. On the constant-LR baseline the rules
# disagreed -- the registered blend kept update 9,000 while both source-macro
# and worst-source preferred 8,361, which was 1.8x better on worst source and
# lost only because its PromptShield term happened to be 13x lower. Keeping one
# candidate per rule costs ~27 MB each and makes that call reviewable once the
# real gates are computed.
_ALTERNATE_SELECTION_RULES = {
    "source_macro_only": lambda row: row["validation_morgott_source_macro_bce"],
    "worst_source": lambda row: row["validation_worst_source_bce"],
}

_METRIC_LEGEND = """\
## What these curves mean

All losses are binary cross-entropy: **lower is better**, and BCE is unbounded
above, so a source sitting at 5 is confidently wrong, not merely uncertain.

### Which checkpoint gets kept

Exactly one `selection_rules/ACTIVE_<rule>_blend` curve decides which checkpoint
is packaged. `micro` is the registered July rule: `0.5 * (Morgott row-micro BCE
+ PromptShield BCE)`. `source_macro` is the later context-campaign rule and
replaces the Morgott term with an unweighted mean across sources. Older runs
retain the historical `ACTIVE_registered_blend` key; new runs name the formula
explicitly so unlike selectors are never overlaid.

Both blends are screening heuristics, not low-FPR operating metrics. The run
keeps the lowest-scoring checkpoint rather than the last one, so a rising tail
costs time rather than quality.

So the alternates are logged beside it, and their weights retained on disk
under `alternate-selections/`, unregistered:

- `selection_rules/alt_source_macro_only` -- uses only Morgott's source macro,
  with no PromptShield term.
- `selection_rules/alt_worst_source` -- minimises the worst single source, the
  closest proxy for the standing zero-finance-false-positive rule.

Where the three disagree is where the selection rule is doing real work, and
worth checking against the actual gates (AUROC, TPR at 1% FPR, recall, finance
false positives) -- none of which is the BCE these rules minimise.

### Directional validation losses

- `validation/validation_morgott_positive_source_label_macro_bce` and the
  matching negative metric -- unweighted means over sources that actually have
  that label, so missed attacks and false flags cannot cancel each other.

The pooled, per-source-label, worst-source, and PromptShield values remain in
the curve artifact for selection and provenance. They are intentionally not
expanded into hundreds of dashboard charts.

### Finance false-flag watch

Only the negative-label BCEs for `banking77`, `harper_valley_bank`, and `tatqa`
are surfaced as `val_bce_false_flags/*`. High means increasing confidence in
flagging ordinary finance traffic. All other source and direction values stay
in the curve artifact for targeted analysis.

`checkpoint_diagnostics/positive_recall_at_empirical_1pct_row_fpr` and
`checkpoint_diagnostics/finance_false_positives_at_empirical_1pct_row_fpr` are
descriptive checkpoint-only diagnostics. Their threshold is derived only from
the checkpoint-selection negatives; calibration rows are never scored during
training and this diagnostic never participates in checkpoint selection.

### Training

`train/loss` is the sum of three weighted terms (canonical + PromptShield +
pair ranking), preserving the historical graph. `train/clip_fraction` and the
pre-clip gradient norm make extreme canonical weights visible instead of
silently flattening them.
`performance/examples_per_second` and optimizer updates/second are synchronized
25-update window rates, not asynchronous launch rates.
`train/head_lr` and `train/adapter_lr` show linear warmup then cosine decay.

Tip: the dashboard takes a regex in the sidebar, or `&metric_filter=` in the
URL. Use `^(selection_rules|checkpoint_diagnostics|validation)/` for the
headline view and `^val_bce_false_flags/` for the three finance curves.
"""


class _RunTracker:
    """Optional Trackio logging.

    Trackio is local-first: runs land in a SQLite store under
    `~/.cache/huggingface/trackio` and are viewed with `trackio show`. Nothing
    leaves the box unless `--trackio-space` names a Hugging Face Space. Only
    scalars and the recorded configuration are logged; corpus text, prompts,
    row identities, and credentials never are.
    """

    def __init__(self, args: argparse.Namespace, *, run_name: str, config: dict):
        self._run = None
        if not args.trackio:
            return
        import trackio

        if not args.resume:
            # Without --resume, Trackio uses resume="never", which opens a NEW
            # run id under the SAME name and does not uniquify. The dashboard
            # keys on the name, so a restart-from-zero silently overlays a dead
            # partial curve on the live one -- three had to be deleted by hand
            # on 2026-08-07 before anyone noticed. Fail instead, and name both
            # of the intents the operator might have had.
            try:
                from trackio.sqlite_storage import SQLiteStorage

                clash = SQLiteStorage.get_latest_run_record_by_name(
                    TRACKIO_PROJECT, run_name
                )
            except Exception as error:
                raise RuntimeError(
                    "cannot prove the Trackio run name is unused; refusing to "
                    "start a curve that could overlay an existing run"
                ) from error
            if clash:
                raise ValueError(
                    f"Trackio run {run_name!r} already exists in project "
                    f"{TRACKIO_PROJECT!r}. Pass --resume to continue it, "
                    f"choose a distinct --run-name, or delete the old run. "
                    f"Starting over would merge both curves under one name."
                )
        self._run = trackio.init(
            project=TRACKIO_PROJECT,
            name=run_name,
            group=args.trackio_group or None,
            space_id=args.trackio_space or None,
            config=config,
            # A progress checkpoint is the authority to resume.  Never create
            # a fresh telemetry run merely because its matching Trackio row is
            # missing: that would make the training and dashboard histories
            # disagree about whether this is a continuation.
            resume="must" if args.resume else "never",
            # Trackio samples device utilisation itself, which is the cheapest
            # way to see whether the GPU is actually the bottleneck.
            auto_log_gpu=True,
            auto_log_cpu=False,
        )

    @staticmethod
    def _scalars(metrics: dict) -> dict:
        return {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        if self._run is None:
            return
        self._run.log(self._scalars(metrics), step=step)

    def describe_metrics(self) -> None:
        """Log a compact legend because Trackio charts carry no prose."""
        if self._run is None:
            return
        import trackio

        self._run.log({"notes": trackio.Markdown(_METRIC_LEGEND)})

    def log_validation(self, row: dict, *, step: int) -> None:
        """Log only decision and directional headline validation metrics.

        The complete row, including counts and the descriptive threshold,
        remains in the curve artifact. Keeping those bookkeeping values out of
        Trackio prevents them from becoming redundant dashboard charts.
        """
        if self._run is None:
            return
        metrics = {
            f"validation/{key}": float(row[key])
            for key in _TRACKIO_VALIDATION_METRICS
            if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)
        }
        # Every rule's score, side by side, so the retention decision is visible
        # rather than implicit. The ACTIVE_ prefix names the one that actually
        # picks the saved weights; the alt_ rules only retain candidates.
        selection_rule = row.get("selection_rule")
        if selection_rule not in CHECKPOINT_SELECTION_RULES:
            raise ValueError("validation row has no supported selection rule")
        metrics[f"selection_rules/ACTIVE_{selection_rule}_blend"] = float(
            row["selection_loss"]
        )
        for name, score in _ALTERNATE_SELECTION_RULES.items():
            metrics[f"selection_rules/alt_{name}"] = float(score(row))
        operating_point = row.get("validation_checkpoint_operating_point")
        if isinstance(operating_point, dict):
            recall = operating_point.get("positive_recall")
            finance_false_positives = operating_point.get("finance_false_positives")
            if isinstance(recall, (int, float)) and not isinstance(recall, bool):
                metrics[
                    "checkpoint_diagnostics/positive_recall_at_empirical_1pct_row_fpr"
                ] = float(recall)
            if isinstance(finance_false_positives, (int, float)) and not isinstance(
                finance_false_positives, bool
            ):
                metrics[
                    "checkpoint_diagnostics/"
                    "finance_false_positives_at_empirical_1pct_row_fpr"
                ] = float(finance_false_positives)
        self._run.log(metrics, step=step)

    def log_finance_false_flag_bces(self, values: dict, *, step: int) -> None:
        """Surface only decision-relevant finance negatives.

        The complete nested source-label summary remains in the on-disk curve
        artifact. Expanding it here created hundreds of low-value charts.
        """
        if self._run is None:
            return
        metrics = {}
        for source in _TRACKIO_FINANCE_NEGATIVE_SOURCES:
            directions = values.get(source)
            if directions is None:
                continue
            if not isinstance(directions, dict):
                raise ValueError("source-label validation summary must be nested")
            summary = directions.get("negative")
            if not isinstance(summary, dict) or summary.get("bce") is None:
                continue
            metrics[f"val_bce_false_flags/{source}"] = float(summary["bce"])
        if metrics:
            self._run.log(metrics, step=step)

    def finish(self, summary: dict | None = None) -> None:
        if self._run is None:
            return
        if summary:
            self._run.log(
                self._scalars({f"selected/{k}": v for k, v in summary.items()})
            )
        self._run.finish()


def _training_trackio_metrics(
    latest: dict[str, float],
    averaged: dict[str, float],
    *,
    peak_vram_gib: float,
    head_lr: float,
    adapter_lr: float | None,
    optimizer_updates_per_second: float,
    examples_per_second: float,
) -> dict[str, float]:
    """Build the compact, stable training dashboard schema."""
    metrics = {
        # Preserve this historical name and its last-update semantics.
        "train/loss": latest["primary_loss"],
        "train/canonical_primary_loss": averaged["canonical_primary_loss"],
        "train/promptshield_loss": averaged["promptshield_loss"],
        "train/pair_loss": averaged["pair_loss"],
        "train/pre_clip_gradient_norm": averaged["pre_clip_gradient_norm"],
        "train/clip_fraction": averaged["gradient_clipped"],
        "train/peak_vram_gib": peak_vram_gib,
        "train/head_lr": head_lr,
        "performance/optimizer_updates_per_second": optimizer_updates_per_second,
        "performance/examples_per_second": examples_per_second,
    }
    if adapter_lr is not None:
        metrics["train/adapter_lr"] = adapter_lr
    return metrics


def _usable_cpus() -> int:
    """CPUs this container may actually use.

    `os.cpu_count()` and `sched_getaffinity` both report the host's cores
    inside a quota-limited container -- 128 against a real budget of 13 here --
    so oversubscribing by 10x is the default outcome unless the cgroup quota is
    read directly.
    """
    quotas = []
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            quotas.append(int(quota) / int(period))
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            quotas.append(quota / period)
    except (OSError, ValueError):
        pass
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    available = available or os.cpu_count() or 1
    return max(1, int(min([available, *quotas]) if quotas else available))


@lru_cache(maxsize=1)
def _verified_fa2_variant() -> Path:
    """Resolve the pinned local FA2 variant and verify its executable digest.

    The kernel is executable code fetched through the Hugging Face kernel
    cache.  Pinning the repository revision is necessary but not sufficient:
    fail closed unless the binary selected for this Torch/CUDA platform is the
    exact artifact recorded by the attention-kernel audit.  Local-only
    resolution also prevents a training launch from silently fetching a new
    binary.
    """
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    repo_id, revision = FA2_KERNEL.rsplit("@", 1)
    snapshot = (
        Path(constants.HF_HUB_CACHE)
        / repo_folder_name(repo_id=repo_id, repo_type="kernel")
        / "snapshots"
        / revision
    )
    binary = snapshot / FA2_KERNEL_BINARY
    if not binary.is_file():
        raise FileNotFoundError(
            "pinned FA2 executable is not present in the Hugging Face kernel cache: "
            f"{binary}"
        )
    digest = file_sha256(binary)
    if digest != FA2_KERNEL_SHA256:
        raise RuntimeError(
            "pinned FA2 executable digest mismatch: "
            f"expected {FA2_KERNEL_SHA256}, got {digest}"
        )
    return binary.parent


def _load_base_model_with_attention(attention: str):
    """`core.load_base_model` with a selectable attention kernel.

    `core.py` is SHA-pinned inside every registered artifact and `inference.py`
    compares its `ATTENTION_IMPLEMENTATION` constant against each recorded run,
    so an alternative kernel is built here instead of edited in there. Only the
    training path is affected; maintained inference keeps the pinned loader.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("mmBERT requires a CUDA device")
    if attention == FA2_KERNEL:
        _verified_fa2_variant()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    encoder = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation=attention,
        dtype=torch.bfloat16,
    ).to("cuda")
    return encoder, tokenizer


class _EncodingCache:
    """Memoise `strict_normalize` plus tokenisation, keyed on the raw text.

    Both are pure functions of the text and the pinned tokenizer revision, so
    memoising them cannot change what the encoder sees. It removes the dominant
    CPU cost: `strict_normalize` runs twice per canonical row per epoch and
    re-normalises the whole PromptShield and pair pools on every draw, roughly
    9.6M calls per run over far fewer distinct strings.

    Keyed on text rather than row id so it needs no uniqueness assumption
    across the canonical, PromptShield, and pair populations.
    """

    def __init__(self, tokenizer, *, max_tokens: int = MAX_TOKENS) -> None:
        if type(max_tokens) is not int or max_tokens not in SUPPORTED_MAX_TOKENS:
            raise ValueError(f"max tokens must be one of {SUPPORTED_MAX_TOKENS}")
        self._tokenizer = tokenizer
        self.max_tokens = max_tokens
        self._encoded: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._encoded)

    def encode(self, texts: list[str]) -> list[list[int]]:
        missing = []
        for text in texts:
            if text not in self._encoded:
                missing.append(text)
        if missing:
            # Deduplicate within the request too; a batch can repeat a text.
            unique = list(dict.fromkeys(missing))
            encoded = self._tokenizer(
                [strict_normalize(text) for text in unique],
                add_special_tokens=True,
                max_length=self.max_tokens,
                truncation=True,
            )["input_ids"]
            for text, ids in zip(unique, encoded, strict=True):
                self._encoded[text] = np.asarray(ids, dtype=np.int32)
        return [self._encoded[text].tolist() for text in texts]

    def warm(self, texts, *, workers: int = 0) -> None:
        """Populate the cache up front, off the training critical path.

        `strict_normalize` is ~205 ns/char of pure Python and is the single
        largest CPU cost in a run. It is a pure function, so it fans out across
        processes safely; `imap` with a chunk size preserves order, and the
        tokenizer is a Rust extension that already threads internally.

        Known ceiling: `Pool.imap` hangs forever if a worker dies abruptly
        before Python 3.13, unlike the batched executor idiom
        `_pooled_overlap_pairs` uses, which raises `BrokenProcessPool`.
        Switching this pool over is deferred with its test update as a
        follow-up.
        """
        pending = list(
            dict.fromkeys(text for text in texts if text not in self._encoded)
        )
        if not pending:
            return
        if workers and workers > 1 and len(pending) > 4096:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
            os.environ.setdefault("RAYON_NUM_THREADS", str(workers))
            from multiprocessing import get_context

            # The model, CUDA runtime, and Trackio may already own background
            # threads. Forking that process can inherit poisoned locks or a
            # CUDA context; spawn imports a clean worker containing only the
            # top-level pure normalizer.
            with get_context("spawn").Pool(workers) as workforce:
                normalized = list(
                    workforce.imap(strict_normalize, pending, chunksize=512)
                )
            for start in range(0, len(pending), 1024):
                window = slice(start, start + 1024)
                encoded = self._tokenizer(
                    normalized[window],
                    add_special_tokens=True,
                    max_length=self.max_tokens,
                    truncation=True,
                )["input_ids"]
                for text, ids in zip(pending[window], encoded, strict=True):
                    self._encoded[text] = np.asarray(ids, dtype=np.int32)
            return
        for start in range(0, len(pending), 1024):
            self.encode(pending[start : start + 1024])


def _cached_batch_logits(
    encoder,
    tokenizer,
    head,
    texts: list[str],
    *,
    train_encoder: bool,
    cache: _EncodingCache | None,
    pad_to_multiple_of: int | None = None,
    column: int | None = 0,
    max_tokens: int | None = None,
):
    """`core.batch_logits` with normalisation and tokenisation memoised.

    Mirrors `core.batch_logits` exactly; `tokenizer.pad` is the same code path
    `tokenizer(..., padding=True)` uses internally, so the encoder input is
    bitwise identical. `tests.test_mmbert_cuda_equivalence` asserts that.

    Falls back to the pinned implementation when no cache is supplied, so the
    hash-locked module stays the single definition of the forward pass.
    """
    import torch

    if max_tokens is None:
        max_tokens = getattr(cache, "max_tokens", MAX_TOKENS)
    if type(max_tokens) is not int or max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(f"max tokens must be one of {SUPPORTED_MAX_TOKENS}")
    cache_max_tokens = getattr(cache, "max_tokens", max_tokens)
    if cache_max_tokens != max_tokens:
        raise ValueError("encoding cache context cap differs from the scoring cap")
    if (
        max_tokens == MAX_TOKENS
        and cache is None
        and pad_to_multiple_of is None
        and column == 0
    ):
        return batch_logits(
            encoder, tokenizer, head, texts, train_encoder=train_encoder
        )
    if cache is None:
        cache = _EncodingCache(tokenizer, max_tokens=max_tokens)
    encoded = tokenizer.pad(
        {"input_ids": cache.encode(texts)},
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    ).to("cuda")
    context = nullcontext() if train_encoder else torch.no_grad()
    with context, torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = encoder(**encoded).last_hidden_state
        features = _pool_features(hidden, encoded["attention_mask"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = head(features)
    return logits if column is None else logits[:, column]


class _EpochStream:
    """Replay the identical canonical epoch stream without re-reading the view.

    `training_rows` yields rows in file order with weights derived from fixed
    counts, so every epoch produces the same sequence and only `shuffled`
    re-seeds. Re-reading costs a full JSON parse, a SHA-256 over the whole view,
    and ~1.07M leakage-hash normalisations per epoch.

    The cache is published only after a complete pass, so an aborted first epoch
    cannot serve a truncated stream.
    """

    def __init__(self, factory, *, expected_rows: int) -> None:
        self._factory = factory
        self._expected = expected_rows
        self._rows: list[dict] | None = None

    @property
    def cached(self) -> bool:
        return self._rows is not None

    def __iter__(self) -> Iterator[dict]:
        if self._rows is not None:
            yield from self._rows
            return
        collected: list[dict] = []
        for row in self._factory():
            collected.append(row)
            yield row
        if len(collected) != self._expected:
            raise ValueError(
                f"cached canonical epoch stream changed: {len(collected)} rows, "
                f"expected {self._expected}"
            )
        self._rows = collected


class _MetricWindow:
    """Accumulate detached CUDA scalars and synchronize once per log window."""

    def __init__(self) -> None:
        self._totals: dict[str, object] = {}
        self._latest: dict[str, object] = {}
        self.updates = 0
        self.examples = 0

    def add(self, metrics: dict[str, object], *, examples: int) -> None:
        if set(metrics) != set(self._totals) and self._totals:
            raise ValueError("training metric keys changed within a log window")
        for key, value in metrics.items():
            detached = value.detach()
            if detached.ndim:
                raise ValueError("training metrics must be scalar tensors")
            self._totals[key] = (
                detached if key not in self._totals else self._totals[key] + detached
            )
            self._latest[key] = detached
        self.updates += 1
        self.examples += examples

    def drain(self) -> tuple[dict[str, float], dict[str, float], int, int]:
        if not self.updates:
            raise ValueError("cannot drain an empty training metric window")
        import torch

        keys = sorted(self._totals)
        # One packed device-to-host transfer is one synchronization point. The
        # old loop called float(loss) on every optimizer update and serialized
        # every scalar again at each checkpoint.
        values = (
            torch.stack(
                [
                    *[self._totals[key] for key in keys],
                    *[self._latest[key] for key in keys],
                ]
            )
            .float()
            .cpu()
            .tolist()
        )
        width = len(keys)
        totals = dict(zip(keys, values[:width], strict=True))
        latest = dict(zip(keys, values[width:], strict=True))
        updates, examples = self.updates, self.examples
        self.__init__()
        return totals, latest, updates, examples


def _configure_compiled_backward_autocast() -> None:
    """Make compiled backward match eager FP32 loss/backward semantics."""
    from torch._functorch import config as functorch_config

    # The default (`same_as_forward`) captures backward under the BF16 forward
    # autocast state. PyTorch documents `off` as the eager-equivalent policy.
    functorch_config.backward_pass_autocast = "off"


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


def _resume_progress_valid(
    *,
    next_epoch: object,
    epoch_updates: object,
    epoch_canonical_seen: object,
    epoch_loss_sum: object,
    epoch_loss_count: object,
    updates: object,
    curve: list[dict],
    best: dict | None,
    epochs: int,
    updates_per_epoch: int,
) -> bool:
    """Return whether a loaded checkpoint names one coherent resume point."""

    return not (
        type(next_epoch) is not int
        or type(epoch_updates) is not int
        or type(epoch_canonical_seen) is not int
        or type(epoch_loss_count) is not int
        or not isinstance(epoch_loss_sum, (int, float))
        or not math.isfinite(epoch_loss_sum)
        or epoch_loss_sum < 0
        or not 0 <= next_epoch <= epochs
        or not 0 <= epoch_updates < updates_per_epoch
        or (next_epoch == epochs and epoch_updates != 0)
        or updates != next_epoch * updates_per_epoch + epoch_updates
        or updates == 0
        or sum(not row.get("interim") for row in curve) != next_epoch
        or epoch_loss_count != epoch_updates
        or (best is None and next_epoch > 0)
    )


def _skip_resumed_batches(
    batch_iterator: Iterator[list[dict]],
    *,
    batches_consumed: int,
    canonical_seen: int,
) -> int:
    skipped = 0
    for _ in range(batches_consumed):
        batch = next(batch_iterator, None)
        if batch is None:
            raise ValueError("resume position exceeds the epoch stream")
        skipped += len(batch)
    if skipped != canonical_seen:
        raise ValueError(
            f"resumed epoch replay saw {skipped} canonical rows, "
            f"expected {canonical_seen}"
        )
    return skipped


def _candidate_weights(
    *, mode: str, head, encoder, epoch: int, updates: int, loss: float
) -> dict:
    return {
        "loss": float(loss),
        "epoch": epoch,
        "updates": updates,
        "head": _cpu_state(head),
        "adapter": _adapter_state(encoder) if mode == "lora" else None,
        "encoder": None,
    }


def _save_alternates(run_directory: Path, alternates: dict, *, mode: str) -> None:
    """Write the runner-up checkpoints beside the run, with a readable index.

    These are deliberately not registered models and not loadable by maintained
    inference. They exist so the selection rule can be revisited against the
    real gates -- AUROC, TPR at 1% FPR, recall, finance false positives -- none
    of which is the BCE the rule actually minimises.
    """
    import torch

    if not alternates:
        return
    directory = run_directory / "alternate-selections"
    directory.mkdir(parents=True, exist_ok=True)
    for name, candidate in alternates.items():
        torch.save(candidate, directory / f"{name}.pt")
    (directory / "index.json").write_text(
        json.dumps(
            {
                "purpose": "runner-up checkpoints under alternate selection rules",
                "registered": False,
                "mode": mode,
                "rules": {
                    name: {
                        "loss": candidate["loss"],
                        "epoch": candidate["epoch"],
                        "updates": candidate.get("updates", 0),
                    }
                    for name, candidate in alternates.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _snapshot_dir(output: Path, run_name: str) -> Path:
    """Dot-prefixed sibling of the run, never the run directory itself.

    Launchers and operators treat `RUNS/<name>` as the completed-run marker,
    so writing snapshots inside it mid-run could make a relaunch mistake an
    active or interrupted run for a completed one.
    """
    return output / f".{run_name}.snapshots"


def _should_snapshot(args: argparse.Namespace, updates: int) -> bool:
    """Periodic retention plus one pre-registered comparison checkpoint."""
    comparison_update = getattr(args, "comparison_update", 0)
    return bool(
        (args.snapshot_every and updates % args.snapshot_every == 0)
        or (comparison_update and updates == comparison_update)
    )


def _save_snapshot(
    directory: Path,
    row: dict,
    *,
    training_identity: dict,
    mode: str,
    head,
    encoder,
    epoch: int,
    updates: int,
) -> None:
    """Retain a validation point's weights regardless of any selection rule.

    `best` and each entry of `alternates` hold a single slot, so the weights of
    every checkpoint those rules reject are destroyed as training continues.
    That is only safe when the selection signal is stable, and on the 2026-08-06
    ladder it is not: adjacent `source_macro` validations 500 updates apart
    ranged from 0.27 to 0.96, so a running minimum latches onto whichever early
    point was luckiest and the weights needed to revisit that call are gone.

    Snapshots make selection reviewable against the real gates -- AUROC, TPR at
    1% FPR, recall, finance false positives -- rather than irreversible. Like
    `alternate-selections` they are research artifacts, deliberately outside
    `model-artifacts.json` and not loadable by maintained inference.
    """
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    payload = _candidate_weights(
        mode=mode,
        head=head,
        encoder=encoder,
        epoch=epoch,
        updates=updates,
        loss=row["selection_loss"],
    )
    payload["training_identity"] = copy.deepcopy(training_identity)
    payload["metrics"] = dict(row)
    path = directory / f"update-{updates:06d}.pt"
    # The same atomic write as the progress checkpoint: a snapshot truncated by
    # a kill is worse than a missing one, because it still reads as available.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_snapshot_index(directory: Path, curve: list) -> None:
    """Let reselection read the whole curve without loading a single tensor."""
    if not directory.is_dir():
        return
    kept = {
        int(path.stem.removeprefix("update-")) for path in directory.glob("update-*.pt")
    }
    (directory / "index.json").write_text(
        json.dumps(
            {
                "purpose": "unconditional per-validation weight snapshots",
                "registered": False,
                "points": [
                    {
                        "updates": row["updates"],
                        "epoch": row["epoch"],
                        "file": f"update-{row['updates']:06d}.pt",
                        "selection_loss": row["selection_loss"],
                        "source_macro_only": row["validation_morgott_source_macro_bce"],
                        "worst_source": row["validation_worst_source"],
                        "worst_source_bce": row["validation_worst_source_bce"],
                        "interim": bool(row.get("interim")),
                        "pre_registered_comparison": bool(
                            row.get("pre_registered_comparison")
                        ),
                    }
                    for row in curve
                    if row["updates"] in kept
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _update_alternates(
    alternates: dict, row: dict, *, mode: str, head, encoder, epoch: int, updates: int
) -> None:
    """Retain the best checkpoint under each alternate selection rule.

    See `_ALTERNATE_SELECTION_RULES` for why: the registered rule is one opinion
    and it discards the weights of every checkpoint it rejects.
    """
    for name, score in _ALTERNATE_SELECTION_RULES.items():
        value = score(row)
        current = alternates.get(name)
        if current is None or value < current["loss"]:
            alternates[name] = _candidate_weights(
                mode=mode,
                head=head,
                encoder=encoder,
                epoch=epoch,
                updates=updates,
                loss=value,
            )


def _lr_multiplier(step: int, *, total_updates: int, warmup_updates: int) -> float:
    """Linear warmup then cosine decay, as a multiplier on each group's base LR.

    The constant-rate baseline peaked at update 9,000 of 25,083 and was still
    degrading when it was stopped.
    """
    if warmup_updates > 0 and step < warmup_updates:
        return (step + 1) / warmup_updates
    span = max(total_updates - warmup_updates, 1)
    progress = min(max(step - warmup_updates, 0) / span, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _shuffled_chunks(items: list, size: int, rng) -> list[list]:
    chunks = [items[start : start + size] for start in range(0, len(items), size)]
    return [chunks[index] for index in rng.permutation(len(chunks))]


def length_grouped_batches(rows, batch_size: int, *, key, factor: int, rng):
    """Yield length-homogeneous batches, in shuffled order.

    Padding is charged per microbatch at its longest row, so batches drawn at
    random from this corpus -- median 75 characters against a mean of 661 --
    spend 1.37 padded tokens per real one. Sorting a megabatch and cutting it
    into batches takes that to ~1.00, measured at 27% fewer tokens processed
    per update across the three populations.

    The shuffle afterwards is what keeps it honest. Without it every epoch
    would walk short batches to long ones, correlating batch content with
    training time -- and in this corpus length tracks source, which is exactly
    what `source_macro` selection grades on.
    """
    for megabatch in batches(rows, batch_size * factor):
        megabatch.sort(key=key)
        yield from _shuffled_chunks(megabatch, batch_size, rng)


class _LengthGroupedCycle:
    """Length-grouped batching layered over an index cycle.

    Optional grouping preserves per-batch proportions while sorting each group
    by length. It is a drop-in for the cycle: `take`, `state_dict` and
    `load_state_dict` all match, so checkpointing is unchanged.
    """

    def __init__(self, cycle, rows: list, *, key, factor: int, seed: int, group=None):
        self._cycle = cycle
        self._rows = rows
        self._key = key
        self._group = group
        self._factor = factor
        self._rng = np.random.default_rng(seed)
        self._queue: list[list[int]] = []

    def take(self, count: int) -> list[int]:
        if not self._queue:
            drawn = [int(index) for index in self._cycle.take(count * self._factor)]
            if self._group is None:
                drawn.sort(key=lambda index: self._key(self._rows[index]))
                self._queue = _shuffled_chunks(drawn, count, self._rng)
            else:
                grouped: dict[object, list[int]] = {}
                for index in drawn:
                    grouped.setdefault(self._group(self._rows[index]), []).append(index)
                chunks = []
                for indices in grouped.values():
                    indices.sort(key=lambda index: self._key(self._rows[index]))
                    chunk_size, remainder = divmod(len(indices), self._factor)
                    if not chunk_size or remainder:
                        raise ValueError(
                            "length grouping cannot preserve group balance"
                        )
                    chunks.append(
                        [
                            indices[start : start + chunk_size]
                            for start in range(0, len(indices), chunk_size)
                        ]
                    )
                self._queue = [
                    list(chain.from_iterable(group[batch] for group in chunks))
                    for batch in range(self._factor)
                ]
                self._rng.shuffle(self._queue)
        return self._queue.pop()

    def state_dict(self) -> dict:
        return {
            "schema_version": 1,
            "cycle": self._cycle.state_dict(),
            "grouped": self._group is not None,
            "queue": [list(batch) for batch in self._queue],
            "rng": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state.get("schema_version") != 1
            or state.get("grouped", False) != (self._group is not None)
            or "queue" not in state
        ):
            raise ValueError("length-grouped cycle state contract failed")
        self._cycle.load_state_dict(state["cycle"])
        self._queue = [list(batch) for batch in state["queue"]]
        self._rng.bit_generator.state = copy.deepcopy(state["rng"])


def _progress_state(
    *,
    mode: str,
    completed_epochs: int,
    epoch_updates: int,
    epoch_loss_sum: float,
    epoch_loss_count: int,
    epoch_canonical_seen: int,
    updates: int,
    promptshield_draws: int,
    pair_draws: int,
    runtime_seconds: float,
    curve: list,
    best: dict | None,
    alternates: dict,
    head,
    encoder,
    optimizer,
    scheduler,
    promptshield_cycle,
    pair_cycle,
) -> dict:
    import torch

    return {
        "next_epoch": completed_epochs,
        "epoch_updates": epoch_updates,
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_loss_count": epoch_loss_count,
        "epoch_canonical_seen": epoch_canonical_seen,
        "updates": updates,
        "promptshield_draws": promptshield_draws,
        "pair_draws": pair_draws,
        "runtime_seconds": runtime_seconds,
        "curve": curve,
        "best": best,
        "alternates": alternates,
        "head": _cpu_state(head),
        "adapter": _adapter_state(encoder) if mode == "lora" else None,
        "encoder": None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "promptshield_cycle": promptshield_cycle.state_dict(),
        "pair_cycle": pair_cycle.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
    }


def _head_contract() -> dict:
    return {
        "outputs": 1,
        "columns": {"0": "instruction_subversion"},
        "primary_column": 0,
        "architecture": "legacy_sequential_binary_v1",
    }


def _training_source_paths() -> tuple[Path, ...]:
    """Local Python sources whose behavior can change a training run.

    Checkpoint identity and final-result provenance must use the same inventory:
    otherwise a resumed run can silently mix source versions while the final
    artifact records only the last process's files.
    """

    module = Path(__file__).resolve()
    package = module.parents[2]
    return (
        module,
        module.with_name("core.py"),
        module.with_name("data.py"),
        module.with_name("external_data.py"),
        package / "data.py",
        package / "normalization.py",
        package / "overlap.py",
    )


def _prep_dependency_paths() -> tuple[Path, ...]:
    """Imported local sources that determine the prepared-corpus payload."""

    training_sources = _training_source_paths()
    names = {
        "src/morgott/models/mmbert/data.py",
        "src/morgott/models/mmbert/external_data.py",
        "src/morgott/data.py",
        "src/morgott/normalization.py",
        "src/morgott/overlap.py",
    }
    root = Path(__file__).resolve().parents[4]
    selected = tuple(
        path for path in training_sources if str(path.relative_to(root)) in names
    )
    if len(selected) != len(names):
        raise AssertionError("prepared-corpus source inventory is incomplete")
    return selected


def _training_identity(
    args: argparse.Namespace,
    data: TrainingData,
    *,
    run_name: str | None = None,
    optimizer_fused: bool | None = None,
) -> dict:
    max_tokens = _runtime_max_tokens(args)
    return {
        "schema_version": 5,
        "run_name": run_name or _resolved_run_name(args),
        "mode": args.mode,
        "arm": _arm(args),
        "head_contract": _head_contract(),
        "attention": args.attention,
        "selection_rule": args.selection_rule,
        "validation_interval": args.validation_interval,
        "comparison_update": getattr(args, "comparison_update", 0),
        "snapshot_every": args.snapshot_every,
        "checkpoint_interval": args.checkpoint_interval,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "compiled": bool(args.compile_encoder),
        "compiled_backward_autocast": ("off" if args.compile_encoder else None),
        "encoding_cache": not args.no_encoding_cache,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "max_tokens": max_tokens,
        "token_budget": args.microbatch_size * max_tokens,
        "shuffle_buffer": args.shuffle_buffer,
        "length_grouped": bool(args.length_grouped),
        "length_group_factor": args.length_group_factor,
        "warmup_fraction": args.warmup_fraction,
        "head_learning_rate": args.head_learning_rate,
        "adapter_learning_rate": args.adapter_learning_rate,
        "pair_ranking_weight": args.pair_ranking_weight,
        "optimizer": {
            "name": "AdamW",
            "betas": list(ADAMW_BETAS),
            "eps": ADAMW_EPS,
            "weight_decay": ADAMW_WEIGHT_DECAY,
            "fused": optimizer_fused,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "gradient_checkpointing": (
            args.mode == "lora" and not args.no_gradient_checkpointing
        ),
        "data": {
            "manifest_sha256": data.data_manifest_sha256,
            "external_manifest_sha256": data.external_manifest_sha256,
            "pair_archive_sha256": file_sha256(args.pairs),
            "additional_pair_archive_sha256": (
                file_sha256(args.additional_pairs)
                if args.additional_pairs is not None
                else None
            ),
            "routing_views": {
                split: spec["sha256"] for split, (_, spec) in data.views.items()
            },
            "populations": _report(data),
        },
        "sources": source_provenance(*_training_source_paths()),
    }


def _references(views: dict, external: dict):
    for split in ("validation", "dev_test"):
        path, spec = views[split]
        yield from canonical_rows(path, spec, split=split, eligible_only=False)
    for row in external["promptshield_validation"]:
        yield {**row, "_candidate_dataset": "promptshield_validation"}
    yield from external["promptshield_test"]
    yield from external["sep"]


# The only functions in this module that prepare the corpus. Everything else
# here -- the training loop, the save path, the CLI, the Trackio wiring -- can
# change without altering a single prepared row.
_PREP_SOURCE_FUNCTIONS = (
    "_pooled_overlap_pairs",
    "_prepare_training_data",
    "_references",
)


def _prep_source_digest() -> str:
    """Hash just the corpus-preparing sources in this module.

    Hashing the whole file meant any unrelated fix threw away a 502 MB cache and
    18 minutes of single-threaded work over 2.1M rows; that happened four times
    on 2026-08-07. Read from the file with `ast` rather than
    `inspect.getsource` on the live objects, so patching `_prepare_training_data`
    -- which the cache tests legitimately do -- still yields a stable key.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    segments = sorted(
        ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in _PREP_SOURCE_FUNCTIONS
    )
    if len(segments) != len(_PREP_SOURCE_FUNCTIONS):
        raise ValueError("prep cache key cannot locate its source functions")
    return hashlib.sha256("\n".join(segments).encode("utf-8")).hexdigest()


def _prep_cache_key(
    data_dir: Path,
    external_dir: Path,
    pair_archive: Path,
    seed: int,
    additional_pair_archive: Path | None,
) -> str:
    """Digest of everything the prepared corpus depends on.

    Every input is hashed, not stat'd: the data contract fails closed on a
    changed digest, and a cache keyed on mtime would quietly hand back a corpus
    that no longer matches the manifest. The source files that do the preparing
    are hashed too, so editing the guard or the normaliser invalidates the
    cache without anyone remembering to bump a version.
    """
    parts = [
        "prep-cache-v1",
        str(seed),
        file_sha256(data_dir / "manifest.json"),
        file_sha256(external_dir / "manifest.json"),
        file_sha256(pair_archive),
        file_sha256(additional_pair_archive) if additional_pair_archive else "none",
    ]
    parts.append(_prep_source_digest())
    for path in _prep_dependency_paths():
        parts.append(file_sha256(path))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _prep_cache_physical_inputs(
    data_dir: Path,
    external_dir: Path,
) -> list[tuple[str, Path, str]]:
    """Resolve the seven manifest-pinned files a prepared cache represents."""

    result = [
        (f"routing:{split}", path, spec["sha256"])
        for split, (path, spec) in sorted(routing_views(data_dir).items())
    ]
    manifest = json.loads((external_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXTERNAL_DATA_SCHEMA_VERSION:
        raise ValueError("unsupported external data manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("external data manifest has no outputs")
    root = external_dir.resolve()
    for name in (
        "promptshield_train",
        "promptshield_validation",
        "promptshield_test",
        "sep",
    ):
        spec = outputs.get(name)
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("path"), str)
            or not isinstance(spec.get("sha256"), str)
            or not isinstance(spec.get("rows"), int)
        ):
            raise ValueError(f"invalid external manifest entry: {name}")
        path = (external_dir / spec["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"external manifest path escapes {external_dir}: {name}")
        result.append((f"external:{name}", path, spec["sha256"]))
    if len(result) != 7:
        raise AssertionError("prepared-cache physical input inventory changed")
    return result


def _verify_prep_cache_physical_inputs(
    data_dir: Path,
    external_dir: Path,
) -> None:
    """Hash every physical manifest input before unpickling a cache hit.

    The seven files are independent and `file_sha256` releases the GIL inside
    hashlib, so a small thread pool overlaps the reads (measured 1.85x on the
    current 9.85 GB). Every check still runs, and the outcomes are examined
    strictly in input order, so the raised errors and messages are the ones
    the serial loop produced.
    """

    inputs = _prep_cache_physical_inputs(data_dir, external_dir)
    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in [executor.submit(file_sha256, path) for _, path, _ in inputs]:
            try:
                outcomes.append((future.result(), None))
            except OSError as error:
                outcomes.append((None, error))
    mismatches = []
    for (name, path, expected), (observed, error) in zip(inputs, outcomes, strict=True):
        if error is not None:
            raise ValueError(
                f"prepared corpus cache input is unavailable: {name}: {path}"
            ) from error
        if observed != expected:
            mismatches.append(f"{name}: expected {expected}, got {observed}")
    if mismatches:
        raise ValueError(
            "prepared corpus cache input hash mismatch: " + "; ".join(mismatches)
        )


def prepare_training_data(
    data_dir: Path,
    external_dir: Path,
    pair_archive: Path,
    *,
    seed: int = 42,
    additional_pair_archive: Path | None = None,
    cache_dir: Path | None = None,
) -> TrainingData:
    """Prepare the corpus, reusing a digest-keyed cache when one matches.

    The preparation itself costs ~17 minutes, almost all of it per-row overlap
    work: `_overlap_values` measures 299 us/row, so one pass over the 1,073,230
    canonical training rows is 5.4 minutes and the build makes several. Nothing
    about it varies between runs on the same corpus, so every restart paid it
    again -- four times in one afternoon on 2026-08-06.
    """
    if cache_dir is None:
        return _prepare_training_data(
            data_dir,
            external_dir,
            pair_archive,
            seed=seed,
            additional_pair_archive=additional_pair_archive,
        )
    key = _prep_cache_key(
        data_dir, external_dir, pair_archive, seed, additional_pair_archive
    )
    payload = cache_dir / f"{key}.pickle"
    digest = cache_dir / f"{key}.sha256"
    if payload.is_file() and digest.is_file():
        recorded = digest.read_text(encoding="utf-8").strip()
        if file_sha256(payload) == recorded:
            _verify_prep_cache_physical_inputs(data_dir, external_dir)
            with payload.open("rb") as handle:
                data = pickle.load(handle)
            if isinstance(data, TrainingData):
                print(f"prepared corpus cache hit: {payload.name}", flush=True)
                return data
        # A corrupt or truncated cache is discarded rather than trusted.
        print("prepared corpus cache failed verification; rebuilding", flush=True)
        payload.unlink(missing_ok=True)
        digest.unlink(missing_ok=True)
    data = _prepare_training_data(
        data_dir,
        external_dir,
        pair_archive,
        seed=seed,
        additional_pair_archive=additional_pair_archive,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_dir, suffix=".partial", delete=False
    ) as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    # Publish the digest only after the payload is in place, so an interrupted
    # write can never be read back as valid.
    temporary.replace(payload)
    digest.write_text(file_sha256(payload) + "\n", encoding="utf-8")
    print(f"prepared corpus cached: {payload.name}", flush=True)
    return data


_OVERLAP_POOL_BATCH = 512


def _pooled_overlap_pairs(
    rows: Iterable[dict],
    *,
    workers: int | None = None,
    batch_size: int = _OVERLAP_POOL_BATCH,
) -> Iterator[tuple[dict, tuple[str, str, int | None]]]:
    """Pair each row with its `_overlap_values` triple, hashed in parallel.

    The triples are pure functions of the row text, so batches fan out across
    spawn workers much as `_EncodingCache.warm` and the routing build already
    do; executor `map` returns results in input order, so the sequential
    guard, owner, and dedup consumers observe the unchanged row order with
    identical values, and it raises `BrokenProcessPool` instead of hanging the
    build if a worker dies (`Pool.map` blocks forever on that before Python
    3.13, and this machine has an OOM history). A stream shorter than one
    batch never pays the pool spawn, which keeps preflights and small test
    corpora serial. Worker count stays modest by default because prep shares
    the machine with training and builds; `MORGOTT_OVERLAP_WORKERS` (a
    positive integer, clamped to the usable CPU quota) overrides it on larger
    hosts.
    """
    if workers is None:
        raw = os.environ.get("MORGOTT_OVERLAP_WORKERS")
        if raw is None:
            workers = min(4, _usable_cpus())
        else:
            try:
                workers = int(raw)
            except ValueError:
                workers = 0
            if workers < 1:
                raise ValueError(
                    f"MORGOTT_OVERLAP_WORKERS must be a positive integer, got {raw!r}"
                )
            workers = min(workers, _usable_cpus())
    iterator = iter(rows)
    first = list(islice(iterator, batch_size))
    if workers <= 1 or len(first) < batch_size:
        for row in chain(first, iterator):
            yield row, _overlap_values(row)
        return
    from multiprocessing import get_context

    # Spawn, not fork: the caller may already own CUDA or tracker threads.
    # ponytail: per-batch map barrier idles workers while the next batch is
    # read from disk, overlap with submit if the barrier ever dominates.
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        for batch in chain([first], batches(iterator, batch_size)):
            yield from zip(
                batch,
                pool.map(_overlap_values, batch, chunksize=16),
                strict=True,
            )


def _prepare_training_data(
    data_dir: Path,
    external_dir: Path,
    pair_archive: Path,
    *,
    seed: int = 42,
    additional_pair_archive: Path | None = None,
) -> TrainingData:
    views = routing_views(data_dir)
    external, _ = external_rows(external_dir)
    original_pairs = matched_pairs(pair_archive)
    if additional_pair_archive is not None:
        original_pairs += additional_matched_pairs(additional_pair_archive)
    candidates = {
        "promptshield": external["promptshield_train"],
        "promptshield_validation": external["promptshield_validation"],
        "pairs": [row for pair in original_pairs for row in pair],
    }
    guard = OverlapGuard(())
    kept, small_removed = filter_small_training_sets(
        candidates,
        _pooled_overlap_pairs(_references(views, external)),
        reference_guard=guard,
        precomputed=True,
    )
    guard.add(chain(kept["promptshield"], kept["promptshield_validation"]))
    train_path, train_spec = views["train"]
    (
        counts,
        group_counts,
        canonical_removed,
        owners,
        pair_rows,
        pair_train_removed,
    ) = profile_canonical(
        _pooled_overlap_pairs(canonical_rows(train_path, train_spec, split="train")),
        guard,
        {"pairs": kept["pairs"]},
        precomputed=True,
    )
    kept_pair_ids = {row["id"] for row in pair_rows["pairs"]}
    pairs = [
        pair
        for pair in original_pairs
        if pair[0]["id"] in kept_pair_ids and pair[1]["id"] in kept_pair_ids
    ]
    dev_path, dev_spec = views["dev_test"]
    validation_guard = OverlapGuard(())
    validation_guard.add(
        _pooled_overlap_pairs(
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
        ),
        precomputed=True,
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
    ) = profile_canonical(
        _pooled_overlap_pairs(validation_candidates),
        validation_guard,
        {},
        precomputed=True,
    )
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
        "leakage_fingerprint": data.validation_partition["leakage_fingerprint"],
        "promptshield_validation_rows": len(data.promptshield_validation),
        "removed_for_overlap": data.removed,
    }


def _validate_population(args: argparse.Namespace, report: dict) -> None:
    """Pin the seed-invariant counts; bound the seed-dependent partition.

    `partition_validation_records` is seeded with `seed + 1`, so only
    `checkpoint_rows` and `calibration_rows` move across seeds. Their sum is
    invariant, and at the baseline seed both keep their exact historical values.
    """
    additional_pairs = getattr(args, "additional_pairs", None)
    expected = (
        ADDITIONAL_PAIR_POPULATION if additional_pairs is not None else FULL_POPULATION
    )
    pinned = {key: report.get(key) for key in PINNED_POPULATION_KEYS}
    if pinned != {key: expected[key] for key in PINNED_POPULATION_KEYS}:
        raise ValueError(f"full-mixture population contract failed: {pinned!r}")

    checkpoint_rows = report.get("checkpoint_rows")
    calibration_rows = report.get("calibration_rows")
    partition = {
        "checkpoint_rows": checkpoint_rows,
        "calibration_rows": calibration_rows,
    }
    if (checkpoint_rows or 0) + (calibration_rows or 0) != VALIDATION_PARTITION_ROWS:
        raise ValueError(f"validation partition contract failed: {partition!r}")
    if args.seed == BASELINE_SEED:
        if partition != {key: expected[key] for key in partition}:
            raise ValueError(f"validation partition contract failed: {partition!r}")
        return
    fraction = checkpoint_rows / VALIDATION_PARTITION_ROWS
    low, high = CHECKPOINT_FRACTION_BOUNDS
    if not low <= fraction <= high:
        raise ValueError(f"validation partition contract failed: {partition!r}")


def _validate_arm(args: argparse.Namespace) -> dict:
    """Enforce the learning rates pinned to each maintained recipe."""
    arm = _arm(args)
    expected_head, expected_adapter = BASELINE_ARM_RATES[args.mode]
    if (args.head_learning_rate, args.adapter_learning_rate) != (
        expected_head,
        expected_adapter,
    ):
        raise ValueError(
            f"pinned learning rates for {args.mode!r} are "
            f"{expected_head} and {expected_adapter}"
        )
    return arm


def _validate_full_recipe(args: argparse.Namespace, report: dict) -> None:
    _runtime_max_tokens(args)
    additional_pairs = getattr(args, "additional_pairs", None)
    additional_pairs_valid = additional_pairs is None or (
        args.mode == "lora"
        and additional_pairs.is_file()
        and file_sha256(additional_pairs) == ADDITIONAL_PAIR_ARCHIVE_SHA256
    )
    if not additional_pairs_valid:
        raise ValueError("full-mixture additional-pair contract failed")
    _validate_population(args, report)
    _validate_arm(args)
    expected_updates = (
        math.ceil(report["canonical_rows"] / args.batch_size) * args.epochs
    )
    comparison_update = getattr(args, "comparison_update", 0)
    if comparison_update > expected_updates:
        raise ValueError(
            f"comparison update {comparison_update} exceeds "
            f"the {expected_updates}-update training run"
        )

    recipe = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "shuffle_buffer": args.shuffle_buffer,
        "pair_ranking_weight": args.pair_ranking_weight,
        "gradient_checkpointing": (
            args.mode == "lora" and not args.no_gradient_checkpointing
        ),
    }
    if (
        recipe != PINNED_RECIPE
        or (args.mode == "frozen" and args.no_gradient_checkpointing)
        or (args.resume and args.preflight_only)
    ):
        raise ValueError(f"full-mixture configuration contract failed: {recipe!r}")

    execution = {"seed": args.seed, "microbatch_size": args.microbatch_size}
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if (
        args.microbatch_size < 2
        or args.microbatch_size % 2
        or args.microbatch_size > args.batch_size
    ):
        raise ValueError(f"execution contract failed: {execution!r}")


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
    cache: _EncodingCache | None = None,
    pad_to_multiple_of: int | None = None,
    max_tokens: int = MAX_TOKENS,
) -> object:
    import torch

    total = None
    for batch in batches(rows, microbatch_size):
        logits = _cached_batch_logits(
            encoder,
            tokenizer,
            head,
            [row["text"] for row in batch],
            train_encoder=train_encoder,
            cache=cache,
            pad_to_multiple_of=pad_to_multiple_of,
            max_tokens=max_tokens,
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
    cache: _EncodingCache | None = None,
    pad_to_multiple_of: int | None = None,
    max_tokens: int = MAX_TOKENS,
) -> object:
    import torch

    total = None
    pair_microbatch = max(1, microbatch_size // 2)
    for batch in batches(pairs, pair_microbatch):
        benign = [pair[0] for pair in batch]
        attack = [pair[1] for pair in batch]
        logits = _cached_batch_logits(
            encoder,
            tokenizer,
            head,
            [row["text"] for row in [*benign, *attack]],
            train_encoder=train_encoder,
            cache=cache,
            pad_to_multiple_of=pad_to_multiple_of,
            max_tokens=max_tokens,
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


def _validation_logits(
    encoder,
    tokenizer,
    head,
    texts: list[str],
    *,
    batch_size: int,
    max_tokens: int = MAX_TOKENS,
    cache: _EncodingCache | None = None,
) -> np.ndarray:
    """Cap-aware validation logits through the cached scoring path.

    Both caps run through `_cached_batch_logits`: with the trainer's warmed
    encoding cache it reuses the memoised tokens instead of re-normalising the
    same ~30k texts on every pass, and without one it falls back to the pinned
    `batch_logits`, so the hash-locked module stays the single forward-pass
    definition. Padding stays exact (`pad_to_multiple_of=None`):
    `tests.test_mmbert_cuda_equivalence` asserts the cached path is bitwise
    identical to `batch_logits` only without a padding multiple, so the
    recorded BCE curve and checkpoint selection cannot move.
    """
    import torch

    encoder.eval()
    head.eval()
    outputs = []
    with torch.no_grad():
        for batch in batches(texts, batch_size):
            values = _cached_batch_logits(
                encoder,
                tokenizer,
                head,
                batch,
                train_encoder=False,
                cache=cache,
                max_tokens=max_tokens,
            )
            outputs.append(values.float().cpu().numpy())
    if not outputs:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(outputs).astype(np.float64)


_CHECKPOINT_ROW_FPR_TARGET = 0.01
_FINANCE_SOURCES = frozenset({"banking77", "harper_valley_bank", "tatqa"})


def _checkpoint_empirical_operating_point(
    rows: list[dict],
    logits: np.ndarray,
    *,
    target_row_fpr: float = _CHECKPOINT_ROW_FPR_TARGET,
) -> dict:
    """A conservative descriptive threshold derived from checkpoint negatives.

    With budget = floor(target * negatives), the threshold is one representable
    float above descending_negative_logits[budget]. Using >= therefore admits
    at most budget negatives even when several tie at the boundary. This never
    reads calibration rows and never selects weights.
    """

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    if (
        logits.ndim != 1
        or logits.shape != labels.shape
        or not np.isfinite(logits).all()
        or not np.isin(labels, (0, 1)).all()
        or not 0.0 < target_row_fpr < 1.0
    ):
        raise ValueError("checkpoint operating point requires finite binary rows")
    negative = labels == 0
    positive = labels == 1
    if not negative.any() or not positive.any():
        raise ValueError("checkpoint operating point requires both labels")

    negative_logits = np.sort(logits[negative])[::-1]
    false_positive_budget = int(math.floor(target_row_fpr * len(negative_logits)))
    boundary = negative_logits[false_positive_budget]
    threshold = float(np.nextafter(boundary, np.inf))
    flagged = logits >= threshold
    false_positives = int(np.count_nonzero(flagged & negative))
    if false_positives > false_positive_budget:
        raise AssertionError("checkpoint threshold exceeded its row-FPR budget")

    finance_negative = np.asarray(
        [row.get("source") in _FINANCE_SOURCES and row["label"] == 0 for row in rows],
        dtype=bool,
    )
    true_positives = int(np.count_nonzero(flagged & positive))
    positive_rows = int(np.count_nonzero(positive))
    return {
        "role": "checkpoint_selection_descriptive_only",
        "target_row_fpr": float(target_row_fpr),
        "threshold_rule": "nextafter(descending_negative_logits[budget], +inf)",
        "threshold_logit": threshold,
        "threshold_probability": float(
            1.0 / (1.0 + np.exp(-np.clip(threshold, -80.0, 80.0)))
        ),
        "negative_rows": int(np.count_nonzero(negative)),
        "false_positive_budget": false_positive_budget,
        "false_positives": false_positives,
        "observed_row_fpr": false_positives / int(np.count_nonzero(negative)),
        "positive_rows": positive_rows,
        "true_positives": true_positives,
        "false_negatives": positive_rows - true_positives,
        "positive_recall": true_positives / positive_rows,
        "finance_negative_rows": int(np.count_nonzero(finance_negative)),
        "finance_false_positives": int(np.count_nonzero(flagged & finance_negative)),
    }


def _primary_validation_summary(
    rows: list[dict],
    logits: np.ndarray,
    *,
    checkpoint_diagnostics: bool,
) -> dict:
    """Preserve pooled/source BCE and add label-direction-aware summaries."""

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    pooled = _bce_from_logits(labels, logits)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.get("source", "unknown"), []).append(index)

    by_source = {}
    by_source_label = {}
    directional: dict[str, list[float]] = {"negative": [], "positive": []}
    for source, indices in sorted(grouped.items()):
        source_indices = np.asarray(indices, dtype=np.int64)
        by_source[source] = _bce_from_logits(
            labels[source_indices], logits[source_indices]
        )
        by_source_label[source] = {}
        for label, name in ((0, "negative"), (1, "positive")):
            selected = source_indices[labels[source_indices] == label]
            bce = (
                _bce_from_logits(labels[selected], logits[selected])
                if len(selected)
                else None
            )
            by_source_label[source][name] = {
                "rows": int(len(selected)),
                "bce": bce,
            }
            if bce is not None:
                directional[name].append(bce)

    result = {
        "bce": pooled,
        "by_source": by_source,
        "by_source_label": by_source_label,
        "negative_source_label_macro_bce": float(np.mean(directional["negative"]))
        if directional["negative"]
        else None,
        "positive_source_label_macro_bce": float(np.mean(directional["positive"]))
        if directional["positive"]
        else None,
    }
    if checkpoint_diagnostics:
        result["operating_point"] = _checkpoint_empirical_operating_point(rows, logits)
    return result


def _validation_bce_by_source(
    encoder,
    tokenizer,
    head,
    rows: list[dict],
    *,
    batch_size: int,
    max_tokens: int = MAX_TOKENS,
    cache: _EncodingCache | None = None,
    checkpoint_diagnostics: bool = False,
) -> dict:
    """Score once and return pooled, legacy source, and directional BCE."""

    logits = _validation_logits(
        encoder,
        tokenizer,
        head,
        [row["text"] for row in rows],
        batch_size=batch_size,
        max_tokens=max_tokens,
        cache=cache,
    )
    return _primary_validation_summary(
        rows,
        logits,
        checkpoint_diagnostics=checkpoint_diagnostics,
    )


def _selection_loss(bces: dict[str, float], by_source: dict[str, dict], rule: str):
    """Equal-domain mean of Morgott and PromptShield validation BCE.

    `micro` reproduces the registered rule exactly. `source_macro` replaces the
    Morgott term with the unweighted mean across its sources, so a large source
    can no longer mask a regression on every small one.
    """
    if rule == "micro":
        return 0.5 * sum(bces.values())
    sources = by_source.get("morgott") or {}
    morgott = sum(sources.values()) / len(sources) if sources else bces["morgott"]
    return 0.5 * (morgott + bces["promptshield"])


def _validation_row(
    encoder,
    tokenizer,
    head,
    data: TrainingData,
    args: argparse.Namespace,
    *,
    epoch: int,
    updates: int,
    training_loss: float,
    canonical_seen: int,
    encoding_cache: _EncodingCache | None = None,
) -> dict:
    """One point on the validation curve, at an epoch or mid-epoch boundary."""
    was_training = encoder.training
    encoder.eval()
    head.eval()
    try:
        max_tokens = _runtime_max_tokens(args)
        morgott_summary = _validation_bce_by_source(
            encoder,
            tokenizer,
            head,
            data.checkpoint,
            batch_size=args.microbatch_size,
            max_tokens=max_tokens,
            cache=encoding_cache,
            checkpoint_diagnostics=True,
        )
        promptshield_summary = _validation_bce_by_source(
            encoder,
            tokenizer,
            head,
            data.promptshield_validation,
            batch_size=args.microbatch_size,
            max_tokens=max_tokens,
            cache=encoding_cache,
        )
    finally:
        encoder.train(was_training)
        head.train()
    morgott = morgott_summary["bce"]
    morgott_sources = morgott_summary["by_source"]
    promptshield = promptshield_summary["bce"]
    bces = {"morgott": morgott, "promptshield": promptshield}
    by_source = {"morgott": morgott_sources}
    worst = max(morgott_sources.items(), key=lambda item: item[1], default=(None, None))
    return {
        "epoch": epoch,
        "updates": updates,
        "training_loss": training_loss,
        "canonical_rows_seen": canonical_seen,
        **{f"validation_{name}_bce": value for name, value in bces.items()},
        "validation_macro_bce": 0.5 * sum(bces.values()),
        "validation_morgott_source_macro_bce": (
            sum(morgott_sources.values()) / len(morgott_sources)
            if morgott_sources
            else morgott
        ),
        "validation_worst_source": worst[0],
        "validation_morgott_by_source_label": morgott_summary["by_source_label"],
        "validation_morgott_negative_source_label_macro_bce": morgott_summary[
            "negative_source_label_macro_bce"
        ],
        "validation_morgott_positive_source_label_macro_bce": morgott_summary[
            "positive_source_label_macro_bce"
        ],
        "validation_worst_source_bce": worst[1],
        "validation_morgott_by_source": morgott_sources,
        "validation_checkpoint_operating_point": morgott_summary["operating_point"],
        "selection_rule": args.selection_rule,
        "selection_loss": _selection_loss(bces, by_source, args.selection_rule),
        "pre_registered_comparison": bool(
            getattr(args, "comparison_update", 0) and updates == args.comparison_update
        ),
    }


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


def _selected_checkpoint_provenance(
    curve: list[dict],
    *,
    selected_epoch: int,
    selected_updates: int,
    selection_rule: str,
) -> dict:
    """Bind packaged weights to one exact validation point.

    Epoch alone is not an identity once interim validation is enabled: Arm 6
    has dozens of epoch-3 checkpoints, including the selected update 17,000
    and the source-macro alternate at update 23,000.  Publication therefore
    fails closed unless the selected state maps to one unique curve row.  The
    selection role and the pre-registered-comparison role are intentionally
    separate because one checkpoint may have both (as Arm 6 did by chance).
    """
    if (
        type(selected_epoch) is not int
        or selected_epoch < 1
        or type(selected_updates) is not int
        or selected_updates < 1
    ):
        raise ValueError("selected checkpoint epoch and updates must be positive")
    matches = [
        row
        for row in curve
        if row.get("epoch") == selected_epoch and row.get("updates") == selected_updates
    ]
    if len(matches) != 1:
        raise ValueError("selected checkpoint must identify one validation point")
    row = matches[0]
    loss = row.get("selection_loss")
    if (
        row.get("selection_rule") != selection_rule
        or not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
        or type(row.get("pre_registered_comparison")) is not bool
        or ("interim" in row and type(row["interim"]) is not bool)
    ):
        raise ValueError("selected checkpoint validation contract failed")
    return {
        "epoch": selected_epoch,
        "updates": selected_updates,
        "selection_role": "secondary",
        "selection_rule": selection_rule,
        "selection_loss": float(loss),
        "validation_point_role": (
            "periodic_validation" if row.get("interim", False) else "epoch_final"
        ),
        "pre_registered_comparison": row["pre_registered_comparison"],
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
    alternates: dict,
    selected_epoch: int,
    selected_updates: int,
    args: argparse.Namespace,
    data: TrainingData,
    seconds: float,
    run_name: str | None = None,
    training_identity: dict | None = None,
    optimizer_fused: bool | None = None,
) -> Path:
    import torch
    from safetensors.torch import save_file

    name = run_name or _resolved_run_name(args)
    comparison_update = getattr(args, "comparison_update", 0)
    max_tokens = _runtime_max_tokens(args)
    current_identity = _training_identity(
        args,
        data,
        run_name=name,
        optimizer_fused=optimizer_fused,
    )
    if training_identity is not None and training_identity != current_identity:
        raise ValueError("training source or identity changed before publication")
    training_identity = current_identity
    selected_checkpoint = _selected_checkpoint_provenance(
        curve,
        selected_epoch=selected_epoch,
        selected_updates=selected_updates,
        selection_rule=args.selection_rule,
    )
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
            "run_name": name,
            "training_identity": training_identity,
            "generic_target": "instruction_subversion",
            "head_contract": _head_contract(),
            "positive_classes": list(INSTRUCTION_SUBVERSION_TAGS),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "serving_attention_implementation": ATTENTION_IMPLEMENTATION,
            "training_attention_implementation": args.attention,
            "normalization": "strict",
            "max_tokens": max_tokens,
            "token_budget": args.microbatch_size * max_tokens,
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
                "max_tokens": max_tokens,
                "token_budget": args.microbatch_size * max_tokens,
                "mixed_precision": "bfloat16",
                "gradient_checkpointing": (
                    mode == "lora" and not args.no_gradient_checkpointing
                ),
                "head_learning_rate": args.head_learning_rate,
                "adapter_learning_rate": (
                    args.adapter_learning_rate if mode == "lora" else None
                ),
                "optimizer": {
                    "name": "AdamW",
                    "betas": list(ADAMW_BETAS),
                    "eps": ADAMW_EPS,
                    "weight_decay": ADAMW_WEIGHT_DECAY,
                    "fused": optimizer_fused,
                    "gradient_clip_norm": GRADIENT_CLIP_NORM,
                },
                "shuffle_buffer": args.shuffle_buffer,
                "warmup_fraction": args.warmup_fraction,
                "warmup_updates": (
                    max(
                        int(
                            math.ceil(report["canonical_rows"] / args.batch_size)
                            * args.epochs
                            * args.warmup_fraction
                        ),
                        1,
                    )
                    if args.warmup_fraction > 0
                    else 0
                ),
                "length_grouped": bool(args.length_grouped),
                "length_group_factor": args.length_group_factor,
                "within_batch_order": "raw_character_length_ascending",
                "updates": math.ceil(report["canonical_rows"] / args.batch_size)
                * args.epochs,
                "selected_epoch": selected_epoch,
                "selected_updates": selected_updates,
                "selected_checkpoint": selected_checkpoint,
                "comparison_protocol": {
                    "pre_registered_update": comparison_update or None,
                    "pre_registered_snapshot": bool(comparison_update),
                    "complete_all_updates": True,
                    "early_stopping": False,
                    "selection_rule_role": "secondary",
                    "epoch_final_role": "descriptive",
                },
                "checkpoint_selection": (
                    "minimum equal-domain mean of Morgott and PromptShield "
                    "validation BCE"
                ),
                "curve": curve,
            },
            "arm": _arm(args),
            "execution": {
                "run_name": name,
                "baseline": BASELINE_EXECUTION,
                "seed": seed,
                "microbatch_size": args.microbatch_size,
                "max_tokens": max_tokens,
                "token_budget": args.microbatch_size * max_tokens,
                "attention_implementation": args.attention,
                "pad_to_multiple_of": args.pad_to_multiple_of,
                "compiled": bool(args.compile_encoder),
                "compiled_backward_autocast": ("off" if args.compile_encoder else None),
                "encoding_cache": not args.no_encoding_cache,
                "checkpoint_interval": args.checkpoint_interval,
                "comparison_update": comparison_update,
                "snapshot_every": args.snapshot_every,
                "validation_partition_seed": seed + 1,
                "deviations": {
                    key: value
                    for key, value in (
                        ("seed", seed),
                        ("microbatch_size", args.microbatch_size),
                        ("max_tokens", max_tokens),
                    )
                    if (MAX_TOKENS if key == "max_tokens" else BASELINE_EXECUTION[key])
                    != value
                },
                "gradient_accumulation": (
                    "exact: the classification loss is normalised by the whole "
                    "optimiser batch and the pair loss by len(batch)/len(pairs), "
                    "so the summed gradient does not depend on the microbatch "
                    "partition"
                ),
                "microbatch_caveat": (
                    "not bitwise-neutral: dropout draw shapes, per-microbatch "
                    "padding, and accumulation order all depend on the "
                    "partition, so runs at different microbatch sizes are "
                    "statistically equivalent but not reproducible against "
                    "each other"
                ),
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
                "weights_provenance": {
                    "source": "training.selected_checkpoint",
                    "epoch": selected_epoch,
                    "updates": selected_updates,
                },
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
                "additional_pair_archive_sha256": (
                    file_sha256(args.additional_pairs)
                    if args.additional_pairs is not None
                    else None
                ),
                **source_provenance(*_training_source_paths()),
            },
            "limitations": [
                "This is development evidence, not a production calibration.",
                "PromptShield and SEP are already-open development benchmarks.",
                f"Inputs are truncated to the first {max_tokens} normalized tokens.",
                "The score is advisory and is not approved for blocking.",
            ],
        }
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Alternates are part of the completed-run package, not a follow-up
        # publication. A failure anywhere in this write leaves only the hidden
        # temporary directory, which the finally block removes; the single
        # rename below is the completed-run marker for every artifact.
        _save_alternates(temporary, alternates, mode=mode)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def train(args: argparse.Namespace, data: TrainingData) -> Path:
    import torch

    run_name = _resolved_run_name(args)
    max_tokens = _runtime_max_tokens(args)
    train_path, train_spec = data.views["train"]
    canonical_stream = _EpochStream(
        lambda: training_rows(
            canonical_rows(train_path, train_spec, split="train"),
            data.canonical_counts,
            data.canonical_group_counts,
            data.canonical_owners,
        ),
        expected_rows=sum(data.canonical_counts.values()),
    )
    torch.manual_seed(args.seed)
    encoder, tokenizer = (
        load_base_model()
        if args.attention == ATTENTION_IMPLEMENTATION
        else _load_base_model_with_attention(
            FA2_KERNEL if args.attention == "fa2" else args.attention
        )
    )
    encoding_cache = (
        None
        if args.no_encoding_cache
        else _EncodingCache(tokenizer, max_tokens=max_tokens)
    )
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
    # Ada has TF32 tensor cores; the head and adapters are the only fp32
    # tensors, and their matmuls do not need full fp32 mantissa.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Fused AdamW folds the per-parameter elementwise work into one kernel.
    try:
        optimizer = torch.optim.AdamW(
            parameters,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            weight_decay=ADAMW_WEIGHT_DECAY,
            fused=True,
        )
    except (RuntimeError, ValueError):
        optimizer = torch.optim.AdamW(
            parameters,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            weight_decay=ADAMW_WEIGHT_DECAY,
        )
    optimizer_fused = bool(optimizer.defaults.get("fused", False))
    tracker = _RunTracker(
        args,
        run_name=run_name,
        config={
            "run_name": run_name,
            "arm": _arm(args),
            "recipe": PINNED_RECIPE,
            "head_contract": _head_contract(),
            "seed": args.seed,
            "microbatch_size": args.microbatch_size,
            "max_tokens": max_tokens,
            "token_budget": args.microbatch_size * max_tokens,
            "attention": args.attention,
            "compiled": bool(args.compile_encoder),
            "compiled_backward_autocast": ("off" if args.compile_encoder else None),
            "pad_to_multiple_of": args.pad_to_multiple_of,
            "selection_rule": args.selection_rule,
            "validation_interval": args.validation_interval,
            "snapshot_every": args.snapshot_every,
            "comparison_update": args.comparison_update,
            "warmup_fraction": args.warmup_fraction,
            "length_grouped": bool(args.length_grouped),
            "length_group_factor": args.length_group_factor,
            "data_manifest_sha256": data.data_manifest_sha256,
            "pair_archive_sha256": file_sha256(args.pairs),
            "additional_pair_archive_sha256": (
                file_sha256(args.additional_pairs)
                if args.additional_pairs is not None
                else None
            ),
            "optimizer": {
                "name": "AdamW",
                "betas": list(ADAMW_BETAS),
                "eps": ADAMW_EPS,
                "weight_decay": ADAMW_WEIGHT_DECAY,
                "fused": optimizer_fused,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
            },
        },
    )
    tracker.describe_metrics()
    if args.compile_encoder:
        # Measured +49.7% at microbatch 24. Left on torch's automatic dynamic
        # detection rather than forcing `dynamic=True`: length-sorted padding
        # produces many distinct shapes, and letting inductor specialise first
        # is what the throughput sweep actually measured.
        # In-place `.compile()`, not `torch.compile(m)`: the wrapper form
        # prefixes every state_dict key with `_orig_mod.`, which would corrupt
        # the saved head and adapter and break peft state extraction.
        _configure_compiled_backward_autocast()
        encoder.compile()
        head.compile()
    if encoding_cache is not None:
        encoding_cache.warm(
            chain(
                (row["text"] for row in data.promptshield),
                (row["text"] for pair in data.pairs for row in pair),
                (row["text"] for row in data.checkpoint),
                (row["text"] for row in data.promptshield_validation),
            ),
            workers=args.prewarm_workers,
        )
        print(f"encoding cache warmed: {len(encoding_cache)} texts", flush=True)
    promptshield_cycle = BalancedIndexCycle(
        np.asarray([row["label"] for row in data.promptshield], dtype=np.int64),
        seed=args.seed + 10_001,
    )
    pair_cycle = PairIndexCycle(len(data.pairs), seed=args.seed + 20_003)
    if args.length_grouped:
        promptshield_cycle = _LengthGroupedCycle(
            promptshield_cycle,
            data.promptshield,
            key=lambda row: len(row["text"]),
            group=lambda row: row["label"],
            factor=args.length_group_factor,
            seed=args.seed + 30_011,
        )
        pair_cycle = _LengthGroupedCycle(
            pair_cycle,
            data.pairs,
            # A pair is padded to its longer side, so that is what it costs.
            key=lambda pair: max(len(pair[0]["text"]), len(pair[1]["text"])),
            factor=args.length_group_factor,
            seed=args.seed + 40_013,
        )
    best = None
    alternates: dict[str, dict] = {}
    curve = []
    updates = 0
    promptshield_draws = 0
    pair_draws = 0
    next_epoch = 0
    prior_seconds = 0.0
    started = time.perf_counter()
    updates_per_epoch = math.ceil(sum(data.canonical_counts.values()) / args.batch_size)
    expected_updates = updates_per_epoch * args.epochs
    warmup_updates = (
        max(int(expected_updates * args.warmup_fraction), 1)
        if args.warmup_fraction > 0
        else 0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            _lr_multiplier(
                step,
                total_updates=expected_updates,
                warmup_updates=warmup_updates,
            )
            if args.warmup_fraction > 0
            else 1.0
        ),
    )
    checkpoint_name = run_name
    checkpoint = args.output / f".{checkpoint_name}.checkpoint.pt"
    identity = _training_identity(
        args,
        data,
        run_name=run_name,
        optimizer_fused=optimizer_fused,
    )

    resume_epoch_updates = 0
    resume_epoch_loss_sum = 0.0
    resume_epoch_loss_count = 0
    resume_epoch_canonical_seen = 0
    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        state = _load_checkpoint(checkpoint, identity=identity)
        next_epoch = state["next_epoch"]
        resume_epoch_updates = state["epoch_updates"]
        resume_epoch_loss_sum = state["epoch_loss_sum"]
        resume_epoch_loss_count = state["epoch_loss_count"]
        resume_epoch_canonical_seen = state["epoch_canonical_seen"]
        updates = state["updates"]
        promptshield_draws = state["promptshield_draws"]
        pair_draws = state["pair_draws"]
        prior_seconds = state["runtime_seconds"]
        curve = state["curve"]
        best = state["best"]
        alternates = state.get("alternates") or {}
        head.load_state_dict(state["head"], strict=True)
        if train_encoder:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(encoder, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        promptshield_cycle.load_state_dict(state["promptshield_cycle"])
        pair_cycle.load_state_dict(state["pair_cycle"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        if not _resume_progress_valid(
            next_epoch=next_epoch,
            epoch_updates=resume_epoch_updates,
            epoch_canonical_seen=resume_epoch_canonical_seen,
            epoch_loss_sum=resume_epoch_loss_sum,
            epoch_loss_count=resume_epoch_loss_count,
            updates=updates,
            curve=curve,
            best=best,
            epochs=args.epochs,
            updates_per_epoch=updates_per_epoch,
        ):
            raise ValueError("resume checkpoint progress contract failed")
        print(
            f"resuming at epoch {next_epoch + 1}/{args.epochs}, "
            f"update {updates}/{expected_updates}, "
            f"epoch batch {resume_epoch_updates}/{updates_per_epoch}",
            flush=True,
        )
    elif checkpoint.exists():
        raise FileExistsError(
            f"checkpoint already exists; pass --resume to use it: {checkpoint}"
        )

    snapshots = (
        _snapshot_dir(args.output, checkpoint_name)
        if args.snapshot_every or args.comparison_update
        else None
    )

    for epoch_index in range(next_epoch, args.epochs):
        epoch = epoch_index + 1
        encoder.train(train_encoder)
        head.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        metric_window = _MetricWindow()
        canonical_seen = 0
        if canonical_stream.cached and file_sha256(train_path) != train_spec["sha256"]:
            # The cache replaces the per-epoch reread, so re-assert the digest
            # that reread used to verify.
            raise ValueError("canonical train view changed during training")
        stream = shuffled(
            canonical_stream,
            seed=args.seed + epoch_index,
            buffer_size=args.shuffle_buffer,
        )
        batch_iterator = (
            length_grouped_batches(
                stream,
                args.batch_size,
                key=lambda row: len(row["text"]),
                factor=args.length_group_factor,
                # Per epoch, so a resumed epoch replays the same batch order
                # that `_skip_resumed_batches` is about to walk past.
                rng=np.random.default_rng(args.seed + 50_021 + epoch_index),
            )
            if args.length_grouped
            else batches(stream, args.batch_size)
        )
        if epoch_index == next_epoch and resume_epoch_updates:
            canonical_seen = _skip_resumed_batches(
                batch_iterator,
                batches_consumed=resume_epoch_updates,
                canonical_seen=resume_epoch_canonical_seen,
            )
            epoch_loss_sum = float(resume_epoch_loss_sum)
            epoch_loss_count = resume_epoch_loss_count
        # Exclude stream construction and resume replay from optimizer
        # throughput.  The timer is reset again after each validation and
        # checkpoint so those I/O-heavy pauses do not depress the next window.
        metric_window_started = time.perf_counter()
        for morgott in batch_iterator:
            morgott.sort(key=lambda row: len(row["text"]))
            canonical_seen += len(morgott)
            optimizer.zero_grad(set_to_none=True)
            canonical_primary = _classification_backward(
                encoder,
                tokenizer,
                head,
                morgott,
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
                cache=encoding_cache,
                pad_to_multiple_of=args.pad_to_multiple_of,
                max_tokens=max_tokens,
            )
            promptshield_indices = promptshield_cycle.take(args.batch_size // 2)
            promptshield_draws += len(promptshield_indices)
            promptshield_batch = sorted(
                [data.promptshield[int(index)] for index in promptshield_indices],
                key=lambda row: len(row["text"]),
            )
            promptshield_loss = _classification_backward(
                encoder,
                tokenizer,
                head,
                promptshield_batch,
                coefficient=DOMAIN_WEIGHT,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
                cache=encoding_cache,
                pad_to_multiple_of=args.pad_to_multiple_of,
                max_tokens=max_tokens,
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
            pair_loss = _pair_backward(
                encoder,
                tokenizer,
                head,
                pair_batch,
                ranking_weight=args.pair_ranking_weight,
                microbatch_size=args.microbatch_size,
                train_encoder=train_encoder,
                cache=encoding_cache,
                pad_to_multiple_of=args.pad_to_multiple_of,
                max_tokens=max_tokens,
            )
            primary_loss = canonical_primary + promptshield_loss + pair_loss
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for group in parameters for parameter in group["params"]],
                GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
            gradient_clipped = (gradient_norm > GRADIENT_CLIP_NORM).to(
                dtype=torch.float32
            )
            optimizer.step()
            scheduler.step()
            updates += 1
            metric_window.add(
                {
                    "primary_loss": primary_loss,
                    "canonical_primary_loss": canonical_primary,
                    "promptshield_loss": promptshield_loss,
                    "pair_loss": pair_loss,
                    "pre_clip_gradient_norm": gradient_norm,
                    "gradient_clipped": gradient_clipped,
                },
                examples=(len(morgott) + len(promptshield_batch) + 2 * len(pair_batch)),
            )
            epoch_boundary = updates % updates_per_epoch == 0
            validation_due = bool(
                args.validation_interval
                and updates % args.validation_interval == 0
                and not epoch_boundary
            )
            checkpoint_due = bool(
                updates % args.checkpoint_interval == 0 and not epoch_boundary
            )
            log_due = bool(
                updates % 25 == 0 or epoch_boundary or updates == expected_updates
            )
            window_metrics = None
            if log_due or validation_due or checkpoint_due:
                totals, latest, window_updates, window_examples = metric_window.drain()
                window_seconds = max(
                    time.perf_counter() - metric_window_started,
                    np.finfo(np.float64).eps,
                )
                # Preserve the historical three-part training-loss curve.
                epoch_loss_sum += totals["primary_loss"]
                epoch_loss_count += window_updates
                window_metrics = {
                    key: value / window_updates for key, value in totals.items()
                }
                if log_due:
                    tracker.log(
                        _training_trackio_metrics(
                            latest,
                            window_metrics,
                            peak_vram_gib=torch.cuda.max_memory_reserved() / (1 << 30),
                            head_lr=optimizer.param_groups[0]["lr"],
                            adapter_lr=(
                                optimizer.param_groups[1]["lr"]
                                if len(optimizer.param_groups) > 1
                                else None
                            ),
                            optimizer_updates_per_second=(
                                window_updates / window_seconds
                            ),
                            examples_per_second=window_examples / window_seconds,
                        ),
                        step=updates,
                    )
            if updates % 100 == 0 or updates == expected_updates:
                elapsed = prior_seconds + time.perf_counter() - started
                eta = elapsed / updates * (expected_updates - updates)
                print(
                    f"epoch {epoch}/{args.epochs} update "
                    f"{updates}/{expected_updates} "
                    f"loss={window_metrics['primary_loss']:.5f} "
                    f"elapsed={elapsed / 3600:.2f}h "
                    f"eta={eta / 3600:.2f}h "
                    f"peak_vram={torch.cuda.max_memory_reserved() / (1 << 30):.2f}GiB",
                    flush=True,
                )
            if validation_due:
                interim = _validation_row(
                    encoder,
                    tokenizer,
                    head,
                    data,
                    args,
                    epoch=epoch,
                    updates=updates,
                    training_loss=epoch_loss_sum / epoch_loss_count,
                    canonical_seen=canonical_seen,
                    encoding_cache=encoding_cache,
                )
                interim["interim"] = True
                curve.append(interim)
                tracker.log_validation(interim, step=updates)
                tracker.log_finance_false_flag_bces(
                    interim["validation_morgott_by_source_label"],
                    step=updates,
                )
                print(
                    f"epoch {epoch} update {updates}/{expected_updates} "
                    f"selection_loss={interim['selection_loss']:.6f} "
                    f"worst_source={interim['validation_worst_source']}",
                    flush=True,
                )
                _update_alternates(
                    alternates,
                    interim,
                    mode=args.mode,
                    head=head,
                    encoder=encoder,
                    epoch=epoch,
                    updates=updates,
                )
                if snapshots is not None and _should_snapshot(args, updates):
                    _save_snapshot(
                        snapshots,
                        interim,
                        training_identity=identity,
                        mode=args.mode,
                        head=head,
                        encoder=encoder,
                        epoch=epoch,
                        updates=updates,
                    )
                    _write_snapshot_index(snapshots, curve)
                if best is None or interim["selection_loss"] < best["loss"]:
                    best = _candidate_weights(
                        mode=args.mode,
                        head=head,
                        encoder=encoder,
                        epoch=epoch,
                        updates=updates,
                        loss=interim["selection_loss"],
                    )
            if checkpoint_due:
                _save_checkpoint(
                    checkpoint,
                    identity=identity,
                    state=_progress_state(
                        mode=args.mode,
                        completed_epochs=epoch_index,
                        epoch_updates=updates - epoch_index * updates_per_epoch,
                        epoch_loss_sum=epoch_loss_sum,
                        epoch_loss_count=epoch_loss_count,
                        epoch_canonical_seen=canonical_seen,
                        updates=updates,
                        promptshield_draws=promptshield_draws,
                        pair_draws=pair_draws,
                        runtime_seconds=(prior_seconds + time.perf_counter() - started),
                        curve=curve,
                        best=best,
                        alternates=alternates,
                        head=head,
                        encoder=encoder,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        promptshield_cycle=promptshield_cycle,
                        pair_cycle=pair_cycle,
                    ),
                )
                print(
                    f"saved progress checkpoint at update "
                    f"{updates}/{expected_updates}: {checkpoint}",
                    flush=True,
                )
            if window_metrics is not None:
                metric_window_started = time.perf_counter()

        if canonical_seen != sum(data.canonical_counts.values()):
            raise ValueError(
                f"epoch {epoch} saw {canonical_seen} canonical rows, "
                f"expected {sum(data.canonical_counts.values())}"
            )
        if metric_window.updates or epoch_loss_count != updates_per_epoch:
            raise ValueError("training loss accumulation contract failed")

        row = _validation_row(
            encoder,
            tokenizer,
            head,
            data,
            args,
            epoch=epoch,
            updates=updates,
            training_loss=epoch_loss_sum / epoch_loss_count,
            canonical_seen=canonical_seen,
            encoding_cache=encoding_cache,
        )
        curve.append(row)
        tracker.log_validation(row, step=updates)
        tracker.log_finance_false_flag_bces(
            row["validation_morgott_by_source_label"], step=updates
        )
        _update_alternates(
            alternates,
            row,
            mode=args.mode,
            head=head,
            encoder=encoder,
            epoch=epoch,
            updates=updates,
        )
        # Epoch boundaries snapshot unconditionally: they are the points the
        # constant-LR baseline actually peaked at, and there are only three.
        if snapshots is not None:
            _save_snapshot(
                snapshots,
                row,
                training_identity=identity,
                mode=args.mode,
                head=head,
                encoder=encoder,
                epoch=epoch,
                updates=updates,
            )
            _write_snapshot_index(snapshots, curve)
        if best is None or row["selection_loss"] < best["loss"]:
            best = _candidate_weights(
                mode=args.mode,
                head=head,
                encoder=encoder,
                epoch=epoch,
                updates=updates,
                loss=row["selection_loss"],
            )
        elapsed = prior_seconds + time.perf_counter() - started
        _save_checkpoint(
            checkpoint,
            identity=identity,
            state=_progress_state(
                mode=args.mode,
                completed_epochs=epoch,
                epoch_updates=0,
                epoch_loss_sum=0.0,
                epoch_loss_count=0,
                epoch_canonical_seen=0,
                updates=updates,
                promptshield_draws=promptshield_draws,
                pair_draws=pair_draws,
                runtime_seconds=elapsed,
                curve=curve,
                best=best,
                alternates=alternates,
                head=head,
                encoder=encoder,
                optimizer=optimizer,
                scheduler=scheduler,
                promptshield_cycle=promptshield_cycle,
                pair_cycle=pair_cycle,
            ),
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
    tracking_summary = {
        "selected_epoch": best["epoch"],
        "selected_updates": best.get("updates", 0),
        "selection_loss": best["loss"],
        "selection_rule": args.selection_rule,
        **{
            f"alternate_{name}_updates": candidate.get("updates", 0)
            for name, candidate in alternates.items()
        },
        **{
            f"alternate_{name}_loss": candidate["loss"]
            for name, candidate in alternates.items()
        },
    }
    head.load_state_dict(best["head"], strict=True)
    if train_encoder:
        from peft import (
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )

        set_peft_model_state_dict(encoder, best["adapter"])
        restored = get_peft_model_state_dict(encoder)
        selected = best["adapter"]
        if restored.keys() != selected.keys() or any(
            not torch.equal(restored[name].cpu(), selected[name]) for name in restored
        ):
            raise ValueError("restored encoder differs from the selected epoch")
    destination = _save_run(
        args.output,
        mode=args.mode,
        seed=args.seed,
        alternates=alternates,
        head=head,
        encoder=encoder,
        report=_report(data),
        curve=curve,
        selected_epoch=best["epoch"],
        selected_updates=best["updates"],
        args=args,
        data=data,
        seconds=prior_seconds + time.perf_counter() - started,
        run_name=run_name,
        training_identity=identity,
        optimizer_fused=optimizer_fused,
    )
    tracker.finish(tracking_summary)
    return destination


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
    parser.add_argument("--additional-pairs", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mmbert/runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument(
        "--max-tokens",
        type=int,
        choices=SUPPORTED_MAX_TOKENS,
        default=MAX_TOKENS,
        help="normalized-token context cap; 512 preserves the historical baseline",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument(
        "--prep-cache",
        type=Path,
        default=Path("artifacts/mmbert/prep-cache"),
        help="reuse a digest-keyed prepared corpus instead of rebuilding it; "
        "the key covers every input digest and the preparing source files, so a "
        "changed corpus or a changed guard misses rather than serves stale data",
    )
    parser.add_argument(
        "--no-prep-cache",
        dest="prep_cache",
        action="store_const",
        const=None,
        help="always rebuild the prepared corpus",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.05,
        help="share of total updates spent in linear LR warmup; 0 disables both "
        "warmup and cosine decay and restores a constant rate",
    )
    parser.add_argument(
        "--no-length-grouped",
        dest="length_grouped",
        action="store_false",
        help="draw batches at random instead of grouping similar lengths; "
        "costs ~27%% more processed tokens per update",
    )
    parser.add_argument(
        "--length-group-factor",
        type=int,
        default=32,
        help="megabatch size as a multiple of the batch size; larger groups "
        "pad less but correlate batch content more strongly with length",
    )
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--pair-ranking-weight", type=float, default=0.25)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "fa2", "eager"),
        default=ATTENTION_IMPLEMENTATION,
        help="training-only attention kernel; maintained inference stays pinned",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        help=(
            "bucket padded length to a multiple; collapses the distinct-shape "
            "space for torch.compile at the cost of some masked positions"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_encoder",
        help="torch.compile the encoder; pair with --pad-to-multiple-of",
    )
    parser.add_argument(
        "--trackio",
        action="store_true",
        help="log scalars and config to a local Trackio store; never logs corpus text",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "single identity used for the output directory, progress checkpoint, "
            "snapshots, result metadata, and Trackio label; defaults to a derived "
            "arm name"
        ),
    )
    parser.add_argument(
        "--trackio-group",
        default="",
        help="group related arms, e.g. the capacity ladder",
    )
    parser.add_argument(
        "--trackio-space",
        default="",
        help="optional Hugging Face Space to mirror to; local-only when unset",
    )
    parser.add_argument(
        "--prewarm-workers",
        type=int,
        default=max(1, _usable_cpus() - 1),
        help="processes used to normalise the encoding cache before training",
    )
    parser.add_argument(
        "--selection-rule",
        choices=CHECKPOINT_SELECTION_RULES,
        default="micro",
        help=(
            "checkpoint selection metric; 'micro' reproduces the registered "
            "rule, 'source_macro' stops a large source masking small-source "
            "regressions"
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=0,
        help=(
            "validate every N updates as well as at epoch boundaries; 0 keeps "
            "the registered epoch-only cadence"
        ),
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help=(
            "also retain every validation point's weights every N updates, "
            "beside the run as .<name>.snapshots/; 0 keeps only the three "
            "rule-selected candidates"
        ),
    )
    parser.add_argument(
        "--comparison-update",
        type=int,
        default=0,
        help=(
            "pre-registered validation update whose weights are retained "
            "unconditionally; 0 disables the fixed comparison point"
        ),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=CHECKPOINT_UPDATE_INTERVAL,
        help="updates between atomic progress checkpoints for --resume",
    )
    parser.add_argument(
        "--no-encoding-cache",
        action="store_true",
        help="disable the normalisation/tokenisation cache (slower; for A/B)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_name = _resolved_run_name(args)
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
    if not math.isfinite(args.warmup_fraction) or not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup fraction must be finite and in [0, 1)")
    if (
        args.length_group_factor < 1
        or args.checkpoint_interval < 1
        or args.prewarm_workers < 0
        or args.validation_interval < 0
        or args.comparison_update < 0
        or (args.pad_to_multiple_of is not None and args.pad_to_multiple_of < 1)
    ):
        raise ValueError("training intervals and grouping sizes are invalid")
    if args.snapshot_every < 0:
        raise ValueError("snapshot interval must be non-negative")
    # A snapshot only exists where a validation row does, so an interval that
    # is not a multiple of the validation cadence would silently never fire.
    if args.snapshot_every and (
        not args.validation_interval or args.snapshot_every % args.validation_interval
    ):
        raise ValueError(
            "--snapshot-every must be a positive multiple of --validation-interval"
        )
    if args.comparison_update and (
        not args.validation_interval
        or args.comparison_update % args.validation_interval
    ):
        raise ValueError(
            "--comparison-update must be a positive multiple of --validation-interval"
        )
    if not args.preflight_only:
        _preflight_execution(args, run_name)
    data = prepare_training_data(
        args.data_dir,
        args.external_dir,
        args.pairs,
        seed=args.seed,
        additional_pair_archive=args.additional_pairs,
        cache_dir=args.prep_cache,
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
