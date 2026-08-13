#!/usr/bin/env python3
"""Score one retained mmBERT snapshot on the frozen SWE-rebench V2 pairs.

The runner transports the canonical-calibration threshold from the completed
full-panel evaluation of the exact same snapshot and evaluation context cap.
It writes aggregate metrics and numeric score journals only; prompt text and
row or repository identities remain in the ignored input archives.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.mmbert_evaluation_contract import (
    canonical_sha256 as _canonical_sha256,
)
from experiments.mmbert_evaluation_contract import (
    read_json_object,
    score_artifact,
    strict_json_loads,
    transported_threshold,
)
from morgott.models.mmbert import evaluate as mmbert_evaluate
from morgott.models.mmbert.core import file_sha256, pool, source_provenance
from morgott.models.mmbert.head_contract import resolve_head_contract
from morgott.models.mmbert.score_journal import (
    ScoreJournal,
    ScoreJournalSpec,
    require_disjoint_paths,
)
from morgott.models.mmbert.serving import WINDOW_OVERLAP
from morgott.normalization import strict_normalize

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PURPOSE = "fixed-checkpoint repository-held-out SWE-rebench V2 comparison"
DESCRIPTIVE_FPR_BUDGET = "1.0000%"
EXPECTED_PAIR_MANIFEST_SHA256 = (
    "1f3a7dfdf9a35c4910f2762cc8b971dc54cdf1af1036b754a2753b4d2f0fb45b"
)
EXPECTED_SOURCE_REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
PAIR_SPLITS = ("validation", "dev_test")
LENGTH_EDGES = (2_048, 4_096, 8_192)
HIGH_GATE = 0.99999
JOURNAL_SHARD_ROWS = 64
MAX_JSON_BYTES = 64 << 20


@dataclass(frozen=True)
class PairSplit:
    """One verified pair archive and its rows, retained only in memory."""

    name: str
    path: Path
    sha256: str
    content_sha256: str
    expected_pairs: int
    expected_repositories: int
    rows: tuple[dict, ...]


@dataclass(frozen=True)
class EvaluationBinding:
    """Text-free binding between a run, snapshot, and full-panel evaluation."""

    run_result: dict
    full_evaluation: dict
    run_result_sha256: str
    snapshot_sha256: str
    snapshot_update: int
    snapshot_epoch: int
    snapshot_role: str
    full_evaluation_sha256: str
    full_score_path: Path
    full_score_sha256: str
    training_max_tokens: int
    evaluation_max_tokens: int
    evaluation_model_sha256: str
    full_scoring_sha256: str
    evaluation_identity_sha256: str
    threshold: float


def _read_json_object(path: Path) -> tuple[dict, str]:
    return read_json_object(path, max_bytes=MAX_JSON_BYTES)


def _gzip_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_rows(path: Path) -> tuple[dict, ...]:
    rows = []
    instance_ids = set()
    required = {
        "attack",
        "attack_span",
        "attack_span_start",
        "benign",
        "channel",
        "instance_id",
        "repository",
        "source",
        "source_revision",
    }
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = strict_json_loads(line)
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError(f"invalid pair-row schema in {path}")
            if (
                not isinstance(row["instance_id"], str)
                or not row["instance_id"]
                or row["instance_id"] in instance_ids
                or not isinstance(row["repository"], str)
                or not row["repository"]
                or not isinstance(row["benign"], str)
                or not row["benign"].strip()
                or not isinstance(row["attack"], str)
                or not row["attack"].strip()
            ):
                raise ValueError(f"invalid or duplicate pair row in {path}")
            instance_ids.add(row["instance_id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"pair archive is empty: {path}")
    return tuple(rows)


def _load_pair_inputs(root: Path) -> tuple[dict, str, dict[str, PairSplit]]:
    root = root.resolve()
    manifest, manifest_sha256 = _read_json_object(root / "manifest.json")
    if (
        manifest_sha256 != EXPECTED_PAIR_MANIFEST_SHA256
        or manifest.get("schema_version") != 1
        or manifest.get("split_unit") != "repository"
        or not isinstance(manifest.get("source"), dict)
        or manifest["source"].get("revision") != EXPECTED_SOURCE_REVISION
        or not isinstance(manifest.get("outputs"), dict)
    ):
        raise ValueError("SWE-rebench V2 pair manifest contract changed")

    loaded = {}
    repositories = {}
    for split in PAIR_SPLITS:
        spec = manifest["outputs"].get(split)
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("path"), str)
            or Path(spec["path"]).is_absolute()
            or not isinstance(spec.get("sha256"), str)
            or not isinstance(spec.get("content_sha256"), str)
            or type(spec.get("pairs")) is not int
            or spec["pairs"] < 1
            or type(spec.get("repositories")) is not int
            or spec["repositories"] < 1
        ):
            raise ValueError(f"invalid pair manifest split: {split}")
        path = (root / split / spec["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"pair archive is missing or escapes its root: {split}")
        archive_sha256 = file_sha256(path)
        content_sha256 = _gzip_content_sha256(path)
        if archive_sha256 != spec["sha256"] or content_sha256 != spec["content_sha256"]:
            raise ValueError(f"pair archive digest mismatch: {split}")
        rows = _pair_rows(path)
        split_repositories = {row["repository"] for row in rows}
        if (
            len(rows) != spec["pairs"]
            or len(split_repositories) != spec["repositories"]
        ):
            raise ValueError(f"pair archive population changed: {split}")
        repositories[split] = split_repositories
        loaded[split] = PairSplit(
            name=split,
            path=path,
            sha256=archive_sha256,
            content_sha256=content_sha256,
            expected_pairs=spec["pairs"],
            expected_repositories=spec["repositories"],
            rows=rows,
        )
    if repositories["validation"] & repositories["dev_test"]:
        raise ValueError("pair repositories cross validation and dev-test")
    return manifest, manifest_sha256, loaded


def _score_artifact(
    full_evaluation_path: Path, full_evaluation: dict
) -> tuple[Path, str]:
    return score_artifact(full_evaluation_path, full_evaluation)


def _transported_threshold(full_evaluation: dict) -> float:
    return transported_threshold(
        full_evaluation,
        target=DESCRIPTIVE_FPR_BUDGET,
        allow_one=False,
        require_canonical_recall=False,
    )


def _bind_full_evaluation(
    run: Path,
    snapshot: Path,
    full_evaluation_path: Path,
    *,
    evaluation_max_tokens: int,
    require_update: int | None = None,
    require_additional_pairs_sha256: str | None = None,
) -> EvaluationBinding:
    run = run.resolve()
    snapshot = snapshot.resolve()
    full_evaluation_path = full_evaluation_path.resolve()
    result, run_result_sha256 = _read_json_object(run / "result.json")
    full_evaluation, full_evaluation_sha256 = _read_json_object(full_evaluation_path)
    training_max_tokens = mmbert_evaluate._training_max_tokens(result)
    if evaluation_max_tokens not in mmbert_evaluate.SUPPORTED_MAX_TOKENS:
        raise ValueError("unsupported evaluation context cap")
    snapshot_update = mmbert_evaluate._snapshot_update_from_path(snapshot)
    if not snapshot.is_file():
        raise FileNotFoundError(f"snapshot does not exist: {snapshot}")
    snapshot_sha256 = file_sha256(snapshot)
    if require_update is not None and snapshot_update != require_update:
        raise ValueError("snapshot does not match the required fixed update")

    checkpoint = full_evaluation.get("evaluated_checkpoint")
    scores = full_evaluation.get("scores")
    inputs = full_evaluation.get("inputs")
    if (
        type(full_evaluation.get("schema_version")) is not int
        or full_evaluation.get("schema_version") not in {1, 2}
        or full_evaluation.get("purpose") != "advisory mmBERT development evaluation"
        or full_evaluation.get("advisory_only") is not True
        or full_evaluation.get("run_result_sha256") != run_result_sha256
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != snapshot_sha256
        or checkpoint.get("update") != snapshot_update
        or type(checkpoint.get("epoch")) is not int
        or checkpoint["epoch"] < 1
        or checkpoint.get("role")
        not in {"pre_registered_comparison", "periodic_validation", "epoch_final"}
        or full_evaluation.get("training_max_tokens") != training_max_tokens
        or full_evaluation.get("evaluation_max_tokens") != evaluation_max_tokens
        or full_evaluation.get("native_context_evaluation")
        is not (training_max_tokens == evaluation_max_tokens)
        or not isinstance(scores, dict)
        or not isinstance(inputs, dict)
    ):
        raise ValueError("full evaluation does not bind the requested snapshot and cap")

    report_schema = full_evaluation["schema_version"]
    base_model = full_evaluation.get("base_model")
    if (report_schema == mmbert_evaluate.EVALUATION_SCHEMA_VERSION) != (
        base_model is not None
    ):
        raise ValueError("full evaluation base-model identity is incomplete")
    expected_model_sha256 = mmbert_evaluate._evaluation_model_sha256(
        run_result_sha256,
        snapshot_sha256,
        base_model=base_model,
    )
    expected_full_scoring_sha256 = scores.get("scoring_sha256")
    if not isinstance(expected_full_scoring_sha256, str) or (
        mmbert_evaluate._SHA256.fullmatch(expected_full_scoring_sha256) is None
    ):
        raise ValueError("full evaluation has no valid scoring identity")
    expected_evaluation_identity = mmbert_evaluate._expected_evaluation_identity_sha256(
        full_evaluation,
        model_sha256=expected_model_sha256,
        scoring_sha256=expected_full_scoring_sha256,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
    )
    if (
        full_evaluation.get("evaluation_model_sha256") != expected_model_sha256
        or scores.get("scoring_sha256") != expected_full_scoring_sha256
        or scores.get("evaluation_identity_sha256") != expected_evaluation_identity
        or full_evaluation.get("evaluation_identity_sha256")
        != expected_evaluation_identity
        or scores.get("training_max_tokens") != training_max_tokens
        or scores.get("evaluation_max_tokens") != evaluation_max_tokens
    ):
        raise ValueError("full evaluation model or scoring identity changed")

    threshold = _transported_threshold(full_evaluation)
    if (
        require_additional_pairs_sha256 is not None
        and inputs.get("additional_pair_archive_sha256")
        != require_additional_pairs_sha256
    ):
        raise ValueError("full evaluation used a different additional-pair archive")

    full_score_path, full_score_sha256 = _score_artifact(
        full_evaluation_path,
        full_evaluation,
    )
    return EvaluationBinding(
        run_result=result,
        full_evaluation=full_evaluation,
        run_result_sha256=run_result_sha256,
        snapshot_sha256=snapshot_sha256,
        snapshot_update=snapshot_update,
        snapshot_epoch=checkpoint["epoch"],
        snapshot_role=checkpoint["role"],
        full_evaluation_sha256=full_evaluation_sha256,
        full_score_path=full_score_path,
        full_score_sha256=full_score_sha256,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
        evaluation_model_sha256=expected_model_sha256,
        full_scoring_sha256=expected_full_scoring_sha256,
        evaluation_identity_sha256=expected_evaluation_identity,
        threshold=threshold,
    )


def _longcode_scoring_sha256(evaluation_max_tokens: int) -> str:
    return _canonical_sha256(
        {
            "aggregation": "maximum primary-head probability over ordered windows",
            "evaluation_max_tokens": evaluation_max_tokens,
            "overlap_tokens": WINDOW_OVERLAP,
            "provenance": source_provenance(
                Path(__file__),
                ROOT / "experiments/mmbert_evaluation_contract.py",
                Path(mmbert_evaluate.__file__),
                Path(mmbert_evaluate.__file__).with_name("core.py"),
                Path(mmbert_evaluate.__file__).with_name("head_contract.py"),
                Path(mmbert_evaluate.__file__).with_name("score_journal.py"),
                Path(mmbert_evaluate.__file__).resolve().parents[2]
                / "normalization.py",
            ),
        }
    )


def _evaluation_identity_sha256(
    *,
    binding: EvaluationBinding,
    journal_model_sha256: str,
    pair_manifest_sha256: str,
    scoring_sha256: str,
    batch_size: int,
) -> str:
    return _canonical_sha256(
        {
            "batch_size": batch_size,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "full_evaluation_identity_sha256": binding.evaluation_identity_sha256,
            "full_evaluation_sha256": binding.full_evaluation_sha256,
            "full_score_sha256": binding.full_score_sha256,
            "full_scoring_sha256": binding.full_scoring_sha256,
            "journal_model_sha256": journal_model_sha256,
            "pair_manifest_sha256": pair_manifest_sha256,
            "scoring_sha256": scoring_sha256,
            "threshold": binding.threshold,
            "training_max_tokens": binding.training_max_tokens,
        }
    )


def _journal_panel_sha256(split: PairSplit, side: str) -> str:
    return _canonical_sha256(
        {
            "archive_sha256": split.sha256,
            "content_sha256": split.content_sha256,
            "rows": split.expected_pairs,
            "side": side,
            "split": split.name,
        }
    )


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _window_max_score(
    encoder,
    tokenizer,
    head,
    text: str,
    *,
    primary_column: int,
    batch_size: int,
    max_tokens: int,
) -> tuple[float, int]:
    import torch

    encoded = tokenizer(
        strict_normalize(text),
        add_special_tokens=True,
        max_length=max_tokens,
        stride=WINDOW_OVERLAP,
        truncation=True,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    windows = int(input_ids.shape[0])
    if windows < 1:
        raise ValueError("tokenizer emitted no long-code windows")
    best = -math.inf
    with torch.no_grad():
        for start in range(0, windows, batch_size):
            batch = {
                "input_ids": input_ids[start : start + batch_size].to("cuda"),
                "attention_mask": attention_mask[start : start + batch_size].to("cuda"),
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encoder(**batch).last_hidden_state
                features = pool(hidden, batch["attention_mask"])
                logits = head(features)
            if (
                logits.ndim != 2
                or logits.shape[0] != len(batch["input_ids"])
                or primary_column >= logits.shape[1]
            ):
                raise ValueError(
                    "head output does not match its primary-column contract"
                )
            best = max(best, float(logits[:, primary_column].float().max()))
    return _stable_sigmoid(best), windows


def _score_side(
    split: PairSplit,
    side: str,
    *,
    encoder,
    tokenizer,
    head,
    primary_column: int,
    batch_size: int,
    max_tokens: int,
    model_sha256: str,
    scoring_sha256: str,
    journal_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    journal = ScoreJournal(
        journal_root / split.name / side,
        ScoreJournalSpec(
            model_sha256=model_sha256,
            panel_sha256=_journal_panel_sha256(split, side),
            scoring_sha256=scoring_sha256,
            rows=len(split.rows),
            batch_size=batch_size,
            columns=("score", "window_count"),
        ),
    )
    for start, stop in journal.missing_ranges(JOURNAL_SHARD_ROWS):
        values = np.empty((stop - start, 2), dtype=np.float64)
        for offset, row in enumerate(split.rows[start:stop]):
            score, windows = _window_max_score(
                encoder,
                tokenizer,
                head,
                row[side],
                primary_column=primary_column,
                batch_size=batch_size,
                max_tokens=max_tokens,
            )
            values[offset] = (score, windows)
        journal.append(values, start=start)
        print(
            f"{split.name}/{side}: scored {stop}/{len(split.rows)}",
            flush=True,
        )
    values = journal.scores()
    return values[:, 0], values[:, 1].astype(np.int64)


def _auroc(benign: np.ndarray, attack: np.ndarray) -> float:
    labels = np.concatenate((np.zeros(len(benign)), np.ones(len(attack))))
    scores = np.concatenate((benign, attack))
    return float(roc_auc_score(labels, scores))


def _slice_metrics(
    benign: np.ndarray,
    attack: np.ndarray,
    threshold: float,
    indexes: list[int],
) -> dict:
    if not indexes:
        return {"pairs": 0}
    clean = benign[indexes]
    attacked = attack[indexes]
    return {
        "pairs": len(indexes),
        "clean_flag_rate": float((clean >= threshold).mean()),
        "attack_recall": float((attacked >= threshold).mean()),
        "both_correct": float(((clean < threshold) & (attacked >= threshold)).mean()),
        "clean_high_gate_rate": float((clean >= HIGH_GATE).mean()),
        "attack_high_gate_rate": float((attacked >= HIGH_GATE).mean()),
        "pair_ordering": float((attacked > clean).mean()),
        "auroc": _auroc(clean, attacked),
    }


def _length_bucket(text: str) -> str:
    length = len(strict_normalize(text))
    for edge in LENGTH_EDGES:
        if length < edge:
            return f"under_{edge}_chars"
    return f"at_least_{LENGTH_EDGES[-1]}_chars"


def _metrics(
    rows: tuple[dict, ...],
    benign: np.ndarray,
    attack: np.ndarray,
    benign_windows: np.ndarray,
    attack_windows: np.ndarray,
    threshold: float,
) -> dict:
    all_indexes = list(range(len(rows)))
    by_length = defaultdict(list)
    by_repository = defaultdict(list)
    for index, row in enumerate(rows):
        by_length[_length_bucket(row["benign"])].append(index)
        by_repository[row["repository"]].append(index)
    repository_metrics = [
        _slice_metrics(benign, attack, threshold, indexes)
        for indexes in by_repository.values()
    ]
    return {
        "overall": _slice_metrics(benign, attack, threshold, all_indexes),
        "by_length": {
            name: _slice_metrics(benign, attack, threshold, indexes)
            for name, indexes in sorted(by_length.items())
        },
        "repository_macro": {
            "repositories": len(repository_metrics),
            "clean_flag_rate": statistics.fmean(
                value["clean_flag_rate"] for value in repository_metrics
            ),
            "attack_recall": statistics.fmean(
                value["attack_recall"] for value in repository_metrics
            ),
            "both_correct": statistics.fmean(
                value["both_correct"] for value in repository_metrics
            ),
            "pair_ordering": statistics.fmean(
                value["pair_ordering"] for value in repository_metrics
            ),
        },
        "operation": {
            "benign_windows": int(benign_windows.sum()),
            "attack_windows": int(attack_windows.sum()),
            "maximum_benign_windows": int(benign_windows.max()),
            "maximum_attack_windows": int(attack_windows.max()),
        },
    }


def _assert_inputs_unchanged(
    *,
    run: Path,
    snapshot: Path,
    full_evaluation_path: Path,
    binding: EvaluationBinding,
    pair_root: Path,
    pair_manifest_sha256: str,
    splits: dict[str, PairSplit],
) -> None:
    if (
        file_sha256(run / "result.json") != binding.run_result_sha256
        or file_sha256(snapshot) != binding.snapshot_sha256
        or file_sha256(full_evaluation_path) != binding.full_evaluation_sha256
        or file_sha256(binding.full_score_path) != binding.full_score_sha256
        or file_sha256(pair_root / "manifest.json") != pair_manifest_sha256
        or any(file_sha256(split.path) != split.sha256 for split in splits.values())
    ):
        raise ValueError(
            "a bound run, evaluation, or pair artifact changed during scoring"
        )


def evaluate(
    run: Path,
    *,
    snapshot: Path,
    full_evaluation_path: Path,
    evaluation_max_tokens: int,
    pair_root: Path,
    output: Path,
    score_journal: Path,
    batch_size: int,
    require_update: int | None = None,
    require_additional_pairs_sha256: str | None = None,
) -> Path:
    import torch

    run = run.resolve()
    snapshot = snapshot.resolve()
    full_evaluation_path = full_evaluation_path.resolve()
    pair_root = pair_root.resolve()
    output = output.resolve()
    score_journal = score_journal.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    require_disjoint_paths(output, score_journal)
    binding = _bind_full_evaluation(
        run,
        snapshot,
        full_evaluation_path,
        evaluation_max_tokens=evaluation_max_tokens,
        require_update=require_update,
        require_additional_pairs_sha256=require_additional_pairs_sha256,
    )
    pair_manifest, pair_manifest_sha256, splits = _load_pair_inputs(pair_root)
    expected_additional = pair_manifest["outputs"]["train"]["sha256"]
    if (
        require_additional_pairs_sha256 is not None
        and require_additional_pairs_sha256 != expected_additional
    ):
        raise ValueError("required training-pair hash differs from the frozen manifest")

    result, encoder, tokenizer, head, base_model = mmbert_evaluate._load_run(run)
    recorded_base_model = binding.full_evaluation.get("base_model")
    if recorded_base_model is not None and recorded_base_model != base_model:
        raise ValueError("loaded base model differs from the full evaluation")
    checkpoint = mmbert_evaluate._load_snapshot(
        snapshot,
        result=result,
        encoder=encoder,
        head=head,
    )
    if (
        checkpoint["sha256"] != binding.snapshot_sha256
        or checkpoint["update"] != binding.snapshot_update
        or checkpoint["epoch"] != binding.snapshot_epoch
        or checkpoint["role"] != binding.snapshot_role
    ):
        raise ValueError("loaded snapshot differs from the full-evaluation binding")
    journal_model_sha256 = mmbert_evaluate._evaluation_model_sha256(
        binding.run_result_sha256,
        binding.snapshot_sha256,
        base_model=base_model,
    )
    head_contract = resolve_head_contract(result)
    scoring_sha256 = _longcode_scoring_sha256(evaluation_max_tokens)
    evaluation_identity_sha256 = _evaluation_identity_sha256(
        binding=binding,
        journal_model_sha256=journal_model_sha256,
        pair_manifest_sha256=pair_manifest_sha256,
        scoring_sha256=scoring_sha256,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    metrics = {}
    for split_name in PAIR_SPLITS:
        split = splits[split_name]
        benign, benign_windows = _score_side(
            split,
            "benign",
            encoder=encoder,
            tokenizer=tokenizer,
            head=head,
            primary_column=head_contract.primary_column,
            batch_size=batch_size,
            max_tokens=evaluation_max_tokens,
            model_sha256=journal_model_sha256,
            scoring_sha256=scoring_sha256,
            journal_root=score_journal,
        )
        attack, attack_windows = _score_side(
            split,
            "attack",
            encoder=encoder,
            tokenizer=tokenizer,
            head=head,
            primary_column=head_contract.primary_column,
            batch_size=batch_size,
            max_tokens=evaluation_max_tokens,
            model_sha256=journal_model_sha256,
            scoring_sha256=scoring_sha256,
            journal_root=score_journal,
        )
        metrics[split_name] = _metrics(
            split.rows,
            benign,
            attack,
            benign_windows,
            attack_windows,
            binding.threshold,
        )
    runtime_seconds = time.perf_counter() - started
    _assert_inputs_unchanged(
        run=run,
        snapshot=snapshot,
        full_evaluation_path=full_evaluation_path,
        binding=binding,
        pair_root=pair_root,
        pair_manifest_sha256=pair_manifest_sha256,
        splits=splits,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "advisory_only": True,
        "evaluation_identity_sha256": evaluation_identity_sha256,
        "checkpoint": {
            "run_name": run.name,
            "run_result_sha256": binding.run_result_sha256,
            "snapshot_sha256": binding.snapshot_sha256,
            "update": binding.snapshot_update,
            "epoch": binding.snapshot_epoch,
            "role": binding.snapshot_role,
            "training_max_tokens": binding.training_max_tokens,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "native_context_evaluation": (
                binding.training_max_tokens == binding.evaluation_max_tokens
            ),
            "base_model": base_model,
            "evaluation_model_sha256": binding.evaluation_model_sha256,
            "journal_model_sha256": journal_model_sha256,
        },
        "full_panel_evaluation": {
            "sha256": binding.full_evaluation_sha256,
            "score_artifact_sha256": binding.full_score_sha256,
            "scoring_sha256": binding.full_scoring_sha256,
            "evaluation_identity_sha256": binding.evaluation_identity_sha256,
        },
        "threshold": {
            "source": "exact-checkpoint full-panel canonical calibration only",
            "component_fpr_budget": DESCRIPTIVE_FPR_BUDGET,
            "value": binding.threshold,
            "selection_on_swe_rebench": False,
            "fixed_high_gate": HIGH_GATE,
        },
        "inputs": {
            "pair_manifest_sha256": pair_manifest_sha256,
            "source_revision": pair_manifest["source"]["revision"],
            "training_pair_archive_sha256": expected_additional,
            "splits": {
                name: {
                    "archive_sha256": split.sha256,
                    "content_sha256": split.content_sha256,
                    "pairs": split.expected_pairs,
                    "repositories": split.expected_repositories,
                }
                for name, split in splits.items()
            },
        },
        "scoring": {
            "aggregation": "maximum primary-head probability over ordered windows",
            "window_tokens": evaluation_max_tokens,
            "overlap_tokens": WINDOW_OVERLAP,
            "scoring_sha256": scoring_sha256,
            "batch_size": batch_size,
            "score_journal_numeric_only": True,
        },
        "metrics": metrics,
        "runtime": {
            "seconds": runtime_seconds,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "provenance": source_provenance(
            Path(__file__),
            ROOT / "experiments/mmbert_evaluation_contract.py",
            Path(mmbert_evaluate.__file__),
            Path(mmbert_evaluate.__file__).with_name("core.py"),
            Path(mmbert_evaluate.__file__).with_name("head_contract.py"),
            Path(mmbert_evaluate.__file__).with_name("score_journal.py"),
            Path(mmbert_evaluate.__file__).resolve().parents[2] / "normalization.py",
        ),
        "limitations": [
            "SWE-rebench V2 is already-open development evidence, not a prospective final test.",
            "Repository grouping holds out task contexts but not attack templates.",
            "The transported threshold is descriptive and is not approved for blocking.",
            "Neither pair split is used for checkpoint or threshold selection.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".longcode-eval-"))
    try:
        (temporary / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def _default_output_name(
    *,
    update: int,
    training_max_tokens: int,
    evaluation_max_tokens: int,
) -> str:
    return (
        f"longcode-evaluation-update-{update}-"
        f"trainctx{training_max_tokens}-evalctx{evaluation_max_tokens}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--full-evaluation", type=Path, required=True)
    parser.add_argument(
        "--evaluation-max-tokens",
        type=int,
        choices=mmbert_evaluate.SUPPORTED_MAX_TOKENS,
        required=True,
    )
    parser.add_argument(
        "--pair-root",
        type=Path,
        default=Path("artifacts/mmbert_lpft_new_data_rebuilt"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-journal", type=Path)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--require-update", type=int)
    parser.add_argument("--require-additional-pairs-sha256")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.require_update is not None and args.require_update < 1:
        raise ValueError("required update must be positive")
    if args.require_additional_pairs_sha256 is not None and (
        len(args.require_additional_pairs_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.require_additional_pairs_sha256
        )
    ):
        raise ValueError("required additional-pair identity must be a SHA-256 digest")
    result = mmbert_evaluate._read_run_result(args.run)
    training_max_tokens = mmbert_evaluate._training_max_tokens(result)
    update = mmbert_evaluate._snapshot_update_from_path(args.snapshot)
    output_name = _default_output_name(
        update=update,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=args.evaluation_max_tokens,
    )
    output = args.output or args.run / output_name
    required_suffix = (
        f"trainctx{training_max_tokens}-evalctx{args.evaluation_max_tokens}"
    )
    if required_suffix not in output.name:
        raise ValueError("long-code output name must include " + required_suffix)
    score_journal = (
        args.score_journal or output.parent / f".{output.name}.score-journal"
    )
    print(
        evaluate(
            args.run,
            snapshot=args.snapshot,
            full_evaluation_path=args.full_evaluation,
            evaluation_max_tokens=args.evaluation_max_tokens,
            pair_root=args.pair_root,
            output=output,
            score_journal=score_journal,
            batch_size=args.batch_size,
            require_update=args.require_update,
            require_additional_pairs_sha256=args.require_additional_pairs_sha256,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
