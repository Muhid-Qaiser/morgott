"""Evaluate one explicit mmBERT checkpoint on the frozen red-team reserve.

The canonical calibration threshold is never selected here.  Current evidence
requires the completed full-evaluation artifact for the same retained update
snapshot and context cap; historical implicit-512 packaged-selected wrappers
remain read-compatible.  This runner verifies its run, head, checkpoint, score
artifact, and input hashes before it loads a model.  The final artifact contains
aggregate numeric evidence only; raw prompt text and row IDs remain in the
existing reserve archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.guard_baselines import run as guard_run
from experiments.mmbert_evaluation_contract import (
    canonical_sha256 as _canonical_sha256,
)
from experiments.mmbert_evaluation_contract import (
    read_json_object,
    score_artifact,
    transported_threshold,
)
from morgott.models.mmbert import evaluate as mmbert_evaluate
from morgott.models.mmbert.core import (
    MAX_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
    source_provenance,
)
from morgott.models.mmbert.data import batches, routing_views
from morgott.models.mmbert.head_contract import resolve_head_contract
from morgott.models.mmbert.score_journal import (
    ScoreJournal,
    ScoreJournalSpec,
    require_disjoint_paths,
)
from morgott.normalization import strict_normalize

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
PURPOSE = "advisory mmBERT checkpoint red-team reserve evaluation"
LEGACY_ARM6_PURPOSE = "advisory Arm 6 snapshot red-team reserve evaluation"
FULL_EVALUATION_PURPOSE = "advisory mmBERT development evaluation"
ARM6_RUN_NAME = "mmbert-lora-full-s42-mb24-nolengthgroup-harmful-balanced"
NOHARM_RUN_NAME = "mmbert-lora-full-s42-mb24-nolengthgroup-noharm-current-control"
SHARED_TARGET = "1.0000%"
PRIMARY_SCORE_COLUMNS = ("score",)
MULTITASK_SCORE_COLUMNS = ("score", "harmful_intent_score")
# Backward-compatible name used by the existing Arm 6 tests and artifacts.
SCORE_COLUMNS = MULTITASK_SCORE_COLUMNS
ALLOWED_CHECKPOINT_ROLES = frozenset(
    {"pre_registered_comparison", "periodic_validation", "epoch_final"}
)
PACKAGED_CHECKPOINT_ROLE = "packaged_selected"
MAX_JSON_BYTES = 16 << 20
FULL_EVALUATION_CONTEXT_CONTRACT = "cap_explicit_v1"
LEGACY_FULL_EVALUATION_CONTEXT_CONTRACT = "legacy_implicit_512_v1"


@dataclass(frozen=True)
class EvaluationBinding:
    """Text-free identity tying one checkpoint to one full evaluation."""

    result: dict
    full_evaluation: dict
    run_result_sha256: str
    full_evaluation_sha256: str
    full_score_sha256: str
    checkpoint_sha256: str
    checkpoint_kind: str
    evaluation_model_sha256: str
    update: int
    epoch: int
    role: str
    threshold: float
    batch_size: int
    head_outputs: int
    score_columns: tuple[str, ...]
    training_max_tokens: int = MAX_TOKENS
    evaluation_max_tokens: int = MAX_TOKENS
    native_context_evaluation: bool = True
    full_evaluation_context_contract: str = LEGACY_FULL_EVALUATION_CONTEXT_CONTRACT
    full_scoring_sha256: str | None = None
    full_evaluation_identity_sha256: str | None = None


def _read_json(path: Path) -> tuple[dict, str]:
    return read_json_object(path, max_bytes=MAX_JSON_BYTES)


def _expected_inputs(
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    additional_pairs: Path,
) -> dict:
    views = routing_views(data_dir)
    return {
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "external_manifest_sha256": file_sha256(external_dir / "manifest.json"),
        "pair_archive_sha256": file_sha256(pairs),
        "additional_pair_archive_sha256": file_sha256(additional_pairs),
        "routing_views": {
            split: {"sha256": spec["sha256"], "rows": spec["rows"]}
            for split, (_, spec) in views.items()
        },
    }


def _score_columns(outputs: int) -> tuple[str, ...]:
    if outputs == 1:
        return PRIMARY_SCORE_COLUMNS
    if outputs == 2:
        return MULTITASK_SCORE_COLUMNS
    raise ValueError("unsupported head width")


def _validate_run_contract(run: Path, result: dict):
    """Return the strict one- or two-output contract for an eligible run."""

    contract = resolve_head_contract(result)
    identity = result.get("training_identity")
    objective = result.get("objective")
    harmful_objective = (
        objective.get("harmful_intent") if isinstance(objective, dict) else object()
    )
    identity_harmful = (
        identity.get("harmful_objective") if isinstance(identity, dict) else object()
    )
    expected_harmful = contract.outputs == 2
    if (
        result.get("purpose") != "maintained full-data advisory mmBERT training"
        or not isinstance(result.get("run_name"), str)
        or result["run_name"] != run.name
        or result.get("model_id") != MODEL_ID
        or result.get("model_revision") != MODEL_REVISION
        or result.get("adaptation") != "lora"
        or result.get("generic_target") != "instruction_subversion"
        or result.get("head_contract") is None
        or not isinstance(identity, dict)
        or identity.get("run_name") != result["run_name"]
        or identity.get("head_contract") != result["head_contract"]
        or "harmful_objective" not in identity
        or identity.get("length_grouped") is not False
        or not isinstance(objective, dict)
        or "harmful_intent" not in objective
        or (identity_harmful is not None) is not expected_harmful
        or (harmful_objective is not None) is not expected_harmful
        or identity_harmful != harmful_objective
    ):
        raise ValueError(
            "completed run is not a strict no-length-grouping mmBERT LoRA run"
        )
    return contract


def _artifact_path(run: Path, relative: object, *, description: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"packaged {description} path is invalid")
    path = (run / relative).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        raise ValueError(f"packaged {description} artifact is missing or escapes run")
    return path


def _packaged_checkpoint(run: Path, result: dict) -> tuple[dict, str]:
    """Validate and hash the exact packaged selected checkpoint."""

    run = run.resolve()
    training = result.get("training")
    artifact = result.get("artifact")
    if not isinstance(training, dict) or not isinstance(artifact, dict):
        raise ValueError("completed run has no packaged checkpoint contract")
    selected = training.get("selected_checkpoint")
    curve = training.get("curve")
    if not isinstance(selected, dict) or set(selected) != {
        "epoch",
        "updates",
        "selection_role",
        "selection_rule",
        "selection_loss",
        "validation_point_role",
        "pre_registered_comparison",
    }:
        raise ValueError("completed run has no strict selected-checkpoint contract")
    epoch = selected.get("epoch")
    update = selected.get("updates")
    loss = selected.get("selection_loss")
    if (
        type(epoch) is not int
        or epoch < 1
        or type(update) is not int
        or update < 1
        or training.get("selected_epoch") != epoch
        or training.get("selected_updates") != update
        or selected.get("selection_role") != "secondary"
        or not isinstance(selected.get("selection_rule"), str)
        or not selected["selection_rule"]
        or not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
        or selected.get("validation_point_role")
        not in {"periodic_validation", "epoch_final"}
        or type(selected.get("pre_registered_comparison")) is not bool
        or not isinstance(curve, list)
    ):
        raise ValueError("packaged selected-checkpoint provenance is invalid")
    matches = [
        row
        for row in curve
        if isinstance(row, dict)
        and row.get("epoch") == epoch
        and row.get("updates") == update
    ]
    if (
        len(matches) != 1
        or matches[0].get("selection_loss") != loss
        or matches[0].get("selection_rule") != selected["selection_rule"]
        or matches[0].get("pre_registered_comparison")
        != selected["pre_registered_comparison"]
        or (
            "periodic_validation" if matches[0].get("interim", False) else "epoch_final"
        )
        != selected["validation_point_role"]
    ):
        raise ValueError("packaged checkpoint is not one unique validation point")
    weights = artifact.get("weights_provenance")
    if weights != {
        "source": "training.selected_checkpoint",
        "epoch": epoch,
        "updates": update,
    }:
        raise ValueError("packaged weights do not bind the selected checkpoint")

    head = _artifact_path(run, artifact.get("head"), description="head")
    head_sha256 = file_sha256(head)
    if head_sha256 != artifact.get("head_sha256"):
        raise ValueError("packaged head hash mismatch")
    adapter_name = artifact.get("adapter")
    adapter_files = artifact.get("adapter_files")
    if (
        not isinstance(adapter_name, str)
        or not adapter_name
        or Path(adapter_name).is_absolute()
        or not isinstance(adapter_files, dict)
        or not adapter_files
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(digest, str)
            or len(digest) != 64
            for name, digest in adapter_files.items()
        )
    ):
        raise ValueError("packaged LoRA adapter contract is invalid")
    adapter = (run / adapter_name).resolve()
    if not adapter.is_relative_to(run) or not adapter.is_dir():
        raise ValueError("packaged LoRA adapter is missing or escapes run")
    actual_adapter_files = {
        path.name: file_sha256(path)
        for path in sorted(adapter.iterdir())
        if path.is_file()
    }
    if actual_adapter_files != adapter_files:
        raise ValueError("packaged LoRA adapter hash mismatch")

    identity = {
        "kind": "packaged_selected",
        "selected_checkpoint": selected,
        "weights_provenance": weights,
        "artifacts": {
            "head": artifact["head"],
            "head_sha256": head_sha256,
            "adapter": adapter_name,
            "adapter_files": actual_adapter_files,
        },
    }
    return identity, _canonical_sha256(identity)


def _score_artifact(
    full_path: Path,
    report: dict,
    score_columns: tuple[str, ...],
) -> tuple[Path, str]:
    return score_artifact(
        full_path,
        report,
        score_columns=score_columns,
        slice_names=("calibration", "dev_test", "promptshield", "sep"),
    )


def _validate_threshold(report: dict) -> float:
    return transported_threshold(
        report,
        target=SHARED_TARGET,
        allow_one=True,
        require_canonical_recall=True,
    )


def _full_evaluation_context_contract(report: dict) -> str:
    """Classify a full evaluation as wholly legacy or wholly cap-explicit.

    The historical 512-token artifacts predate all context and scoring identity
    fields.  Mixing either contract with only part of the other is ambiguous and
    therefore rejected rather than inferred.
    """

    scores = report.get("scores")
    runtime = report.get("runtime")
    locations = (
        (report, "training_max_tokens"),
        (report, "evaluation_max_tokens"),
        (report, "native_context_evaluation"),
        (report, "evaluation_identity_sha256"),
        (scores, "scoring_sha256"),
        (scores, "evaluation_identity_sha256"),
        (scores, "training_max_tokens"),
        (scores, "evaluation_max_tokens"),
        (runtime, "training_max_tokens"),
        (runtime, "evaluation_max_tokens"),
        (runtime, "native_context_evaluation"),
    )
    present = [
        isinstance(container, dict) and key in container for container, key in locations
    ]
    if all(present):
        return FULL_EVALUATION_CONTEXT_CONTRACT
    if not any(present):
        return LEGACY_FULL_EVALUATION_CONTEXT_CONTRACT
    raise ValueError("full evaluation has partial context/scoring identity metadata")


def validate_binding(
    run: Path,
    snapshot: Path | None,
    full_evaluation: Path,
    *,
    expected_inputs: dict,
    evaluation_max_tokens: int | None = None,
) -> EvaluationBinding:
    """Validate every text-free artifact before model loading."""

    run = run.resolve()
    full_evaluation = full_evaluation.resolve()

    result, run_result_sha256 = _read_json(run / "result.json")
    report, full_evaluation_sha256 = _read_json(full_evaluation)
    contract = _validate_run_contract(run, result)
    score_columns = _score_columns(contract.outputs)
    training_max_tokens = mmbert_evaluate._training_max_tokens(result)
    if evaluation_max_tokens is not None and (
        type(evaluation_max_tokens) is not int
        or evaluation_max_tokens not in mmbert_evaluate.SUPPORTED_MAX_TOKENS
    ):
        raise ValueError("unsupported evaluation context cap")
    context_contract = _full_evaluation_context_contract(report)
    if context_contract == LEGACY_FULL_EVALUATION_CONTEXT_CONTRACT:
        if evaluation_max_tokens is not None:
            raise ValueError(
                "explicit context evaluation requires a current cap-aware "
                "full evaluation"
            )
        if training_max_tokens != MAX_TOKENS:
            raise ValueError("legacy full evaluation is valid only for implicit 512")
        evaluation_max_tokens = MAX_TOKENS
    elif evaluation_max_tokens is None:
        raise ValueError(
            "current full evaluation requires explicit --evaluation-max-tokens"
        )
    if context_contract == FULL_EVALUATION_CONTEXT_CONTRACT and snapshot is None:
        raise ValueError(
            "current cap-aware reserve evaluation requires a retained snapshot"
        )
    native_context_evaluation = training_max_tokens == evaluation_max_tokens

    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") not in {1, 2}
        or (
            report.get("schema_version") == 2
            and context_contract != FULL_EVALUATION_CONTEXT_CONTRACT
        )
        or report.get("purpose") != FULL_EVALUATION_PURPOSE
        or report.get("advisory_only") is not True
        or report.get("model_id") != MODEL_ID
        or report.get("model_revision") != MODEL_REVISION
        or report.get("adaptation") != "lora"
        or report.get("head_contract") != result.get("head_contract")
        or report.get("run_result_sha256") != run_result_sha256
        or report.get("inputs") != expected_inputs
    ):
        raise ValueError("full evaluation does not match the completed mmBERT run")

    report_schema = report["schema_version"]
    base_model = report.get("base_model")
    if (report_schema == mmbert_evaluate.EVALUATION_SCHEMA_VERSION) != (
        base_model is not None
    ):
        raise ValueError("full evaluation base-model identity is incomplete")

    if snapshot is not None:
        snapshot = snapshot.resolve()
        update = mmbert_evaluate._snapshot_update_from_path(snapshot)
        if not snapshot.is_file():
            raise FileNotFoundError(f"snapshot does not exist: {snapshot}")
        checkpoint_sha256 = file_sha256(snapshot)
        checkpoint = report.get("evaluated_checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != {"sha256", "update", "epoch", "role"}
            or checkpoint.get("sha256") != checkpoint_sha256
            or checkpoint.get("update") != update
            or type(checkpoint.get("epoch")) is not int
            or checkpoint["epoch"] < 1
            or checkpoint.get("role") not in ALLOWED_CHECKPOINT_ROLES
        ):
            raise ValueError("full evaluation checkpoint differs from the snapshot")
        evaluation_model_sha256 = mmbert_evaluate._evaluation_model_sha256(
            run_result_sha256,
            checkpoint_sha256,
            base_model=base_model,
        )
        if report.get("evaluation_model_sha256") != evaluation_model_sha256:
            raise ValueError("full evaluation model identity is invalid")
        checkpoint_kind = "retained_update_snapshot"
        epoch = checkpoint["epoch"]
        role = checkpoint["role"]
    else:
        if "evaluated_checkpoint" in report or "evaluation_model_sha256" in report:
            raise ValueError(
                "a snapshot full evaluation requires the matching --snapshot"
            )
        _, checkpoint_sha256 = _packaged_checkpoint(run, result)
        training = result["training"]
        selected = training["selected_checkpoint"]
        update = selected["updates"]
        epoch = selected["epoch"]
        role = PACKAGED_CHECKPOINT_ROLE
        checkpoint_kind = "packaged_selected"
        evaluation_model_sha256 = mmbert_evaluate._evaluation_model_sha256(
            run_result_sha256,
            None,
            base_model=base_model,
        )

    scores = report.get("scores")
    runtime = report.get("runtime")
    if context_contract == FULL_EVALUATION_CONTEXT_CONTRACT:
        full_scoring_sha256 = (
            scores.get("scoring_sha256") if isinstance(scores, dict) else None
        )
        if not isinstance(full_scoring_sha256, str) or (
            mmbert_evaluate._SHA256.fullmatch(full_scoring_sha256) is None
        ):
            raise ValueError("full evaluation has no valid scoring identity")
        full_evaluation_identity_sha256 = (
            mmbert_evaluate._expected_evaluation_identity_sha256(
                report,
                model_sha256=evaluation_model_sha256,
                scoring_sha256=full_scoring_sha256,
                training_max_tokens=training_max_tokens,
                evaluation_max_tokens=evaluation_max_tokens,
            )
        )
        if (
            report.get("training_max_tokens") != training_max_tokens
            or report.get("evaluation_max_tokens") != evaluation_max_tokens
            or report.get("native_context_evaluation") is not native_context_evaluation
            or report.get("evaluation_identity_sha256")
            != full_evaluation_identity_sha256
            or not isinstance(scores, dict)
            or scores.get("scoring_sha256") != full_scoring_sha256
            or scores.get("evaluation_identity_sha256")
            != full_evaluation_identity_sha256
            or scores.get("training_max_tokens") != training_max_tokens
            or scores.get("evaluation_max_tokens") != evaluation_max_tokens
            or not isinstance(runtime, dict)
            or runtime.get("training_max_tokens") != training_max_tokens
            or runtime.get("evaluation_max_tokens") != evaluation_max_tokens
            or runtime.get("native_context_evaluation") is not native_context_evaluation
        ):
            raise ValueError(
                "full evaluation model, context, or scoring identity changed"
            )
    else:
        full_scoring_sha256 = None
        full_evaluation_identity_sha256 = None

    _, full_score_sha256 = _score_artifact(
        full_evaluation,
        report,
        score_columns,
    )
    threshold = _validate_threshold(report)
    runtime = report.get("runtime")
    batch_size = runtime.get("batch_size") if isinstance(runtime, dict) else None
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("full evaluation has no valid scoring batch size")
    return EvaluationBinding(
        result=result,
        full_evaluation=report,
        run_result_sha256=run_result_sha256,
        full_evaluation_sha256=full_evaluation_sha256,
        full_score_sha256=full_score_sha256,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_kind=checkpoint_kind,
        evaluation_model_sha256=evaluation_model_sha256,
        update=update,
        epoch=epoch,
        role=role,
        threshold=threshold,
        batch_size=batch_size,
        head_outputs=contract.outputs,
        score_columns=score_columns,
        training_max_tokens=training_max_tokens,
        evaluation_max_tokens=evaluation_max_tokens,
        native_context_evaluation=native_context_evaluation,
        full_evaluation_context_contract=context_contract,
        full_scoring_sha256=full_scoring_sha256,
        full_evaluation_identity_sha256=full_evaluation_identity_sha256,
    )


def _source_provenance() -> dict:
    return source_provenance(
        Path(__file__),
        ROOT / "experiments/guard_baselines/run.py",
        ROOT / "experiments/mmbert_evaluation_contract.py",
        ROOT / "src/morgott/models/mmbert/core.py",
        ROOT / "src/morgott/models/mmbert/evaluate.py",
        ROOT / "src/morgott/models/mmbert/head_contract.py",
        ROOT / "src/morgott/models/mmbert/score_journal.py",
        ROOT / "src/morgott/normalization.py",
    )


def _scoring_sha256(binding: EvaluationBinding, provenance: dict) -> str:
    return _canonical_sha256(
        {
            "contract": "mmbert-redteam-reserve-head-aware-scoring-v3",
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "head_contract": binding.result["head_contract"],
            "score_columns": list(binding.score_columns),
            "provenance": provenance,
        }
    )


def _journal_model_sha256(
    binding: EvaluationBinding,
    base_model: dict[str, str],
) -> str:
    return _canonical_sha256(
        {
            "loaded_model_sha256": mmbert_evaluate._evaluation_model_sha256(
                binding.run_result_sha256,
                (
                    binding.checkpoint_sha256
                    if binding.checkpoint_kind == "retained_update_snapshot"
                    else None
                ),
                base_model=base_model,
            ),
            "run_result_sha256": binding.run_result_sha256,
            "evaluation_model_sha256": binding.evaluation_model_sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "checkpoint_kind": binding.checkpoint_kind,
            "full_evaluation_sha256": binding.full_evaluation_sha256,
            "full_score_sha256": binding.full_score_sha256,
            "full_evaluation_context_contract": (
                binding.full_evaluation_context_contract
            ),
            "full_scoring_sha256": binding.full_scoring_sha256,
            "full_evaluation_identity_sha256": (
                binding.full_evaluation_identity_sha256
            ),
            "training_max_tokens": binding.training_max_tokens,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "native_context_evaluation": binding.native_context_evaluation,
            "head_contract": binding.result["head_contract"],
            "batch_size": binding.batch_size,
        }
    )


def _evaluation_identity_sha256(
    *,
    binding: EvaluationBinding,
    journal_model_sha256: str,
    panel_sha256: str,
    reserve_identity_sha256: str,
    scoring_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "journal_model_sha256": journal_model_sha256,
            "checkpoint_update": binding.update,
            "checkpoint_epoch": binding.epoch,
            "checkpoint_role": binding.role,
            "reserve_archive_sha256": guard_run.REDTEAM_SHA256,
            "reserve_identity_sha256": reserve_identity_sha256,
            "reserve_score_panel_sha256": panel_sha256,
            "batch_size": binding.batch_size,
            "score_columns": list(binding.score_columns),
            "scoring_sha256": scoring_sha256,
            "training_max_tokens": binding.training_max_tokens,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "native_context_evaluation": binding.native_context_evaluation,
            "full_evaluation_identity_sha256": (
                binding.full_evaluation_identity_sha256
            ),
            "full_scoring_sha256": binding.full_scoring_sha256,
        }
    )


def _assert_artifacts_unchanged(
    *,
    run: Path,
    snapshot: Path | None,
    full_evaluation: Path,
    redteam: Path,
    binding: EvaluationBinding,
) -> None:
    _, full_evaluation_sha256 = _read_json(full_evaluation)
    _, full_score_sha256 = _score_artifact(
        full_evaluation,
        binding.full_evaluation,
        binding.score_columns,
    )
    checkpoint_unchanged = (
        file_sha256(snapshot) == binding.checkpoint_sha256
        if snapshot is not None
        else _packaged_checkpoint(run, binding.result)[1] == binding.checkpoint_sha256
    )
    if (
        file_sha256(run / "result.json") != binding.run_result_sha256
        or not checkpoint_unchanged
        or full_evaluation_sha256 != binding.full_evaluation_sha256
        or full_score_sha256 != binding.full_score_sha256
        or file_sha256(redteam) != guard_run.REDTEAM_SHA256
    ):
        raise ValueError(
            "a bound run, checkpoint, evaluation, or reserve artifact changed"
        )


def _truncation_flags(
    tokenizer,
    rows: list[dict],
    *,
    batch_size: int,
    max_tokens: int,
) -> np.ndarray:
    if (
        type(max_tokens) is not int
        or max_tokens not in mmbert_evaluate.SUPPORTED_MAX_TOKENS
    ):
        raise ValueError("unsupported truncation context cap")
    flags = []
    for batch in batches(rows, 512):
        for start in range(0, len(batch), batch_size):
            texts = [
                strict_normalize(row["text"])
                for row in batch[start : start + batch_size]
            ]
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                max_length=max_tokens + 1,
                padding=False,
                return_length=True,
                truncation=True,
            )
            lengths = encoded.get("length")
            if not isinstance(lengths, list) or len(lengths) != len(texts):
                raise ValueError("tokenizer did not return aligned lengths")
            flags.extend(length > max_tokens for length in lengths)
    return np.asarray(flags, dtype=bool)


def _quantiles(scores: np.ndarray) -> dict:
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("score summary requires one finite non-empty vector")
    return {
        "mean": float(scores.mean()),
        "p05": float(np.quantile(scores, 0.05)),
        "p25": float(np.quantile(scores, 0.25)),
        "p50": float(np.quantile(scores, 0.50)),
        "p75": float(np.quantile(scores, 0.75)),
        "p95": float(np.quantile(scores, 0.95)),
    }


def _flag_summary(scores: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    return {
        "rows": len(scores),
        "flagged": int(flagged.sum()),
        "flag_rate": float(flagged.mean()) if len(scores) else None,
        "threshold": threshold,
        "score_quantiles": _quantiles(scores) if len(scores) else None,
        "fpr": None,
        "precision": None,
        "roc_auc": None,
        "pr_auc": None,
    }


def _by_field(
    rows: list[dict],
    scores: np.ndarray,
    field: str,
    summary,
) -> dict:
    values = np.asarray([row[field] for row in rows])
    return {
        str(value): summary(scores[values == value])
        for value in sorted(set(values.tolist()), key=str)
    }


def _primary_report(rows: list[dict], scores: np.ndarray, threshold: float) -> dict:
    summarize = lambda values: _flag_summary(values, threshold)  # noqa: E731
    basis = np.asarray([row["subversion_basis"] for row in rows])
    attested = basis != "None"
    bare_harmful = ~attested
    attested_summary = summarize(scores[attested])
    return {
        "target": "instruction_subversion",
        "aggregate": summarize(scores),
        "subversion_attested": {
            **attested_summary,
            "recall": attested_summary["flag_rate"],
            "metric_interpretation": "recall on source-attested subversion rows",
        },
        "bare_harmful_control": {
            **summarize(scores[bare_harmful]),
            "metric_interpretation": (
                "off-target flag rate on harmful requests without attested "
                "instruction subversion; not a benign FPR"
            ),
        },
        "by_subversion_basis": _by_field(rows, scores, "subversion_basis", summarize),
        "by_prompt_kind": _by_field(rows, scores, "prompt_kind", summarize),
        "by_attack_mode": _by_field(rows, scores, "attack_mode", summarize),
        "by_category": _by_field(rows, scores, "category", summarize),
        "by_channel": _by_field(rows, scores, "input_channel", summarize),
    }


def _auxiliary_report(rows: list[dict], scores: np.ndarray) -> dict:
    return {
        "target": "harmful_intent",
        "role": "unlabelled descriptive diagnostic only",
        "known_label_rows": 0,
        "unknown_masked_rows": len(rows),
        "threshold": None,
        "metrics": None,
        "aggregate": _quantiles(scores),
        "by_subversion_basis": _by_field(rows, scores, "subversion_basis", _quantiles),
        "by_prompt_kind": _by_field(rows, scores, "prompt_kind", _quantiles),
        "limitation": (
            "The reserve supplies no canonical harmful_intent or benign security "
            "tags. No harmful label, BCE, AUROC, AP, or operating threshold is "
            "invented from campaign metadata."
        ),
    }


def _score_matrix_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _build_report(
    *,
    binding: EvaluationBinding,
    rows: list[dict],
    head_scores: np.ndarray,
    panel_sha256: str,
    journal: ScoreJournal,
    journal_model_sha256: str,
    base_model: dict[str, str],
    truncation: np.ndarray,
    runtime_seconds: float,
    resumed_rows: int,
    device: str,
    peak_reserved_bytes: int,
    provenance: dict,
    scoring_sha256: str,
) -> dict:
    if (
        head_scores.shape
        != (
            len(rows),
            binding.head_outputs,
        )
        or not np.isfinite(head_scores).all()
    ):
        raise ValueError(
            "reserve scoring did not return the declared finite head columns"
        )
    if truncation.shape != (len(rows),):
        raise ValueError("truncation flags are misaligned")
    primary = _primary_report(rows, head_scores[:, 0], binding.threshold)
    canonical_metrics = binding.full_evaluation["canonical_dev_test"]["metrics"]
    attested_recall = primary["subversion_attested"]["flag_rate"]
    reserve_identity_sha256 = mmbert_evaluate._identity_sha256(rows)
    evaluation_identity_sha256 = _evaluation_identity_sha256(
        binding=binding,
        journal_model_sha256=journal_model_sha256,
        panel_sha256=panel_sha256,
        reserve_identity_sha256=reserve_identity_sha256,
        scoring_sha256=scoring_sha256,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            LEGACY_ARM6_PURPOSE
            if binding.result["run_name"] == ARM6_RUN_NAME
            and binding.head_outputs == 2
            and binding.checkpoint_kind == "retained_update_snapshot"
            and binding.full_evaluation_context_contract
            == LEGACY_FULL_EVALUATION_CONTEXT_CONTRACT
            else PURPOSE
        ),
        "advisory_only": True,
        "promotion_authorized": False,
        "status": "scored",
        "training_max_tokens": binding.training_max_tokens,
        "evaluation_max_tokens": binding.evaluation_max_tokens,
        "native_context_evaluation": binding.native_context_evaluation,
        "evaluation_identity_sha256": evaluation_identity_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model": base_model,
        "run_name": binding.result["run_name"],
        "adaptation": "lora",
        "head_contract": binding.result["head_contract"],
        "evaluated_checkpoint": {
            "sha256": binding.checkpoint_sha256,
            "update": binding.update,
            "epoch": binding.epoch,
            "role": binding.role,
        },
        "threshold_evidence": {
            "target": SHARED_TARGET,
            "threshold": binding.threshold,
            "protocol": "canonical calibration components only",
            "full_evaluation_sha256": binding.full_evaluation_sha256,
            "full_evaluation_score_sha256": binding.full_score_sha256,
            "run_result_sha256": binding.run_result_sha256,
            "evaluation_model_sha256": binding.evaluation_model_sha256,
            "checkpoint_kind": binding.checkpoint_kind,
            "training_max_tokens": binding.training_max_tokens,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "native_context_evaluation": binding.native_context_evaluation,
            "full_evaluation_context_contract": (
                binding.full_evaluation_context_contract
            ),
            "full_evaluation_scoring_sha256": binding.full_scoring_sha256,
            "full_evaluation_identity_sha256": (
                binding.full_evaluation_identity_sha256
            ),
            "calibration_row_identity_sha256": binding.full_evaluation["calibration"][
                "row_identity_sha256"
            ],
        },
        "full_panel_evaluation": {
            "sha256": binding.full_evaluation_sha256,
            "score_artifact_sha256": binding.full_score_sha256,
            "model_sha256": binding.evaluation_model_sha256,
            "scoring_sha256": binding.full_scoring_sha256,
            "evaluation_identity_sha256": (binding.full_evaluation_identity_sha256),
            "context_contract": binding.full_evaluation_context_contract,
        },
        "reserve": {
            "archive_sha256": guard_run.REDTEAM_SHA256,
            "rows": len(rows),
            "row_identity_sha256": reserve_identity_sha256,
            "score_panel_sha256": panel_sha256,
            "population": "frozen first-party red-team reserve",
            "label_contract": (
                "positive-only campaign panel containing source-attested "
                "instruction subversion and bare harmful off-target controls"
            ),
            "raw_prompt_material_persisted": False,
            "reading_rules": list(guard_run.REDTEAM_READING),
        },
        "instruction_subversion": primary,
        **(
            {"harmful_intent": _auxiliary_report(rows, head_scores[:, 1])}
            if binding.head_outputs == 2
            else {}
        ),
        "contamination_control": {
            "canonical_dev_test_recall": canonical_metrics["recall"],
            "redteam_subversion_attested_recall": attested_recall,
            "delta_against_subversion_attested": (
                canonical_metrics["recall"] - attested_recall
            ),
            "interpretation": (
                "The two populations differ in composition and publication "
                "status. This delta is descriptive and cannot attribute a gap "
                "to contamination."
            ),
        },
        "truncation": {
            "max_input_tokens": binding.evaluation_max_tokens,
            "rows": len(rows),
            "truncated_rows": int(truncation.sum()),
            "truncated_fraction": float(truncation.mean()),
        },
        "scores": {
            "storage": "text-free numeric score journal only",
            "columns": list(binding.score_columns),
            "dtype": "float64",
            "shape": [len(rows), binding.head_outputs],
            "sha256": _score_matrix_sha256(head_scores),
            "journal_identity_sha256": journal.identity_sha256,
            "journal_model_sha256": journal_model_sha256,
            "scoring_sha256": scoring_sha256,
            "evaluation_identity_sha256": evaluation_identity_sha256,
            "training_max_tokens": binding.training_max_tokens,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "native_context_evaluation": binding.native_context_evaluation,
        },
        "runtime": {
            "seconds": runtime_seconds,
            "seconds_scope": (
                "current invocation only; resumed journal shards retain no timing"
            ),
            "rows": len(rows),
            "scored_rows_current_invocation": len(rows) - resumed_rows,
            "resumed_rows": resumed_rows,
            "batch_size": binding.batch_size,
            "training_max_tokens": binding.training_max_tokens,
            "evaluation_max_tokens": binding.evaluation_max_tokens,
            "native_context_evaluation": binding.native_context_evaluation,
            "device": device,
            "peak_reserved_bytes": peak_reserved_bytes,
        },
        "inputs": binding.full_evaluation["inputs"],
        "provenance": provenance,
        "limitations": [
            "The reserve is consumed development evidence, not a prospective final test.",
            "The panel has no matched benign denominator; aggregate flag rate is not FPR.",
            "Campaign category is confounded with attack mode.",
            (
                "This is the run-native context decision cell."
                if binding.native_context_evaluation
                else (
                    "This is an optional cross-cap diagnostic; it must not replace "
                    "either run-native context decision cell."
                )
            ),
            "Scores are advisory and do not authorize blocking or promotion.",
        ],
    }


def evaluate_reserve(
    run: Path,
    *,
    snapshot: Path | None,
    full_evaluation: Path,
    redteam: Path,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    additional_pairs: Path,
    output: Path,
    score_journal: Path,
    batch_size: int | None,
    evaluation_max_tokens: int | None,
) -> Path:
    """Score only the reserve and atomically publish a text-free report."""

    import torch

    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    require_disjoint_paths(output, score_journal)
    expected_inputs = _expected_inputs(
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        additional_pairs=additional_pairs,
    )
    binding = validate_binding(
        run,
        snapshot,
        full_evaluation,
        expected_inputs=expected_inputs,
        evaluation_max_tokens=evaluation_max_tokens,
    )
    if batch_size is not None and batch_size != binding.batch_size:
        raise ValueError(
            "reserve batch size must match the full evaluation to preserve "
            "padded score semantics"
        )
    rows = guard_run._redteam_rows(redteam)
    panel_sha256 = guard_run._journal_panel_sha256(
        guard_run.REDTEAM_SHA256,
        "redteam_reserve",
        rows,
    )
    result, encoder, tokenizer, head, base_model = mmbert_evaluate._load_run(run)
    recorded_base_model = binding.full_evaluation.get("base_model")
    if recorded_base_model is not None and recorded_base_model != base_model:
        raise ValueError("loaded base model differs from the full evaluation")
    if result != binding.result:
        raise ValueError("completed run changed after preflight validation")
    if snapshot is not None:
        checkpoint = mmbert_evaluate._load_snapshot(
            snapshot,
            result=result,
            encoder=encoder,
            head=head,
        )
        if checkpoint != {
            "sha256": binding.checkpoint_sha256,
            "update": binding.update,
            "epoch": binding.epoch,
            "role": binding.role,
        }:
            raise ValueError("loaded snapshot differs from full evaluation evidence")
    elif _packaged_checkpoint(run.resolve(), result)[1] != binding.checkpoint_sha256:
        raise ValueError("loaded package differs from full evaluation evidence")

    provenance = _source_provenance()
    scoring_sha256 = _scoring_sha256(binding, provenance)
    journal_model_sha256 = _journal_model_sha256(binding, base_model)
    journal = ScoreJournal(
        score_journal,
        ScoreJournalSpec(
            model_sha256=journal_model_sha256,
            panel_sha256=panel_sha256,
            scoring_sha256=scoring_sha256,
            rows=len(rows),
            batch_size=binding.batch_size,
            columns=binding.score_columns,
        ),
    )
    resumed_rows = journal.completed_rows

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    scored = mmbert_evaluate._score(
        rows,
        encoder,
        tokenizer,
        head,
        batch_size=binding.batch_size,
        journal=journal,
        score_columns=binding.score_columns,
        max_tokens=binding.evaluation_max_tokens,
    )
    truncation = _truncation_flags(
        tokenizer,
        rows,
        batch_size=binding.batch_size,
        max_tokens=binding.evaluation_max_tokens,
    )
    runtime_seconds = time.perf_counter() - started
    _assert_artifacts_unchanged(
        run=run,
        snapshot=snapshot,
        full_evaluation=full_evaluation,
        redteam=redteam,
        binding=binding,
    )
    if _source_provenance() != provenance:
        raise ValueError("scoring source provenance changed during evaluation")
    report = _build_report(
        binding=binding,
        rows=rows,
        head_scores=scored["head_scores"],
        panel_sha256=panel_sha256,
        journal=journal,
        journal_model_sha256=journal_model_sha256,
        base_model=base_model,
        truncation=truncation,
        runtime_seconds=runtime_seconds,
        resumed_rows=resumed_rows,
        device=torch.cuda.get_device_name(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        provenance=provenance,
        scoring_sha256=scoring_sha256,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".mmbert-redteam-"))
    try:
        (temporary / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help=(
            "retained update-N.pt snapshot; omission is supported only for a "
            "historical legacy packaged-selected evaluation"
        ),
    )
    parser.add_argument("--full-evaluation", type=Path)
    parser.add_argument(
        "--evaluation-max-tokens",
        type=int,
        choices=mmbert_evaluate.SUPPORTED_MAX_TOKENS,
        help=(
            "required for current cap-aware evaluations; omission is accepted "
            "only for wholly legacy implicit-512 full-evaluation artifacts"
        ),
    )
    parser.add_argument("--redteam", type=Path, default=guard_run.REDTEAM_PATH)
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
        "--additional-pairs",
        type=Path,
        default=Path("artifacts/mmbert_lpft_new_data_rebuilt/train/pairs.jsonl.gz"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--score-journal", type=Path)
    parser.add_argument("--batch-size", type=int)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.evaluation_max_tokens is not None and args.snapshot is None:
        parser.error("--evaluation-max-tokens requires --snapshot")
    update = (
        mmbert_evaluate._snapshot_update_from_path(args.snapshot)
        if args.snapshot is not None
        else None
    )
    legacy_name = (
        f"redteam-reserve-evaluation-update-{update}"
        if update is not None
        else "redteam-reserve-evaluation"
    )
    if args.evaluation_max_tokens is None:
        name = legacy_name
        default_full_name = (
            f"evaluation-update-{update}" if update is not None else "evaluation"
        )
    else:
        result = mmbert_evaluate._read_run_result(args.run)
        training_max_tokens = mmbert_evaluate._training_max_tokens(result)
        context_suffix = (
            f"trainctx{training_max_tokens}-evalctx{args.evaluation_max_tokens}"
        )
        name = f"{legacy_name}-{context_suffix}"
        default_full_name = (
            f"evaluation-update-{update}-{context_suffix}"
            if update is not None
            else f"evaluation-{context_suffix}"
        )
    full_evaluation = (
        args.full_evaluation or args.run / default_full_name / "evaluation.json"
    )
    output = args.output or args.run / name
    score_journal = args.score_journal or args.run / f".{name}.score-journal"
    if args.evaluation_max_tokens is not None and context_suffix not in output.name:
        raise ValueError("cap-aware output name must include " + context_suffix)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(
        evaluate_reserve(
            args.run,
            snapshot=args.snapshot,
            full_evaluation=full_evaluation,
            redteam=args.redteam,
            data_dir=args.data_dir,
            external_dir=args.external_dir,
            pairs=args.pairs,
            additional_pairs=args.additional_pairs,
            output=output,
            score_journal=score_journal,
            batch_size=args.batch_size,
            evaluation_max_tokens=args.evaluation_max_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
