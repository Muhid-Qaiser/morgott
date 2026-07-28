"""Train the full-combined frozen-encoder generic detector head.

The training pool contains every retained canonical generic-injection row,
PromptShield train, and generated matched pairs.
Three predeclared objectives separate data inclusion from reweighting while
preserving the same deterministic batches and optimizer-update schedule.
An optional aligned pair-ranking term targets the known within-pair failure.

Validation selects checkpoints.
This runner never evaluates a test set or calibrates a blocking threshold.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from prepare_combined_generic import REPO_ROOT, TARGET, canonical_record, file_sha256
from train_combined_generic_head import (
    VALIDATION_FEATURE_RECORD_CHUNK,
    VALIDATION_PREDICTION_BATCH_SIZE,
    _bce_from_logits,
    _binary_metrics,
    _save_head,
    _scores,
    extract_features,
    load_records,
    new_head,
    predict_logits,
    resolve_model_revision,
    validate_selection_report,
)

DEFAULT_SELECTION = REPO_ROOT / "artifacts/combined_generic/full_selection_s42"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/combined_generic/full_runs"
DEFAULT_FEATURE_CACHE = REPO_ROOT / "artifacts/combined_generic/feature_cache"
FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_CACHE_REPORT = "cache_report.json"
FEATURE_CACHE_STATE = "cache_state.json"
FEATURE_CACHE_DATA = "canonical_features.uint16"
OBJECTIVES = ("canonical_uniform", "full_uniform", "full_balanced")


def objective_spec(
    name: str,
    *,
    canonical_rows: int,
    promptshield_labels: np.ndarray,
    matched_pair_rows: int,
) -> dict:
    """Return the exact loss weights for one predeclared causal control."""
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective: {name}")
    labels = np.asarray(promptshield_labels)
    counts = Counter(int(label) for label in labels)
    if (
        canonical_rows < 1
        or counts.keys() != {0, 1}
        or sum(counts.values()) != len(labels)
        or matched_pair_rows < 2
        or matched_pair_rows % 2
    ):
        raise ValueError("objective populations must be non-empty binary datasets")

    rows = {
        "morgott": canonical_rows,
        "promptshield": len(labels),
        "matched_pairs": matched_pair_rows,
    }
    if name == "canonical_uniform":
        coefficients = {
            "morgott": 1.0,
            "promptshield": 0.0,
            "matched_pairs": 0.0,
        }
    elif name == "full_uniform":
        total = sum(rows.values())
        coefficients = {domain: count / total for domain, count in rows.items()}
    else:
        coefficients = {domain: 1.0 / 3.0 for domain in rows}

    uniform = name != "full_balanced"
    correction = (
        {str(label): 2.0 * counts[label] / len(labels) for label in (0, 1)}
        if uniform
        else {"0": 1.0, "1": 1.0}
    )
    return {
        "name": name,
        "unique_training_rows": rows,
        "canonical_weighting": (
            "uniform_per_row" if uniform else "label_source_group_balanced"
        ),
        "domain_bce_coefficients": coefficients,
        "domain_bce_coefficients_definition": (
            "unique row proportion"
            if name == "full_uniform"
            else "predeclared fixed coefficients"
        ),
        "promptshield_sampling": "class_balanced_cycle",
        "promptshield_class_loss_correction": correction,
        "promptshield_class_loss_correction_definition": (
            "2 * observed class rows / PromptShield rows" if uniform else "none"
        ),
        "matched_pair_sampling": "complete_pair_cycle",
        "cross_objective_schedule_comparability": (
            "requires identical recorded seed, epochs, domain batch sizes, "
            "learning rate, and shuffle buffer"
        ),
    }


def training_objective_loss(
    *,
    canonical_logits,
    canonical_targets,
    canonical_weights,
    promptshield_logits,
    promptshield_targets,
    benign_logits,
    attack_logits,
    objective: dict,
    pair_ranking_weight: float,
) -> dict:
    """Compute one update's predeclared objective from aligned domain batches."""
    import torch

    name = objective.get("name")
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective: {name}")
    if pair_ranking_weight < 0:
        raise ValueError("pair ranking weight must be non-negative")
    if name == "canonical_uniform" and pair_ranking_weight:
        raise ValueError("canonical_uniform cannot use pair ranking")

    canonical_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        canonical_logits,
        canonical_targets,
        reduction="none",
    )
    if objective["canonical_weighting"] == "label_source_group_balanced":
        canonical_loss = torch.mean(canonical_losses * canonical_weights)
    else:
        canonical_loss = canonical_losses.mean()

    promptshield_losses = torch.nn.functional.binary_cross_entropy_with_logits(
        promptshield_logits,
        promptshield_targets,
        reduction="none",
    )
    correction = objective["promptshield_class_loss_correction"]
    promptshield_weights = torch.where(
        promptshield_targets == 0,
        float(correction["0"]),
        float(correction["1"]),
    )
    promptshield_loss = torch.mean(promptshield_losses * promptshield_weights)
    pair_bce = 0.5 * (
        torch.nn.functional.softplus(benign_logits).mean()
        + torch.nn.functional.softplus(-attack_logits).mean()
    )
    coefficients = objective["domain_bce_coefficients"]
    domain_bce = (
        float(coefficients["morgott"]) * canonical_loss
        + float(coefficients["promptshield"]) * promptshield_loss
        + float(coefficients["matched_pairs"]) * pair_bce
    )
    ranking_loss = torch.nn.functional.softplus(-(attack_logits - benign_logits)).mean()
    return {
        "total": domain_bce + pair_ranking_weight * ranking_loss,
        "domain_bce": domain_bce,
        "ranking": ranking_loss,
        "morgott_bce": canonical_loss,
        "promptshield_bce": promptshield_loss,
        "matched_pair_bce": pair_bce,
    }


def run_directory_name(
    model_id: str,
    *,
    objective: str,
    pair_ranking_weight: float,
    seed: int,
) -> str:
    """Return an output tag that names every causal objective choice."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if seed < 0 or not math.isfinite(pair_ranking_weight) or pair_ranking_weight < 0:
        raise ValueError("invalid run identity")
    if objective == "canonical_uniform" and pair_ranking_weight:
        raise ValueError("canonical_uniform cannot use pair ranking")
    model_tag = re.sub(r"[^a-z0-9]+", "-", model_id.casefold()).strip("-")
    objective_tag = objective.replace("_", "-")
    ranking_tag = str(float(pair_ranking_weight)).replace(".", "p")
    return f"{model_tag}_objective-{objective_tag}_pair-rank-{ranking_tag}_s{seed}"


def _verify_file(path: Path, expected_sha256: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"input hash mismatch for {path}: expected {expected_sha256}, "
            f"found {actual}"
        )


def _resolve_repo_artifact(spec: dict) -> Path:
    path = (REPO_ROOT / spec["path"]).resolve()
    if not path.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        raise ValueError(f"artifact path escapes artifacts: {path}")
    _verify_file(path, spec["sha256"])
    return path


def _resolve_repo_input(spec: dict) -> Path:
    path = (REPO_ROOT / spec["path"]).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"input path escapes repository: {path}")
    _verify_file(path, spec["sha256"])
    return path


def _verify_unchanged(paths: list[tuple[Path, str]]) -> None:
    for path, expected_sha256 in paths:
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"provenance input changed while training: {path}")


def _stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _bfloat16_to_uint16(features) -> np.ndarray:
    """Return a little-endian copy of the exact CPU BF16 bit patterns."""
    import torch

    if (
        features.device.type != "cpu"
        or features.dtype != torch.bfloat16
        or features.ndim != 2
    ):
        raise ValueError("feature cache requires a two-dimensional CPU BF16 tensor")
    bits = features.detach().contiguous().view(torch.uint16).numpy()
    return np.asarray(bits, dtype="<u2").copy(order="C")


def _uint16_to_bfloat16(values: np.ndarray):
    """Restore a CPU BF16 tensor from little-endian stored bit patterns."""
    import torch

    stored = np.asarray(values, dtype="<u2")
    if stored.ndim != 2:
        raise ValueError("stored feature rows must be two-dimensional")
    native = np.array(stored, dtype=np.uint16, order="C", copy=True)
    return torch.from_numpy(native).view(torch.bfloat16)


def validate_feature_cache_artifact(
    directory: Path,
    *,
    expected_spec: dict | None,
) -> dict:
    """Verify a completed feature cache and return its resolved artifact paths."""
    directory = directory.resolve()
    report_path = directory / FEATURE_CACHE_REPORT
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported feature cache schema")
    spec = report.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("feature cache report has no spec")
    if report.get("spec_sha256") != _stable_json_sha256(spec):
        raise ValueError("feature cache spec hash mismatch")
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("feature cache spec mismatch")
    data = report.get("data")
    if not isinstance(data, dict):
        raise ValueError("feature cache report has no data descriptor")
    if data.get("file") != FEATURE_CACHE_DATA:
        raise ValueError("feature cache data filename is invalid")
    data_path = (directory / data["file"]).resolve()
    if data_path.parent != directory:
        raise ValueError("feature cache data path escapes its directory")
    shape = data.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value < 1 for value in shape)
        or spec.get("rows") != shape[0]
        or spec.get("feature_width") != shape[1]
        or data.get("storage_dtype") != "uint16_le"
        or data.get("logical_dtype") != "bfloat16"
    ):
        raise ValueError("feature cache shape or dtype contract failed")
    expected_bytes = shape[0] * shape[1] * np.dtype("<u2").itemsize
    if (
        data.get("bytes") != expected_bytes
        or data_path.stat().st_size != expected_bytes
    ):
        raise ValueError("feature cache byte size mismatch")
    actual_sha256 = file_sha256(data_path)
    if data.get("sha256") != actual_sha256:
        raise ValueError("feature cache data hash mismatch")
    return {
        "directory": directory,
        "report_path": report_path,
        "data_path": data_path,
        "report": report,
    }


class CanonicalFeatureCache:
    """Read-only BF16 feature matrix backed by a verified raw memmap."""

    def __init__(self, artifact: dict) -> None:
        shape = tuple(artifact["report"]["data"]["shape"])
        self.artifact = artifact
        self.rows, self.feature_width = shape
        self._matrix = np.memmap(
            artifact["data_path"],
            mode="r",
            dtype="<u2",
            shape=shape,
        )

    def take(self, indices: np.ndarray):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("feature cache indices must be a non-empty vector")
        if indices.min() < 0 or indices.max() >= self.rows:
            raise IndexError("feature cache index is out of bounds")
        return _uint16_to_bfloat16(self._matrix[indices])

    def close(self) -> None:
        matrix = self._matrix
        self._matrix = None
        mmap = getattr(matrix, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _load_jsonl_spec(spec: dict) -> list[dict]:
    path = _resolve_repo_artifact(spec)
    records = load_records(path, spec["sha256"])
    if len(records) != spec["rows"]:
        raise ValueError(
            f"{path} row count mismatch: expected {spec['rows']}, found {len(records)}"
        )
    counts = Counter(str(row["generic_label"]) for row in records)
    if dict(sorted(counts.items())) != spec["labels"]:
        raise ValueError(f"{path} label counts do not match its report")
    return records


def _pair_indices(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_pair = defaultdict(dict)
    for index, record in enumerate(records):
        if (
            record.get("dataset") != "matched_pairs"
            or record.get("source") != "matched_pairs_generated"
            or record.get("pair_family") != "generated_matched_instruction_subversion"
        ):
            raise ValueError(f"invalid matched-pair provenance: {record.get('id')}")
        pair_id = record.get("pair_id")
        role = record.get("pair_role")
        if not isinstance(pair_id, str) or role not in {"benign", "attack"}:
            raise ValueError(f"invalid pair fields: {record.get('id')}")
        if role in by_pair[pair_id]:
            raise ValueError(f"duplicate {role} half for {pair_id}")
        expected_label = 0 if role == "benign" else 1
        if (
            record["generic_label"] != expected_label
            or record.get("pair_label") != expected_label
        ):
            raise ValueError(f"pair label mismatch: {record.get('id')}")
        by_pair[pair_id][role] = index
    incomplete = sorted(
        pair_id
        for pair_id, roles in by_pair.items()
        if set(roles) != {"benign", "attack"}
    )
    if incomplete:
        raise ValueError(f"incomplete matched pairs: {incomplete[:3]}")
    pair_ids = sorted(by_pair)
    benign = np.asarray([by_pair[pair_id]["benign"] for pair_id in pair_ids])
    attack = np.asarray([by_pair[pair_id]["attack"] for pair_id in pair_ids])
    return benign, attack


def _validate_full_inputs(
    connection: sqlite3.Connection,
    canonical_spec: dict,
    promptshield: list[dict],
    matched_pairs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    database_rows = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    if database_rows != canonical_spec["rows"]:
        raise ValueError(
            f"canonical index rows: expected {canonical_spec['rows']}, "
            f"found {database_rows}"
        )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
    required_columns = {
        "id",
        "generic_label",
        "source",
        "group_id",
        "normalized_text_sha256",
        "strict_text_sha256",
        "byte_offset",
        "objective_weight",
    }
    if not required_columns <= columns:
        raise ValueError(
            f"canonical index is missing columns: {sorted(required_columns - columns)}"
        )
    missing_weights = connection.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        WHERE objective_weight IS NULL OR objective_weight <= 0
        """
    ).fetchone()[0]
    if missing_weights:
        raise ValueError(f"canonical index has {missing_weights} invalid weights")
    for record in promptshield:
        if (
            record.get("dataset") != "promptshield"
            or record.get("channel") is not None
            or record.get("subtype_training_eligible") is not False
        ):
            raise ValueError(
                f"PromptShield provenance contract failed: {record.get('id')}"
            )
    benign, attack = _pair_indices(matched_pairs)
    external_hashes = [
        record["strict_text_sha256"] for record in [*promptshield, *matched_pairs]
    ]
    if len(external_hashes) != len(set(external_hashes)):
        raise ValueError("external fitting datasets have strict-text overlap")
    for start in range(0, len(external_hashes), 500):
        values = external_hashes[start : start + 500]
        placeholders = ",".join("?" for _ in values)
        overlap = connection.execute(
            f"""
            SELECT strict_text_sha256
            FROM candidates
            WHERE strict_text_sha256 IN ({placeholders})
            LIMIT 1
            """,
            values,
        ).fetchone()
        if overlap is not None:
            raise ValueError("canonical and external fitting texts overlap")
    return benign, attack


def canonical_feature_cache_spec(
    *,
    selection_report_path: Path,
    selection_report_sha256: str,
    canonical_spec: dict,
    canonical_input_spec: dict,
    model_id: str,
    model_revision: str,
    hidden_size: int,
    max_tokens: int,
    token_budget: int,
    feature_record_chunk: int,
    runner_sha256: str,
    head_helper_sha256: str,
    strict_normalizer_sha256: str,
    canonical_projection_sha256: str,
    packages: dict[str, str],
) -> dict:
    if hidden_size < 1 or feature_record_chunk < 1:
        raise ValueError("encoder dimensions and feature chunk must be positive")
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "selection_report": {
            "path": str(selection_report_path.relative_to(REPO_ROOT)),
            "sha256": selection_report_sha256,
        },
        "canonical_index": {
            "path": canonical_spec["path"],
            "sha256": canonical_spec["sha256"],
            "rows": canonical_spec["rows"],
        },
        "canonical_source": {
            "path": canonical_input_spec["path"],
            "sha256": canonical_input_spec["sha256"],
            "rows": canonical_input_spec["rows"],
        },
        "model_id": model_id,
        "model_revision": model_revision,
        "attention_implementation": "sdpa",
        "normalization": "strict",
        "pooling": "cls_masked_mean_masked_max",
        "max_tokens": max_tokens,
        "token_budget": token_budget,
        "feature_record_chunk": feature_record_chunk,
        "rows": canonical_spec["rows"],
        "hidden_size": hidden_size,
        "feature_width": hidden_size * 3,
        "logical_dtype": "bfloat16",
        "storage_dtype": "uint16_le",
        "helpers": {
            "runner_sha256": runner_sha256,
            "head_helper_sha256": head_helper_sha256,
            "strict_normalizer_sha256": strict_normalizer_sha256,
            "canonical_projection_sha256": canonical_projection_sha256,
        },
        "packages": dict(sorted(packages.items())),
    }


def _digest_file_prefix(path: Path, byte_count: int):
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        while remaining:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                raise ValueError("feature cache prefix is shorter than its state")
            digest.update(block)
            remaining -= len(block)
    return digest


def _iter_canonical_source_chunks(
    connection: sqlite3.Connection,
    canonical_path: Path,
    *,
    start_index: int,
    chunk_rows: int,
):
    if start_index < 0 or chunk_rows < 1:
        raise ValueError("invalid canonical feature-cache range")
    chunk = []
    cursor = connection.execute(
        """
        SELECT id, generic_label, normalized_text_sha256,
               strict_text_sha256, byte_offset
        FROM candidates
        ORDER BY byte_offset
        LIMIT -1 OFFSET ?
        """,
        (start_index,),
    )
    with canonical_path.open("rb") as source:
        for feature_index, (
            expected_id,
            expected_label,
            expected_normalized,
            expected_strict,
            offset,
        ) in enumerate(cursor, start=start_index):
            source.seek(offset)
            canonical = json.loads(source.readline())
            row = canonical_record(canonical)
            if (
                row["id"] != expected_id
                or row["generic_label"] != expected_label
                or row["normalized_text_sha256"] != expected_normalized
                or row["strict_text_sha256"] != expected_strict
            ):
                raise ValueError(f"canonical source mismatch at offset {offset}")
            row["feature_index"] = feature_index
            chunk.append(row)
            if len(chunk) == chunk_rows:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _initialize_or_resume_feature_cache(
    directory: Path,
    spec: dict,
) -> tuple[Path, Path, int, hashlib._Hash]:
    data_path = directory / FEATURE_CACHE_DATA
    state_path = directory / FEATURE_CACHE_STATE
    expected_bytes = spec["rows"] * spec["feature_width"] * np.dtype("<u2").itemsize
    if not state_path.exists() and not data_path.exists():
        with data_path.open("xb") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        state = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "spec": spec,
            "completed_rows": 0,
            "prefix_sha256": hashlib.sha256(b"").hexdigest(),
        }
        _atomic_write_json(state_path, state)
    elif not state_path.exists() or not data_path.exists():
        raise ValueError("incomplete feature cache has no resumable state")

    state = json.loads(state_path.read_text())
    if (
        state.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION
        or state.get("spec") != spec
    ):
        raise ValueError("feature cache state spec mismatch")
    completed = state.get("completed_rows")
    if type(completed) is not int or not 0 <= completed <= spec["rows"]:
        raise ValueError("feature cache completed-row state is invalid")
    if data_path.stat().st_size != expected_bytes:
        raise ValueError("partial feature cache byte size mismatch")
    prefix_bytes = completed * spec["feature_width"] * np.dtype("<u2").itemsize
    digest = _digest_file_prefix(data_path, prefix_bytes)
    if state.get("prefix_sha256") != digest.hexdigest():
        raise ValueError("partial feature cache prefix hash mismatch")
    return data_path, state_path, completed, digest


def prepare_canonical_feature_cache(
    cache_root: Path,
    spec: dict,
    *,
    connection: sqlite3.Connection,
    canonical_path: Path,
    encoder,
    tokenizer,
    chunk_rows: int,
) -> tuple[CanonicalFeatureCache, dict]:
    """Build once, resume safely, and verify before exposing cached features."""
    if chunk_rows < 1 or spec.get("feature_record_chunk") != chunk_rows:
        raise ValueError("feature cache chunk size does not match its pinned spec")
    key = _stable_json_sha256(spec)
    model_tag = re.sub(r"[^a-z0-9]+", "-", spec["model_id"].casefold()).strip("-")
    directory = cache_root / f"{model_tag}_{key[:20]}"
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / FEATURE_CACHE_REPORT
    if report_path.exists():
        artifact = validate_feature_cache_artifact(
            directory,
            expected_spec=spec,
        )
        return CanonicalFeatureCache(artifact), {
            "cache_hit": True,
            "resumed_from_rows": spec["rows"],
            "encoded_rows": 0,
        }

    lock_path = directory / ".build.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"feature cache is already being built: {directory}"
            ) from error
        if report_path.exists():
            artifact = validate_feature_cache_artifact(
                directory,
                expected_spec=spec,
            )
            return CanonicalFeatureCache(artifact), {
                "cache_hit": True,
                "resumed_from_rows": spec["rows"],
                "encoded_rows": 0,
            }

        data_path, state_path, completed, digest = _initialize_or_resume_feature_cache(
            directory, spec
        )
        resumed_from = completed
        matrix = np.memmap(
            data_path,
            mode="r+",
            dtype="<u2",
            shape=(spec["rows"], spec["feature_width"]),
        )
        try:
            chunks = _iter_canonical_source_chunks(
                connection,
                canonical_path,
                start_index=completed,
                chunk_rows=chunk_rows,
            )
            for records in chunks:
                features = extract_features(
                    encoder,
                    tokenizer,
                    records,
                    max_tokens=spec["max_tokens"],
                    token_budget=spec["token_budget"],
                    record_chunk=chunk_rows,
                )
                stored = _bfloat16_to_uint16(features)
                expected_shape = (len(records), spec["feature_width"])
                if stored.shape != expected_shape:
                    raise ValueError(
                        f"feature chunk shape mismatch: expected {expected_shape}, "
                        f"found {stored.shape}"
                    )
                start = completed
                completed += len(records)
                if completed > spec["rows"]:
                    raise ValueError("feature cache produced too many rows")
                matrix[start:completed] = stored
                matrix.flush()
                with data_path.open("r+b") as data_handle:
                    os.fsync(data_handle.fileno())
                digest.update(stored.tobytes(order="C"))
                _atomic_write_json(
                    state_path,
                    {
                        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                        "spec": spec,
                        "completed_rows": completed,
                        "prefix_sha256": digest.hexdigest(),
                    },
                )
                print(
                    f"cached {completed}/{spec['rows']} canonical feature rows",
                    flush=True,
                )
        finally:
            matrix.flush()
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if completed != spec["rows"]:
            raise ValueError(
                f"feature cache stopped at {completed} rows; expected {spec['rows']}"
            )
        data_sha256 = file_sha256(data_path)
        if data_sha256 != digest.hexdigest():
            raise ValueError("completed feature cache hash differs from streamed hash")
        report = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "spec": spec,
            "spec_sha256": key,
            "data": {
                "file": FEATURE_CACHE_DATA,
                "sha256": data_sha256,
                "bytes": data_path.stat().st_size,
                "shape": [spec["rows"], spec["feature_width"]],
                "storage_dtype": "uint16_le",
                "logical_dtype": "bfloat16",
            },
        }
        _atomic_write_json(report_path, report)
        state_path.unlink(missing_ok=True)
        artifact = validate_feature_cache_artifact(
            directory,
            expected_spec=spec,
        )
        return CanonicalFeatureCache(artifact), {
            "cache_hit": False,
            "resumed_from_rows": resumed_from,
            "encoded_rows": completed - resumed_from,
        }


class BalancedIndexCycle:
    """Deterministic class-balanced cycling for a repeatedly sampled domain."""

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


def iter_canonical_batches(
    connection: sqlite3.Connection,
    *,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
):
    """Emit cached-feature indices through a bounded deterministic shuffle."""
    if batch_size < 1 or shuffle_buffer < batch_size:
        raise ValueError("shuffle buffer must be at least one batch")
    rng = random.Random(seed)
    buffer = []
    batch = []

    def emit(record: dict):
        batch.append(record)
        if len(batch) == batch_size:
            result = list(batch)
            batch.clear()
            return result
        return None

    cursor = connection.execute(
        """
        SELECT id, generic_label, objective_weight
        FROM candidates
        ORDER BY byte_offset
        """
    )
    for feature_index, (row_id, generic_label, weight) in enumerate(cursor):
        row = {
            "id": row_id,
            "generic_label": generic_label,
            "objective_weight": weight,
            "feature_index": feature_index,
        }
        if len(buffer) < shuffle_buffer:
            buffer.append(row)
            continue
        slot = rng.randrange(len(buffer))
        outgoing = buffer[slot]
        buffer[slot] = row
        if result := emit(outgoing):
            yield result
    rng.shuffle(buffer)
    for row in buffer:
        if result := emit(row):
            yield result
    if batch:
        yield list(batch)


def _feature_labels(records: list[dict]) -> np.ndarray:
    return np.asarray([record["generic_label"] for record in records], dtype=np.int64)


def _train(
    canonical_feature_cache: CanonicalFeatureCache,
    connection: sqlite3.Connection,
    *,
    canonical_rows: int,
    promptshield_records: list[dict],
    promptshield_features,
    pair_features,
    pair_benign: np.ndarray,
    pair_attack: np.ndarray,
    validation_morgott_features,
    validation_morgott_labels: np.ndarray,
    validation_promptshield_features,
    validation_promptshield_labels: np.ndarray,
    hidden_size: int,
    seed: int,
    epochs: int,
    morgott_batch_size: int,
    promptshield_batch_size: int,
    pair_batch_pairs: int,
    objective_name: str,
    pair_ranking_weight: float,
    learning_rate: float,
    shuffle_buffer: int,
):
    import torch

    if (
        canonical_feature_cache.rows != canonical_rows
        or canonical_feature_cache.feature_width != hidden_size * 3
    ):
        raise ValueError("canonical feature cache does not match the training head")
    head = new_head(hidden_size, seed).to("cuda")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    promptshield_labels = _feature_labels(promptshield_records)
    objective = objective_spec(
        objective_name,
        canonical_rows=canonical_rows,
        promptshield_labels=promptshield_labels,
        matched_pair_rows=len(pair_benign) * 2,
    )
    if objective_name == "canonical_uniform" and pair_ranking_weight:
        raise ValueError("canonical_uniform cannot use pair ranking")
    promptshield_cycle = BalancedIndexCycle(
        promptshield_labels,
        seed=seed + 10_001,
    )
    pair_cycle = PairIndexCycle(len(pair_benign), seed=seed + 20_003)
    curve = []
    best = None
    updates = 0
    promptshield_draws = 0
    pair_draws = 0
    for epoch in range(epochs):
        losses = []
        bce_losses = []
        ranking_losses = []
        canonical_seen = 0
        head.train()
        batches = iter_canonical_batches(
            connection,
            batch_size=morgott_batch_size,
            shuffle_buffer=shuffle_buffer,
            seed=seed + epoch,
        )
        for canonical_records in batches:
            canonical_seen += len(canonical_records)
            feature_indices = np.asarray(
                [row["feature_index"] for row in canonical_records],
                dtype=np.int64,
            )
            canonical_features = canonical_feature_cache.take(feature_indices).to(
                "cuda"
            )
            canonical_targets = torch.tensor(
                [row["generic_label"] for row in canonical_records],
                dtype=torch.float32,
                device="cuda",
            )
            canonical_weights = torch.tensor(
                [row["objective_weight"] for row in canonical_records],
                dtype=torch.float32,
                device="cuda",
            )
            promptshield_indices = promptshield_cycle.take(promptshield_batch_size)
            promptshield_draws += len(promptshield_indices)
            promptshield_batch = promptshield_features[promptshield_indices].to("cuda")
            promptshield_targets = torch.from_numpy(
                promptshield_labels[promptshield_indices]
            ).to(device="cuda", dtype=torch.float32)
            selected_pairs = pair_cycle.take(pair_batch_pairs)
            pair_draws += len(selected_pairs)
            benign_batch = pair_features[pair_benign[selected_pairs]].to("cuda")
            attack_batch = pair_features[pair_attack[selected_pairs]].to("cuda")

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                canonical_logits = head(canonical_features)[:, 0]
                promptshield_logits = head(promptshield_batch)[:, 0]
                benign_logits = head(benign_batch)[:, 0]
                attack_logits = head(attack_batch)[:, 0]
                objective_losses = training_objective_loss(
                    canonical_logits=canonical_logits,
                    canonical_targets=canonical_targets,
                    canonical_weights=canonical_weights,
                    promptshield_logits=promptshield_logits,
                    promptshield_targets=promptshield_targets,
                    benign_logits=benign_logits,
                    attack_logits=attack_logits,
                    objective=objective,
                    pair_ranking_weight=pair_ranking_weight,
                )
                loss = objective_losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach().cpu()))
            bce_losses.append(float(objective_losses["domain_bce"].detach().cpu()))
            ranking_losses.append(float(objective_losses["ranking"].detach().cpu()))
        if canonical_seen != canonical_rows:
            raise ValueError(
                f"epoch {epoch + 1} saw {canonical_seen} canonical rows, "
                f"expected {canonical_rows}"
            )

        morgott_logits = predict_logits(head, validation_morgott_features)
        promptshield_logits = predict_logits(head, validation_promptshield_features)
        morgott_bce = _bce_from_logits(
            validation_morgott_labels,
            morgott_logits,
        )
        promptshield_bce = _bce_from_logits(
            validation_promptshield_labels,
            promptshield_logits,
        )
        macro_bce = 0.5 * (morgott_bce + promptshield_bce)
        curve.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(losses)),
                "mean_objective_weighted_domain_bce": float(np.mean(bce_losses)),
                "mean_pair_ranking_loss": float(np.mean(ranking_losses)),
                "canonical_rows_seen": canonical_seen,
                "validation_morgott_bce": morgott_bce,
                "validation_promptshield_bce": promptshield_bce,
                "validation_macro_bce": macro_bce,
            }
        )
        key = (macro_bce, epoch + 1)
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "epoch": epoch + 1,
                "state": {
                    name: value.detach().contiguous().cpu().clone()
                    for name, value in head.state_dict().items()
                },
            }
    expected_updates = math.ceil(canonical_rows / morgott_batch_size) * epochs
    if updates != expected_updates:
        raise ValueError(f"expected {expected_updates} updates, found {updates}")
    head.load_state_dict(best["state"])
    return head, {
        "epochs": epochs,
        "updates": updates,
        "selected_epoch": best["epoch"],
        "canonical_rows_per_epoch": canonical_rows,
        "promptshield_draws": promptshield_draws,
        "matched_pair_draws": pair_draws,
        "morgott_batch_size": morgott_batch_size,
        "promptshield_batch_size": promptshield_batch_size,
        "pair_batch_pairs": pair_batch_pairs,
        "learning_rate": learning_rate,
        "shuffle_buffer": shuffle_buffer,
        "objective": objective,
        "pair_ranking_weight": pair_ranking_weight,
        "gradient_clip_norm": 1.0,
        "loss": (
            "objective-weighted domain BCE "
            "+ pair_ranking_weight * aligned_pair_softplus"
        ),
        "checkpoint_selection": (
            "minimum equal-domain mean of matched Morgott and PromptShield "
            "validation BCE"
        ),
        "curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-dir",
        default=str(DEFAULT_SELECTION.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--model-id", default="jhu-clsp/mmBERT-base")
    parser.add_argument("--model-revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--morgott-batch-size", type=int, default=128)
    parser.add_argument("--promptshield-batch-size", type=int, default=64)
    parser.add_argument("--pair-batch-pairs", type=int, default=32)
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        required=True,
        help="required causal control; recorded verbatim in the run artifact",
    )
    parser.add_argument(
        "--pair-ranking-weight",
        type=float,
        default=0.0,
        help="run 0.0 as the BCE control before the predeclared 0.25 ablation",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--shuffle-buffer", type=int, default=8192)
    parser.add_argument(
        "--feature-cache-root",
        default=str(DEFAULT_FEATURE_CACHE),
    )
    parser.add_argument("--feature-cache-chunk-rows", type=int, default=256)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    try:
        revision = resolve_model_revision(args.model_id, args.model_revision)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.seed < 0
        or args.epochs < 1
        or args.morgott_batch_size < 1
        or args.promptshield_batch_size < 2
        or args.promptshield_batch_size % 2
        or args.pair_batch_pairs < 1
        or not math.isfinite(args.pair_ranking_weight)
        or args.pair_ranking_weight < 0
        or args.learning_rate <= 0
        or args.feature_cache_chunk_rows < 1
    ):
        parser.error("invalid training hyperparameters")
    if args.max_tokens < 2 or args.token_budget < args.max_tokens:
        parser.error("token budget must be at least max tokens")
    if args.shuffle_buffer < args.morgott_batch_size:
        parser.error("shuffle buffer must be at least the Morgott batch size")
    if args.objective == "canonical_uniform" and args.pair_ranking_weight:
        parser.error("canonical_uniform cannot use pair ranking")

    selection_dir = (REPO_ROOT / args.selection_dir).resolve()
    report_path = selection_dir / "full_selection_report.json"
    report_sha256 = file_sha256(report_path)
    report = json.loads(report_path.read_text())
    validate_selection_report(report, selection_dir=selection_dir, full=True)

    canonical_spec = report["outputs"]["morgott_train_index"]
    database_path = _resolve_repo_artifact(canonical_spec)
    canonical_path = _resolve_repo_input(report["inputs"]["canonical_train"])
    promptshield_records = _load_jsonl_spec(report["outputs"]["promptshield"])
    pair_records = _load_jsonl_spec(report["outputs"]["matched_pairs"])
    validation_morgott_records = _load_jsonl_spec(report["validation"]["morgott"])
    validation_promptshield_records = _load_jsonl_spec(
        report["validation"]["promptshield"]
    )

    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
    )
    pair_benign, pair_attack = _validate_full_inputs(
        connection,
        canonical_spec,
        promptshield_records,
        pair_records,
    )

    output_root = Path(args.output_root).resolve()
    if not output_root.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        parser.error("--output-root must be inside the artifacts directory")
    output = output_root / run_directory_name(
        args.model_id,
        objective=args.objective,
        pair_ranking_weight=args.pair_ranking_weight,
        seed=args.seed,
    )
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    feature_cache_root = Path(args.feature_cache_root).resolve()
    if not feature_cache_root.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        parser.error("--feature-cache-root must be inside the artifacts directory")

    runner_path = Path(__file__).resolve()
    head_helper_path = REPO_ROOT / "experiments/train_combined_generic_head.py"
    strict_normalizer_path = REPO_ROOT / "experiments/strict_normalize.py"
    canonical_projection_path = REPO_ROOT / "experiments/prepare_combined_generic.py"
    runner_sha256 = file_sha256(runner_path)
    head_helper_sha256 = file_sha256(head_helper_path)
    strict_normalizer_sha256 = file_sha256(strict_normalizer_path)
    canonical_projection_sha256 = file_sha256(canonical_projection_path)
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "safetensors",
            "scikit-learn",
            "torch",
            "transformers",
        )
    }
    tracked_inputs = [
        (report_path, report_sha256),
        (database_path, canonical_spec["sha256"]),
        (canonical_path, report["inputs"]["canonical_train"]["sha256"]),
        (
            (REPO_ROOT / report["outputs"]["promptshield"]["path"]).resolve(),
            report["outputs"]["promptshield"]["sha256"],
        ),
        (
            (REPO_ROOT / report["outputs"]["matched_pairs"]["path"]).resolve(),
            report["outputs"]["matched_pairs"]["sha256"],
        ),
        (
            (REPO_ROOT / report["validation"]["morgott"]["path"]).resolve(),
            report["validation"]["morgott"]["sha256"],
        ),
        (
            (REPO_ROOT / report["validation"]["promptshield"]["path"]).resolve(),
            report["validation"]["promptshield"]["sha256"],
        ),
        (runner_path, runner_sha256),
        (head_helper_path, head_helper_sha256),
        (strict_normalizer_path, strict_normalizer_sha256),
        (canonical_projection_path, canonical_projection_sha256),
    ]

    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    encoder = AutoModel.from_pretrained(
        args.model_id,
        revision=revision,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to("cuda")
    encoder.eval()
    encoder.gradient_checkpointing_disable()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    hidden_size = encoder.config.hidden_size
    feature_cache_spec = canonical_feature_cache_spec(
        selection_report_path=report_path,
        selection_report_sha256=report_sha256,
        canonical_spec=canonical_spec,
        canonical_input_spec=report["inputs"]["canonical_train"],
        model_id=args.model_id,
        model_revision=revision,
        hidden_size=hidden_size,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
        feature_record_chunk=args.feature_cache_chunk_rows,
        runner_sha256=runner_sha256,
        head_helper_sha256=head_helper_sha256,
        strict_normalizer_sha256=strict_normalizer_sha256,
        canonical_projection_sha256=canonical_projection_sha256,
        packages=packages,
    )
    canonical_feature_cache, feature_cache_activity = prepare_canonical_feature_cache(
        feature_cache_root,
        feature_cache_spec,
        connection=connection,
        canonical_path=canonical_path,
        encoder=encoder,
        tokenizer=tokenizer,
        chunk_rows=args.feature_cache_chunk_rows,
    )
    feature_cache_artifact = canonical_feature_cache.artifact
    tracked_inputs.extend(
        [
            (
                feature_cache_artifact["report_path"],
                file_sha256(feature_cache_artifact["report_path"]),
            ),
            (
                feature_cache_artifact["data_path"],
                feature_cache_artifact["report"]["data"]["sha256"],
            ),
        ]
    )
    promptshield_features = extract_features(
        encoder,
        tokenizer,
        promptshield_records,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
    )
    pair_features = extract_features(
        encoder,
        tokenizer,
        pair_records,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
    )
    validation_morgott_features = extract_features(
        encoder,
        tokenizer,
        validation_morgott_records,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
        record_chunk=VALIDATION_FEATURE_RECORD_CHUNK,
    )
    validation_promptshield_features = extract_features(
        encoder,
        tokenizer,
        validation_promptshield_records,
        max_tokens=args.max_tokens,
        token_budget=args.token_budget,
        record_chunk=VALIDATION_FEATURE_RECORD_CHUNK,
    )
    validation_morgott_labels = _feature_labels(validation_morgott_records)
    validation_promptshield_labels = _feature_labels(validation_promptshield_records)
    del encoder
    torch.cuda.empty_cache()

    try:
        head, training = _train(
            canonical_feature_cache,
            connection,
            canonical_rows=canonical_spec["rows"],
            promptshield_records=promptshield_records,
            promptshield_features=promptshield_features,
            pair_features=pair_features,
            pair_benign=pair_benign,
            pair_attack=pair_attack,
            validation_morgott_features=validation_morgott_features,
            validation_morgott_labels=validation_morgott_labels,
            validation_promptshield_features=validation_promptshield_features,
            validation_promptshield_labels=validation_promptshield_labels,
            hidden_size=hidden_size,
            seed=args.seed,
            epochs=args.epochs,
            morgott_batch_size=args.morgott_batch_size,
            promptshield_batch_size=args.promptshield_batch_size,
            pair_batch_pairs=args.pair_batch_pairs,
            objective_name=args.objective,
            pair_ranking_weight=args.pair_ranking_weight,
            learning_rate=args.learning_rate,
            shuffle_buffer=args.shuffle_buffer,
        )
    finally:
        canonical_feature_cache.close()
        connection.close()
    objective = training.pop("objective")

    morgott_logits = predict_logits(
        head,
        validation_morgott_features,
        batch_size=VALIDATION_PREDICTION_BATCH_SIZE,
    )
    promptshield_logits = predict_logits(
        head,
        validation_promptshield_features,
        batch_size=VALIDATION_PREDICTION_BATCH_SIZE,
    )
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    elapsed = time.perf_counter() - started
    morgott_scores = _scores(morgott_logits)
    promptshield_scores = _scores(promptshield_logits)
    arrays = {
        "validation_morgott_selection_scores.npy": morgott_scores,
        "validation_morgott_selection_labels.npy": validation_morgott_labels,
        "validation_promptshield_scores.npy": promptshield_scores,
        "validation_promptshield_labels.npy": validation_promptshield_labels,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_root))
    try:
        temporary_head_path = temporary / "head.safetensors"
        head_sha256 = _save_head(head, temporary_head_path)
        for name, values in arrays.items():
            np.save(temporary / name, values)

        probe = validation_morgott_features[:64]
        probe_logits = predict_logits(head, probe)
        reloaded = new_head(hidden_size, args.seed).to("cuda")
        reloaded.load_state_dict(load_file(str(temporary_head_path)))
        reloaded_logits = predict_logits(reloaded, probe)
        roundtrip_delta = float(np.max(np.abs(probe_logits - reloaded_logits)))
        if roundtrip_delta > 1e-6:
            raise ValueError(f"head roundtrip mismatch: {roundtrip_delta}")

        result = {
            "schema_version": 1,
            "purpose": (
                "artifact-only full-combined generic instruction-subversion "
                "frozen-encoder experiment"
            ),
            "generic_target": TARGET,
            "model_id": args.model_id,
            "model_revision": revision,
            "attention_implementation": "sdpa",
            "normalization": "strict",
            "feature_width": hidden_size * 3,
            "max_tokens": args.max_tokens,
            "token_budget": args.token_budget,
            "canonical_feature_record_chunk": args.feature_cache_chunk_rows,
            "validation_feature_record_chunk": VALIDATION_FEATURE_RECORD_CHUNK,
            "validation_prediction_batch_size": VALIDATION_PREDICTION_BATCH_SIZE,
            "seed": args.seed,
            "objective": objective,
            "training": training,
            "validation": {
                "morgott_selection": _binary_metrics(
                    validation_morgott_labels,
                    morgott_scores,
                ),
                "promptshield": _binary_metrics(
                    validation_promptshield_labels,
                    promptshield_scores,
                ),
            },
            "runtime": {
                "seconds": elapsed,
                "peak_reserved_bytes": peak_reserved_bytes,
                "canonical_feature_cache": feature_cache_activity,
            },
            "artifact": {
                "head": str((output / temporary_head_path.name).relative_to(REPO_ROOT)),
                "head_sha256": head_sha256,
                "roundtrip_probe_rows": len(probe),
                "roundtrip_max_abs_logit_delta": roundtrip_delta,
                "arrays": {name: file_sha256(temporary / name) for name in arrays},
            },
            "provenance": {
                "full_selection_report": str(report_path.relative_to(REPO_ROOT)),
                "full_selection_report_sha256": report_sha256,
                "runner_sha256": runner_sha256,
                "head_helper_sha256": head_helper_sha256,
                "strict_normalizer_sha256": strict_normalizer_sha256,
                "canonical_projection_sha256": canonical_projection_sha256,
                "objective_spec_sha256": _stable_json_sha256(objective),
                "canonical_feature_cache": {
                    "report": str(
                        feature_cache_artifact["report_path"].relative_to(REPO_ROOT)
                    ),
                    "report_sha256": file_sha256(feature_cache_artifact["report_path"]),
                    "data": str(
                        feature_cache_artifact["data_path"].relative_to(REPO_ROOT)
                    ),
                    "data_sha256": feature_cache_artifact["report"]["data"]["sha256"],
                    "spec_sha256": feature_cache_artifact["report"]["spec_sha256"],
                },
                "packages": packages,
            },
            "limitations": [
                (
                    f"The selected {objective['name']} objective is a development "
                    "ablation, not a production recipe."
                ),
                "No subtype or PromptShield input-channel label is inferred.",
                "PromptShield validation has only 497 negatives before joint filtering.",
                "This runner selects checkpoints on validation but does not calibrate "
                "an operating threshold.",
                "No held-out test is scored by this runner.",
                "The learned score is advisory and is not approved for blocking.",
            ],
        }
        (temporary / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        _verify_unchanged(tracked_inputs)
        if output.exists():
            raise FileExistsError(f"refusing to replace existing output: {output}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"{objective['name']} selected epoch {training['selected_epoch']}; "
        f"validation AUC morgott "
        f"{result['validation']['morgott_selection']['roc_auc']:.4f}, "
        f"PromptShield {result['validation']['promptshield']['roc_auc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
