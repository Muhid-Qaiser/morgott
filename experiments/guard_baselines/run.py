"""Score pinned third-party guard baselines on the incumbent's own row identities.

Two populations, one shared threshold protocol:

(a) the standard panel -- canonical dev_test plus PromptShield test plus SEP,
    thresholded from the same canonical calibration components the maintained
    evaluator uses, so every baseline is read at a comparable operating point;
(b) the archived first-party red-team reserve, which is positive-only and has
    zero exact overlap with any routing row, as a contamination control.

Every number here is advisory development evidence. No baseline result can
authorize a model promotion.

Usage:
    uv run --locked --extra encoder python -m experiments.guard_baselines.run \
        --baseline modernguard-1
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from morgott.models.mmbert.core import file_sha256, source_provenance
from morgott.models.mmbert.data import (
    batches,
    canonical_rows,
    external_rows,
    routing_views,
)
from morgott.models.mmbert.evaluate import (
    _REAL_FINANCE_SOURCES,
    _by_subtype,
    _by_value,
    _identity_sha256,
    _metrics,
    _pair_metrics,
    _real_finance_mask,
    _score_panel_sha256,
    _select_component_thresholds,
)
from morgott.models.mmbert.score_journal import (
    ScoreJournal,
    ScoreJournalSpec,
    require_disjoint_paths,
)
from morgott.models.mmbert.train import FULL_POPULATION, prepare_training_data

from .adapters import (
    BASELINES,
    PANEL_ORDER_BATCHING,
    RENDERED_LENGTH_BATCHING,
    ExtractionUnavailable,
    build_baseline,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
SEED = 42
SHARED_TARGET = "1.0000%"
JOURNAL_COLUMNS = ("score", "overflow")
JOURNAL_TARGET_ROWS = 512

# The git-tracked integrity anchor for the reserve is data-archive/SHA256SUMS.
# The payload itself is gitignored and lives in Azure.
REDTEAM_PATH = Path("data-archive/redteam/redteam_campaigns_20260806.jsonl.gz")
REDTEAM_SHA256 = "685e75c25509634b57dc2ccaf20e2c952873391c10fa2af0c4bc1440df76ed7e"
REDTEAM_ROWS = 5_112
REDTEAM_PULL = "scripts/azsync.sh pull data-archive"

# Seed-invariant population keys, mirroring train.py's own contract check.
PINNED_POPULATION_KEYS = (
    "canonical_rows",
    "promptshield_rows",
    "matched_pairs",
)

REDTEAM_READING = (
    "Positive-only: there is no benign denominator, so FPR, precision, AUROC, "
    "and PR-AUC are undefined here and are reported as null.",
    "The aggregate is a flag rate across two label classes, not attack recall. "
    "docs/data-contract.md treats a harmful request without instruction "
    "subversion as a separate label, and roughly two thirds of this corpus is "
    "exactly that. Read by_subversion_basis, not the aggregate.",
    "verdict and breached are outcome metadata for one target model on one "
    "day. They are never detector labels.",
    "category is confounded with attack_mode, so per-category slices measure "
    "the campaign, not the topic.",
    "Frozen evaluation reserve. This harness only reads it; it must never "
    "enter training.",
)


def _panel_sha256(slices: dict[str, list[dict]]) -> str:
    """Bind the named, ordered scoring inputs for the complete guard panel."""

    digest = hashlib.sha256()
    digest.update(b"guard-scoring-panel-v2\n")
    for name in sorted(slices):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(_score_panel_sha256(slices[name]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_input_sha256(
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    redteam: Path | None,
) -> dict[str, str | None]:
    return {
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "external_manifest_sha256": file_sha256(external_dir / "manifest.json"),
        "pair_archive_sha256": file_sha256(pairs),
        "redteam_reserve_sha256": (
            file_sha256(redteam) if redteam is not None else None
        ),
    }


def _require_unchanged_evaluation_inputs(
    expected: dict[str, str | None],
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    redteam: Path | None,
) -> None:
    if (
        _evaluation_input_sha256(
            data_dir=data_dir,
            external_dir=external_dir,
            pairs=pairs,
            redteam=redteam,
        )
        != expected
    ):
        raise ValueError("guard-baseline inputs changed during evaluation")


def _journal_panel_sha256(
    panel_sha256: str,
    label: str,
    rows: list[dict],
) -> str:
    """Bind a journal to ordered text and every metric/slicing field.

    The journal manifest receives only the resulting digest. Raw row IDs and
    prompt text are replaced with their hashes before entering the digest, and
    every other field is included so a metadata-only relabel or re-slice cannot
    silently reuse old scores.
    """
    digest = hashlib.sha256()
    digest.update(panel_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\n")
    for row in rows:
        private = {}
        for key, value in row.items():
            if key == "text":
                private["text_sha256"] = hashlib.sha256(
                    str(value).encode("utf-8")
                ).hexdigest()
            elif key == "id":
                private["id_sha256"] = _canonical_sha256(value)
            else:
                private[key] = value
        digest.update(
            json.dumps(
                private,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _journal_model_sha256(baseline) -> str:
    return _canonical_sha256(
        {
            "contract": "guard-baseline-model-v1",
            "description": baseline.describe(),
        }
    )


def _journal_scoring_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).with_name("adapters.py"),
        ROOT / "src/morgott/models/mmbert/core.py",
        ROOT / "src/morgott/models/mmbert/inference.py",
        ROOT / "src/morgott/models/mmbert/score_journal.py",
        ROOT / "src/morgott/normalization.py",
        ROOT / "uv.lock",
    )
    return _canonical_sha256(
        {
            "contract": "guard-baseline-score-and-overflow-v2",
            "sources": {
                str(path.resolve().relative_to(ROOT)): file_sha256(path)
                for path in paths
            },
        }
    )


def _batching_config(baseline) -> dict:
    batching = getattr(baseline, "batching", None)
    if callable(batching):
        config = batching()
    else:
        # Test doubles and legacy external adapters retain panel-order behavior.
        config = {
            "strategy": PANEL_ORDER_BATCHING,
            "bucket_rows": None,
            "sort_key": None,
            "restore_order_before_journal_append": False,
        }
    if not isinstance(config, dict):
        raise ValueError(f"{baseline.spec.slug} returned invalid batching metadata")
    required = {
        "strategy",
        "bucket_rows",
        "sort_key",
        "restore_order_before_journal_append",
    }
    if set(config) != required:
        raise ValueError(
            f"{baseline.spec.slug} batching metadata must contain {sorted(required)!r}"
        )
    strategy = config["strategy"]
    if strategy == PANEL_ORDER_BATCHING:
        if config != {
            "strategy": PANEL_ORDER_BATCHING,
            "bucket_rows": None,
            "sort_key": None,
            "restore_order_before_journal_append": False,
        }:
            raise ValueError(f"{baseline.spec.slug} has invalid panel-order batching")
    elif strategy == RENDERED_LENGTH_BATCHING:
        if (
            type(config["bucket_rows"]) is not int
            or config["bucket_rows"] < 1
            or config["sort_key"]
            != "exact_rendered_token_count_then_original_row_offset"
            or config["restore_order_before_journal_append"] is not True
        ):
            raise ValueError(
                f"{baseline.spec.slug} has invalid rendered-length batching"
            )
        if not callable(getattr(baseline, "prepare_for_scoring", None)) or not callable(
            getattr(baseline, "score_prepared", None)
        ):
            raise ValueError(
                f"{baseline.spec.slug} rendered-length batching requires "
                "prepare_for_scoring and score_prepared"
            )
    else:
        raise ValueError(
            f"{baseline.spec.slug} names an unknown batching strategy: {strategy!r}"
        )
    return dict(config)


def _open_score_journal(
    root: Path,
    *,
    panel_sha256: str,
    label: str,
    rows: list[dict],
    batch_size: int,
    model_sha256: str,
    scoring_sha256: str,
) -> ScoreJournal:
    return ScoreJournal(
        root / label,
        ScoreJournalSpec(
            model_sha256=model_sha256,
            panel_sha256=_journal_panel_sha256(panel_sha256, label, rows),
            scoring_sha256=scoring_sha256,
            rows=len(rows),
            batch_size=batch_size,
            columns=JOURNAL_COLUMNS,
        ),
    )


def _redteam_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"the red-team evaluation reserve is missing: {path}\n"
            f"It is gitignored and lives in Azure. Fetch it with `{REDTEAM_PULL}`, "
            "then verify with `sha256sum -c data-archive/SHA256SUMS`."
        )
    if file_sha256(path) != REDTEAM_SHA256:
        raise ValueError(
            f"red-team reserve digest mismatch: {path}\n"
            "data-archive/SHA256SUMS is the integrity anchor; re-pull with "
            f"`{REDTEAM_PULL}`."
        )
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    # Mechanically positive so the shared metric helpers apply.
                    # The honest slice is by_subversion_basis, not this label.
                    "label": 1,
                    "source": "redteam_campaigns",
                    "input_channel": row["input_channel"],
                    "group_id": row["split_group_id"],
                    "security_tags": (),
                    "prompt_kind": str(row["prompt_kind"]),
                    "attack_mode": str(row["attack_mode"]),
                    "category": str(row["category"]),
                    "subversion_basis": str(row["subversion_basis"]),
                }
            )
    if len(rows) != REDTEAM_ROWS:
        raise ValueError(f"red-team reserve row count changed: {len(rows)}")
    return rows


def _check_population(prepared, *, allow_drift: bool) -> dict:
    observed = {
        "canonical_rows": sum(prepared.canonical_counts.values()),
        "promptshield_rows": len(prepared.promptshield),
        "matched_pairs": len(prepared.pairs),
        "calibration_rows": len(prepared.calibration),
        "checkpoint_rows": len(prepared.checkpoint),
    }
    expected = {key: FULL_POPULATION[key] for key in PINNED_POPULATION_KEYS}
    drift = {
        key: {"observed": observed[key], "expected": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if drift and not allow_drift:
        raise ValueError(
            "the corpus moved under this panel; every stored comparison in "
            f"artifacts/comparisons/ is now on different rows: {drift!r}\n"
            "Re-run every baseline, or pass --allow-population-drift to record "
            "the drift and continue."
        )
    return {"observed": observed, "pinned": expected, "drift": drift or None}


def build_panel(
    *,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    redteam: Path | None,
    allow_drift: bool,
    prep_cache: Path | None = Path("artifacts/mmbert/prep-cache"),
) -> dict:
    if redteam is not None and not redteam.is_file():
        _redteam_rows(redteam)
    input_sha256 = _evaluation_input_sha256(
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        redteam=redteam,
    )
    views = routing_views(data_dir)
    external, _ = external_rows(external_dir)
    prepared = prepare_training_data(
        data_dir,
        external_dir,
        pairs,
        seed=SEED,
        cache_dir=prep_cache,
    )
    population = _check_population(prepared, allow_drift=allow_drift)
    calibration = prepared.calibration
    del prepared
    gc.collect()

    dev_path, dev_spec = views["dev_test"]
    slices = {
        "calibration": calibration,
        "canonical_dev_test": list(
            canonical_rows(dev_path, dev_spec, split="dev_test")
        ),
        "promptshield_test": external["promptshield_test"],
        "sep": external["sep"],
    }
    if any(not rows for rows in slices.values()):
        raise ValueError("a standard panel slice is empty")
    # Assert the pinned real-finance population before any model is loaded, so
    # a moved corpus fails here rather than after scoring 460k rows.
    _real_finance_mask(
        {
            "labels": np.asarray(
                [row["label"] for row in slices["canonical_dev_test"]],
                dtype=np.int8,
            ),
            "channels": np.asarray(
                [row["input_channel"] for row in slices["canonical_dev_test"]]
            ),
            "sources": np.asarray(
                [row["source"] for row in slices["canonical_dev_test"]]
            ),
        }
    )
    panel = {
        "slices": slices,
        "redteam": _redteam_rows(redteam) if redteam is not None else None,
        "views": views,
        "population": population,
        "input_sha256": input_sha256,
        "panel_sha256": _panel_sha256(slices),
        "row_identity_sha256": {
            name: _identity_sha256(rows) for name, rows in slices.items()
        },
    }
    _require_unchanged_evaluation_inputs(
        input_sha256,
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        redteam=redteam,
    )
    return panel


def score_rows(
    baseline,
    rows: list[dict],
    *,
    batch_size: int,
    label: str,
    journal: ScoreJournal | None = None,
) -> dict:
    if not rows:
        raise ValueError(f"{baseline.spec.slug} evaluation population is empty")
    started = time.perf_counter()
    resumed_rows = journal.completed_rows if journal is not None else 0
    batching = _batching_config(baseline)
    strategy = batching["strategy"]
    bucket_rows = batching["bucket_rows"]
    runtime_batching = {
        **batching,
        "model_batches_current_invocation": 0,
        "prepared_rows_current_invocation": (
            0 if strategy == RENDERED_LENGTH_BATCHING else None
        ),
        "rendered_tokens_current_invocation": (
            0 if strategy == RENDERED_LENGTH_BATCHING else None
        ),
        "padded_tokens_current_invocation": (
            0 if strategy == RENDERED_LENGTH_BATCHING else None
        ),
        "panel_order_padded_tokens_current_invocation": (
            0 if strategy == RENDERED_LENGTH_BATCHING else None
        ),
    }

    def score_panel_order(start: int, stop: int) -> np.ndarray:
        scores = []
        overflow = []
        for batch in batches(rows[start:stop], batch_size):
            values, truncated = baseline.score([row["text"] for row in batch])
            runtime_batching["model_batches_current_invocation"] += 1
            if len(values) != len(batch) or len(truncated) != len(batch):
                raise ValueError(f"{baseline.spec.slug} returned a misaligned batch")
            scores.extend(values)
            overflow.extend(truncated)
        return np.column_stack(
            (
                np.asarray(scores, dtype=np.float64),
                np.asarray(overflow, dtype=np.float64),
            )
        )

    def score_length_bucket(start: int, stop: int) -> np.ndarray:
        texts = [row["text"] for row in rows[start:stop]]
        prompt_ids, overflow = baseline.prepare_for_scoring(texts)
        if len(prompt_ids) != len(texts) or len(overflow) != len(texts):
            raise ValueError(
                f"{baseline.spec.slug} returned misaligned prepared inputs"
            )
        try:
            lengths = np.asarray([len(ids) for ids in prompt_ids], dtype=np.int64)
        except TypeError as error:
            raise ValueError(
                f"{baseline.spec.slug} returned invalid prepared input IDs"
            ) from error
        if (
            lengths.shape != (len(texts),)
            or np.any(lengths < 1)
            or np.any(lengths > baseline.spec.max_tokens)
        ):
            raise ValueError(
                f"{baseline.spec.slug} returned empty or over-cap prepared input IDs"
            )

        # A fixed contiguous bucket keeps the journal append-only and resumable.
        # The secondary key makes ties explicit rather than relying on an
        # implementation detail of the sorting algorithm.
        order = sorted(
            range(len(texts)),
            key=lambda offset: (int(lengths[offset]), offset),
        )
        scores = np.empty(len(texts), dtype=np.float64)
        for offset in range(0, len(order), batch_size):
            positions = order[offset : offset + batch_size]
            values = np.asarray(
                baseline.score_prepared([prompt_ids[index] for index in positions]),
                dtype=np.float64,
            )
            runtime_batching["model_batches_current_invocation"] += 1
            if values.shape != (len(positions),):
                raise ValueError(
                    f"{baseline.spec.slug} returned a misaligned prepared batch"
                )
            scores[positions] = values
            runtime_batching["padded_tokens_current_invocation"] += len(
                positions
            ) * int(lengths[positions].max())

        panel_order_padded = 0
        for offset in range(0, len(texts), batch_size):
            batch_lengths = lengths[offset : offset + batch_size]
            panel_order_padded += len(batch_lengths) * int(batch_lengths.max())
        runtime_batching["prepared_rows_current_invocation"] += len(texts)
        runtime_batching["rendered_tokens_current_invocation"] += int(lengths.sum())
        runtime_batching["panel_order_padded_tokens_current_invocation"] += (
            panel_order_padded
        )
        return np.column_stack(
            (
                scores,
                np.asarray(overflow, dtype=np.float64),
            )
        )

    def score_range(start: int, stop: int) -> np.ndarray:
        if strategy == PANEL_ORDER_BATCHING:
            return score_panel_order(start, stop)
        chunks = [
            score_length_bucket(bucket_start, min(bucket_start + bucket_rows, stop))
            for bucket_start in range(start, stop, bucket_rows)
        ]
        return np.concatenate(chunks, axis=0)

    if journal is None:
        values = score_range(0, len(rows))
    else:
        if (
            strategy == RENDERED_LENGTH_BATCHING
            and resumed_rows != len(rows)
            and resumed_rows % bucket_rows
        ):
            raise ValueError(
                f"{baseline.spec.slug} journal does not end on a complete "
                f"{bucket_rows}-row rendered-length bucket"
            )
        # Panel-order scoring keeps the historical model-batch-aligned shard
        # behavior. Length-aware scoring uses its fixed bucket as the shard so
        # resumption cannot change sorting or padding composition.
        shard_rows = (
            bucket_rows
            if strategy == RENDERED_LENGTH_BATCHING
            else batch_size * max(1, JOURNAL_TARGET_ROWS // batch_size)
        )
        for index, (start, stop) in enumerate(journal.missing_ranges(shard_rows)):
            journal.append(score_range(start, stop), start=start)
            if index % 25 == 0 or stop == len(rows):
                print(f"  {label}: {stop}/{len(rows)}", flush=True)
        values = journal.scores()

    scores = np.asarray(values[:, 0], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError(f"{baseline.spec.slug} produced a non-finite score")
    overflow_numeric = np.asarray(values[:, 1], dtype=np.float64)
    if not np.isin(overflow_numeric, (0.0, 1.0)).all():
        raise ValueError(f"{baseline.spec.slug} produced an invalid overflow flag")
    overflow = overflow_numeric.astype(bool)
    seconds = time.perf_counter() - started
    scored_rows = len(rows) - resumed_rows
    if strategy == RENDERED_LENGTH_BATCHING:
        rendered_tokens = runtime_batching["rendered_tokens_current_invocation"]
        padded_tokens = runtime_batching["padded_tokens_current_invocation"]
        panel_order_padded = runtime_batching[
            "panel_order_padded_tokens_current_invocation"
        ]
        runtime_batching["padding_fraction_current_invocation"] = (
            (padded_tokens - rendered_tokens) / padded_tokens if padded_tokens else None
        )
        runtime_batching["padded_token_reduction_fraction_vs_panel_order"] = (
            (panel_order_padded - padded_tokens) / panel_order_padded
            if panel_order_padded
            else None
        )
    else:
        runtime_batching["padding_fraction_current_invocation"] = None
        runtime_batching["padded_token_reduction_fraction_vs_panel_order"] = None
    return {
        "labels": np.asarray([row["label"] for row in rows], dtype=np.int8),
        "scores": scores,
        "sources": np.asarray([row["source"] for row in rows]),
        "channels": np.asarray([row["input_channel"] for row in rows]),
        "pair_ids": [row.get("pair_id") for row in rows],
        "tags": [row.get("security_tags", ()) for row in rows],
        "records": rows,
        "overflow": overflow,
        "truncation": {
            "max_input_tokens": baseline.spec.max_tokens,
            "rows": len(rows),
            "truncated_rows": int(overflow.sum()),
            "truncated_fraction": float(overflow.mean()) if len(rows) else None,
        },
        "runtime": {
            "seconds": seconds,
            "scored_rows": scored_rows,
            "resumed_rows": resumed_rows,
            "rows_per_second": (
                scored_rows / seconds if seconds and scored_rows else None
            ),
            "score_journal_identity_sha256": (
                journal.identity_sha256 if journal is not None else None
            ),
            "batching": runtime_batching,
        },
    }


def _slice_report(scored: dict, threshold: float, native: float | None) -> dict:
    report = {
        "row_identity_sha256": _identity_sha256(scored["records"]),
        "truncation": scored["truncation"],
        "runtime": scored["runtime"],
        "shared_calibration_protocol": _metrics(
            scored["labels"],
            scored["scores"],
            threshold,
        ),
    }
    if native is not None:
        report["native_cutoff"] = _metrics(scored["labels"], scored["scores"], native)
    return report


def _quantiles(scores: np.ndarray) -> dict:
    return {
        f"p{int(q * 100)}": float(np.quantile(scores, q))
        for q in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def _redteam_slices(scored: dict) -> None:
    for key in ("prompt_kind", "attack_mode", "category", "subversion_basis"):
        scored[key] = np.asarray([row[key] for row in scored["records"]])


def _redteam_report(scored: dict, threshold: float, native: float | None) -> dict:
    _redteam_slices(scored)
    report = {
        **_slice_report(scored, threshold, native),
        "population": "first-party red-team evaluation reserve, positive-only",
        "flag_rate_shared_protocol": float(
            (scored["scores"] >= threshold).mean(),
        ),
        "score_quantiles": _quantiles(scored["scores"]),
        "by_subversion_basis": _by_value(scored, "subversion_basis", threshold),
        "by_prompt_kind": _by_value(scored, "prompt_kind", threshold),
        "by_attack_mode": _by_value(scored, "attack_mode", threshold),
        "by_category": _by_value(scored, "category", threshold),
        "by_channel": _by_value(scored, "channels", threshold),
        "reading_rules": list(REDTEAM_READING),
    }
    return report


def _contamination_control(dev: dict, redteam: dict, threshold: float) -> dict:
    """Recall on already-open dev rows against recall on unpublished attacks.

    Published third-party work measured Qwen3Guard falling from 85.3% to 33.8%
    on prompts not derived from public datasets. The two populations differ in
    composition as well as provenance, so this delta is descriptive: it cannot
    separate contamination from a genuine distribution shift.
    """
    _redteam_slices(redteam)
    positives = dev["labels"] == 1
    dev_recall = float((dev["scores"][positives] >= threshold).mean())
    attested = redteam["subversion_basis"] != "None"
    subverted_recall = (
        float((redteam["scores"][attested] >= threshold).mean())
        if attested.any()
        else None
    )
    return {
        "canonical_dev_test_recall": dev_recall,
        "redteam_flag_rate": float((redteam["scores"] >= threshold).mean()),
        "redteam_subversion_attested_recall": subverted_recall,
        "redteam_subversion_attested_rows": int(attested.sum()),
        "delta_against_subversion_attested": (
            dev_recall - subverted_recall if subverted_recall is not None else None
        ),
        "interpretation": (
            "A large positive delta is consistent with the dev panel being "
            "already-open and the reserve being unpublished, but the two "
            "populations differ in composition as well as provenance, so this "
            "is descriptive and not an attribution of contamination."
        ),
    }


def _historical_reference(baseline) -> dict | None:
    """The incumbent's recorded metrics, on the corpus it was scored on then."""
    reference = baseline.spec.historical_evaluation
    if reference is None:
        return None
    path = ROOT / reference
    if not path.is_file():
        raise FileNotFoundError(f"registered evaluation is missing: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": reference,
        "sha256": file_sha256(path),
        "population_differs": True,
        "same_rows": False,
        "warning": (
            "These numbers were computed on the PREVIOUS corpus and are NOT on "
            "the rows scored above. Never place them in one table with the "
            "fresh re-score without this qualifier."
        ),
        "recorded": {
            name: recorded[name]["metrics"]
            for name in ("canonical_dev_test", "promptshield_test", "sep")
        },
        "recorded_sep_pairs": recorded["sep"]["pairs"],
        "recorded_real_finance_negatives": recorded["real_finance_negatives"][
            "metrics"
        ],
        "recorded_thresholds": recorded["thresholds"],
    }


def evaluate(
    slug: str,
    *,
    panel: dict,
    output: Path,
    batch_size: int,
    data_dir: Path,
    external_dir: Path,
    pairs: Path,
    redteam: Path | None,
    score_journal: Path | None = None,
) -> Path:
    import torch

    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if score_journal is not None:
        require_disjoint_paths(output, score_journal)
    input_sha256 = panel["input_sha256"]
    _require_unchanged_evaluation_inputs(
        input_sha256,
        data_dir=data_dir,
        external_dir=external_dir,
        pairs=pairs,
        redteam=redteam,
    )
    baseline = build_baseline(slug, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{slug}-"))
    try:
        try:
            baseline.load()
        except ExtractionUnavailable as error:
            report = _unavailable_report(baseline, error, panel)
            _write_report(temporary, report)
            _require_unchanged_evaluation_inputs(
                input_sha256,
                data_dir=data_dir,
                external_dir=external_dir,
                pairs=pairs,
                redteam=redteam,
            )
            os.replace(temporary, output)
            return output

        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model_sha256 = _journal_model_sha256(baseline)
        scoring_sha256 = _journal_scoring_sha256()

        def journal_for(name: str, rows: list[dict]) -> ScoreJournal | None:
            if score_journal is None:
                return None
            return _open_score_journal(
                score_journal,
                panel_sha256=panel["panel_sha256"],
                label=name,
                rows=rows,
                batch_size=batch_size,
                model_sha256=model_sha256,
                scoring_sha256=scoring_sha256,
            )

        scored = {
            name: score_rows(
                baseline,
                rows,
                batch_size=batch_size,
                label=name,
                journal=journal_for(name, rows),
            )
            for name, rows in panel["slices"].items()
        }
        thresholds, evidence = _select_component_thresholds(
            scored["calibration"]["scores"],
            scored["calibration"]["labels"],
            scored["calibration"]["records"],
        )
        if SHARED_TARGET not in thresholds:
            raise ValueError(f"{slug}: the one-percent component threshold is missing")
        shared = thresholds[SHARED_TARGET]
        native = baseline.spec.native_threshold
        dev = scored["canonical_dev_test"]
        finance = _real_finance_mask(dev)
        redteam_scored = (
            score_rows(
                baseline,
                panel["redteam"],
                batch_size=batch_size,
                label="redteam",
                journal=journal_for("redteam", panel["redteam"]),
            )
            if panel["redteam"] is not None
            else None
        )
        elapsed = time.perf_counter() - started

        report = {
            "schema_version": SCHEMA_VERSION,
            "purpose": "multi-baseline advisory guard-model comparison",
            "advisory_only": True,
            "promotion_authorized": False,
            "status": "scored",
            **baseline.describe(),
            "score_journal": {
                "enabled": score_journal is not None,
                "content": "numeric scores and overflow flags only",
                "model_sha256": model_sha256,
                "scoring_sha256": scoring_sha256,
                "columns": list(JOURNAL_COLUMNS),
            },
            "panel": {
                "definition": (
                    "canonical dev_test plus PromptShield test plus SEP, with "
                    "the canonical calibration components supplying the "
                    "threshold"
                ),
                "panel_sha256": panel["panel_sha256"],
                "row_identity_sha256": panel["row_identity_sha256"],
                "rows": {name: len(rows) for name, rows in panel["slices"].items()},
                "total_rows": sum(len(rows) for rows in panel["slices"].values()),
                "population_contract": panel["population"],
            },
            "thresholds": {
                "source": "same canonical calibration row identities",
                "protocol": (
                    "component-level target FPR within each trusted channel, "
                    "Bonferroni corrected, selected on calibration only"
                ),
                "shared": shared,
                "selected": thresholds,
                "evidence": evidence,
            },
            "calibration": _slice_report(scored["calibration"], shared, native),
            "canonical_dev_test": {
                **_slice_report(dev, shared, native),
                "by_source": _by_value(dev, "sources", shared),
                "by_channel": _by_value(dev, "channels", shared),
                "by_instruction_subtype": _by_subtype(dev, shared),
            },
            "promptshield_test": _slice_report(
                scored["promptshield_test"],
                shared,
                native,
            ),
            "sep": {
                **_slice_report(scored["sep"], shared, native),
                "pairs_shared_calibration_protocol": _pair_metrics(
                    scored["sep"],
                    shared,
                ),
            },
            "real_finance_negatives": {
                "sources": sorted(_REAL_FINANCE_SOURCES),
                "note": (
                    "The incumbent's best property is zero false positives "
                    "here. These rows are ordinary finance conversation, and "
                    "topic vocabulary is never itself a deny rule."
                ),
                "shared_calibration_protocol": _metrics(
                    np.zeros(int(finance.sum()), dtype=np.int8),
                    dev["scores"][finance],
                    shared,
                ),
                "by_source": {
                    source: _metrics(
                        np.zeros(
                            int((finance & (dev["sources"] == source)).sum()),
                            dtype=np.int8,
                        ),
                        dev["scores"][finance & (dev["sources"] == source)],
                        shared,
                    )
                    for source in sorted(_REAL_FINANCE_SOURCES)
                },
            },
            "redteam_reserve": (
                _redteam_report(redteam_scored, shared, native)
                if redteam_scored is not None
                else {"status": "skipped_by_flag"}
            ),
            "contamination_control": (
                _contamination_control(dev, redteam_scored, shared)
                if redteam_scored is not None
                else None
            ),
            "historical_reference": _historical_reference(baseline),
            "runtime": {
                "seconds": elapsed,
                "seconds_scope": (
                    "current invocation only; completed journal shards do not "
                    "retain timing"
                ),
                "rows": sum(len(value["labels"]) for value in scored.values())
                + (len(redteam_scored["labels"]) if redteam_scored else 0),
                "scored_rows_current_invocation": sum(
                    value["runtime"]["scored_rows"] for value in scored.values()
                )
                + (redteam_scored["runtime"]["scored_rows"] if redteam_scored else 0),
                "resumed_rows": sum(
                    value["runtime"]["resumed_rows"] for value in scored.values()
                )
                + (redteam_scored["runtime"]["resumed_rows"] if redteam_scored else 0),
                "batch_size": batch_size,
                "device": torch.cuda.get_device_name(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "by_slice": {name: value["runtime"] for name, value in scored.items()},
            },
            "truncation": {
                "max_input_tokens": baseline.spec.max_tokens,
                "note": (
                    "Context length is the variable under study, so every "
                    "slice records how many of its rows exceeded this limit."
                ),
                "by_slice": {
                    name: value["truncation"] for name, value in scored.items()
                },
            },
            "inputs": {
                **input_sha256,
                "routing_views": {
                    split: {"sha256": spec["sha256"], "rows": spec["rows"]}
                    for split, (_, spec) in panel["views"].items()
                },
            },
            "provenance": source_provenance(
                Path(__file__),
                Path(__file__).with_name("adapters.py"),
                ROOT / "src/morgott/models/mmbert/core.py",
                ROOT / "src/morgott/models/mmbert/data.py",
                ROOT / "src/morgott/models/mmbert/evaluate.py",
                ROOT / "src/morgott/models/mmbert/inference.py",
                ROOT / "src/morgott/models/mmbert/train.py",
                ROOT / "src/morgott/normalization.py",
            ),
            "limitations": [
                "These are already-open development baselines, not a "
                "prospective final test.",
                "PromptShield and SEP are published benchmarks; third-party "
                "training-source overlap is undisclosed for every baseline.",
                "Each baseline uses its own native tokenization and template, "
                "which differ from mmBERT strict normalization.",
                "No baseline result can authorize a model promotion.",
            ],
        }
        _write_arrays(temporary, scored, redteam_scored)
        report["scores"] = {
            "path": "scores.npz",
            "sha256": file_sha256(temporary / "scores.npz"),
        }
        _write_report(temporary, report)
        baseline.unload()
        gc.collect()
        torch.cuda.empty_cache()
        _require_unchanged_evaluation_inputs(
            input_sha256,
            data_dir=data_dir,
            external_dir=external_dir,
            pairs=pairs,
            redteam=redteam,
        )
        os.replace(temporary, output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _write_arrays(temporary: Path, scored: dict, redteam_scored: dict | None) -> None:
    """Persist the overflow flags too, so truncation stays re-sliceable."""
    populations = dict(scored)
    if redteam_scored is not None:
        populations["redteam"] = redteam_scored
    arrays = {}
    for name, value in populations.items():
        arrays[f"{name}_labels"] = value["labels"]
        arrays[f"{name}_scores"] = value["scores"]
        arrays[f"{name}_overflow"] = value["overflow"]
    np.savez(temporary / "scores.npz", **arrays)


def _write_report(temporary: Path, report: dict) -> None:
    (temporary / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _unavailable_report(baseline, error: ExtractionUnavailable, panel: dict) -> dict:
    """Record a baseline whose documented score path does not exist.

    A guard whose scalar cannot be extracted is a recorded gap, never a
    fabricated score or a silent omission from the ladder.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "multi-baseline advisory guard-model comparison",
        "advisory_only": True,
        "promotion_authorized": False,
        "status": "extraction_unavailable",
        "reason": str(error),
        **baseline.describe(),
        "panel": {
            "panel_sha256": panel["panel_sha256"],
            "row_identity_sha256": panel["row_identity_sha256"],
        },
        "inputs": panel["input_sha256"],
        "limitations": [
            "No score was produced. This baseline is absent from the "
            "comparison, not zero-scoring on it.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=sorted(BASELINES))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--panel-only", action="store_true")
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
        "--prep-cache",
        type=Path,
        default=Path("artifacts/mmbert/prep-cache"),
        help="verified prepared-corpus cache shared with the mmBERT trainer",
    )
    parser.add_argument(
        "--no-prep-cache",
        action="store_true",
        help="rebuild the prepared corpus instead of using the shared cache",
    )
    parser.add_argument("--redteam", type=Path, default=REDTEAM_PATH)
    parser.add_argument("--skip-redteam", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--require-panel-sha256")
    parser.add_argument("--allow-population-drift", action="store_true")
    journal_group = parser.add_mutually_exclusive_group()
    journal_group.add_argument(
        "--score-journal",
        type=Path,
        help=(
            "text-free resumable numeric journal root; defaults beside the "
            "new output directory"
        ),
    )
    journal_group.add_argument(
        "--no-score-journal",
        action="store_true",
        help="disable resumable numeric score journaling",
    )
    args = parser.parse_args()

    if args.list:
        for slug in sorted(BASELINES):
            spec = BASELINES[slug]
            print(f"{slug}\t{spec.repo_id}@{spec.revision[:12]}\t{spec.max_tokens} tok")
        return 0
    if not args.panel_only and args.baseline is None:
        parser.error("--baseline is required unless --list or --panel-only is given")

    panel = build_panel(
        data_dir=args.data_dir,
        external_dir=args.external_dir,
        pairs=args.pairs,
        redteam=None if args.skip_redteam else args.redteam,
        allow_drift=args.allow_population_drift,
        prep_cache=None if args.no_prep_cache else args.prep_cache,
    )
    if args.require_panel_sha256 and args.require_panel_sha256 != panel["panel_sha256"]:
        raise ValueError(
            "panel identity mismatch: this is not the row set the pinned "
            f"digest describes. expected {args.require_panel_sha256}, "
            f"assembled {panel['panel_sha256']}"
        )
    if args.panel_only:
        print(
            json.dumps(
                {
                    "panel_sha256": panel["panel_sha256"],
                    "row_identity_sha256": panel["row_identity_sha256"],
                    "rows": {name: len(rows) for name, rows in panel["slices"].items()},
                    "redteam_rows": (
                        len(panel["redteam"]) if panel["redteam"] else None
                    ),
                    "population_contract": panel["population"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    spec = BASELINES[args.baseline]
    batch_size = args.batch_size or spec.batch_size
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    output = args.output or Path("artifacts/comparisons") / args.baseline
    score_journal = None
    if not args.no_score_journal:
        score_journal = (
            args.score_journal or output.parent / f".{output.name}.score-journal"
        )
    print(
        evaluate(
            args.baseline,
            panel=panel,
            output=output,
            batch_size=batch_size,
            data_dir=args.data_dir,
            external_dir=args.external_dir,
            pairs=args.pairs,
            redteam=None if args.skip_redteam else args.redteam,
            score_journal=score_journal,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
