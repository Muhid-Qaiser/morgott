"""Prepare the full generic instruction-subversion training recipe.

This is deliberately separate from ``prepare_combined_generic.py``.
That script prepares an update-matched causal ablation.
This script prepares the eventual all-data recipe:

* every eligible, leakage-filtered canonical ``injection_label`` training row;
* leakage-filtered PromptShield train rows;
* leakage-filtered generated matched pairs, kept as pair atoms.

All outputs are research artifacts.
The canonical corpus is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from prepare_combined_generic import (
    DATA_DIR,
    PROMPTSHIELD_TEST,
    PROMPTSHIELD_TEST_SHA256,
    REPO_ROOT,
    SEP,
    SEP_SHA256,
    TARGET,
    LeakageIndex,
    _add_jsonl_references,
    _candidate_counts,
    _common_external_row,
    _json_bytes,
    _load_jsonl,
    _population_summary,
    _serialize_counts,
    _stream_train_candidates,
    _verify_file,
    _write_exclusion,
    _write_rows,
    canonical_record,
    create_candidate_table,
    file_sha256,
    strict_hash,
)

from morgott.data import manifest_output_path, text_hash
from morgott.overlap import NEAR_METHOD

PAIR_ARCHIVE = REPO_ROOT / "artifacts/matched_pairs/pairs_20260726T105034Z.jsonl"
PAIR_ARCHIVE_SHA256 = "8ec5c1c77b378688b190722f7d1fc51e9bef819ee9670948d2658f4a37082158"
PAIR_ARCHIVE_ROWS = 11_046
DEFAULT_BASE_SELECTION = REPO_ROOT / "artifacts/combined_generic/selection_s42"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/combined_generic/full_selection_s42"


def _verify_source_hashes(
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"source changed during run: {name}: {path}")


def validate_base_selection(base_report: dict, *, seed: int) -> None:
    """Require the base selection to have been built from current pinned inputs."""
    if (
        base_report.get("schema_version") != 2
        or base_report.get("purpose")
        != "artifact-only update-matched generic instruction-subversion experiment"
        or base_report.get("generic_target") != TARGET
        or base_report.get("seed") != seed
        or base_report["eligibility"].get("label_field") != "injection_label"
        or base_report["eligibility"].get("routing_label_used") is not False
    ):
        raise ValueError("base selection target, purpose, or seed is stale")
    expected_helpers = {
        "runner_sha256": REPO_ROOT / "experiments/prepare_combined_generic.py",
        "strict_normalizer_sha256": REPO_ROOT / "experiments/strict_normalize.py",
        "overlap_module_sha256": REPO_ROOT / "src/morgott/overlap.py",
        "canonical_text_helper_sha256": REPO_ROOT / "src/morgott/data.py",
    }
    for field, path in expected_helpers.items():
        if base_report["provenance"].get(field) != file_sha256(path):
            raise ValueError(f"base selection {field} is stale")

    manifest_path = DATA_DIR / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "path": str(manifest_path.relative_to(REPO_ROOT)),
        "sha256": manifest_hash,
    }
    if base_report["inputs"].get("manifest") != expected_manifest:
        raise ValueError("base selection manifest is stale")
    for split in ("train", "validation", "dev_test"):
        current = manifest["routing_views"][split]
        path = manifest_output_path(DATA_DIR, current)
        expected = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": current["sha256"],
            "rows": current["rows"],
        }
        if base_report["inputs"]["routing_views"].get(split) != expected:
            raise ValueError(f"base selection {split} routing view is stale")
        _verify_file(path, current["sha256"])
    for name, path, digest in (
        ("promptshield_test", PROMPTSHIELD_TEST, PROMPTSHIELD_TEST_SHA256),
        ("sep", SEP, SEP_SHA256),
    ):
        spec = base_report["inputs"].get(name)
        if (
            spec is None
            or spec.get("path") != str(path.relative_to(REPO_ROOT))
            or spec.get("sha256") != digest
        ):
            raise ValueError(f"base selection {name} input is stale")
        _verify_file(path, digest)
    promptshield_report = (
        REPO_ROOT / "artifacts/promptshield_training/filter_report.json"
    )
    promptshield_spec = base_report["inputs"].get("promptshield_filter_report")
    if (
        promptshield_spec is None
        or promptshield_spec.get("path")
        != str(promptshield_report.relative_to(REPO_ROOT))
        or promptshield_spec.get("sha256") != file_sha256(promptshield_report)
    ):
        raise ValueError("base selection PromptShield filter report is stale")


def matched_pair_records(
    pair: dict,
    *,
    pair_index: int,
    archive_sha256: str,
) -> list[dict]:
    """Project one generated pair without inventing subtype annotations."""
    required = {
        "attack",
        "attack_span",
        "benign",
        "category",
        "channel",
        "domain",
        "generator_lab",
        "generator_model",
        "label_basis",
        "language",
        "task",
        "technique",
    }
    missing = sorted(required - pair.keys())
    if missing:
        raise ValueError(f"matched pair {pair_index} is missing fields: {missing}")
    if pair["channel"] not in {"direct_user", "untrusted_content"}:
        raise ValueError(
            f"matched pair {pair_index} has unsupported channel: {pair['channel']!r}"
        )
    if pair["label_basis"] != "model_generated":
        raise ValueError(
            f"matched pair {pair_index} has unexpected label basis: "
            f"{pair['label_basis']!r}"
        )
    pair_id = f"genpair:{archive_sha256[:12]}:{pair_index}"
    rows = []
    for role, label in (("benign", 0), ("attack", 1)):
        text = pair[role]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"matched pair {pair_index} has empty {role} text")
        payload_span = None
        reported_attack_span = None
        if role == "attack":
            attack_span = pair["attack_span"]
            if attack_span is not None and not isinstance(attack_span, str):
                raise ValueError(
                    f"matched pair {pair_index} has a non-string attack span"
                )
            reported_attack_span = attack_span
            if attack_span:
                start = text.find(attack_span)
                if start >= 0:
                    payload_span = [start, start + len(attack_span)]
        rows.append(
            {
                "schema_version": 1,
                "id": f"{pair_id}:{role}",
                "text": text,
                "generic_target": TARGET,
                "generic_label": label,
                "dataset": "matched_pairs",
                "source": "matched_pairs_generated",
                "source_id": f"{pair_index}:{role}",
                "group_id": pair_id,
                "group_basis": "generated_matched_pair",
                "channel": pair["channel"],
                "channel_basis": "matched_pair_generation_spec",
                "label_basis": pair["label_basis"],
                "data_role": "train",
                "normalized_text_sha256": text_hash(text),
                "strict_text_sha256": strict_hash(text),
                "origin_sources": ["matched_pairs_generated"],
                "pair_id": pair_id,
                "pair_role": role,
                "pair_label": label,
                "pair_family": "generated_matched_instruction_subversion",
                "category": pair["category"],
                "generator_lab": pair["generator_lab"],
                "generator_model": pair["generator_model"],
                "generated_at": pair.get("generated_at"),
                "language": pair["language"],
                "task": pair["task"],
                "domain": pair["domain"],
                "technique": pair["technique"],
                "known_payload_span": payload_span is not None,
                "payload_char_span": payload_span,
                "reported_attack_span": reported_attack_span,
            }
        )
    if rows[0]["strict_text_sha256"] == rows[1]["strict_text_sha256"]:
        raise ValueError(f"matched pair {pair_index} has identical strict text")
    return rows


def admit_pair_atom(
    records: list[dict],
    index: LeakageIndex,
    *,
    dataset: str,
) -> tuple[bool, list[dict | None]]:
    """Admit both pair halves or neither, then index both only on success."""
    if (
        len(records) != 2
        or {row.get("pair_role") for row in records} != {"benign", "attack"}
        or len({row.get("pair_id") for row in records}) != 1
    ):
        raise ValueError("pair atom must contain one benign and one attack half")
    matches = [index.match(row) for row in records]
    if any(match is not None for match in matches):
        return False, matches
    for row in records:
        index.add(row, dataset=dataset)
    return True, matches


def add_group_balanced_weights(connection: sqlite3.Connection) -> dict:
    """Add weights that equalize label, source, and lineage-group mass."""
    labels = connection.execute(
        "SELECT COUNT(DISTINCT generic_label) FROM candidates"
    ).fetchone()[0]
    rows = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    if labels != 2 or rows == 0:
        raise ValueError(
            f"canonical candidates require two non-empty labels, found {labels}"
        )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
    if "objective_weight" in columns:
        raise ValueError("candidate table already has objective weights")
    connection.execute("ALTER TABLE candidates ADD COLUMN objective_weight REAL")
    connection.executescript(
        """
        CREATE TEMP TABLE group_stats AS
        SELECT generic_label, source, group_id, COUNT(*) AS group_rows
        FROM candidates
        GROUP BY generic_label, source, group_id;

        CREATE UNIQUE INDEX temp.group_stats_key
        ON group_stats(generic_label, source, group_id);

        CREATE TEMP TABLE source_stats AS
        SELECT generic_label, source, COUNT(*) AS source_groups
        FROM group_stats
        GROUP BY generic_label, source;

        CREATE UNIQUE INDEX temp.source_stats_key
        ON source_stats(generic_label, source);

        CREATE TEMP TABLE label_stats AS
        SELECT generic_label, COUNT(*) AS label_sources
        FROM source_stats
        GROUP BY generic_label;

        CREATE UNIQUE INDEX temp.label_stats_key
        ON label_stats(generic_label);

        CREATE TEMP TABLE candidate_weights (
            id TEXT PRIMARY KEY,
            objective_weight REAL NOT NULL
        ) WITHOUT ROWID;
        """
    )
    connection.execute(
        """
        INSERT INTO candidate_weights(id, objective_weight)
        SELECT c.id,
               CAST(? AS REAL) /
               (? * ls.label_sources * ss.source_groups * gs.group_rows)
        FROM candidates AS c
        JOIN group_stats AS gs
          ON gs.generic_label = c.generic_label
         AND gs.source = c.source
         AND gs.group_id = c.group_id
        JOIN source_stats AS ss
          ON ss.generic_label = c.generic_label
         AND ss.source = c.source
        JOIN label_stats AS ls
          ON ls.generic_label = c.generic_label
        """,
        (rows, labels),
    )
    connection.execute(
        """
        UPDATE candidates
        SET objective_weight = (
            SELECT objective_weight
            FROM candidate_weights
            WHERE candidate_weights.id = candidates.id
        )
        """
    )
    missing = connection.execute(
        "SELECT COUNT(*) FROM candidates WHERE objective_weight IS NULL"
    ).fetchone()[0]
    if missing:
        raise ValueError(f"{missing} candidates are missing objective weights")
    connection.execute("CREATE INDEX candidates_byte_offset ON candidates(byte_offset)")
    connection.execute(
        "CREATE INDEX candidates_normalized_text ON candidates(normalized_text_sha256)"
    )
    connection.commit()
    weight_sum, minimum, maximum = connection.execute(
        """
        SELECT SUM(objective_weight),
               MIN(objective_weight),
               MAX(objective_weight)
        FROM candidates
        """
    ).fetchone()
    if abs(weight_sum - rows) > max(1e-8, rows * 1e-10):
        raise ValueError(f"objective weights sum to {weight_sum}, expected {rows}")
    return {
        "rows": rows,
        "labels": labels,
        "sum": weight_sum,
        "minimum": minimum,
        "maximum": maximum,
        "definition": (
            "N / (number_of_labels * sources_in_label * "
            "groups_in_label_source * rows_in_group)"
        ),
    }


def _artifact_from_report(base_dir: Path, spec: dict) -> Path:
    path = (REPO_ROOT / spec["path"]).resolve()
    if not path.is_relative_to(base_dir):
        raise ValueError(f"base-selection artifact escapes its directory: {path}")
    _verify_file(path, spec["sha256"])
    return path


def _refilter_generic_rows(
    rows: list[dict],
    index: LeakageIndex,
    *,
    dataset: str,
    exclusions,
    exclusion_counts: Counter,
) -> list[dict]:
    kept = []
    for row in rows:
        match = index.match(row)
        if match is None:
            kept.append(row)
            continue
        exclusion_counts[(dataset, match["reason"], match["reference_dataset"])] += 1
        _write_exclusion(
            exclusions,
            dataset=dataset,
            row=row,
            match=match,
        )
    for row in kept:
        index.add(row, dataset=dataset)
    return kept


def _index_canonical_candidates(
    index: LeakageIndex,
    connection: sqlite3.Connection,
    canonical_path: Path,
) -> int:
    indexed = 0
    cursor = connection.execute(
        """
        SELECT id, generic_label, normalized_text_sha256,
               strict_text_sha256, byte_offset
        FROM candidates
        ORDER BY byte_offset
        """
    )
    with canonical_path.open("rb") as source:
        for (
            expected_id,
            expected_label,
            expected_normalized,
            expected_strict,
            offset,
        ) in cursor:
            source.seek(offset)
            canonical = json.loads(source.readline())
            row = canonical_record(canonical)
            if (
                row["id"] != expected_id
                or row["generic_label"] != expected_label
                or row["normalized_text_sha256"] != expected_normalized
                or row["strict_text_sha256"] != expected_strict
            ):
                raise ValueError(f"canonical index mismatch at offset {offset}")
            index.add(row, dataset="morgott_train")
            indexed += 1
    return indexed


def _write_filtered_pairs(
    output_path: Path,
    exclusions,
    index: LeakageIndex,
    *,
    exclusion_counts: Counter,
) -> tuple[int, Counter]:
    _verify_file(PAIR_ARCHIVE, PAIR_ARCHIVE_SHA256)
    pairs_kept = 0
    row_counts = Counter()
    rows_seen = 0
    with (
        PAIR_ARCHIVE.open(encoding="utf-8") as source,
        output_path.open("wb") as destination,
    ):
        for pair_index, line in enumerate(source):
            if not line.strip():
                continue
            rows_seen += 1
            records = matched_pair_records(
                json.loads(line),
                pair_index=pair_index,
                archive_sha256=PAIR_ARCHIVE_SHA256,
            )
            admitted, matches = admit_pair_atom(
                records,
                index,
                dataset="matched_pairs_generated",
            )
            if not admitted:
                actual_match = next(match for match in matches if match is not None)
                for row, match in zip(records, matches, strict=True):
                    resolved = match or {
                        "reason": "paired_half_excluded",
                        "reference_dataset": actual_match["reference_dataset"],
                    }
                    exclusion_counts[
                        (
                            "matched_pairs_generated",
                            resolved["reason"],
                            resolved["reference_dataset"],
                        )
                    ] += 1
                    _write_exclusion(
                        exclusions,
                        dataset="matched_pairs_generated",
                        row=row,
                        match=resolved,
                    )
                continue
            for row in records:
                destination.write(_json_bytes(row))
                row_counts[
                    (
                        row["generic_label"],
                        row["channel"],
                        row["category"],
                        row["generator_lab"],
                        row["known_payload_span"],
                    )
                ] += 1
            pairs_kept += 1
    if rows_seen != PAIR_ARCHIVE_ROWS:
        raise ValueError(
            f"matched-pair archive rows: expected {PAIR_ARCHIVE_ROWS}, "
            f"found {rows_seen}"
        )
    return pairs_kept, row_counts


def _database_summary(path: Path, connection: sqlite3.Connection) -> dict:
    rows = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    labels = {
        str(label): count
        for label, count in connection.execute(
            """
            SELECT generic_label, COUNT(*)
            FROM candidates
            GROUP BY generic_label
            ORDER BY generic_label
            """
        )
    }
    sources = {
        source: count
        for source, count in connection.execute(
            """
            SELECT source, COUNT(*)
            FROM candidates
            GROUP BY source
            ORDER BY source
            """
        )
    }
    channels = {
        str(channel): count
        for channel, count in connection.execute(
            """
            SELECT channel, COUNT(*)
            FROM candidates
            GROUP BY channel
            ORDER BY channel
            """
        )
    }
    groups = connection.execute(
        "SELECT COUNT(DISTINCT group_id) FROM candidates"
    ).fetchone()[0]
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "table": "candidates",
        "rows": rows,
        "labels": labels,
        "sources": sources,
        "channels": channels,
        "unique_groups": groups,
        "text_storage": "byte offsets into the hash-pinned canonical train view",
    }


def _build(
    output: Path,
    *,
    published_output: Path,
    base_selection: Path,
    seed: int,
    provenance: dict[str, str],
) -> None:
    base_report_path = base_selection / "selection_report.json"
    base_report = json.loads(base_report_path.read_text())
    validate_base_selection(base_report, seed=seed)
    base_specs = base_report["outputs"]
    validation_partition = base_report.get("validation_partition", {})
    component_calibration = validation_partition.get("component_calibration", {})
    if (
        validation_partition.get("target_checkpoint_fraction") != 0.2
        or validation_partition.get("promptshield_used_for_threshold") is not False
        or "validation_morgott_calibration" not in base_specs
        or component_calibration.get("component_id_field") != "validation_component_id"
        or component_calibration.get("family_confidence") != 0.95
        or component_calibration.get("per_channel_confidence") != 0.975
        or component_calibration.get("multiplicity_correction") != "Bonferroni"
        or component_calibration.get("family_scope")
        != "the two trusted channels, with a separate family for each target"
        or component_calibration.get("pooled_negative_role")
        != "empirical diagnostic only"
    ):
        raise ValueError("base selection validation partition is stale")
    promptshield_train_path = _artifact_from_report(
        base_selection,
        base_specs["promptshield"],
    )
    promptshield_validation_path = _artifact_from_report(
        base_selection,
        base_specs["validation_promptshield"],
    )
    _verify_file(
        promptshield_validation_path,
        base_specs["validation_promptshield"]["sha256"],
    )
    for name in (
        "validation_morgott_selection",
        "validation_morgott_calibration",
    ):
        spec = base_specs[name]
        path = _artifact_from_report(base_selection, spec)
        _verify_file(path, spec["sha256"])
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        if rows != spec["rows"]:
            raise ValueError(f"base selection {name} row count is stale")
    promptshield_train = _load_jsonl(
        promptshield_train_path,
        base_specs["promptshield"]["sha256"],
    )

    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical_paths = {}
    for split in ("train", "validation", "dev_test"):
        spec = manifest["routing_views"][split]
        path = manifest_output_path(DATA_DIR, spec)
        _verify_file(path, spec["sha256"])
        canonical_paths[split] = path
    promptshield_test = _load_jsonl(
        PROMPTSHIELD_TEST,
        PROMPTSHIELD_TEST_SHA256,
    )
    sep_rows = _load_jsonl(SEP, SEP_SHA256)

    output.mkdir(exist_ok=True)
    exclusions_path = output / "exclusions.jsonl"
    exclusion_counts = Counter()
    references = Counter()
    index = LeakageIndex()

    print("indexing held-out and validation references", flush=True)
    references["morgott_dev_test"] = _add_jsonl_references(
        index,
        canonical_paths["dev_test"],
        dataset="morgott_dev_test",
        text_field="text",
        canonical=True,
    )
    for row in promptshield_test:
        index.add(
            _common_external_row(row, text_field="prompt"),
            dataset="promptshield_test",
        )
    references["promptshield_test"] = len(promptshield_test)
    for row in sep_rows:
        index.add(
            _common_external_row(row, text_field="text"),
            dataset="sep",
        )
    references["sep"] = len(sep_rows)
    references["morgott_validation"] = _add_jsonl_references(
        index,
        canonical_paths["validation"],
        dataset="morgott_validation",
        text_field="text",
        canonical=True,
    )
    references["promptshield_validation"] = _add_jsonl_references(
        index,
        promptshield_validation_path,
        dataset="promptshield_validation",
        text_field="text",
        canonical=True,
    )
    database_path = output / "morgott_train_index.sqlite"
    connection = sqlite3.connect(database_path)
    create_candidate_table(connection)
    with exclusions_path.open("wb") as exclusions:
        print(
            "refiltering base PromptShield train against current references", flush=True
        )
        promptshield_train = _refilter_generic_rows(
            promptshield_train,
            index,
            dataset="promptshield_train",
            exclusions=exclusions,
            exclusion_counts=exclusion_counts,
        )
        promptshield_path = output / "promptshield.jsonl"
        _write_rows(promptshield_path, promptshield_train)
        references["promptshield_train"] = len(promptshield_train)

        print("filtering every eligible canonical train row", flush=True)
        _stream_train_candidates(
            canonical_paths["train"],
            connection,
            index,
            seed=seed,
            exclusions=exclusions,
            exclusion_counts=exclusion_counts,
        )
        train_available = _candidate_counts(connection)
        weight_summary = add_group_balanced_weights(connection)

        print("indexing retained canonical train rows", flush=True)
        references["morgott_train"] = _index_canonical_candidates(
            index,
            connection,
            canonical_paths["train"],
        )

        print("filtering generated pairs as two-row atoms", flush=True)
        pairs_path = output / "matched_pairs.jsonl"
        pairs_kept, pair_counts = _write_filtered_pairs(
            pairs_path,
            exclusions,
            index,
            exclusion_counts=exclusion_counts,
        )

    connection.execute("VACUUM")
    connection.close()
    database_connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
    )
    canonical_spec = _database_summary(database_path, database_connection)
    database_connection.close()
    pair_spec = _population_summary(
        pairs_path,
        published_path=published_output / pairs_path.name,
    )
    promptshield_spec = _population_summary(
        promptshield_path,
        published_path=published_output / promptshield_path.name,
    )
    if pair_spec["rows"] != pairs_kept * 2:
        raise ValueError("matched-pair output lost pair atomicity")
    exclusions_spec = {
        "path": str((published_output / exclusions_path.name).relative_to(REPO_ROOT)),
        "sha256": file_sha256(exclusions_path),
        "rows": sum(exclusion_counts.values()),
        "by_candidate_reason_reference": _serialize_counts(exclusion_counts),
    }
    report = {
        "schema_version": 2,
        "purpose": (
            "artifact-only full generic instruction-subversion training recipe"
        ),
        "generic_target": TARGET,
        "seed": seed,
        "canonical_corpus_modified": False,
        "eligibility": {
            "canonical_pool": "all retained rows after eligibility and leakage filters",
            "routing_training_eligible": True,
            "input_channel": ["direct_user", "untrusted_content"],
            "label_field": "injection_label",
            "labels": [0, 1],
            "routing_label_used": False,
            "promptshield_subtypes_assigned": False,
            "matched_pairs_are_weak_supervision": True,
        },
        "training_recipe": {
            "domains": [
                "morgott_canonical",
                "promptshield_train",
                "matched_pairs_generated",
            ],
            "pair_atoms_preserved": True,
            "pair_ranking_capable": True,
            "canonical_objective_weights": weight_summary,
            "predeclared_objectives": {
                "canonical_uniform": "Morgott-only uniform-row BCE control",
                "full_uniform": (
                    "all-domain BCE weighted by each domain's unique-row share"
                ),
                "full_balanced": (
                    "equal-domain BCE with label/source/group-balanced Morgott "
                    "and class-balanced PromptShield"
                ),
            },
            "comparison_protocol": {
                "bce_controls_first": [
                    "canonical_uniform",
                    "full_uniform",
                    "full_balanced",
                ],
                "pair_ranking_after_best_full_data_bce_objective": True,
                "pair_ranking_weight": 0.25,
            },
        },
        "leakage": {
            "precedence": [
                "morgott_dev_test+promptshield_test+sep",
                "promptshield_validation+morgott_validation",
                "promptshield_train",
                "morgott_train",
                "matched_pairs_generated",
            ],
            "pair_exclusion_is_atomic": True,
            "normalization_exact": (
                "morgott.data.normalize_text SHA-256 or canonical precomputed hash"
            ),
            "strict_exact": "experiments.strict_normalize SHA-256",
            "near": NEAR_METHOD,
            "references": dict(sorted(references.items())),
            "exclusions": exclusions_spec,
        },
        "available_after_filter": {
            "canonical_by_label_source": _serialize_counts(train_available),
            "matched_pair_rows_by_label_channel_category_lab_known_span": (
                _serialize_counts(pair_counts)
            ),
        },
        "inputs": {
            "base_update_matched_selection": {
                "path": str(base_report_path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(base_report_path),
            },
            "manifest": {
                "path": str(manifest_path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(manifest_path),
            },
            "canonical_train": {
                "path": str(canonical_paths["train"].relative_to(REPO_ROOT)),
                "sha256": manifest["routing_views"]["train"]["sha256"],
                "rows": manifest["routing_views"]["train"]["rows"],
            },
            "routing_views": {
                split: {
                    "path": str(canonical_paths[split].relative_to(REPO_ROOT)),
                    "sha256": manifest["routing_views"][split]["sha256"],
                    "rows": manifest["routing_views"][split]["rows"],
                }
                for split in ("train", "validation", "dev_test")
            },
            "promptshield_train": base_specs["promptshield"],
            "promptshield_validation": base_specs["validation_promptshield"],
            "matched_pairs": {
                "path": str(PAIR_ARCHIVE.relative_to(REPO_ROOT)),
                "sha256": PAIR_ARCHIVE_SHA256,
                "pairs": PAIR_ARCHIVE_ROWS,
            },
            "promptshield_test": {
                "path": str(PROMPTSHIELD_TEST.relative_to(REPO_ROOT)),
                "sha256": PROMPTSHIELD_TEST_SHA256,
                "rows": len(promptshield_test),
            },
            "sep": {
                "path": str(SEP.relative_to(REPO_ROOT)),
                "sha256": SEP_SHA256,
                "rows": len(sep_rows),
            },
        },
        "outputs": {
            "morgott_train_index": {
                **canonical_spec,
                "path": str(
                    (published_output / database_path.name).relative_to(REPO_ROOT)
                ),
            },
            "promptshield": promptshield_spec,
            "matched_pairs": pair_spec,
        },
        "validation": {
            "selection_report": str(base_report_path.relative_to(REPO_ROOT)),
            "morgott": base_specs["validation_morgott_selection"],
            "morgott_calibration": base_specs["validation_morgott_calibration"],
            "promptshield": base_specs["validation_promptshield"],
            "checkpoint_selection_only": ["morgott", "promptshield"],
            "threshold_calibration_only": "morgott_calibration",
            "promptshield_used_for_threshold": False,
            "component_calibration": component_calibration,
        },
        "provenance": provenance,
        "limitations": [
            "PromptShield publishes no row-level source-family or channel lineage.",
            "Generated pairs are model-generated weak supervision.",
            "The SimHash near-overlap check is conservative and not exhaustive.",
            "Validation components reduce known lineage and near-overlap "
            "dependence, but components and recurring source families are not IID "
            "or sampled from a deployment distribution.",
            "No held-out test is scored or used by this preparation step.",
            "The learned score remains advisory and is not approved for blocking.",
        ],
    }
    (output / "full_selection_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-selection",
        default=str(DEFAULT_BASE_SELECTION.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    base_selection = (REPO_ROOT / args.base_selection).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else REPO_ROOT / f"artifacts/combined_generic/full_selection_s{args.seed}"
    )
    artifacts_root = (REPO_ROOT / "artifacts").resolve()
    if not base_selection.is_relative_to(artifacts_root):
        parser.error("--base-selection must be inside the artifacts directory")
    if not output.is_relative_to(artifacts_root):
        parser.error("--output must be inside the artifacts directory")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    source_paths = {
        "runner_sha256": Path(__file__).resolve(),
        "base_preparation_runner_sha256": (
            REPO_ROOT / "experiments/prepare_combined_generic.py"
        ),
        "strict_normalizer_sha256": (REPO_ROOT / "experiments/strict_normalize.py"),
        "overlap_module_sha256": REPO_ROOT / "src/morgott/overlap.py",
        "canonical_text_helper_sha256": REPO_ROOT / "src/morgott/data.py",
    }
    provenance = {name: file_sha256(path) for name, path in source_paths.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _build(
            temporary,
            published_output=output,
            base_selection=base_selection,
            seed=args.seed,
            provenance=provenance,
        )
        _verify_source_hashes(source_paths, provenance)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    report = json.loads((output / "full_selection_report.json").read_text())
    print(
        f"wrote {output}; "
        f"{report['outputs']['morgott_train_index']['rows']} canonical rows and "
        f"{report['outputs']['matched_pairs']['rows'] // 2} matched pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
