"""Prepare update-matched generic instruction-subversion experiment artifacts.

This artifact-only recipe never modifies the canonical corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from strict_normalize import strict_normalize

from morgott.data import manifest_output_path, text_hash
from morgott.overlap import NEAR_METHOD, NearIndex

TARGET = "instruction_subversion"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROMPTSHIELD_DIR = REPO_ROOT / "artifacts/promptshield_training"
PROMPTSHIELD_TEST = REPO_ROOT / "artifacts/external_eval_data/promptshield/test.jsonl"
PROMPTSHIELD_TEST_SHA256 = (
    "c763dcde8cc9921613476887b43f12917229d1e5e6cfa29c07ee5dc36311abf6"
)
SEP = REPO_ROOT / "artifacts/external_eval_data/sep/sep.jsonl"
SEP_SHA256 = "0ddcfa5a7963f65f9fc8fdf63af10b9052685f87f0142c243a42a394d6e31a89"


def strict_hash(text: str) -> str:
    return hashlib.sha256(strict_normalize(text).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source_hashes(
    paths: dict[str, Path],
    expected: dict[str, str],
) -> None:
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"source changed during run: {name}: {path}")


def _verify_file(path: Path, expected_sha256: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"input hash mismatch for {path}: expected {expected_sha256}, "
            f"found {actual}"
        )


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict]:
    _verify_file(path, expected_sha256)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _common_external_row(row: dict, *, text_field: str) -> dict:
    text = row[text_field]
    return {
        "id": row["id"],
        "text": text,
        "label": row.get("label"),
        "source": row.get("source"),
        "normalized_text_sha256": (
            row.get("normalized_text_sha256") or text_hash(text)
        ),
    }


def _origins(row: dict) -> list[dict]:
    origins = row.get("origins")
    return origins if isinstance(origins, list) and origins else [row]


def _weak_origin(origin: dict) -> bool:
    basis = str(origin.get("label_basis", "")).casefold()
    return "weak" in basis or "unverified" in basis


def canonical_is_eligible(row: dict) -> bool:
    """Return whether a canonical row has supported generic injection evidence."""
    return (
        row.get("routing_training_eligible") is True
        and row.get("input_channel") in {"direct_user", "untrusted_content"}
        and type(row.get("injection_label")) is int
        and row["injection_label"] in {0, 1}
        and row.get("security_label") != "uncertain"
        and not all(_weak_origin(origin) for origin in _origins(row))
    )


def canonical_record(row: dict) -> dict:
    if not canonical_is_eligible(row):
        raise ValueError(f"ineligible canonical row: {row.get('id')}")
    origins = _origins(row)
    return {
        "schema_version": 1,
        "id": row["id"],
        "text": row["text"],
        "generic_target": TARGET,
        "generic_label": row["injection_label"],
        "dataset": "morgott",
        "source": row["source"],
        "source_id": row.get("source_id"),
        "group_id": row["split_group_id"],
        "channel": row["input_channel"],
        "channel_basis": "trusted_corpus_metadata",
        "label_basis": row.get("label_basis"),
        "data_role": row.get("data_role"),
        "normalized_text_sha256": row["normalized_text_sha256"],
        "strict_text_sha256": strict_hash(row["text"]),
        "origin_sources": sorted(
            {
                origin["source"]
                for origin in origins
                if isinstance(origin.get("source"), str)
            }
        ),
    }


def promptshield_record(row: dict, *, data_role: str = "train") -> dict:
    label = row.get("label")
    if type(label) is not int or label not in {0, 1}:
        raise ValueError(f"invalid PromptShield label: {label!r}")
    text = row["prompt"]
    return {
        "schema_version": 1,
        "id": row["id"],
        "text": text,
        "generic_target": TARGET,
        "generic_label": label,
        "dataset": "promptshield",
        "source": "promptshield",
        "source_id": row["id"],
        "group_id": row["id"],
        "group_basis": "row_id_no_published_lineage",
        "channel": None,
        "channel_basis": "not_published",
        "subtype_training_eligible": False,
        "label_basis": "promptshield_binary_source_label",
        "data_role": data_role,
        "normalized_text_sha256": text_hash(text),
        "strict_text_sha256": strict_hash(text),
        "origin_sources": ["promptshield"],
    }


def _stable_rank(seed: int, *parts: object) -> int:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def create_candidate_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE candidates (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            group_id TEXT NOT NULL,
            generic_label INTEGER NOT NULL CHECK (generic_label IN (0, 1)),
            channel TEXT,
            normalized_text_sha256 TEXT NOT NULL,
            strict_text_sha256 TEXT NOT NULL UNIQUE,
            source_rank INTEGER NOT NULL,
            group_rank INTEGER NOT NULL,
            row_rank INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX candidates_label_source "
        "ON candidates(generic_label, source, group_id)"
    )
    connection.execute(
        """
        CREATE TABLE strict_conflicts (
            strict_text_sha256 TEXT PRIMARY KEY
        )
        """
    )


def insert_candidate(
    connection: sqlite3.Connection,
    row: dict,
    *,
    offset: int,
    seed: int,
) -> dict:
    label = row["generic_label"]
    source = row["source"]
    group_id = row["group_id"]
    if type(label) is not int or label not in {0, 1}:
        raise ValueError(f"invalid generic label: {label!r}")
    strict_text_sha256 = row["strict_text_sha256"]
    conflict = connection.execute(
        """
        SELECT 1
        FROM strict_conflicts
        WHERE strict_text_sha256 = ?
        """,
        (strict_text_sha256,),
    ).fetchone()
    if conflict is not None:
        return {"status": "conflict", "removed_id": None}
    existing = connection.execute(
        """
        SELECT id, generic_label, normalized_text_sha256
        FROM candidates
        WHERE strict_text_sha256 = ?
        """,
        (strict_text_sha256,),
    ).fetchone()
    if existing is not None:
        existing_id, existing_label, existing_normalized = existing
        if existing_label == label:
            return {
                "status": "duplicate",
                "retained_id": existing_id,
            }
        connection.execute(
            "DELETE FROM candidates WHERE strict_text_sha256 = ?",
            (strict_text_sha256,),
        )
        connection.execute(
            "INSERT INTO strict_conflicts (strict_text_sha256) VALUES (?)",
            (strict_text_sha256,),
        )
        return {
            "status": "conflict",
            "removed_id": existing_id,
            "removed_normalized_text_sha256": existing_normalized,
        }

    connection.execute(
        """
        INSERT INTO candidates (
            id, source, group_id, generic_label, channel,
            normalized_text_sha256, strict_text_sha256, source_rank,
            group_rank, row_rank, byte_offset
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            source,
            group_id,
            label,
            row.get("channel"),
            row["normalized_text_sha256"],
            strict_text_sha256,
            _stable_rank(seed, "source", label, source),
            _stable_rank(seed, "group", label, source, group_id),
            _stable_rank(
                seed,
                "row",
                label,
                source,
                group_id,
                row["normalized_text_sha256"],
                row["id"],
            ),
            offset,
        ),
    )
    return {"status": "inserted"}


def _partition_source_balanced(
    rows: list[dict],
    *,
    count: int,
    label: int,
    seed: int,
    group_owners: dict[tuple[str, str], str],
) -> tuple[list[dict], list[dict]]:
    by_side_source = {
        "m1": defaultdict(list),
        "m2": defaultdict(list),
    }
    for row in rows:
        side = group_owners[(row["source"], row["group_id"])]
        by_side_source[side][row["source"]].append(row)
    selected = {"m1": [], "m2": []}
    for side in ("m1", "m2"):
        sources = sorted(
            by_side_source[side],
            key=lambda source: (
                _stable_rank(seed, "partition-source", label, side, source),
                source,
            ),
        )
        positions = {source: 0 for source in sources}
        while len(selected[side]) < count:
            progressed = False
            for source in sources:
                position = positions[source]
                source_rows = by_side_source[side][source]
                if position == len(source_rows):
                    continue
                selected[side].append(source_rows[position])
                positions[source] += 1
                progressed = True
                if len(selected[side]) == count:
                    break
            if not progressed:
                break
    m1 = selected["m1"]
    m2 = selected["m2"]
    if len(m1) != count or len(m2) != count:
        raise ValueError(
            f"could not group-disjoint partition label {label}: "
            f"expected {count} per half, found {len(m1)} and {len(m2)}"
        )
    return m1, m2


def _candidate_pool(
    connection: sqlite3.Connection,
    *,
    per_label: dict[int, int],
) -> dict[int, list[dict]]:
    candidate_rows = {}
    for label in (0, 1):
        count = per_label[label]
        cursor = connection.execute(
            """
            WITH per_group AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY generic_label, source, group_id
                        ORDER BY row_rank, id
                    ) AS group_position
                FROM candidates
                WHERE generic_label = ?
            ),
            per_source AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY generic_label, source
                        ORDER BY group_position, group_rank, row_rank, id
                    ) AS source_position
                FROM per_group
            )
            SELECT id, source, group_id, generic_label, channel,
                   normalized_text_sha256, byte_offset,
                   source_position, group_position, source_rank, group_rank
            FROM per_source
            ORDER BY source_position, source_rank, group_position,
                     group_rank, row_rank, id
            LIMIT ?
            """,
            (label, count * 8),
        )
        columns = [value[0] for value in cursor.description]
        rows = [dict(zip(columns, values, strict=True)) for values in cursor]
        if len(rows) < count * 2:
            raise ValueError(
                f"insufficient canonical label {label} rows: "
                f"needed {count * 2}, found {len(rows)}"
            )
        candidate_rows[label] = rows
    return candidate_rows


def _group_owners(
    candidate_rows: dict[int, list[dict]],
    *,
    seed: int,
) -> dict[tuple[str, str], str]:

    groups_by_source = defaultdict(set)
    for rows in candidate_rows.values():
        for row in rows:
            groups_by_source[row["source"]].add(row["group_id"])
    group_owners = {}
    for source, groups in sorted(groups_by_source.items()):
        ordered = sorted(
            groups,
            key=lambda group_id: (
                _stable_rank(seed, "group-owner", source, group_id),
                group_id,
            ),
        )
        swap = _stable_rank(seed, "group-owner-swap", source) % 2
        for index, group_id in enumerate(ordered):
            group_owners[(source, group_id)] = "m1" if (index + swap) % 2 == 0 else "m2"
    return group_owners


def select_balanced_candidates(
    connection: sqlite3.Connection,
    *,
    per_label: dict[int, int],
    seed: int,
    group_owners: dict[tuple[str, str], str] | None = None,
) -> dict[str, list[dict]]:
    """Select two disjoint, source- and group-balanced canonical halves."""
    selected = {"m1": [], "m2": []}
    candidate_rows = _candidate_pool(connection, per_label=per_label)
    group_owners = group_owners or _group_owners(candidate_rows, seed=seed)

    for label in (0, 1):
        count = per_label[label]
        m1, m2 = _partition_source_balanced(
            candidate_rows[label],
            count=count,
            label=label,
            seed=seed,
            group_owners=group_owners,
        )
        selected["m1"].extend(m1)
        selected["m2"].extend(m2)
    for rows in selected.values():
        rows.sort(
            key=lambda row: (
                row["generic_label"],
                row["source"],
                row["source_position"],
                row["group_position"],
                row["id"],
            )
        )
    return selected


class LeakageIndex:
    """Exact, strict-normalized, and conservative near-overlap references."""

    def __init__(self) -> None:
        self._normalized: dict[str, str] = {}
        self._strict: dict[str, str] = {}
        self._near = NearIndex()

    def add(self, row: dict, *, dataset: str) -> None:
        normalized = row.get("normalized_text_sha256") or text_hash(row["text"])
        self._normalized.setdefault(normalized, dataset)
        self._strict.setdefault(strict_hash(row["text"]), dataset)
        self._near.add(row, dataset=dataset)

    def match(self, row: dict) -> dict | None:
        normalized = row.get("normalized_text_sha256") or text_hash(row["text"])
        if dataset := self._normalized.get(normalized):
            return {
                "reason": "normalized_exact",
                "reference_dataset": dataset,
            }
        if dataset := self._strict.get(strict_hash(row["text"])):
            return {
                "reason": "strict_exact",
                "reference_dataset": dataset,
            }
        matches = self._near.query(row)
        if matches:
            return {
                "reason": "near",
                "reference_dataset": matches[0]["dataset"],
                "hamming_distance": matches[0]["hamming_distance"],
            }
        return None

    def filter(self, rows, *, dataset: str) -> tuple[list[dict], list[dict], Counter]:
        kept = []
        excluded = []
        counts = Counter()
        for row in rows:
            match = self.match(row)
            if match is None:
                kept.append(row)
                continue
            counts[(match["reason"], match["reference_dataset"])] += 1
            excluded.append(
                {
                    "candidate_dataset": dataset,
                    "candidate_id": row.get("id"),
                    "candidate_normalized_text_sha256": (
                        row.get("normalized_text_sha256") or text_hash(row["text"])
                    ),
                    **match,
                }
            )
        return kept, excluded, counts


def _json_bytes(row: dict) -> bytes:
    return (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))


def _write_exclusion(
    handle,
    *,
    dataset: str,
    row: dict,
    match: dict,
) -> None:
    text = row["text"]
    handle.write(
        _json_bytes(
            {
                "candidate_dataset": dataset,
                "candidate_id": row.get("id"),
                "candidate_normalized_text_sha256": (
                    row.get("normalized_text_sha256") or text_hash(text)
                ),
                **match,
            }
        )
    )


def _write_removed_conflict(
    handle,
    *,
    dataset: str,
    insertion: dict,
    strict_text_sha256: str,
) -> None:
    removed_id = insertion.get("removed_id")
    if removed_id is None:
        return
    handle.write(
        _json_bytes(
            {
                "candidate_dataset": dataset,
                "candidate_id": removed_id,
                "candidate_normalized_text_sha256": insertion[
                    "removed_normalized_text_sha256"
                ],
                "candidate_strict_text_sha256": strict_text_sha256,
                "reason": "strict_label_conflict",
                "reference_dataset": dataset,
            }
        )
    )


def _record_insertion_exclusion(
    handle,
    *,
    dataset: str,
    row: dict,
    insertion: dict,
    exclusion_counts: Counter,
) -> None:
    status = insertion["status"]
    reason = "strict_label_conflict" if status == "conflict" else "strict_duplicate"
    match = {
        "reason": reason,
        "reference_dataset": dataset,
    }
    exclusion_counts[(dataset, reason, dataset)] += 1
    _write_exclusion(
        handle,
        dataset=dataset,
        row=row,
        match=match,
    )
    if status == "conflict" and insertion.get("removed_id") is not None:
        exclusion_counts[(dataset, reason, dataset)] += 1
        _write_removed_conflict(
            handle,
            dataset=dataset,
            insertion=insertion,
            strict_text_sha256=strict_hash(row["text"]),
        )


def _serialize_counts(counts: Counter) -> dict:
    return {
        "|".join(str(value) for value in key): count
        for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _population_summary(path: Path, *, published_path: Path) -> dict:
    labels = Counter()
    channels = Counter()
    sources = Counter()
    source_labels = Counter()
    groups = Counter()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            labels[str(row["generic_label"])] += 1
            channels[str(row.get("channel"))] += 1
            sources[str(row["source"])] += 1
            source_labels[(str(row["source"]), str(row["generic_label"]))] += 1
            groups[str(row["group_id"])] += 1
    largest_group_rows = max(groups.values(), default=0)
    return {
        "path": str(published_path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "rows": rows,
        "labels": dict(sorted(labels.items())),
        "channels": dict(sorted(channels.items())),
        "sources": dict(sorted(sources.items())),
        "source_labels": _serialize_counts(source_labels),
        "unique_groups": len(groups),
        "largest_group_rows": largest_group_rows,
        "largest_group_share": largest_group_rows / rows if rows else None,
    }


def _add_jsonl_references(
    index: LeakageIndex,
    path: Path,
    *,
    dataset: str,
    text_field: str,
    canonical: bool = False,
) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            common = (
                row if canonical else _common_external_row(row, text_field=text_field)
            )
            index.add(common, dataset=dataset)
            rows += 1
    return rows


def _filter_promptshield(
    rows: list[dict],
    index: LeakageIndex,
    *,
    dataset: str,
    data_role: str,
    exclusions,
    exclusion_counts: Counter,
) -> list[dict]:
    kept = []
    for row in rows:
        common = _common_external_row(row, text_field="prompt")
        match = index.match(common)
        if match is not None:
            exclusion_counts[
                (dataset, match["reason"], match["reference_dataset"])
            ] += 1
            _write_exclusion(
                exclusions,
                dataset=dataset,
                row=common,
                match=match,
            )
            continue
        kept.append(promptshield_record(row, data_role=data_role))
    kept.sort(
        key=lambda row: (
            row["generic_label"],
            row["strict_text_sha256"],
            row["id"],
        )
    )
    return kept


def _stream_filtered_validation(
    source_path: Path,
    connection: sqlite3.Connection,
    index: LeakageIndex,
    *,
    seed: int,
    exclusions,
    exclusion_counts: Counter,
) -> Counter:
    counts = Counter()
    with source_path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not canonical_is_eligible(row):
                continue
            match = index.match(row)
            if match is not None:
                exclusion_counts[
                    (
                        "morgott_validation",
                        match["reason"],
                        match["reference_dataset"],
                    )
                ] += 1
                _write_exclusion(
                    exclusions,
                    dataset="morgott_validation",
                    row=row,
                    match=match,
                )
                continue
            generic = canonical_record(row)
            insertion = insert_candidate(
                connection,
                generic,
                offset=offset,
                seed=seed,
            )
            if insertion["status"] != "inserted":
                _record_insertion_exclusion(
                    exclusions,
                    dataset="morgott_validation",
                    row=row,
                    insertion=insertion,
                    exclusion_counts=exclusion_counts,
                )
                continue
            counts[(generic["generic_label"], generic["source"])] += 1
    connection.commit()
    return counts


def _stream_train_candidates(
    source_path: Path,
    connection: sqlite3.Connection,
    index: LeakageIndex,
    *,
    seed: int,
    exclusions,
    exclusion_counts: Counter,
) -> Counter:
    counts = Counter()
    inserted = 0
    with source_path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not canonical_is_eligible(row):
                continue
            match = index.match(row)
            if match is not None:
                exclusion_counts[
                    ("morgott_train", match["reason"], match["reference_dataset"])
                ] += 1
                _write_exclusion(
                    exclusions,
                    dataset="morgott_train",
                    row=row,
                    match=match,
                )
                continue
            generic = canonical_record(row)
            insertion = insert_candidate(
                connection,
                generic,
                offset=offset,
                seed=seed,
            )
            if insertion["status"] != "inserted":
                _record_insertion_exclusion(
                    exclusions,
                    dataset="morgott_train",
                    row=row,
                    insertion=insertion,
                    exclusion_counts=exclusion_counts,
                )
                continue
            counts[(generic["generic_label"], generic["source"])] += 1
            inserted += 1
            if inserted % 10_000 == 0:
                connection.commit()
    connection.commit()
    return counts


def _candidate_counts(connection: sqlite3.Connection) -> Counter:
    return Counter(
        {
            (label, source): count
            for label, source, count in connection.execute(
                """
                SELECT generic_label, source, COUNT(*)
                FROM candidates
                GROUP BY generic_label, source
                """
            )
        }
    )


def _all_candidate_metadata(connection: sqlite3.Connection) -> list[dict]:
    cursor = connection.execute(
        """
        SELECT id, generic_label, byte_offset
        FROM candidates
        ORDER BY generic_label, source, group_rank, row_rank, id
        """
    )
    columns = [value[0] for value in cursor.description]
    return [dict(zip(columns, values, strict=True)) for values in cursor]


def _rows_at_offsets(
    path: Path,
    selected: list[dict],
    *,
    canonical: bool,
) -> list[dict]:
    rows_by_id = {}
    with path.open("rb") as handle:
        for metadata in sorted(selected, key=lambda row: row["byte_offset"]):
            handle.seek(metadata["byte_offset"])
            row = json.loads(handle.readline())
            if row["id"] != metadata["id"]:
                raise ValueError(
                    f"offset mismatch: expected {metadata['id']}, found {row['id']}"
                )
            generic = canonical_record(row) if canonical else row
            if generic["generic_label"] != metadata["generic_label"]:
                raise ValueError(f"label mismatch at offset for {generic['id']}")
            rows_by_id[generic["id"]] = generic
    return [rows_by_id[metadata["id"]] for metadata in selected]


def _group_near_components(records: list[dict]) -> list[list[dict]]:
    """Return connected components over lineage groups and conservative near text."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_group = {}
    row_by_id = {}
    near = NearIndex()
    for index, row in enumerate(records):
        if row["id"] in row_by_id:
            raise ValueError(f"duplicate record id in component pool: {row['id']}")
        group = (row["source"], row["group_id"])
        if group in first_by_group:
            union(index, first_by_group[group])
        else:
            first_by_group[group] = index
        for match in near.query(row):
            union(index, row_by_id[match["id"]])
        row_by_id[row["id"]] = index
        near.add(row, dataset="partition_pool")

    components = defaultdict(list)
    for index, row in enumerate(records):
        components[find(index)].append(row)
    return list(components.values())


def component_group_owners(
    records: list[dict],
    *,
    seed: int,
) -> dict[tuple[str, str], str]:
    """Keep lineage groups and conservative near components on one side."""
    ordered = sorted(
        _group_near_components(records),
        key=lambda rows: (
            -len(rows),
            _stable_rank(
                seed,
                "near-component",
                *(sorted(row["id"] for row in rows)),
            ),
        ),
    )
    label_mass = {"m1": Counter(), "m2": Counter()}
    source_mass = {"m1": Counter(), "m2": Counter()}
    owners = {}
    for rows in ordered:
        component_labels = Counter(row["generic_label"] for row in rows)
        component_sources = Counter(
            (row["generic_label"], row["source"]) for row in rows
        )

        def imbalance(side: str) -> tuple:
            other = "m2" if side == "m1" else "m1"
            source_imbalance = sum(
                abs(
                    source_mass[side][key]
                    + component_sources[key]
                    - source_mass[other][key]
                )
                for key in component_sources
            )
            label_imbalance = sum(
                abs(
                    label_mass[side][label]
                    + component_labels[label]
                    - label_mass[other][label]
                )
                for label in component_labels
            )
            return (
                source_imbalance,
                label_imbalance,
                sum(label_mass[side].values()),
                _stable_rank(
                    seed,
                    "near-component-side",
                    rows[0]["id"],
                    side,
                ),
                side,
            )

        side = min(("m1", "m2"), key=imbalance)
        label_mass[side].update(component_labels)
        source_mass[side].update(component_sources)
        for row in rows:
            owners[(row["source"], row["group_id"])] = side
    return owners


def partition_validation_records(
    records: list[dict],
    *,
    seed: int,
    checkpoint_fraction: float = 0.2,
) -> tuple[dict[str, list[dict]], dict]:
    """Partition every validation component into checkpoint or calibration roles."""
    if not 0 < checkpoint_fraction < 1:
        raise ValueError("checkpoint fraction must be between zero and one")
    if not records:
        raise ValueError("validation partition requires records")

    totals = Counter((row["generic_label"], row["source"]) for row in records)
    target = {key: count * checkpoint_fraction for key, count in totals.items()}
    checkpoint_counts = Counter()
    roles = {
        "checkpoint_selection": [],
        "calibration": [],
    }
    components = sorted(
        _group_near_components(records),
        key=lambda rows: (
            -len(rows),
            _stable_rank(
                seed,
                "validation-component",
                *(sorted(row["id"] for row in rows)),
            ),
        ),
    )
    for rows in components:
        identity = hashlib.sha256()
        for row in sorted(rows, key=lambda value: value["id"]):
            identity.update(str(row["id"]).encode())
            identity.update(b"\0")
            identity.update(str(row["strict_text_sha256"]).encode())
            identity.update(b"\n")
        component_id = f"validation-component:{identity.hexdigest()}"
        component_rows = [
            {**row, "validation_component_id": component_id} for row in rows
        ]
        component_counts = Counter(
            (row["generic_label"], row["source"]) for row in component_rows
        )
        calibration_cost = sum(
            abs(checkpoint_counts[key] - target[key]) for key in component_counts
        )
        checkpoint_cost = sum(
            abs(checkpoint_counts[key] + count - target[key])
            for key, count in component_counts.items()
        )
        if checkpoint_cost == calibration_cost:
            role = (
                "checkpoint_selection"
                if _stable_rank(
                    seed,
                    "validation-component-role",
                    *(sorted(row["id"] for row in component_rows)),
                )
                % 2
                == 0
                else "calibration"
            )
        else:
            role = (
                "checkpoint_selection"
                if checkpoint_cost < calibration_cost
                else "calibration"
            )
        roles[role].extend(component_rows)
        if role == "checkpoint_selection":
            checkpoint_counts.update(component_counts)

    for rows in roles.values():
        rows.sort(
            key=lambda row: (
                row["generic_label"],
                row["source"],
                row["group_id"],
                row["id"],
            )
        )
    checkpoint = roles["checkpoint_selection"]
    calibration = roles["calibration"]
    checkpoint_ids = {row["id"] for row in checkpoint}
    calibration_ids = {row["id"] for row in calibration}
    checkpoint_normalized = {
        row.get("normalized_text_sha256") or text_hash(row["text"])
        for row in checkpoint
    }
    calibration_normalized = {
        row.get("normalized_text_sha256") or text_hash(row["text"])
        for row in calibration
    }
    checkpoint_strict = {row["strict_text_sha256"] for row in checkpoint}
    calibration_strict = {row["strict_text_sha256"] for row in calibration}
    checkpoint_groups = {(row["source"], row["group_id"]) for row in checkpoint}
    calibration_groups = {(row["source"], row["group_id"]) for row in calibration}
    checkpoint_components = {row["validation_component_id"] for row in checkpoint}
    calibration_components = {row["validation_component_id"] for row in calibration}
    near = NearIndex()
    for row in checkpoint:
        near.add(row, dataset="checkpoint_selection")
    near_overlap = any(near.query(row) for row in calibration)
    disjointness = {
        "row": checkpoint_ids.isdisjoint(calibration_ids),
        "normalized": checkpoint_normalized.isdisjoint(calibration_normalized),
        "strict": checkpoint_strict.isdisjoint(calibration_strict),
        "lineage_group": checkpoint_groups.isdisjoint(calibration_groups),
        "near": not near_overlap,
        "validation_component": checkpoint_components.isdisjoint(
            calibration_components
        ),
    }
    if not all(disjointness.values()):
        raise ValueError(f"validation partition is not disjoint: {disjointness}")
    if len(checkpoint) + len(calibration) != len(records):
        raise ValueError("validation partition lost records")

    by_label_source = {}
    for key, total in sorted(totals.items()):
        label, source = key
        checkpoint_rows = checkpoint_counts[key]
        by_label_source[f"{label}|{source}"] = {
            "total": total,
            "checkpoint_selection": checkpoint_rows,
            "calibration": total - checkpoint_rows,
            "target_checkpoint": target[key],
        }

    def negative_evidence(rows: list[dict]) -> dict:
        negatives = [row for row in rows if row["generic_label"] == 0]
        rows_by_channel = Counter(str(row["channel"]) for row in negatives)
        rows_by_source = Counter(str(row["source"]) for row in negatives)
        components_by_channel = defaultdict(set)
        components_by_source = defaultdict(set)
        for row in negatives:
            component_id = row["validation_component_id"]
            components_by_channel[str(row["channel"])].add(component_id)
            components_by_source[str(row["source"])].add(component_id)
        return {
            "rows_by_channel": dict(sorted(rows_by_channel.items())),
            "components_by_channel": {
                key: len(values)
                for key, values in sorted(components_by_channel.items())
            },
            "rows_by_source": dict(sorted(rows_by_source.items())),
            "components_by_source": {
                key: len(values) for key, values in sorted(components_by_source.items())
            },
        }

    return roles, {
        "target_checkpoint_fraction": checkpoint_fraction,
        "actual_checkpoint_fraction": len(checkpoint) / len(records),
        "total_rows": len(records),
        "checkpoint_selection_rows": len(checkpoint),
        "calibration_rows": len(calibration),
        "components": len(components),
        "component_basis": ["source+group_id", "conservative_near_overlap"],
        "component_calibration": {
            "component_id_field": "validation_component_id",
            "component_id_definition": (
                "SHA-256 over sorted row id and strict-text SHA-256 pairs"
            ),
            "target_unit": (
                "lineage-and-near validation component within trusted channel"
            ),
            "score_aggregation": (
                "maximum negative score per component within trusted channel"
            ),
            "family_confidence": 0.95,
            "per_channel_confidence": 0.975,
            "multiplicity_correction": "Bonferroni",
            "family_scope": (
                "the two trusted channels, with a separate family for each target"
            ),
            "trusted_channels": ["direct_user", "untrusted_content"],
            "pooled_negative_role": "empirical diagnostic only",
            "components_by_role": {
                "checkpoint_selection": len(checkpoint_components),
                "calibration": len(calibration_components),
            },
            "negative_evidence_by_role": {
                role: negative_evidence(rows) for role, rows in roles.items()
            },
            "inference_caveat": (
                "Components and recurring source families are not IID or sampled "
                "from a deployment distribution; confidence bounds are development "
                "evidence, not production guarantees."
            ),
        },
        "by_label_source": by_label_source,
        "disjointness": disjointness,
    }


def _near_disjoint_group_owners(
    connection: sqlite3.Connection,
    canonical_path: Path,
    *,
    per_label: dict[int, int],
    seed: int,
) -> dict[tuple[str, str], str]:
    pool = _candidate_pool(connection, per_label=per_label)
    metadata = [row for label in (0, 1) for row in pool[label]]
    records = _rows_at_offsets(
        canonical_path,
        metadata,
        canonical=True,
    )
    return component_group_owners(records, seed=seed)


def _validate_matched_populations(
    m1: list[dict],
    m2: list[dict],
    promptshield: list[dict],
) -> None:
    populations = {"m1": m1, "m2": m2, "promptshield": promptshield}
    label_counts = {
        name: Counter(row["generic_label"] for row in rows)
        for name, rows in populations.items()
    }
    if len({tuple(sorted(counts.items())) for counts in label_counts.values()}) != 1:
        raise ValueError(f"matched label counts differ: {label_counts}")
    if len({len(rows) for rows in populations.values()}) != 1:
        raise ValueError(
            f"matched row counts differ: "
            f"{ {name: len(rows) for name, rows in populations.items()} }"
        )
    ids = {name: {row["id"] for row in rows} for name, rows in populations.items()}
    if not ids["m1"].isdisjoint(ids["m2"]):
        raise ValueError("canonical matched halves overlap")
    m1_hashes = {row["strict_text_sha256"] for row in m1}
    m2_hashes = {row["strict_text_sha256"] for row in m2}
    if m1_hashes & m2_hashes:
        raise ValueError("canonical matched halves have strict-text overlap")
    m1_groups = {(row["source"], row["group_id"]) for row in m1}
    m2_groups = {(row["source"], row["group_id"]) for row in m2}
    if m1_groups & m2_groups:
        raise ValueError("canonical matched halves have lineage-group overlap")
    near = NearIndex()
    for row in m1:
        near.add(row, dataset="m1")
    near_matches = [row["id"] for row in m2 if near.query(row)]
    if near_matches:
        raise ValueError(
            f"canonical matched halves have {len(near_matches)} near overlaps"
        )
    canonical_hashes = m1_hashes | m2_hashes
    promptshield_hashes = {row["strict_text_sha256"] for row in promptshield}
    if len(promptshield_hashes) != len(promptshield):
        raise ValueError("PromptShield fitting rows contain strict duplicates")
    if canonical_hashes & promptshield_hashes:
        raise ValueError("canonical and PromptShield fitting texts overlap")


def _build(
    output: Path,
    *,
    published_output: Path,
    seed: int,
    provenance: dict[str, str],
) -> None:
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical_paths = {}
    for split in ("train", "validation", "dev_test"):
        spec = manifest["routing_views"][split]
        path = manifest_output_path(DATA_DIR, spec)
        _verify_file(path, spec["sha256"])
        canonical_paths[split] = path

    promptshield_report_path = PROMPTSHIELD_DIR / "filter_report.json"
    promptshield_report = json.loads(promptshield_report_path.read_text())
    promptshield_rows = {}
    for split in ("train", "validation"):
        spec = promptshield_report["outputs"][split]
        promptshield_rows[split] = _load_jsonl(
            REPO_ROOT / spec["path"],
            spec["sha256"],
        )
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

    print("indexing high-priority held-out sets", flush=True)
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

    validation_db = sqlite3.connect(output / "validation_candidates.sqlite")
    create_candidate_table(validation_db)
    with exclusions_path.open("wb") as exclusions:
        promptshield_validation = _filter_promptshield(
            promptshield_rows["validation"],
            index,
            dataset="promptshield_validation",
            data_role="validation",
            exclusions=exclusions,
            exclusion_counts=exclusion_counts,
        )
        validation_promptshield_path = output / "validation_promptshield.jsonl"
        _write_rows(validation_promptshield_path, promptshield_validation)
        for row in promptshield_validation:
            index.add(row, dataset="promptshield_validation")
        references["promptshield_validation"] = len(promptshield_validation)

        print("filtering canonical validation", flush=True)
        _stream_filtered_validation(
            canonical_paths["validation"],
            validation_db,
            index,
            seed=seed,
            exclusions=exclusions,
            exclusion_counts=exclusion_counts,
        )

        print("indexing complete canonical validation", flush=True)
        references["morgott_validation"] = _add_jsonl_references(
            index,
            canonical_paths["validation"],
            dataset="morgott_validation",
            text_field="text",
            canonical=True,
        )

        print("filtering PromptShield train", flush=True)
        promptshield_train = _filter_promptshield(
            promptshield_rows["train"],
            index,
            dataset="promptshield_train",
            data_role="train",
            exclusions=exclusions,
            exclusion_counts=exclusion_counts,
        )
        promptshield_train_path = output / "promptshield.jsonl"
        _write_rows(promptshield_train_path, promptshield_train)
        for row in promptshield_train:
            index.add(row, dataset="promptshield_train")
        references["promptshield_train"] = len(promptshield_train)

        train_db = sqlite3.connect(output / "train_candidates.sqlite")
        create_candidate_table(train_db)
        print("filtering canonical train", flush=True)
        _stream_train_candidates(
            canonical_paths["train"],
            train_db,
            index,
            seed=seed,
            exclusions=exclusions,
            exclusion_counts=exclusion_counts,
        )

    train_available = _candidate_counts(train_db)
    promptshield_train_counts = Counter(
        row["generic_label"] for row in promptshield_train
    )
    if set(promptshield_train_counts) != {0, 1}:
        raise ValueError(
            f"PromptShield train lost a class after filtering: "
            f"{dict(promptshield_train_counts)}"
        )
    print("building lineage- and near-disjoint canonical halves", flush=True)
    train_group_owners = _near_disjoint_group_owners(
        train_db,
        canonical_paths["train"],
        per_label=dict(promptshield_train_counts),
        seed=seed,
    )
    train_selection = select_balanced_candidates(
        train_db,
        per_label=dict(promptshield_train_counts),
        seed=seed,
        group_owners=train_group_owners,
    )
    train_db.close()
    m1 = _rows_at_offsets(
        canonical_paths["train"],
        train_selection["m1"],
        canonical=True,
    )
    m2 = _rows_at_offsets(
        canonical_paths["train"],
        train_selection["m2"],
        canonical=True,
    )
    _validate_matched_populations(m1, m2, promptshield_train)
    m1_path = output / "m1.jsonl"
    m2_path = output / "m2.jsonl"
    _write_rows(m1_path, m1)
    _write_rows(m2_path, m2)

    validation_available = _candidate_counts(validation_db)
    validation_records = _rows_at_offsets(
        canonical_paths["validation"],
        _all_candidate_metadata(validation_db),
        canonical=True,
    )
    validation_roles, validation_partition = partition_validation_records(
        validation_records,
        seed=seed + 1,
        checkpoint_fraction=0.2,
    )
    validation_morgott_selection = [
        {**row, "experiment_role": "checkpoint_selection"}
        for row in validation_roles["checkpoint_selection"]
    ]
    validation_morgott_calibration = [
        {**row, "experiment_role": "calibration"}
        for row in validation_roles["calibration"]
    ]
    validation_morgott_selection_path = output / "validation_morgott_selection.jsonl"
    validation_morgott_calibration_path = (
        output / "validation_morgott_calibration.jsonl"
    )
    _write_rows(
        validation_morgott_selection_path,
        validation_morgott_selection,
    )
    _write_rows(
        validation_morgott_calibration_path,
        validation_morgott_calibration,
    )

    validation_db.close()
    (output / "train_candidates.sqlite").unlink()
    (output / "validation_candidates.sqlite").unlink()
    artifacts = {
        name: _population_summary(
            path,
            published_path=published_output / path.name,
        )
        for name, path in {
            "m1": m1_path,
            "m2": m2_path,
            "promptshield": promptshield_train_path,
            "validation_morgott_selection": validation_morgott_selection_path,
            "validation_morgott_calibration": validation_morgott_calibration_path,
            "validation_promptshield": validation_promptshield_path,
        }.items()
    }
    exclusions_spec = {
        "path": str((published_output / exclusions_path.name).relative_to(REPO_ROOT)),
        "sha256": file_sha256(exclusions_path),
        "rows": sum(exclusion_counts.values()),
        "by_candidate_reason_reference": _serialize_counts(exclusion_counts),
    }
    report = {
        "schema_version": 2,
        "purpose": (
            "artifact-only update-matched generic instruction-subversion experiment"
        ),
        "generic_target": TARGET,
        "seed": seed,
        "canonical_corpus_modified": False,
        "eligibility": {
            "routing_training_eligible": True,
            "input_channel": ["direct_user", "untrusted_content"],
            "label_field": "injection_label",
            "labels": [0, 1],
            "exclude_security_label": "uncertain",
            "exclude_if_all_origins_are_weak_or_unverified": True,
            "routing_label_used": False,
        },
        "sampling": {
            "control": ["m1", "m2"],
            "combined": ["m1", "promptshield"],
            "matched_to_promptshield_by_label": True,
            "canonical_selection": (
                "deterministic round-robin across source, then lineage group, "
                "then seeded row hash; lineage and conservative near components "
                "are assigned wholly to one half"
            ),
            "m1_m2_row_strict_group_and_near_disjoint": True,
        },
        "validation_partition": {
            **validation_partition,
            "checkpoint_selection": [
                "morgott_validation_checkpoint_selection",
                "promptshield_validation",
            ],
            "threshold_calibration": "morgott_validation_calibration_only",
            "promptshield_used_for_threshold": False,
        },
        "leakage": {
            "precedence": [
                "morgott_dev_test+promptshield_test+sep",
                "promptshield_validation+morgott_validation",
                "promptshield_train",
                "morgott_train",
            ],
            "normalization_exact": (
                "morgott.data.normalize_text SHA-256 or canonical precomputed hash"
            ),
            "strict_exact": "experiments.strict_normalize SHA-256",
            "near": NEAR_METHOD,
            "references": dict(sorted(references.items())),
            "exclusions": exclusions_spec,
        },
        "available_after_filter": {
            "train_by_label_source": _serialize_counts(train_available),
            "validation_by_label_source": _serialize_counts(validation_available),
        },
        "inputs": {
            "manifest": {
                "path": str(manifest_path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(manifest_path),
            },
            "routing_views": {
                split: {
                    "path": str(canonical_paths[split].relative_to(REPO_ROOT)),
                    "sha256": manifest["routing_views"][split]["sha256"],
                    "rows": manifest["routing_views"][split]["rows"],
                }
                for split in ("train", "validation", "dev_test")
            },
            "promptshield_filter_report": {
                "path": str(promptshield_report_path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(promptshield_report_path),
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
        "outputs": artifacts,
        "provenance": provenance,
        "limitations": [
            "PromptShield publishes no row-level source-family lineage.",
            "The SimHash near-overlap check is conservative and not exhaustive.",
            "PromptShield validation is checkpoint-selection evidence and is not "
            "an operating-threshold calibration distribution.",
            "PromptShield test, SEP, and canonical dev-test are already-open "
            "development evidence, not prospective final tests.",
            "Validation components reduce known lineage and near-overlap "
            "dependence, but components and recurring source families are not IID "
            "or sampled from a deployment distribution.",
            "No subtype or PromptShield input-channel label is inferred.",
        ],
    }
    (output / "selection_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    output = (
        Path(args.output).resolve()
        if args.output
        else REPO_ROOT / f"artifacts/combined_generic/selection_s{args.seed}"
    )
    artifacts_root = (REPO_ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifacts_root):
        parser.error("--output must be inside the repository artifacts directory")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    source_paths = {
        "runner_sha256": Path(__file__).resolve(),
        "strict_normalizer_sha256": (
            Path(__file__).resolve().parent / "strict_normalize.py"
        ),
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
            seed=args.seed,
            provenance=provenance,
        )
        _verify_source_hashes(source_paths, provenance)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    report = json.loads((output / "selection_report.json").read_text())
    print(
        f"wrote {output}; "
        f"{report['outputs']['promptshield']['rows']} rows per training half"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
