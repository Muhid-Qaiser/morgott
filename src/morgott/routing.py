from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import zlib
from collections import Counter
from collections import defaultdict
from pathlib import Path

from .data import SOURCES, deduplicate, file_sha256, manifest_output_path, text_hash
from .overlap import NEAR_METHOD, NearIndex, fingerprint


TRAIN = 0
VALIDATION = 1
DEV_TEST = 2
PARTITION_NAMES = {
    TRAIN: "train",
    VALIDATION: "validation",
    DEV_TEST: "dev_test",
}
TARGET_RATIOS = {TRAIN: 0.7, VALIDATION: 0.1, DEV_TEST: 0.2}


class _JsonlWriter:
    def __init__(
        self, data_dir: Path, path: Path, *, track_groups: bool = False
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.path = path
        self.handle = tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False)
        self.temporary = Path(self.handle.name)
        self.digest = hashlib.sha256()
        self.rows = 0
        self.routing = Counter()
        self.representative_sources = Counter()
        self.source_memberships = Counter()
        self.representative_source_routing = Counter()
        self.source_membership_routing = Counter()
        self.reasons = Counter()
        self.track_groups = track_groups
        self.current_group: str | None = None
        self.current_group_rows = 0
        self.largest_group: tuple[int, str] = (0, "")

    def _finish_group(self) -> None:
        if self.current_group_rows > self.largest_group[0]:
            self.largest_group = (self.current_group_rows, self.current_group or "")
        self.current_group_rows = 0

    def write(self, row: dict) -> None:
        if self.track_groups:
            group = row["split_group_id"]
            if group != self.current_group:
                self._finish_group()
                self.current_group = group
            self.current_group_rows += 1
        line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        self.handle.write(line)
        self.digest.update(line)
        self.rows += 1
        self.routing[row["routing_label"]] += 1
        self.representative_sources[row["source"]] += 1
        self.representative_source_routing[row["source"], row["routing_label"]] += 1
        memberships = {origin["source"] for origin in row.get("origins", [])} or {
            row["source"]
        }
        self.source_memberships.update(memberships)
        self.source_membership_routing.update(
            (source, row["routing_label"]) for source in memberships
        )
        if row.get("quarantine_reason"):
            self.reasons[row["quarantine_reason"]] += 1

    def finish(self) -> dict:
        if self.track_groups:
            self._finish_group()
        self.handle.close()
        summary = {
            "path": str(self.path.relative_to(self.data_dir)),
            "rows": self.rows,
            "routing_benign": self.routing[0],
            "routing_non_benign": self.routing[1],
            "representative_sources": dict(sorted(self.representative_sources.items())),
            "source_memberships": dict(sorted(self.source_memberships.items())),
            "representative_source_routing": {
                source: {
                    "benign": self.representative_source_routing[source, 0],
                    "non_benign": self.representative_source_routing[source, 1],
                }
                for source in sorted(self.representative_sources)
            },
            "source_membership_routing": {
                source: {
                    "benign": self.source_membership_routing[source, 0],
                    "non_benign": self.source_membership_routing[source, 1],
                }
                for source in sorted(self.source_memberships)
            },
            "sha256": self.digest.hexdigest(),
        }
        if self.reasons:
            summary["reasons"] = dict(sorted(self.reasons.items()))
        if self.track_groups:
            rows, group = self.largest_group
            summary["largest_split_group"] = {
                "rows": rows,
                "partition_share": rows / self.rows if self.rows else 0.0,
                "split_group_sha256": (
                    hashlib.sha256(group.encode()).hexdigest() if group else None
                ),
            }
        return summary

    def publish(self) -> None:
        self.temporary.replace(self.path)

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        self.temporary.unlink(missing_ok=True)


def _open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -131072")
    connection.executescript(
        """
        CREATE TABLE rows (
            seq INTEGER PRIMARY KEY,
            source_role TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            normalized_hash TEXT NOT NULL,
            group_id INTEGER,
            routing_label INTEGER NOT NULL,
            payload BLOB NOT NULL
        );
        """
    )
    return connection


def _ingest_sources(
    connection: sqlite3.Connection,
    data_dir: Path,
    source_outputs: dict[str, dict],
) -> list[str]:
    group_ids: dict[str, int] = {}
    group_names = [""]
    row_batch = []

    def flush() -> None:
        if row_batch:
            connection.executemany(
                "INSERT INTO rows(source_role, eligible, normalized_hash, group_id, "
                "routing_label, payload) VALUES (?, ?, ?, ?, ?, ?)",
                row_batch,
            )
            row_batch.clear()

    for source, output in sorted(source_outputs.items()):
        path = manifest_output_path(data_dir, output)
        digest = file_sha256(path)
        if digest != output["sha256"]:
            raise RuntimeError(
                f"{source} source shard changed: expected {output['sha256']}, got {digest}"
            )
        source_ids = set()
        rows_seen = 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"{source}:{line_number} has a blank row")
                row = json.loads(line)
                rows_seen += 1
                row_id = row.get("id")
                text = row.get("text")
                if (
                    row.get("schema_version") != 5
                    or row.get("source") != source
                    or row.get("source_revision") != SOURCES[source]["revision"]
                    or row.get("license") != SOURCES[source]["license"]
                    or not isinstance(row_id, str)
                    or not row_id
                    or not isinstance(text, str)
                    or not text.strip()
                    or not isinstance(row.get("origins"), list)
                    or not row["origins"]
                ):
                    raise ValueError(f"{source}:{line_number} has an invalid schema")
                if row_id in source_ids:
                    raise ValueError(f"{source}:{line_number} has a duplicate id")
                source_ids.add(row_id)
                role = row.get("source_role")
                eligible = row.get("routing_training_eligible")
                expected = role in {"candidate", "dev_test"}
                if role not in {"candidate", "dev_test", "auxiliary", "uncertain"}:
                    raise ValueError(f"{source}:{line_number} has invalid source role")
                if type(eligible) is not bool or eligible != expected:
                    raise ValueError(
                        f"{source}:{line_number} has inconsistent routing eligibility"
                    )
                normalized_hash = row.get("normalized_text_sha256")
                if normalized_hash != text_hash(text):
                    raise ValueError(f"{source}:{line_number} has invalid text hash")
                routing_label = row.get("routing_label")
                if type(routing_label) is not int or routing_label not in (0, 1):
                    raise ValueError(
                        f"{source}:{line_number} has invalid routing label"
                    )
                if role == "auxiliary":
                    continue
                group_id = None
                if eligible:
                    group_name = row.get("split_group_id")
                    if not isinstance(group_name, str) or not group_name:
                        raise ValueError(f"{source}:{line_number} has no split lineage")
                    group_id = group_ids.get(group_name)
                    if group_id is None:
                        group_id = len(group_names)
                        group_ids[group_name] = group_id
                        group_names.append(group_name)
                row_batch.append(
                    (
                        role,
                        int(eligible),
                        normalized_hash,
                        group_id,
                        routing_label,
                        zlib.compress(line.rstrip(b"\r\n"), level=1),
                    )
                )
                if len(row_batch) >= 2_000:
                    flush()
        flush()
        if rows_seen != output.get("rows"):
            raise RuntimeError(
                f"{source} source shard row count changed: "
                f"expected {output.get('rows')}, got {rows_seen}"
            )
        connection.commit()
    connection.executescript(
        """
        CREATE INDEX rows_hash ON rows(normalized_hash, seq);
        CREATE INDEX rows_group ON rows(group_id);
        CREATE INDEX rows_role_hash ON rows(source_role, normalized_hash, seq);
        """
    )
    return group_names


def _eligible_exact_groups(connection: sqlite3.Connection):
    cursor = connection.execute(
        "SELECT normalized_hash, source_role, group_id, payload "
        "FROM rows INDEXED BY rows_hash WHERE eligible = 1 "
        "ORDER BY normalized_hash, seq"
    )
    current_hash = None
    rows = []
    for normalized_hash, source_role, group_id, payload in cursor:
        if current_hash is not None and normalized_hash != current_hash:
            yield current_hash, rows
            rows = []
        current_hash = normalized_hash
        rows.append((source_role, group_id, json.loads(zlib.decompress(payload))))
    if rows:
        yield current_hash, rows


def _prepare_supervised_rows(
    connection: sqlite3.Connection,
    group_names: list[str],
    quarantine: _JsonlWriter,
) -> dict[str, int]:
    connection.execute(
        "CREATE TABLE merged ("
        "seq INTEGER PRIMARY KEY, normalized_hash TEXT NOT NULL UNIQUE, "
        "fixed_dev_test INTEGER NOT NULL, group_id INTEGER NOT NULL, "
        "source TEXT NOT NULL, routing_label INTEGER NOT NULL, payload BLOB NOT NULL)"
    )
    fixed_groups = {
        group_id
        for (group_id,) in connection.execute(
            "SELECT DISTINCT group_id FROM rows "
            "WHERE eligible = 1 AND source_role = 'dev_test'"
        )
    }
    stats = Counter()
    batch = []
    for normalized_hash, indexed_rows in _eligible_exact_groups(connection):
        group_ids = {group_id for _, group_id, _ in indexed_rows}
        fixed = any(group_id in fixed_groups for group_id in group_ids)
        indexed_rows.sort(
            key=lambda item: (
                item[1] not in fixed_groups,
                item[0] != "dev_test",
                item[2]["source"],
                item[2]["split_group_id"],
                item[2]["id"],
            )
        )
        selected_group = indexed_rows[0][1]
        row = _merge_exact(
            [candidate for _, _, candidate in indexed_rows], quarantine, stats
        )
        if row is None:
            continue
        if len(group_ids) > 1:
            stats["cross_lineage_exact_duplicates"] += 1
            selected_group = len(group_names)
            group_name = f"exact:{normalized_hash}"
            group_names.append(group_name)
        if fixed and any(role == "candidate" for role, _, _ in indexed_rows):
            stats["candidate_exact_rows_fixed_by_official_lineage"] += 1
        row["split_group_id"] = group_names[selected_group]
        batch.append(
            (
                normalized_hash,
                int(fixed),
                selected_group,
                row["source"],
                row["routing_label"],
                zlib.compress(
                    json.dumps(row, ensure_ascii=False, sort_keys=True).encode(), 1
                ),
            )
        )
        if len(batch) == 2_000:
            connection.executemany(
                "INSERT INTO merged(normalized_hash, fixed_dev_test, group_id, source, "
                "routing_label, payload) VALUES (?, ?, ?, ?, ?, ?)",
                batch,
            )
            batch.clear()
    if batch:
        connection.executemany(
            "INSERT INTO merged(normalized_hash, fixed_dev_test, group_id, source, "
            "routing_label, payload) VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
    connection.executescript(
        "CREATE INDEX merged_group ON merged(group_id);"
        "CREATE TABLE supervised_hashes AS SELECT normalized_hash FROM merged;"
        "CREATE UNIQUE INDEX supervised_hash ON supervised_hashes(normalized_hash);"
    )
    return dict(sorted(stats.items()))


def _partition_groups(
    connection: sqlite3.Connection, group_names: list[str]
) -> dict[str, object]:
    vectors: dict[int, Counter] = defaultdict(Counter)
    candidate_counts = Counter()
    for group_id, source, routing_label, rows in connection.execute(
        "SELECT group_id, source, routing_label, COUNT(*) FROM merged "
        "WHERE fixed_dev_test = 0 GROUP BY group_id, source, routing_label "
        "ORDER BY group_id, source, routing_label"
    ):
        stratum = (source, routing_label)
        vectors[group_id][stratum] = rows
        candidate_counts[stratum] += rows
    fixed_counts = Counter(
        {
            (source, routing_label): rows
            for source, routing_label, rows in connection.execute(
                "SELECT source, routing_label, COUNT(*) FROM merged "
                "WHERE fixed_dev_test = 1 GROUP BY source, routing_label "
                "ORDER BY source, routing_label"
            )
        }
    )

    targets = {partition: Counter() for partition in PARTITION_NAMES}
    for stratum, candidate_rows in candidate_counts.items():
        total = candidate_rows + fixed_counts[stratum]
        desired = {
            TRAIN: TARGET_RATIOS[TRAIN] * total,
            VALIDATION: TARGET_RATIOS[VALIDATION] * total,
            DEV_TEST: max(0.0, TARGET_RATIOS[DEV_TEST] * total - fixed_counts[stratum]),
        }
        scale = candidate_rows / sum(desired.values())
        for partition, rows in desired.items():
            targets[partition][stratum] = rows * scale

    total_rows = sum(candidate_counts.values()) + sum(fixed_counts.values())
    candidate_rows = sum(candidate_counts.values())
    candidate_quotas = {
        TRAIN: TARGET_RATIOS[TRAIN] * total_rows,
        VALIDATION: TARGET_RATIOS[VALIDATION] * total_rows,
        DEV_TEST: max(
            0.0,
            TARGET_RATIOS[DEV_TEST] * total_rows - sum(fixed_counts.values()),
        ),
    }
    quota_scale = candidate_rows / sum(candidate_quotas.values())
    candidate_quotas = {
        partition: rows * quota_scale for partition, rows in candidate_quotas.items()
    }
    for _ in range(30):
        for partition in PARTITION_NAMES:
            current = sum(targets[partition].values())
            scale = candidate_quotas[partition] / current if current else 0.0
            for stratum in targets[partition]:
                targets[partition][stratum] *= scale
        for stratum, rows in candidate_counts.items():
            current = sum(targets[partition][stratum] for partition in PARTITION_NAMES)
            scale = rows / current
            for partition in PARTITION_NAMES:
                targets[partition][stratum] *= scale

    assigned = {partition: Counter() for partition in PARTITION_NAMES}
    assignments = []
    ordered_groups = sorted(
        vectors,
        key=lambda group_id: (
            -sum(vectors[group_id].values()),
            hashlib.sha256(group_names[group_id].encode()).digest(),
        ),
    )
    for group_id in ordered_groups:
        vector = vectors[group_id]
        tie_order = sorted(
            PARTITION_NAMES,
            key=lambda partition: hashlib.sha256(
                f"{group_names[group_id]}:{partition}".encode()
            ).digest(),
        )

        def cost(partition: int) -> float:
            return sum(
                (
                    (assigned[partition][stratum] + rows - targets[partition][stratum])
                    ** 2
                    - (assigned[partition][stratum] - targets[partition][stratum]) ** 2
                )
                / max(candidate_counts[stratum], 1)
                for stratum, rows in vector.items()
            )

        partition = min(tie_order, key=cost)
        assigned[partition].update(vector)
        assignments.append((group_id, partition))
    connection.execute(
        "CREATE TABLE partitions (group_id INTEGER PRIMARY KEY, partition INTEGER NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO partitions(group_id, partition) VALUES (?, ?)", assignments
    )
    connection.commit()

    candidate_partition_rows = {
        PARTITION_NAMES[partition]: sum(counts.values())
        for partition, counts in assigned.items()
    }
    official_rows = connection.execute(
        "SELECT COUNT(*) FROM merged WHERE fixed_dev_test = 1"
    ).fetchone()[0]
    final_rows = dict(candidate_partition_rows)
    final_rows["dev_test"] += official_rows
    total_rows = sum(final_rows.values())
    return {
        "partition_method": (
            "exact deduplication, official-dev locking, 70/10/20 iterative "
            "source/label targets, then deterministic weighted lineage placement"
        ),
        "target_ratios": {
            PARTITION_NAMES[partition]: ratio
            for partition, ratio in TARGET_RATIOS.items()
        },
        "candidate_partition_rows": candidate_partition_rows,
        "official_dev_test_rows": official_rows,
        "planned_rows": final_rows,
        "planned_ratios": {
            name: rows / total_rows if total_rows else 0.0
            for name, rows in final_rows.items()
        },
    }


def _partition_rows(connection: sqlite3.Connection, partition: int):
    cursor = connection.execute(
        "SELECT m.payload FROM merged AS m LEFT JOIN partitions AS p "
        "ON p.group_id = m.group_id WHERE "
        "(m.fixed_dev_test = 1 AND ? = ?) OR "
        "(m.fixed_dev_test = 0 AND p.partition = ?) "
        "ORDER BY m.group_id, m.normalized_hash",
        (partition, DEV_TEST, partition),
    )
    for (payload,) in cursor:
        yield json.loads(zlib.decompress(payload))


def _uncertain_exact_groups(connection: sqlite3.Connection):
    cursor = connection.execute(
        "SELECT normalized_hash, payload FROM rows INDEXED BY rows_role_hash "
        "WHERE source_role = 'uncertain' ORDER BY normalized_hash, seq"
    )
    current_hash = None
    rows = []
    for normalized_hash, payload in cursor:
        if current_hash is not None and normalized_hash != current_hash:
            yield current_hash, rows
            rows = []
        current_hash = normalized_hash
        rows.append(json.loads(zlib.decompress(payload)))
    if rows:
        yield current_hash, rows


def _quarantine(
    writer: _JsonlWriter,
    row: dict,
    reason: str,
    matches: list[dict] | None = None,
) -> None:
    quarantined = dict(row)
    quarantined["data_role"] = "quarantine"
    quarantined["quarantine_reason"] = reason
    if matches:
        quarantined["near_match_count"] = len(matches)
        quarantined["near_matches"] = matches[:20]
    writer.write(quarantined)


def _merge_exact(
    rows: list[dict], quarantine: _JsonlWriter, stats: Counter
) -> dict | None:
    stats["input_rows"] += len(rows)
    merged, merge_stats = deduplicate(rows, label_fields=("routing_label",))
    if not merged:
        stats["exact_conflict_rows_quarantined"] += len(rows)
        for row in rows:
            _quarantine(quarantine, row, "exact_label_conflict")
        return None
    stats["exact_duplicate_rows_merged"] += merge_stats["duplicates"]
    return merged[0]


def _write_supervised_views(
    connection: sqlite3.Connection,
    writers: dict[str, _JsonlWriter],
    quarantine: _JsonlWriter,
) -> dict[str, int]:
    stats = Counter()
    dev_index = NearIndex()
    train_index = NearIndex()
    for partition in (DEV_TEST, TRAIN, VALIDATION):
        name = PARTITION_NAMES[partition]
        for row in _partition_rows(connection, partition):
            near_value = fingerprint(row["text"])
            matches = []
            reason = None
            if partition != DEV_TEST and near_value is not None:
                matches = dev_index.query(row, value=near_value)
                if matches:
                    reason = "near_dev_test_overlap"
            if partition == VALIDATION and not matches and near_value is not None:
                matches = train_index.query(row, value=near_value)
                if matches:
                    reason = "near_train_overlap"
            if matches:
                stats[f"{reason}_rows_quarantined"] += 1
                _quarantine(quarantine, row, reason, matches)
                continue
            row["data_role"] = name
            writers[name].write(row)
            if partition == DEV_TEST and near_value is not None:
                dev_index.add(row, dataset="routing_dev_test", value=near_value)
            elif partition == TRAIN and near_value is not None:
                train_index.add(row, dataset="routing_train", value=near_value)
    return dict(sorted(stats.items()))


def _write_uncertain_view(
    connection: sqlite3.Connection,
    writer: _JsonlWriter,
    quarantine: _JsonlWriter,
) -> dict[str, int]:
    stats = Counter()
    lookup = connection.cursor()
    for normalized_hash, exact_rows in _uncertain_exact_groups(connection):
        if lookup.execute(
            "SELECT 1 FROM supervised_hashes WHERE normalized_hash = ?",
            (normalized_hash,),
        ).fetchone():
            stats["exact_supervised_overlap_rows_quarantined"] += len(exact_rows)
            for row in exact_rows:
                _quarantine(quarantine, row, "exact_supervised_overlap")
            continue
        row = _merge_exact(exact_rows, quarantine, stats)
        if row is not None:
            row["data_role"] = "uncertain"
            writer.write(row)
    return dict(sorted(stats.items()))


def materialize_routing_views(
    data_dir: Path,
    source_outputs: dict[str, dict],
    build_dir: Path,
    *,
    invalidate_manifest: Path | None = None,
) -> tuple[dict[str, dict], dict, dict]:
    connection = _open_index(build_dir / "routing.sqlite3")
    writers = {
        name: _JsonlWriter(
            data_dir,
            data_dir / "views" / "routing" / f"{name}.jsonl",
            track_groups=name != "uncertain",
        )
        for name in ("train", "validation", "dev_test", "uncertain")
    }
    quarantine = _JsonlWriter(data_dir, data_dir / "quarantine" / "routing.jsonl")
    all_writers = [*writers.values(), quarantine]
    try:
        group_names = _ingest_sources(connection, data_dir, source_outputs)
        exact_stats = _prepare_supervised_rows(connection, group_names, quarantine)
        partition_stats = _partition_groups(connection, group_names)
        supervised_stats = _write_supervised_views(connection, writers, quarantine)
        uncertain_stats = _write_uncertain_view(
            connection, writers["uncertain"], quarantine
        )
        summaries = {name: writer.finish() for name, writer in writers.items()}
        quarantine_summary = quarantine.finish()
        actual_rows = {
            name: summaries[name]["rows"]
            for name in ("train", "validation", "dev_test")
        }
        actual_total = sum(actual_rows.values())
        actual_ratios = {
            name: rows / actual_total if actual_total else 0.0
            for name, rows in actual_rows.items()
        }
        if invalidate_manifest is not None:
            invalidate_manifest.unlink(missing_ok=True)
        for writer in all_writers:
            writer.publish()
        stats = {
            **exact_stats,
            **partition_stats,
            "actual_rows": actual_rows,
            "actual_ratios": actual_ratios,
            "ratio_deviation": {
                name: actual_ratios[name] - TARGET_RATIOS[partition]
                for partition, name in PARTITION_NAMES.items()
            },
            "supervised": supervised_stats,
            "uncertain": uncertain_stats,
            "near_method": NEAR_METHOD,
            "storage": "temporary SQLite index; canonical JSONL source shards remain authoritative",
        }
        return summaries, quarantine_summary, stats
    except BaseException:
        for writer in all_writers:
            writer.abort()
        raise
    finally:
        connection.close()
