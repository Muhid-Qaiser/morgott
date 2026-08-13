from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ...data import iter_verified_jsonl, text_hash
from ...overlap import (
    LEAKAGE_NORMALIZATION_METHOD,
    NearIndex,
    fingerprint,
    leakage_text_hash,
)
from .core import INSTRUCTION_SUBVERSION_TAGS, file_sha256

PAIR_ARCHIVE_SHA256 = "0aa08878c3096b402cf6ee309a50b730f7150b1aed78682cdf6e42a504da13d3"
PAIR_CONTENT_SHA256 = "8ec5c1c77b378688b190722f7d1fc51e9bef819ee9670948d2658f4a37082158"
EXTERNAL_DATA_SCHEMA_VERSION = 2


@dataclass
class TrainingData:
    """Prepared corpus with a stable module path so every caller can unpickle it."""

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


_strict_hash = leakage_text_hash


def _overlap_values(row: dict) -> tuple[str, str, int | None]:
    text = row["text"]
    return text_hash(text), _strict_hash(text), fingerprint(text)


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes {root}")
    return path


def routing_views(data_dir: Path) -> dict[str, tuple[Path, dict]]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("canonical_row_schema_version") != 5:
        raise ValueError("unsupported canonical row schema")
    result = {}
    for split in ("train", "validation", "dev_test"):
        spec = manifest.get("routing_views", {}).get(split)
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("path"), str)
            or not isinstance(spec.get("sha256"), str)
            or not isinstance(spec.get("rows"), int)
        ):
            raise ValueError(f"invalid routing manifest entry: {split}")
        result[split] = (_safe_path(data_dir, spec["path"]), spec)
    return result


def _model_eligible(row: dict) -> bool:
    origins = row.get("origins")
    origins = origins if isinstance(origins, list) and origins else [row]
    weak = all(
        "weak" in str(origin.get("label_basis", "")).casefold()
        or "unverified" in str(origin.get("label_basis", "")).casefold()
        for origin in origins
    )
    eligible = (
        row.get("routing_training_eligible") is True
        and row.get("input_channel") in {"direct_user", "untrusted_content"}
        and type(row.get("injection_label")) is int
        and row["injection_label"] in (0, 1)
        and row.get("security_label") != "uncertain"
        and not weak
    )
    if not eligible or row["injection_label"] == 0:
        return eligible
    return bool(set(row.get("security_tags", ())) & set(INSTRUCTION_SUBVERSION_TAGS))


def canonical_rows(
    path: Path,
    spec: dict,
    *,
    split: str,
    eligible_only: bool = True,
) -> Iterator[dict]:
    input_count = 0
    for row in iter_verified_jsonl(path, spec["sha256"]):
        if (
            row.get("schema_version") != 5
            or row.get("data_role") != split
            or row.get("routing_label") not in (0, 1)
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or not isinstance(row.get("source"), str)
            or not isinstance(row.get("input_channel"), str)
            or not row["input_channel"]
            or not isinstance(row.get("security_tags"), list)
            or not isinstance(row.get("split_group_id"), str)
            or not row["split_group_id"]
        ):
            raise ValueError(f"invalid canonical {split} row: {row.get('id')}")
        input_count += 1
        if eligible_only and not _model_eligible(row):
            continue
        yield {
            "id": row["id"],
            "text": row["text"],
            "label": row["injection_label"],
            "source": row["source"],
            "input_channel": row["input_channel"],
            "group_id": row["split_group_id"],
            "security_tags": row["security_tags"],
        }
    if input_count != spec["rows"]:
        raise ValueError(f"{split} row count mismatch")


def external_rows(directory: Path) -> tuple[dict[str, list[dict]], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXTERNAL_DATA_SCHEMA_VERSION:
        raise ValueError("unsupported external data manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("external data manifest has no outputs")
    result = {}
    seen = set()
    for name in (
        "promptshield_train",
        "promptshield_validation",
        "promptshield_test",
        "sep",
    ):
        spec = outputs.get(name)
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise ValueError(f"invalid external manifest entry: {name}")
        path = _safe_path(directory, spec["path"])
        rows = list(iter_verified_jsonl(path, spec["sha256"]))
        if len(rows) != spec["rows"]:
            raise ValueError(f"{name} row count mismatch")
        for row in rows:
            if (
                not isinstance(row.get("id"), str)
                or not row["id"]
                or row["id"] in seen
                or not isinstance(row.get("text"), str)
                or not row["text"]
                or row.get("label") not in (0, 1)
                or row.get("input_channel") not in {"direct_user", "untrusted_content"}
                or not isinstance(row.get("source"), str)
                or not isinstance(row.get("source_revision"), str)
                or not isinstance(row.get("license"), str)
            ):
                raise ValueError(f"invalid external row: {row.get('id')}")
            seen.add(row["id"])
        result[name] = rows
    return result, manifest


def _matched_pairs(
    path: Path,
    *,
    source: str,
    expected_archive_sha256: str | None,
    expected_content_sha256: str | None,
) -> list[tuple[dict, dict]]:
    if (
        expected_archive_sha256 is not None
        and file_sha256(path) != expected_archive_sha256
    ):
        raise ValueError("matched-pair archive hash mismatch")
    digest = hashlib.sha256()
    pairs = []
    with gzip.open(path, "rb") as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            raw = json.loads(line)
            if (
                not isinstance(raw.get("benign"), str)
                or not raw["benign"]
                or not isinstance(raw.get("attack"), str)
                or not raw["attack"]
                or raw["benign"] == raw["attack"]
                or raw.get("channel") not in {"direct_user", "untrusted_content"}
            ):
                raise ValueError(f"invalid matched pair: {index}")
            pair_id = f"{source}:{index}"
            common = {
                "source": source,
                "input_channel": raw["channel"],
                "group_id": pair_id,
                "pair_id": pair_id,
            }
            pairs.append(
                (
                    {
                        **common,
                        "id": f"{pair_id}:benign",
                        "text": raw["benign"],
                        "label": 0,
                    },
                    {
                        **common,
                        "id": f"{pair_id}:attack",
                        "text": raw["attack"],
                        "label": 1,
                    },
                )
            )
    if (
        expected_content_sha256 is not None
        and digest.hexdigest() != expected_content_sha256
    ):
        raise ValueError("matched-pair content hash mismatch")
    return pairs


def matched_pairs(path: Path) -> list[tuple[dict, dict]]:
    return _matched_pairs(
        path,
        source="matched_pairs",
        expected_archive_sha256=PAIR_ARCHIVE_SHA256,
        expected_content_sha256=PAIR_CONTENT_SHA256,
    )


def additional_matched_pairs(path: Path) -> list[tuple[dict, dict]]:
    return _matched_pairs(
        path,
        source="additional_matched_pairs",
        expected_archive_sha256=None,
        expected_content_sha256=None,
    )


class OverlapGuard:
    def __init__(self, references: Iterable[dict]) -> None:
        self.normalized = set()
        self.strict = set()
        self.near = NearIndex()
        self.add(references)

    def add(self, references: Iterable[dict]) -> None:
        for row in references:
            self._add(row)

    def _add(
        self,
        row: dict,
        values: tuple[str, str, int | None] | None = None,
    ) -> None:
        normalized, strict, near = values or _overlap_values(row)
        self.normalized.add(normalized)
        self.strict.add(strict)
        if near is not None:
            self.near.add(
                row,
                dataset=row["source"],
                value=near,
                normalized_hash=normalized,
            )

    def reason(
        self,
        row: dict,
        values: tuple[str, str, int | None] | None = None,
    ) -> str | None:
        normalized, strict, near = values or _overlap_values(row)
        if normalized in self.normalized:
            return "normalized_exact"
        if strict in self.strict:
            return "strict_exact"
        if near is not None and self.near.query(
            row,
            value=near,
            normalized_hash=normalized,
        ):
            return "near"
        return None


class _SmallSetFilter:
    def __init__(self, candidates: dict[str, list[dict]]) -> None:
        self.candidates = candidates
        self.near = NearIndex()
        self.normalized: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.strict: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.blocked: dict[tuple[str, str], str] = {}
        for dataset, rows in candidates.items():
            for row in rows:
                identity = (dataset, row["id"])
                normalized, strict, near = _overlap_values(row)
                self.normalized[normalized].add(identity)
                self.strict[strict].add(identity)
                if near is not None:
                    self.near.add(
                        row,
                        dataset=dataset,
                        value=near,
                        normalized_hash=normalized,
                    )

    def block(
        self,
        reference: dict,
        values: tuple[str, str, int | None] | None = None,
    ) -> None:
        normalized, strict, near = values or _overlap_values(reference)
        self_dataset = reference.get("_candidate_dataset")
        for identity in self.normalized.get(normalized, ()):
            if identity[0] != self_dataset:
                self.blocked.setdefault(identity, "normalized_exact")
        for identity in self.strict.get(strict, ()):
            if identity[0] != self_dataset:
                self.blocked.setdefault(identity, "strict_exact")
        matches = (
            self.near.query(reference, value=near, normalized_hash=normalized)
            if near is not None
            else ()
        )
        for match in matches:
            identity = (match["dataset"], match["id"])
            if identity[0] != self_dataset:
                self.blocked.setdefault(identity, "near")

    def result(self) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
        kept = {}
        removed = {}
        for dataset, rows in self.candidates.items():
            kept[dataset] = [
                row for row in rows if (dataset, row["id"]) not in self.blocked
            ]
            removed[dataset] = dict(
                sorted(
                    Counter(
                        self.blocked[(dataset, row["id"])]
                        for row in rows
                        if (dataset, row["id"]) in self.blocked
                    ).items()
                )
            )
        return kept, removed


def filter_small_training_sets(
    candidates: dict[str, list[dict]],
    references: Iterable[dict],
    *,
    reference_guard: OverlapGuard | None = None,
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    candidate_filter = _SmallSetFilter(candidates)
    for reference in references:
        values = _overlap_values(reference)
        candidate_filter.block(reference, values)
        if reference_guard is not None and "_candidate_dataset" not in reference:
            reference_guard._add(reference, values)
    return candidate_filter.result()


def profile_canonical(
    rows: Iterable[dict],
    guard: OverlapGuard,
    candidates: dict[str, list[dict]],
) -> tuple[
    Counter,
    Counter,
    dict[str, int],
    dict[str, tuple | None],
    dict[str, list[dict]],
    dict[str, dict[str, int]],
]:
    counts = Counter()
    group_counts = Counter()
    removed = Counter()
    owners = {}
    candidate_filter = _SmallSetFilter(candidates)
    for row in rows:
        values = _overlap_values(row)
        reason = guard.reason(row, values)
        if reason:
            removed[reason] += 1
            continue
        candidate_filter.block(row, values)
        digest = values[1]
        owner = owners.get(digest)
        if digest not in owners:
            owners[digest] = (
                row["id"],
                row["source"],
                row["label"],
                row["group_id"],
            )
            counts[(row["source"], row["label"])] += 1
            group_counts[(row["label"], row["source"], row["group_id"])] += 1
        elif owner is None:
            removed["strict_label_conflict"] += 1
        elif owner[2] == row["label"]:
            removed["strict_duplicate"] += 1
        else:
            counts[(owner[1], owner[2])] -= 1
            group_counts[(owner[2], owner[1], owner[3])] -= 1
            owners[digest] = None
            removed["strict_label_conflict"] += 2
    counts = Counter({key: value for key, value in counts.items() if value > 0})
    group_counts = Counter(
        {key: value for key, value in group_counts.items() if value > 0}
    )
    if not counts or {label for _, label in counts} != {0, 1}:
        raise ValueError("canonical training population must contain both labels")
    candidate_kept, candidate_removed = candidate_filter.result()
    return (
        counts,
        group_counts,
        dict(sorted(removed.items())),
        owners,
        candidate_kept,
        candidate_removed,
    )


def training_rows(
    rows: Iterable[dict],
    counts: Counter,
    group_counts: Counter,
    owners: dict[str, tuple | None],
) -> Iterator[dict]:
    total = sum(counts.values())
    labels = {label for _, label in counts}
    sources_per_label = Counter(label for _, label in counts)
    groups_per_label_source = Counter(
        (label, source) for label, source, _ in group_counts
    )
    yielded = 0
    for row in rows:
        owner = owners.get(_strict_hash(row["text"]))
        if owner is None or owner[0] != row["id"]:
            continue
        label = row["label"]
        source = row["source"]
        group_id = row["group_id"]
        group_rows = group_counts[(label, source, group_id)]
        yielded += 1
        yield {
            **row,
            "weight": total
            / (
                len(labels)
                * sources_per_label[label]
                * groups_per_label_source[(label, source)]
                * group_rows
            ),
        }
    if yielded != total:
        raise ValueError("canonical training population changed after profiling")


def shuffled(
    rows: Iterable[dict],
    *,
    seed: int,
    buffer_size: int,
) -> Iterator[dict]:
    if buffer_size < 2:
        raise ValueError("shuffle buffer must be at least two")
    randomizer = random.Random(seed)
    buffer = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = randomizer.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    randomizer.shuffle(buffer)
    yield from buffer


def batches(rows: Iterable[dict], size: int) -> Iterator[list[dict]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _stable_rank(seed: int, *parts: object) -> int:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _group_near_components(records: list[dict]) -> list[list[dict]]:
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


def partition_validation_records(
    records: list[dict],
    *,
    seed: int,
    checkpoint_fraction: float = 0.2,
) -> tuple[dict[str, list[dict]], dict]:
    """Partition lineage-and-near components for checkpointing and calibration."""
    if not 0 < checkpoint_fraction < 1:
        raise ValueError("checkpoint fraction must be between zero and one")
    if not records:
        raise ValueError("validation partition requires records")

    prepared = [
        {
            **row,
            "normalized_text_sha256": (
                row.get("normalized_text_sha256") or text_hash(row["text"])
            ),
            "strict_text_sha256": (
                row.get("strict_text_sha256") or _strict_hash(row["text"])
            ),
        }
        for row in records
    ]
    totals = Counter((row["label"], row["source"]) for row in prepared)
    target = {key: count * checkpoint_fraction for key, count in totals.items()}
    checkpoint_counts = Counter()
    roles = {"checkpoint_selection": [], "calibration": []}
    components = sorted(
        _group_near_components(prepared),
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
            identity.update(row["id"].encode())
            identity.update(b"\0")
            identity.update(row["strict_text_sha256"].encode())
            identity.update(b"\n")
        component_id = f"validation-component:{identity.hexdigest()}"
        component_rows = [
            {**row, "validation_component_id": component_id} for row in rows
        ]
        component_counts = Counter(
            (row["label"], row["source"]) for row in component_rows
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
                row["label"],
                row["source"],
                row["group_id"],
                row["id"],
            )
        )
    checkpoint = roles["checkpoint_selection"]
    calibration = roles["calibration"]
    checkpoint_ids = {row["id"] for row in checkpoint}
    calibration_ids = {row["id"] for row in calibration}
    checkpoint_normalized = {row["normalized_text_sha256"] for row in checkpoint}
    calibration_normalized = {row["normalized_text_sha256"] for row in calibration}
    checkpoint_strict = {row["strict_text_sha256"] for row in checkpoint}
    calibration_strict = {row["strict_text_sha256"] for row in calibration}
    checkpoint_groups = {(row["source"], row["group_id"]) for row in checkpoint}
    calibration_groups = {(row["source"], row["group_id"]) for row in calibration}
    checkpoint_components = {row["validation_component_id"] for row in checkpoint}
    calibration_components = {row["validation_component_id"] for row in calibration}
    near = NearIndex()
    for row in checkpoint:
        near.add(row, dataset="checkpoint_selection")
    disjointness = {
        "row": checkpoint_ids.isdisjoint(calibration_ids),
        "normalized": checkpoint_normalized.isdisjoint(calibration_normalized),
        "strict": checkpoint_strict.isdisjoint(calibration_strict),
        "lineage_group": checkpoint_groups.isdisjoint(calibration_groups),
        "near": not any(near.query(row) for row in calibration),
        "validation_component": checkpoint_components.isdisjoint(
            calibration_components
        ),
    }
    if not all(disjointness.values()):
        raise ValueError(f"validation partition is not disjoint: {disjointness}")
    if len(checkpoint) + len(calibration) != len(prepared):
        raise ValueError("validation partition lost records")

    def negative_evidence(rows: list[dict]) -> dict:
        negatives = [row for row in rows if row["label"] == 0]
        rows_by_channel = Counter(row["input_channel"] for row in negatives)
        rows_by_source = Counter(row["source"] for row in negatives)
        components_by_channel = defaultdict(set)
        components_by_source = defaultdict(set)
        for row in negatives:
            component_id = row["validation_component_id"]
            components_by_channel[row["input_channel"]].add(component_id)
            components_by_source[row["source"]].add(component_id)
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

    by_label_source = {}
    for key, total in sorted(totals.items()):
        label, source = key
        selected = checkpoint_counts[key]
        by_label_source[f"{label}|{source}"] = {
            "total": total,
            "checkpoint_selection": selected,
            "calibration": total - selected,
            "target_checkpoint": target[key],
        }
    return roles, {
        "leakage_fingerprint": LEAKAGE_NORMALIZATION_METHOD,
        "target_checkpoint_fraction": checkpoint_fraction,
        "actual_checkpoint_fraction": len(checkpoint) / len(prepared),
        "total_rows": len(prepared),
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
        },
        "by_label_source": by_label_source,
        "disjointness": disjointness,
    }
