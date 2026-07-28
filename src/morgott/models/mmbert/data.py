from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from ...data import iter_verified_jsonl, text_hash
from ...normalization import strict_normalize
from ...overlap import NearIndex, fingerprint
from .core import INSTRUCTION_SUBVERSION_TAGS, file_sha256

PAIR_ARCHIVE_SHA256 = "0aa08878c3096b402cf6ee309a50b730f7150b1aed78682cdf6e42a504da13d3"
PAIR_CONTENT_SHA256 = "8ec5c1c77b378688b190722f7d1fc51e9bef819ee9670948d2658f4a37082158"


def _strict_hash(text: str) -> str:
    return hashlib.sha256(strict_normalize(text).encode()).hexdigest()


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
    if manifest.get("schema_version") != 1:
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


def matched_pairs(path: Path) -> list[tuple[dict, dict]]:
    if file_sha256(path) != PAIR_ARCHIVE_SHA256:
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
            pair_id = f"matched:{index}"
            common = {
                "source": "matched_pairs",
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
    if digest.hexdigest() != PAIR_CONTENT_SHA256:
        raise ValueError("matched-pair content hash mismatch")
    return pairs


class OverlapGuard:
    def __init__(self, references: Iterable[dict]) -> None:
        self.normalized = set()
        self.strict = set()
        self.near = NearIndex()
        for row in references:
            normalized, strict, near = _overlap_values(row)
            self.normalized.add(normalized)
            self.strict.add(strict)
            if near is not None:
                self.near.add(
                    row,
                    dataset=row["source"],
                    value=near,
                    normalized_hash=normalized,
                )

    def add_exact(self, references: Iterable[dict]) -> None:
        for row in references:
            self.normalized.add(text_hash(row["text"]))
            self.strict.add(_strict_hash(row["text"]))

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
) -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    candidate_filter = _SmallSetFilter(candidates)
    for reference in references:
        candidate_filter.block(reference)
    return candidate_filter.result()


def profile_canonical(
    rows: Iterable[dict],
    guard: OverlapGuard,
    candidates: dict[str, list[dict]],
) -> tuple[
    Counter,
    dict[str, int],
    dict[str, tuple | None],
    dict[str, list[dict]],
    dict[str, dict[str, int]],
]:
    counts = Counter()
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
            owners[digest] = (row["id"], row["source"], row["label"])
            counts[(row["source"], row["label"])] += 1
        elif owner is None:
            removed["strict_label_conflict"] += 1
        elif owner[2] == row["label"]:
            removed["strict_duplicate"] += 1
        else:
            counts[(owner[1], owner[2])] -= 1
            owners[digest] = None
            removed["strict_label_conflict"] += 2
    if not counts or {label for _, label in counts} != {0, 1}:
        raise ValueError("canonical training population must contain both labels")
    candidate_kept, candidate_removed = candidate_filter.result()
    return (
        counts,
        dict(sorted(removed.items())),
        owners,
        candidate_kept,
        candidate_removed,
    )


def training_rows(
    rows: Iterable[dict],
    counts: Counter,
    owners: dict[str, tuple | None],
) -> Iterator[dict]:
    total = sum(counts.values())
    strata = len(counts)
    yielded = 0
    for row in rows:
        owner = owners.get(_strict_hash(row["text"]))
        if owner is None or owner[0] != row["id"]:
            continue
        key = (row["source"], row["label"])
        yielded += 1
        yield {**row, "weight": total / (strata * counts[key])}
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


def checkpoint_rows(rows: Iterable[dict]) -> list[dict]:
    selected = [row for row in rows if is_checkpoint_group(row["group_id"])]
    if {row["label"] for row in selected} != {0, 1}:
        raise ValueError("checkpoint split must contain both labels")
    return selected


def is_checkpoint_group(group_id: str) -> bool:
    return hashlib.sha256(group_id.encode()).digest()[0] < 64
