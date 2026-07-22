from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

from .data import normalize_text, text_hash

NEAR_BITS = 128
NEAR_BANDS = 8
NEAR_MAX_HAMMING = 6
NEAR_MIN_WORDS = 5
NEAR_MAX_CHARS = 24_000
NEAR_METHOD = (
    "128-bit SimHash over word unigrams+bigrams, eight 16-bit bands, "
    "Hamming <= 6, five-word minimum; documents over 24,000 characters use "
    "contiguous 8,000-character head, middle, and tail windows"
)
_WORD = re.compile(r"\w+", re.UNICODE)


def fingerprint(text: str) -> int | None:
    if len(text) <= NEAR_MAX_CHARS:
        windows = [text]
    else:
        window = NEAR_MAX_CHARS // 3
        middle = len(text) // 2
        windows = [
            text[:window],
            text[middle - window // 2 : middle + window // 2],
            text[-window:],
        ]
    word_windows = [_WORD.findall(normalize_text(window)) for window in windows]
    if sum(map(len, word_windows)) < NEAR_MIN_WORDS:
        return None
    features = [word for words in word_windows for word in words]
    features.extend(
        f"{left}\0{right}"
        for words in word_windows
        for left, right in zip(words, words[1:])
    )
    digests = b"".join(
        hashlib.blake2b(feature.encode(), digest_size=NEAR_BITS // 8).digest()
        for feature in features
    )
    bits = np.unpackbits(
        np.frombuffer(digests, dtype=np.uint8), bitorder="little"
    ).reshape(-1, NEAR_BITS)
    majority = bits.sum(axis=0) * 2 >= len(features)
    return int.from_bytes(np.packbits(majority, bitorder="little").tobytes(), "little")


class NearIndex:
    """Strict, deterministic near-overlap index; results are not exhaustive."""

    def __init__(self) -> None:
        self.records: list[tuple[int, str, dict]] = []
        self.buckets: list[dict[int, list[int]]] = [
            defaultdict(list) for _ in range(NEAR_BANDS)
        ]

    def add(self, row: dict, *, dataset: str, value: int | None = None) -> None:
        value = fingerprint(row["text"]) if value is None else value
        if value is None:
            return
        normalized_hash = row.get("normalized_text_sha256") or text_hash(row["text"])
        record = {
            "dataset": dataset,
            "id": row.get("id"),
            "source": row.get("source"),
            "label": row.get("label"),
            "routing_label": row.get("routing_label"),
            "normalized_text_sha256": normalized_hash,
        }
        index = len(self.records)
        self.records.append((value, normalized_hash, record))
        for band in range(NEAR_BANDS):
            key = (value >> (band * 16)) & 0xFFFF
            self.buckets[band][key].append(index)

    def query(self, row: dict, *, value: int | None = None) -> list[dict]:
        value = fingerprint(row["text"]) if value is None else value
        if value is None:
            return []
        normalized_hash = row.get("normalized_text_sha256") or text_hash(row["text"])
        candidates: set[int] = set()
        for band in range(NEAR_BANDS):
            key = (value >> (band * 16)) & 0xFFFF
            candidates.update(self.buckets[band].get(key, ()))
        matches = []
        for index in candidates:
            candidate, candidate_hash, record = self.records[index]
            distance = (candidate ^ value).bit_count()
            if candidate_hash != normalized_hash and distance <= NEAR_MAX_HAMMING:
                matches.append({**record, "hamming_distance": distance})
        return sorted(
            matches,
            key=lambda match: (
                match["hamming_distance"],
                str(match["dataset"]),
                str(match["id"]),
            ),
        )


def audit_near_overlaps(
    reference_sets: dict[str, list[dict]], candidate_sets: dict[str, list[dict]]
) -> list[dict]:
    index = NearIndex()
    for dataset, rows in reference_sets.items():
        for row in rows:
            index.add(row, dataset=dataset)

    overlaps = []
    for dataset, rows in candidate_sets.items():
        for row in rows:
            matches = index.query(row)
            if not matches:
                continue
            overlaps.append(
                {
                    "candidate_dataset": dataset,
                    "candidate_id": row.get("id"),
                    "candidate_source": row.get("source"),
                    "candidate_label": row.get("label"),
                    "candidate_routing_label": row.get("routing_label"),
                    "candidate_normalized_text_sha256": row.get(
                        "normalized_text_sha256"
                    )
                    or text_hash(row["text"]),
                    "matches": matches,
                }
            )
    return overlaps
