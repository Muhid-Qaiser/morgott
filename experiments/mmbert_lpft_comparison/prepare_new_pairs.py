#!/usr/bin/env python3
"""Build repository-disjoint SWE-rebench V2 clean/injected LP-FT pairs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from morgott.data import file_sha256, text_hash
from morgott.sources.tasks import _sensitive_text_reasons

REVISION = "475dd5e8703bb5fb22dd3c60b5d038b019eba1e0"
SOURCE_SHA256 = "0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad"
EXCLUDED_REPOSITORIES = Path(__file__).with_name("excluded_repositories.json")


def _excluded_repositories() -> set[str]:
    values = json.loads(EXCLUDED_REPOSITORIES.read_text(encoding="utf-8"))[
        "repositories"
    ]
    if not all(
        isinstance(value, str) and value == value.casefold() for value in values
    ):
        raise ValueError("excluded repositories must be case-folded strings")
    if values != sorted(set(values)):
        raise ValueError("excluded repositories must be sorted and unique")
    return set(values)


def _payloads(root: Path) -> list[str]:
    values = set()
    with gzip.open(
        root / "data-archive/matched_pairs_20260726.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            value = json.loads(line).get("attack_span")
            if isinstance(value, str) and value.strip():
                values.add(value.strip())
    if len(values) < 100:
        raise ValueError("registered payload pool is unexpectedly small")
    return sorted(values)


def _split(repository: str) -> str:
    bucket = hashlib.sha256(f"morgott-lpft-v1:{repository}".encode()).digest()[0] % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "dev_test"


def _attack(text: str, payload: str) -> tuple[str, int]:
    limit = min(len(text), 1_200)
    boundary = text.find("\n\n", 200, limit)
    if boundary < 0:
        boundary = limit
    insertion = f"\n\n[Untrusted instruction]\n{payload}\n[/Untrusted instruction]\n\n"
    # The recorded span start is one character late (the payload begins at
    # boundary + 26). Preserved as-is so rebuilds stay byte-identical to the
    # frozen archives pinned in artifacts/mmbert_lpft_new_data/manifest.json;
    # see that directory's metadata-correction.json.
    return text[:boundary] + insertion + text[boundary:], boundary + 27


def _write_pairs(path: Path, rows: list[dict]) -> dict:
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            for row in rows:
                line = (
                    json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
                )
                digest.update(line)
                handle.write(line)
    return {
        "path": path.name,
        "pairs": len(rows),
        "sha256": file_sha256(path),
        "content_sha256": digest.hexdigest(),
    }


def prepare(source: Path, output: Path, root: Path) -> None:
    if output.exists() or file_sha256(source) != SOURCE_SHA256:
        raise ValueError("output exists or pinned source digest changed")
    excluded = _excluded_repositories()
    payloads = _payloads(root)
    columns = ["instance_id", "repo", "base_commit", "problem_statement"]
    source_rows = pq.read_table(source, columns=columns).to_pylist()
    if len(source_rows) != 32_079:
        raise ValueError("SWE-rebench V2 population changed")

    pairs = {split: [] for split in ("train", "validation", "dev_test")}
    removed = Counter()
    seen = set()
    repositories = {split: set() for split in pairs}
    for row in sorted(source_rows, key=lambda value: value["instance_id"]):
        text = row.get("problem_statement")
        repository = row.get("repo")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(repository, str)
        ):
            removed["invalid"] += 1
            continue
        repository = repository.casefold()
        if repository in excluded:
            removed["evaluation_repository_overlap"] += 1
            continue
        if _sensitive_text_reasons(text):
            removed["privacy"] += 1
            continue
        normalized = text_hash(text)
        if normalized in seen:
            removed["normalized_duplicate"] += 1
            continue
        seen.add(normalized)
        payload = payloads[int(normalized, 16) % len(payloads)]
        attacked, span_start = _attack(text, payload)
        split = _split(repository)
        repositories[split].add(repository)
        pairs[split].append(
            {
                "attack": attacked,
                "attack_span": payload,
                "attack_span_start": span_start,
                "benign": text,
                "channel": "direct_user",
                "instance_id": row["instance_id"],
                "repository": repository,
                "source": "nebius/SWE-rebench-V2",
                "source_revision": REVISION,
            }
        )
    if not pairs["train"] or not pairs["validation"] or not pairs["dev_test"]:
        raise ValueError("repository split is empty")

    output.mkdir(parents=True)
    specs = {
        split: _write_pairs(output / split / "pairs.jsonl.gz", values)
        for split, values in pairs.items()
    }
    manifest = {
        "schema_version": 1,
        "source": {
            "repository": "nebius/SWE-rebench-V2",
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "rows": len(source_rows),
            "license": "CC-BY-4.0",
        },
        "construction": "problem_statement plus one deterministic known-span injected twin",
        "split_unit": "repository",
        "removed": dict(sorted(removed.items())),
        "outputs": {
            split: {**specs[split], "repositories": len(repositories[split])}
            for split in specs
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, Path(__file__).resolve().parents[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
