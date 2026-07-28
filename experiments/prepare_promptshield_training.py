"""Build leakage-reduced PromptShield train and validation artifacts.

The public test split remains untouched.
This script does not modify the canonical morgott corpus.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strict_normalize import strict_normalize  # noqa: E402

from morgott.data import text_hash  # noqa: E402
from morgott.overlap import NearIndex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "artifacts/external_eval_data/promptshield"
OUTPUT = REPO_ROOT / "artifacts/promptshield_training"
INPUTS = {
    "train": (
        "2f3c2d0b5cb79594dc54f5aad542fb5dffe598edc70b37b37dca74a8b74670ad",
        18_909,
    ),
    "validation": (
        "e19f3c21331aa1aab48887ae90d14711cc749f336dfd3ff6aa949f35280a1597",
        1_000,
    ),
    "test": (
        "c763dcde8cc9921613476887b43f12917229d1e5e6cfa29c07ee5dc36311abf6",
        23_516,
    ),
}
EXPECTED = {
    "train": {"rows": 18_284, "labels": {"0": 9_456, "1": 8_828}},
    "validation": {"rows": 998, "labels": {"0": 497, "1": 501}},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(name: str) -> list[dict]:
    path = SOURCE / f"{name}.jsonl"
    expected_hash, expected_rows = INPUTS[name]
    if file_sha256(path) != expected_hash:
        raise ValueError(f"{name} input hash mismatch")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count mismatch")
    return rows


def strict_hash(text: str) -> str:
    return hashlib.sha256(strict_normalize(text).encode()).hexdigest()


def filter_against(
    candidates: list[dict], references: list[dict]
) -> tuple[list[dict], Counter]:
    reference_normalized = {text_hash(row["prompt"]) for row in references}
    reference_strict = {strict_hash(row["prompt"]) for row in references}
    near = NearIndex()
    for row in references:
        near.add(
            {"id": row["id"], "text": row["prompt"], "label": row["label"]},
            dataset="heldout",
        )
    kept = []
    removed = Counter()
    for row in candidates:
        normalized = text_hash(row["prompt"])
        if normalized in reference_normalized:
            removed["normalized_exact"] += 1
        elif strict_hash(row["prompt"]) in reference_strict:
            removed["strict_exact"] += 1
        elif near.query(
            {"id": row["id"], "text": row["prompt"], "label": row["label"]}
        ):
            removed["near"] += 1
        else:
            kept.append(row)
    return kept, removed


def deduplicate_strict(rows: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    kept = []
    for row in rows:
        digest = strict_hash(row["prompt"])
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(row)
    return kept, len(rows) - len(kept)


def write_split(name: str, rows: list[dict]) -> dict:
    path = OUTPUT / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
    }


def main() -> int:
    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    train, train_overlap = filter_against(train, [*validation, *test])
    validation, validation_overlap = filter_against(validation, test)
    train, train_duplicates = deduplicate_strict(train)
    validation, validation_duplicates = deduplicate_strict(validation)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": write_split("train", train),
        "validation": write_split("validation", validation),
    }
    for name, expected in EXPECTED.items():
        actual = {
            "rows": outputs[name]["rows"],
            "labels": outputs[name]["labels"],
        }
        if actual != expected:
            raise ValueError(
                f"{name} filtered population mismatch: "
                f"expected {expected}, found {actual}"
            )
    report = {
        "schema_version": 1,
        "purpose": "PromptShield-trained OOD benchmark development",
        "canonical_corpus_modified": False,
        "inputs": {
            name: {
                "path": str((SOURCE / f"{name}.jsonl").relative_to(REPO_ROOT)),
                "sha256": expected_hash,
                "rows": expected_rows,
            }
            for name, (expected_hash, expected_rows) in INPUTS.items()
        },
        "filter": {
            "train_against_validation_and_test": dict(train_overlap),
            "validation_against_test": dict(validation_overlap),
            "strict_duplicates_removed": {
                "train": train_duplicates,
                "validation": validation_duplicates,
            },
        },
        "outputs": outputs,
        "limitations": [
            "SimHash near matching is conservative and not exhaustive.",
            "PromptShield does not publish row-level source-family lineage.",
            "The public test has already influenced model-development decisions.",
        ],
    }
    (OUTPUT / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["filter"], indent=2))
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
