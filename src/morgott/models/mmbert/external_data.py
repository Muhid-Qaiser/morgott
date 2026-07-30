"""Fetch and project PromptShield and SEP without touching the canonical corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ...data import _fetch, text_hash
from ...normalization import strict_normalize
from ...overlap import NearIndex
from .core import file_sha256, source_provenance
from .data import EXTERNAL_DATA_SCHEMA_VERSION

PROMPTSHIELD_REPOSITORY = "hendzh/PromptShield"
PROMPTSHIELD_REVISION = "a5234cb1f5cdb256600cab64b8c961195b5e8404"
PROMPTSHIELD_LICENSE = "Apache-2.0"
PROMPTSHIELD = {
    "train": (
        "aa33c3ffcc27bd07c0a233b52f1b8c3cbdb30606ce2412da06a88b5290cdc7b6",
        18_909,
    ),
    "validation": (
        "1d93d90d57d3ef44ed0c546fbc04d66324436c5fcd32e7fcb940ceed270fbe77",
        1_000,
    ),
    "test": (
        "526207c2485829d9961407011d7f4cd929569e7f285dc8396b3f385e0608bc70",
        23_516,
    ),
}
PROMPTSHIELD_FILTERED_ROWS = {"train": 18_284, "validation": 998}
SEP_REVISION = "7606c0696f20f5aa433169fd2221f76852d1d4f5"
SEP_URL = (
    "https://raw.githubusercontent.com/"
    "egozverev/Should-It-Be-Executed-Or-Processed/"
    f"{SEP_REVISION}/SEP_dataset/SEP_dataset.json"
)
SEP_RAW_SHA256 = "9f81a52ba089073251793f8499fbb79bb4112e94cf48d31b7f8805e8b44fa3ce"
SEP_ROWS = 18_320
SEP_LICENSE = "CC-BY-4.0"


def _strict_hash(text: str) -> str:
    return hashlib.sha256(strict_normalize(text).encode()).hexdigest()


def _promptshield_rows(split: str, rows: Iterable[dict]) -> list[dict]:
    projected = []
    for index, row in enumerate(rows):
        if (
            set(row) != {"prompt", "label"}
            or not isinstance(row["prompt"], str)
            or not row["prompt"]
            or type(row["label"]) is not int
            or row["label"] not in (0, 1)
        ):
            raise ValueError(f"invalid PromptShield row {split}:{index}")
        projected.append(
            {
                "id": f"promptshield:{split}:{index}",
                "text": row["prompt"],
                "label": row["label"],
                "input_channel": "direct_user",
                "source": "promptshield",
                "source_revision": PROMPTSHIELD_REVISION,
                "license": PROMPTSHIELD_LICENSE,
            }
        )
    return projected


def _sep_rows(rows: Iterable[dict]) -> list[dict]:
    projected = []
    required = {
        "system_prompt_clean",
        "system_prompt_instructed",
        "prompt_clean",
        "prompt_instructed",
        "witness",
        "info",
    }
    for index, row in enumerate(rows):
        if set(row) != required or not isinstance(row["info"], dict):
            raise ValueError(f"invalid SEP row {index}")
        pair_id = f"sep:{index}"
        for suffix, label, field in (
            ("clean", 0, "prompt_clean"),
            ("instructed", 1, "prompt_instructed"),
        ):
            text = row[field]
            if not isinstance(text, str) or not text:
                raise ValueError(f"invalid SEP text {index}:{field}")
            projected.append(
                {
                    "id": f"{pair_id}:{suffix}",
                    "text": text,
                    "label": label,
                    "input_channel": "untrusted_content",
                    "pair_id": pair_id,
                    "source": "sep",
                    "source_revision": SEP_REVISION,
                    "license": SEP_LICENSE,
                }
            )
    return projected


def _filter(
    candidates: list[dict],
    references: list[dict],
) -> tuple[list[dict], dict[str, int]]:
    normalized = {text_hash(row["text"]) for row in references}
    strict = {_strict_hash(row["text"]) for row in references}
    near = NearIndex()
    for row in references:
        near.add(row, dataset="heldout")

    kept = []
    removed = Counter()
    seen = set()
    for row in candidates:
        normalized_hash = text_hash(row["text"])
        strict_hash = _strict_hash(row["text"])
        if normalized_hash in normalized:
            removed["normalized_exact"] += 1
        elif strict_hash in strict:
            removed["strict_exact"] += 1
        elif near.query(row):
            removed["near"] += 1
        elif strict_hash in seen:
            removed["duplicate"] += 1
        else:
            seen.add(strict_hash)
            kept.append(row)
    return kept, dict(sorted(removed.items()))


def _write(path: Path, rows: list[dict]) -> dict:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "rows": len(rows),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
    }


def _prepare(directory: Path) -> dict:
    splits = {}
    raw_inputs = {}
    for split, (expected_hash, expected_rows) in PROMPTSHIELD.items():
        url = (
            "https://huggingface.co/datasets/"
            f"{PROMPTSHIELD_REPOSITORY}/resolve/{PROMPTSHIELD_REVISION}/{split}.json"
        )
        raw, _ = _fetch(url, expected_sha256=expected_hash)
        splits[split] = _promptshield_rows(split, json.loads(raw))
        if len(splits[split]) != expected_rows:
            raise ValueError(f"PromptShield {split} row count mismatch")
        raw_inputs[split] = {"sha256": expected_hash, "rows": expected_rows}

    splits["train"], train_removed = _filter(
        splits["train"],
        [*splits["validation"], *splits["test"]],
    )
    splits["validation"], validation_removed = _filter(
        splits["validation"],
        splits["test"],
    )
    for split, expected_rows in PROMPTSHIELD_FILTERED_ROWS.items():
        if len(splits[split]) != expected_rows:
            raise ValueError(f"PromptShield {split} filtered population mismatch")

    sep_raw, _ = _fetch(SEP_URL, expected_sha256=SEP_RAW_SHA256)
    sep = _sep_rows(json.loads(sep_raw))
    if len(sep) != SEP_ROWS:
        raise ValueError("SEP row count mismatch")

    outputs = {
        f"promptshield_{split}": _write(
            directory / f"promptshield_{split}.jsonl",
            rows,
        )
        for split, rows in splits.items()
    }
    outputs["sep"] = _write(directory / "sep.jsonl", sep)
    manifest = {
        "schema_version": EXTERNAL_DATA_SCHEMA_VERSION,
        "purpose": "mmBERT training and already-open external development evaluation",
        "inputs": {
            "promptshield": {
                "repository": PROMPTSHIELD_REPOSITORY,
                "revision": PROMPTSHIELD_REVISION,
                "license": PROMPTSHIELD_LICENSE,
                "splits": raw_inputs,
            },
            "sep": {
                "url": SEP_URL,
                "revision": SEP_REVISION,
                "sha256": SEP_RAW_SHA256,
                "rows": SEP_ROWS,
                "license": SEP_LICENSE,
            },
        },
        "filter": {
            "promptshield_train_against_validation_and_test": train_removed,
            "promptshield_validation_against_test": validation_removed,
        },
        "outputs": outputs,
        "provenance": source_provenance(
            Path(__file__),
            Path(__file__).with_name("core.py"),
            Path(__file__).resolve().parents[2] / "data.py",
            Path(__file__).resolve().parents[2] / "normalization.py",
            Path(__file__).resolve().parents[2] / "overlap.py",
        ),
        "limitations": [
            "SimHash near matching is conservative and not exhaustive.",
            "PromptShield does not publish row-level source-family lineage.",
            "PromptShield test and SEP are already-open development evidence.",
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".mmbert-data-"))
    try:
        manifest = _prepare(temporary)
        os.replace(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/mmbert/data"))
    args = parser.parse_args()
    manifest = prepare(args.output)
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
