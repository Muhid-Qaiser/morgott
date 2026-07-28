"""Fetch and deterministically project the pinned PromptShield and SEP releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

from morgott.data import text_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/external_eval_data"
PROMPTSHIELD_REPO = "hendzh/PromptShield"
PROMPTSHIELD_REVISION = "a5234cb1f5cdb256600cab64b8c961195b5e8404"
PROMPTSHIELD = {
    "train": {
        "raw_sha256": "aa33c3ffcc27bd07c0a233b52f1b8c3cbdb30606ce2412da06a88b5290cdc7b6",
        "output_sha256": "2f3c2d0b5cb79594dc54f5aad542fb5dffe598edc70b37b37dca74a8b74670ad",
        "rows": 18_909,
    },
    "validation": {
        "raw_sha256": "1d93d90d57d3ef44ed0c546fbc04d66324436c5fcd32e7fcb940ceed270fbe77",
        "output_sha256": "e19f3c21331aa1aab48887ae90d14711cc749f336dfd3ff6aa949f35280a1597",
        "rows": 1_000,
    },
    "test": {
        "raw_sha256": "526207c2485829d9961407011d7f4cd929569e7f285dc8396b3f385e0608bc70",
        "output_sha256": "c763dcde8cc9921613476887b43f12917229d1e5e6cfa29c07ee5dc36311abf6",
        "rows": 23_516,
    },
}
SEP_REVISION = "7606c0696f20f5aa433169fd2221f76852d1d4f5"
SEP_URL = (
    "https://raw.githubusercontent.com/"
    "egozverev/Should-It-Be-Executed-Or-Processed/"
    f"{SEP_REVISION}/SEP_dataset/SEP_dataset.json"
)
SEP_RAW_SHA256 = "9f81a52ba089073251793f8499fbb79bb4112e94cf48d31b7f8805e8b44fa3ce"
SEP_OUTPUT_SHA256 = "0ddcfa5a7963f65f9fc8fdf63af10b9052685f87f0142c243a42a394d6e31a89"
SEP_ROWS = 18_320


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def promptshield_rows(split: str, rows: Iterable[dict]) -> Iterator[dict]:
    if split not in PROMPTSHIELD:
        raise ValueError(f"unknown PromptShield split: {split}")
    for index, row in enumerate(rows):
        if (
            set(row) != {"prompt", "label"}
            or not isinstance(row["prompt"], str)
            or type(row["label"]) is not int
            or row["label"] not in (0, 1)
        ):
            raise ValueError(f"invalid PromptShield row {split}:{index}")
        yield {
            "id": f"promptshield:{split}:{index}",
            "prompt": row["prompt"],
            "label": row["label"],
        }


def sep_rows(rows: Iterable[dict]) -> Iterator[dict]:
    for index, row in enumerate(rows):
        info = row.get("info")
        required = {
            "system_prompt_clean",
            "system_prompt_instructed",
            "prompt_clean",
            "prompt_instructed",
            "witness",
            "info",
        }
        if set(row) != required or not isinstance(info, dict):
            raise ValueError(f"invalid SEP row {index}")
        pair_id = f"sep:{index}"
        for suffix, label, field in (
            ("clean", 0, "prompt_clean"),
            ("instructed", 1, "prompt_instructed"),
        ):
            text = row[field]
            if not isinstance(text, str) or not text:
                raise ValueError(f"invalid SEP text {index}:{field}")
            yield {
                "id": f"{pair_id}:{suffix}",
                "text": text,
                "label": label,
                "system_prompt": row["system_prompt_clean"],
                "original_field": field,
                "pair_id": pair_id,
                "witness": row["witness"],
                "task_domain": info["type"],
                "subtask": info["subtask"],
                "probe_id": info["appended_task_id"],
                "probe_placement": info["appended_type"],
                "is_insistent": info["is_insistent"],
                "normalized_text_sha256": text_hash(text),
                "source": "sep",
            }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _prepare(directory: Path) -> None:
    from huggingface_hub import hf_hub_download

    promptshield = directory / "promptshield"
    sep = directory / "sep"
    promptshield.mkdir(parents=True)
    sep.mkdir()

    for split, spec in PROMPTSHIELD.items():
        raw_path = Path(
            hf_hub_download(
                repo_id=PROMPTSHIELD_REPO,
                repo_type="dataset",
                revision=PROMPTSHIELD_REVISION,
                filename=f"{split}.json",
            )
        )
        raw = raw_path.read_bytes()
        if sha256_bytes(raw) != spec["raw_sha256"]:
            raise ValueError(f"PromptShield {split} raw hash mismatch")
        output = promptshield / f"{split}.jsonl"
        count = _write_jsonl(
            output,
            promptshield_rows(split, json.loads(raw)),
        )
        if count != spec["rows"] or file_sha256(output) != spec["output_sha256"]:
            raise ValueError(f"PromptShield {split} projection mismatch")

    with urllib.request.urlopen(SEP_URL, timeout=60) as response:
        raw = response.read()
    if sha256_bytes(raw) != SEP_RAW_SHA256:
        raise ValueError("SEP raw hash mismatch")
    output = sep / "sep.jsonl"
    count = _write_jsonl(output, sep_rows(json.loads(raw)))
    if count != SEP_ROWS or file_sha256(output) != SEP_OUTPUT_SHA256:
        raise ValueError("SEP projection mismatch")

    (directory / "source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promptshield": {
                    "repository": PROMPTSHIELD_REPO,
                    "revision": PROMPTSHIELD_REVISION,
                    "license": "Apache-2.0",
                    "splits": PROMPTSHIELD,
                },
                "sep": {
                    "url": SEP_URL,
                    "revision": SEP_REVISION,
                    "raw_sha256": SEP_RAW_SHA256,
                    "output_sha256": SEP_OUTPUT_SHA256,
                    "rows": SEP_ROWS,
                    "dataset_license": "CC-BY-4.0",
                    "code_license": "MIT",
                },
            },
            indent=2,
        )
        + "\n"
    )


def prepare(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".external-eval-"))
    try:
        _prepare(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
