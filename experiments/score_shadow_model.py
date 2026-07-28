"""Score JSONL with one retained advisory mmBERT artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from eval_combined_generic_head import (
    _evaluator_source_paths,
    _load_model,
    _score_records,
)

ALLOWED_CHANNELS = {"direct_user", "untrusted_content"}
PINNED_MODEL_ID = "jhu-clsp/mmBERT-base"
PINNED_MODEL_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(root: Path, spec: dict, *, name: str) -> Path:
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{name} path escapes the artifact root")
    expected = spec.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or file_sha256(path) != expected
    ):
        raise ValueError(f"{name} hash mismatch")
    return path


def load_bundle(manifest_path: Path, model_key: str) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("advisory_only") is not True
        or manifest.get("purpose") != "advisory first-pass shadow models"
    ):
        raise ValueError("model manifest contract failed")
    entry = manifest.get("models", {}).get(model_key)
    if not isinstance(entry, dict):
        raise ValueError(f"unknown model key: {model_key}")
    adaptation = entry.get("adaptation")
    if adaptation not in {"frozen", "lora"}:
        raise ValueError("model adaptation contract failed")

    root = manifest_path.parent
    result_path = _verified_path(root, entry["result"], name="result")
    head_path = _verified_path(root, entry["head"], name="head")
    evaluation_path = _verified_path(root, entry["evaluation"], name="evaluation")
    evaluator_path = _verified_path(
        root,
        manifest["evidence"]["evaluator"],
        name="evaluator",
    )
    result = json.loads(result_path.read_text())
    if (
        result.get("model_id") != PINNED_MODEL_ID
        or result.get("model_revision") != PINNED_MODEL_REVISION
        or result.get("attention_implementation") != "sdpa"
        or result.get("normalization") != "strict"
        or result.get("generic_target") != "instruction_subversion"
        or result.get("artifact", {}).get("head_sha256") != entry["head"]["sha256"]
        or result.get("adaptation", "frozen") != adaptation
    ):
        raise ValueError("model result contract failed")

    evaluation = json.loads(evaluation_path.read_text())
    evaluation_inputs = evaluation.get("input_sha256", {})
    expected_adapter = entry.get("adapter", {}).get("files")
    if (
        evaluation.get("model_id") != result["model_id"]
        or evaluation.get("model_revision") != result["model_revision"]
        or evaluation.get("adaptation") != adaptation
        or evaluation_inputs.get("run_result") != entry["result"]["sha256"]
        or evaluation_inputs.get("head") != entry["head"]["sha256"]
        or evaluation_inputs.get("adapter_files") != expected_adapter
    ):
        raise ValueError("model evaluation contract failed")
    source_paths = _evaluator_source_paths(
        full="full-combined" in result.get("purpose", ""),
        adaptation=adaptation,
    )
    if source_paths["evaluator"].resolve() != evaluator_path:
        raise ValueError("registered evaluator path mismatch")
    for name, path in source_paths.items():
        digest = file_sha256(path)
        if evaluation_inputs.get(name) != digest:
            raise ValueError(f"scoring source hash mismatch: {name}")

    adapter_path = None
    if adaptation == "lora":
        adapter = entry.get("adapter")
        adapter_path = (root / adapter["path"]).resolve()
        if not adapter_path.is_relative_to(root) or not adapter_path.is_dir():
            raise ValueError("adapter path escapes the artifact root")
        files = adapter.get("files")
        if files != result.get("artifact", {}).get("adapter_files"):
            raise ValueError("adapter manifest differs from the training result")
        for filename, digest in files.items():
            _verified_path(
                root,
                {
                    "path": str(Path(adapter["path"]) / filename),
                    "sha256": digest,
                },
                name=f"adapter {filename}",
            )

    return {
        "model_key": model_key,
        "adaptation": adaptation,
        "result": result,
        "result_sha256": entry["result"]["sha256"],
        "evaluation_sha256": entry["evaluation"]["sha256"],
        "head_path": head_path,
        "head_sha256": entry["head"]["sha256"],
        "adapter_path": adapter_path,
        "adapter_sha256": (
            dict(entry["adapter"]["files"]) if adaptation == "lora" else None
        ),
    }


def _read_records(path: Path) -> list[dict]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or row_id in seen:
                raise ValueError(f"invalid or duplicate id on line {line_number}")
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise ValueError(f"invalid text on line {line_number}")
            if row.get("input_channel") not in ALLOWED_CHANNELS:
                raise ValueError(f"invalid input_channel on line {line_number}")
            seen.add(row_id)
            records.append(row)
    if not records:
        raise ValueError("input contains no records")
    return records


def score_file(
    manifest_path: Path,
    model_key: str,
    input_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_path}")
    bundle = load_bundle(manifest_path, model_key)
    records = _read_records(input_path)
    encoder, tokenizer, head = _load_model(bundle)
    scores = _score_records(
        encoder,
        tokenizer,
        head,
        records,
        max_tokens=bundle["result"]["max_tokens"],
        token_budget=bundle["result"]["token_budget"],
    )
    if len(scores) != len(records) or any(not math.isfinite(float(x)) for x in scores):
        raise ValueError("model returned invalid scores")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row, score in zip(records, scores, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "input_channel": row["input_channel"],
                            "score": float(score),
                            "model": model_key,
                            "model_id": bundle["result"]["model_id"],
                            "model_revision": bundle["result"]["model_revision"],
                            "artifacts": {
                                "result_sha256": bundle["result_sha256"],
                                "evaluation_sha256": bundle["evaluation_sha256"],
                                "head_sha256": bundle["head_sha256"],
                                "adapter_sha256": bundle["adapter_sha256"],
                            },
                        }
                    )
                    + "\n"
                )
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    score_file(args.manifest, args.model, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
