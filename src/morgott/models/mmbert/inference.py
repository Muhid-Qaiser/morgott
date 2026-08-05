"""Load and score the retained advisory mmBERT shadows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from .core import (
    ATTENTION_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
    load_base_model,
    new_head,
    score_texts,
)

ALLOWED_CHANNELS = {"direct_user", "untrusted_content"}
COMMON_EVIDENCE_SOURCES = {
    "canonical_text_helper",
    "descriptive_threshold_helper",
    "evaluator",
    "full_preparation_helper",
    "generic_preparation_helper",
    "strict_normalizer",
    "training_head_helper",
}
ADAPTATION_EVIDENCE_SOURCE = {
    "frozen": "full_training_helper",
    "lora": "lora_training_runner",
}
RUNTIME_SOURCE_PATHS = (
    "src/morgott/models/mmbert/core.py",
    "src/morgott/normalization.py",
)


def verified_artifact_path(root: Path, spec: object, *, name: str) -> Path:
    if not isinstance(spec, dict):
        raise ValueError(f"{name} registry entry is missing")
    relative = spec.get("path")
    expected = spec.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise ValueError(f"{name} registry entry is invalid")
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or len(expected) != 64
        or file_sha256(path) != expected
    ):
        raise ValueError(f"{name} hash mismatch")
    return path


def _source_provenance(provenance: dict, *, name: str) -> dict:
    sources = provenance.get("sources")
    if (
        not isinstance(sources, dict)
        or not sources
        or not isinstance(provenance.get("uv_lock_sha256"), str)
        or len(provenance["uv_lock_sha256"]) != 64
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            for path, digest in sources.items()
        )
    ):
        raise ValueError(f"{name} source provenance contract failed")
    return sources


def _verify_runtime_sources(root: Path, evidence: dict) -> None:
    sources = {
        name: _source_provenance(evidence[name], name=name)
        for name in ("training", "evaluation")
    }
    for path in RUNTIME_SOURCE_PATHS:
        digests = {source.get(path) for source in sources.values()}
        if None in digests or len(digests) != 1:
            raise ValueError(f"runtime source evidence mismatch: {path}")
        verified_artifact_path(
            root,
            {"path": path, "sha256": digests.pop()},
            name=f"runtime source {path}",
        )


def _verify_historical_evidence(
    manifest: dict,
    evaluation_inputs: dict,
    adaptation: str,
) -> str:
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("model evidence contract failed")
    source_commit = evidence.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("model source commit contract failed")
    sources = evidence.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("model source evidence contract failed")
    required = COMMON_EVIDENCE_SOURCES | {ADAPTATION_EVIDENCE_SOURCE[adaptation]}
    for name in required:
        spec = sources.get(name)
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("path"), str)
            or not isinstance(spec.get("sha256"), str)
            or len(spec["sha256"]) != 64
            or evaluation_inputs.get(name) != spec["sha256"]
        ):
            raise ValueError(f"model source evidence mismatch: {name}")
    if (
        evaluation_inputs.get("calibration_threshold_helper")
        != sources["evaluator"]["sha256"]
    ):
        raise ValueError("model calibration source evidence mismatch")
    return source_commit


def _verify_maintained_evidence(
    root: Path,
    entry: dict,
    result: dict,
    evaluation: dict,
) -> str:
    evidence = entry.get("source_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("training") != result.get("provenance")
        or evidence.get("evaluation") != evaluation.get("provenance")
    ):
        raise ValueError("maintained model source evidence mismatch")
    _verify_runtime_sources(root, evidence)
    encoded = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_bundle(manifest_path: Path, model_key: str) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("advisory_only") is not True
        or manifest.get("purpose") != "advisory first-pass shadow models"
    ):
        raise ValueError("model manifest contract failed")
    entry = manifest.get("models", {}).get(model_key)
    if not isinstance(entry, dict):
        raise ValueError(f"unknown model key: {model_key}")
    adaptation = entry.get("adaptation")
    if adaptation not in ADAPTATION_EVIDENCE_SOURCE:
        raise ValueError("model adaptation contract failed")

    root = manifest_path.parent
    result_path = verified_artifact_path(root, entry["result"], name="result")
    head_path = verified_artifact_path(root, entry["head"], name="head")
    evaluation_path = verified_artifact_path(
        root,
        entry["evaluation"],
        name="evaluation",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("model_id") != MODEL_ID
        or result.get("model_revision") != MODEL_REVISION
        or result.get("attention_implementation") != ATTENTION_IMPLEMENTATION
        or result.get("normalization") != "strict"
        or result.get("generic_target") != "instruction_subversion"
        or result.get("artifact", {}).get("head_sha256") != entry["head"]["sha256"]
        or result.get("adaptation", "frozen") != adaptation
    ):
        raise ValueError("model result contract failed")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    expected_adapter = entry.get("adapter", {}).get("files")
    source_evidence_sha256 = None
    if entry.get("artifact_format") == "maintained-v1":
        evaluation_inputs = evaluation.get("inputs", {})
        training_inputs = result.get("provenance", {})
        required_digests = (
            "data_manifest_sha256",
            "external_manifest_sha256",
            "pair_archive_sha256",
        )
        if (
            result.get("purpose") != "maintained full-data advisory mmBERT training"
            or evaluation.get("purpose") != "advisory mmBERT development evaluation"
            or evaluation.get("model_id") != MODEL_ID
            or evaluation.get("model_revision") != MODEL_REVISION
            or evaluation.get("adaptation") != adaptation
            or evaluation.get("run_result_sha256") != entry["result"]["sha256"]
            or any(
                not isinstance(training_inputs.get(name), str)
                or len(training_inputs[name]) != 64
                or evaluation_inputs.get(name) != training_inputs[name]
                for name in required_digests
            )
        ):
            raise ValueError("model evaluation contract failed")
        source_evidence_sha256 = _verify_maintained_evidence(
            root,
            entry,
            result,
            evaluation,
        )
        source_commit = None
    else:
        evaluation_inputs = evaluation.get("input_sha256", {})
        if (
            evaluation.get("model_id") != MODEL_ID
            or evaluation.get("model_revision") != MODEL_REVISION
            or evaluation.get("adaptation") != adaptation
            or evaluation_inputs.get("run_result") != entry["result"]["sha256"]
            or evaluation_inputs.get("head") != entry["head"]["sha256"]
            or evaluation_inputs.get("adapter_files") != expected_adapter
        ):
            raise ValueError("model evaluation contract failed")
        source_commit = _verify_historical_evidence(
            manifest,
            evaluation_inputs,
            adaptation,
        )

    adapter_path = None
    if adaptation == "lora":
        adapter = entry.get("adapter")
        adapter_path = (root / adapter["path"]).resolve()
        files = adapter.get("files")
        if (
            not adapter_path.is_relative_to(root)
            or not adapter_path.is_dir()
            or files != result.get("artifact", {}).get("adapter_files")
        ):
            raise ValueError("adapter manifest differs from the training result")
        for filename, digest in files.items():
            verified_artifact_path(
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
        "source_commit": source_commit,
        "source_evidence_sha256": source_evidence_sha256,
    }


def _read_records(path: Path) -> list[dict]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "id",
                "text",
                "input_channel",
            }:
                raise ValueError(f"invalid record schema on line {line_number}")
            row_id = row["id"]
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


def _load_model(bundle: dict):
    from safetensors.torch import load_file

    encoder, tokenizer = load_base_model()
    result = bundle["result"]
    if bundle["adaptation"] == "lora":
        from peft import PeftModel, get_peft_model_state_dict

        encoder = PeftModel.from_pretrained(
            encoder,
            bundle["adapter_path"],
            is_trainable=False,
        )
        modules = sorted(
            name
            for name, module in encoder.named_modules()
            if hasattr(module, "lora_A")
        )
        parameters = sum(
            value.numel() for value in get_peft_model_state_dict(encoder).values()
        )
        if (
            modules != sorted(result["lora"]["targeted_modules"])
            or parameters != result["lora"]["adapter_parameters"]
        ):
            raise ValueError("loaded LoRA adapter identity does not match the run")
    encoder.eval()
    encoder.gradient_checkpointing_disable()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    head = new_head(encoder.config.hidden_size, result["seed"]).to("cuda")
    head.load_state_dict(load_file(str(bundle["head_path"])), strict=True)
    head.eval()
    return encoder, tokenizer, head


def _score_records(
    encoder,
    tokenizer,
    head,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
) -> object:
    if max_tokens < 1 or token_budget < max_tokens:
        raise ValueError("invalid scoring token budget")
    return score_texts(
        encoder,
        tokenizer,
        head,
        [record["text"] for record in records],
        batch_size=max(1, token_budget // max_tokens),
    )


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
                            "model_id": MODEL_ID,
                            "model_revision": MODEL_REVISION,
                            "artifacts": {
                                "result_sha256": bundle["result_sha256"],
                                "evaluation_sha256": bundle["evaluation_sha256"],
                                "head_sha256": bundle["head_sha256"],
                                "adapter_sha256": bundle["adapter_sha256"],
                                "source_commit": bundle["source_commit"],
                                "source_evidence_sha256": bundle[
                                    "source_evidence_sha256"
                                ],
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
