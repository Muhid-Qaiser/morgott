"""Verify the maintained advisory mmBERT serving package."""

from __future__ import annotations

import json
from pathlib import Path

from .core import (
    ATTENTION_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
)

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
    artifact_format = entry.get("artifact_format")
    if artifact_format != "maintained-snapshot-v1":
        raise ValueError("unsupported model artifact format")
    adaptation = entry.get("adaptation")
    if adaptation != "lora":
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
        or result.get("adaptation") != adaptation
    ):
        raise ValueError("model result contract failed")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
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
        or any(
            not isinstance(training_inputs.get(name), str)
            or len(training_inputs[name]) != 64
            or evaluation_inputs.get(name) != training_inputs[name]
            for name in required_digests
        )
    ):
        raise ValueError("model evaluation contract failed")
    # Snapshot packaging wraps result.json, so the archived evaluation remains
    # bound to the original training result recorded inside it.
    packaging = result.get("packaging")
    checkpoint = evaluation.get("evaluated_checkpoint")
    weights = result.get("artifact", {}).get("weights_provenance")
    if (
        not isinstance(packaging, dict)
        or packaging.get("format") != "maintained-snapshot-v1"
        or not isinstance(packaging.get("source_result"), dict)
        or not isinstance(packaging.get("source_snapshot"), dict)
        or not isinstance(packaging.get("source_evaluation"), dict)
        or packaging["source_evaluation"].get("sha256") != entry["evaluation"]["sha256"]
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != packaging["source_snapshot"].get("sha256")
        or not isinstance(weights, dict)
        or weights.get("source") != "retained_snapshot"
        or weights.get("snapshot_sha256") != packaging["source_snapshot"].get("sha256")
        or weights.get("source_result_sha256")
        != packaging["source_result"].get("sha256")
        or checkpoint.get("epoch") != weights.get("epoch")
        or checkpoint.get("update") != weights.get("updates")
        or result.get("max_tokens") != 1024
    ):
        raise ValueError("maintained snapshot provenance contract failed")
    evaluation_result_sha256 = packaging["source_result"].get("sha256")
    if evaluation.get("run_result_sha256") != evaluation_result_sha256:
        raise ValueError("model evaluation result binding failed")
    _verify_runtime_sources(
        root,
        {
            "training": training_inputs,
            "evaluation": evaluation.get("provenance"),
        },
    )

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
        "adapter_sha256": dict(entry["adapter"]["files"]),
    }
