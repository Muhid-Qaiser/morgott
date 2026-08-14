"""Materialize the retained 1,024-token checkpoint into safe serving files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from morgott.models.mmbert.core import (
    ATTENTION_IMPLEMENTATION,
    MODEL_ID,
    MODEL_REVISION,
    add_lora,
    file_sha256,
    new_head,
)
from morgott.models.mmbert.evaluate import _load_snapshot

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEY = "mmbert-lora-full-ctx1024-u17000-s42"
SOURCE_RESULT = (
    ROOT / "reports/provenance/mmbert-context-results-20260812/records/"
    "training_runs.train_context_1024.result.json"
)
SOURCE_EVALUATION = (
    ROOT / "reports/provenance/mmbert-context-results-20260812/records/"
    "evaluations.full_panel.3.artifact.json"
)
SOURCE_SNAPSHOT = (
    ROOT / "reports/provenance/mmbert-context-checkpoints-20260812/"
    "trainctx1024-update-017000.pt"
)
SOURCE_RESULT_SHA256 = (
    "f9f683fbf2aa8c5d0ab0490eebdd9707349eccce4b8b69b815bcf02b56957df6"
)
SOURCE_EVALUATION_SHA256 = (
    "e2d6162ada5a67969a4b3bab23f7c83aa1b69cba15f72b4e0a4b10383aff0a9b"
)
SOURCE_SNAPSHOT_SHA256 = (
    "6de8784ecdb3f954f372f3411f9553889a9cfb8d369b72db20597d4924281774"
)


def _verified_json(path: Path, expected_sha256: str, *, name: str) -> dict:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"{name} hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def package_snapshot(
    output: Path = ROOT / "artifacts/models" / MODEL_KEY,
) -> dict:
    """Validate the archived snapshot and publish only safe serving artifacts."""

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel

    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("model artifact must remain inside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to replace model artifact: {output}")

    source_result = _verified_json(
        SOURCE_RESULT,
        SOURCE_RESULT_SHA256,
        name="source training result",
    )
    source_evaluation = _verified_json(
        SOURCE_EVALUATION,
        SOURCE_EVALUATION_SHA256,
        name="source evaluation",
    )
    if not SOURCE_SNAPSHOT.is_file() or file_sha256(SOURCE_SNAPSHOT) != (
        SOURCE_SNAPSHOT_SHA256
    ):
        raise ValueError("source snapshot hash mismatch")
    checkpoint = source_evaluation.get("evaluated_checkpoint")
    if (
        source_result.get("model_id") != MODEL_ID
        or source_result.get("model_revision") != MODEL_REVISION
        or source_result.get("attention_implementation") != ATTENTION_IMPLEMENTATION
        or source_result.get("adaptation") != "lora"
        or source_result.get("max_tokens") != 1024
        or source_evaluation.get("run_result_sha256") != SOURCE_RESULT_SHA256
        or source_evaluation.get("training_max_tokens") != 1024
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != SOURCE_SNAPSHOT_SHA256
        or checkpoint.get("update") != 17000
        or checkpoint.get("epoch") != 3
        or checkpoint.get("role") != "pre_registered_comparison"
    ):
        raise ValueError("retained snapshot provenance contract failed")

    encoder = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        dtype=torch.float32,
    )
    encoder = add_lora(encoder)
    head = new_head(encoder.config.hidden_size, source_result["seed"])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{MODEL_KEY}-"))
    try:
        snapshot_alias = temporary / "update-17000.pt"
        os.link(SOURCE_SNAPSHOT, snapshot_alias)
        restored = _load_snapshot(
            snapshot_alias,
            result=source_result,
            encoder=encoder,
            head=head,
        )
        if restored != {
            "sha256": SOURCE_SNAPSHOT_SHA256,
            "update": 17000,
            "epoch": 3,
            "role": "pre_registered_comparison",
        }:
            raise ValueError("restored snapshot identity changed")
        snapshot_alias.unlink()

        head_path = temporary / "head.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in head.state_dict().items()
            },
            str(head_path),
        )
        adapter = temporary / "adapter"
        encoder.save_pretrained(adapter, safe_serialization=True)
        (adapter / "README.md").unlink()
        adapter_files = {
            path.name: file_sha256(path)
            for path in sorted(adapter.iterdir())
            if path.is_file()
        }
        if set(adapter_files) != {
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise ValueError("PEFT adapter output contract changed")

        evaluation_dir = temporary / "evaluation"
        evaluation_dir.mkdir()
        shutil.copyfile(SOURCE_EVALUATION, evaluation_dir / "evaluation.json")

        result = {
            name: source_result[name]
            for name in (
                "adaptation",
                "attention_implementation",
                "generic_target",
                "max_tokens",
                "model_id",
                "model_revision",
                "normalization",
                "provenance",
                "purpose",
                "schema_version",
                "seed",
                "token_budget",
            )
        }
        result.update(
            artifact={
                "adapter": "adapter",
                "adapter_files": adapter_files,
                "encoder": None,
                "encoder_sha256": None,
                "head": "head.safetensors",
                "head_sha256": file_sha256(head_path),
                "weights_provenance": {
                    "source": "retained_snapshot",
                    "epoch": 3,
                    "updates": 17000,
                    "snapshot_sha256": SOURCE_SNAPSHOT_SHA256,
                    "source_result_sha256": SOURCE_RESULT_SHA256,
                },
            },
            lora={
                name: source_result["lora"][name]
                for name in ("adapter_parameters", "targeted_modules")
            },
            packaging={
                "format": "maintained-snapshot-v1",
                "source_result": {
                    "path": str(SOURCE_RESULT.relative_to(ROOT)),
                    "sha256": SOURCE_RESULT_SHA256,
                },
                "source_snapshot": {
                    "path": str(SOURCE_SNAPSHOT.relative_to(ROOT)),
                    "sha256": SOURCE_SNAPSHOT_SHA256,
                },
                "source_evaluation": {
                    "path": str(SOURCE_EVALUATION.relative_to(ROOT)),
                    "sha256": SOURCE_EVALUATION_SHA256,
                },
            },
        )
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/models" / MODEL_KEY,
    )
    args = parser.parse_args(argv)
    result = package_snapshot(args.output)
    print(
        json.dumps(
            {
                "model_key": MODEL_KEY,
                "head_sha256": result["artifact"]["head_sha256"],
                "adapter_files": result["artifact"]["adapter_files"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
