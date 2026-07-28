"""Evaluate a maintained mmBERT run without promoting it into authorization."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..detector import choose_threshold
from .core import (
    ATTENTION_IMPLEMENTATION,
    INSTRUCTION_SUBVERSION_TAGS,
    MODEL_ID,
    MODEL_REVISION,
    file_sha256,
    load_base_model,
    new_head,
    score_texts,
    source_provenance,
)
from .data import (
    batches,
    canonical_rows,
    external_rows,
    is_checkpoint_group,
    routing_views,
)


def _load_run(run: Path) -> tuple[dict, object, object, object]:
    from safetensors.torch import load_file

    run = run.resolve()
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mode = result.get("adaptation")
    artifact = result.get("artifact", {})
    head_name = artifact.get("head")
    if not isinstance(head_name, str):
        raise ValueError("run has no head artifact")
    head_path = (run / head_name).resolve()
    if (
        result.get("purpose") != "maintained full-data advisory mmBERT training"
        or result.get("model_id") != MODEL_ID
        or result.get("model_revision") != MODEL_REVISION
        or result.get("attention_implementation") != ATTENTION_IMPLEMENTATION
        or result.get("normalization") != "strict"
        or mode not in {"frozen", "lora"}
        or not head_path.is_relative_to(run)
        or file_sha256(head_path) != artifact.get("head_sha256")
    ):
        raise ValueError("run contract failed")

    encoder, tokenizer = load_base_model()
    if mode == "lora":
        from peft import PeftModel, get_peft_model_state_dict

        adapter_name = artifact.get("adapter")
        if not isinstance(adapter_name, str) or not isinstance(
            artifact.get("adapter_files"), dict
        ):
            raise ValueError("run has no LoRA adapter artifact")
        adapter = (run / adapter_name).resolve()
        if not adapter.is_relative_to(run) or not adapter.is_dir():
            raise ValueError("adapter path escapes the run")
        actual = {
            path.name: file_sha256(path)
            for path in sorted(adapter.iterdir())
            if path.is_file()
        }
        if actual != artifact["adapter_files"]:
            raise ValueError("adapter hash mismatch")
        encoder = PeftModel.from_pretrained(encoder, adapter, is_trainable=False)
        modules = sorted(
            name
            for name, module in encoder.named_modules()
            if hasattr(module, "lora_A")
        )
        parameters = sum(
            value.numel() for value in get_peft_model_state_dict(encoder).values()
        )
        if (
            modules != result["lora"]["targeted_modules"]
            or parameters != result["lora"]["adapter_parameters"]
        ):
            raise ValueError("LoRA identity mismatch")
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    head = new_head(encoder.config.hidden_size, result["seed"]).to("cuda")
    head.load_state_dict(load_file(str(head_path)), strict=True)
    head.eval()
    return result, encoder, tokenizer, head


def _score(
    rows,
    encoder,
    tokenizer,
    head,
    *,
    batch_size: int,
) -> dict:
    labels = []
    scores = []
    sources = []
    channels = []
    pair_ids = []
    tags = []
    for batch in batches(rows, 512):
        values = score_texts(
            encoder,
            tokenizer,
            head,
            [row["text"] for row in batch],
            batch_size=batch_size,
        )
        labels.extend(row["label"] for row in batch)
        scores.extend(values)
        sources.extend(row["source"] for row in batch)
        channels.extend(row["input_channel"] for row in batch)
        pair_ids.extend(row.get("pair_id") for row in batch)
        tags.extend(row.get("security_tags", ()) for row in batch)
    if not labels:
        raise ValueError("evaluation population is empty")
    return {
        "labels": np.asarray(labels, dtype=np.int8),
        "scores": np.asarray(scores, dtype=np.float64),
        "sources": np.asarray(sources),
        "channels": np.asarray(channels),
        "pair_ids": pair_ids,
        "tags": tags,
    }


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    result = {
        "rows": len(labels),
        "positive": int(positives.sum()),
        "negative": int(negatives.sum()),
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "recall": true_positive / positives.sum() if positives.any() else None,
        "fpr": false_positive / negatives.sum() if negatives.any() else None,
        "precision": (true_positive / predictions.sum() if predictions.any() else 0.0),
        "roc_auc": None,
        "pr_auc": None,
    }
    if positives.any() and negatives.any():
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
    return result


def _by_value(scored: dict, key: str, threshold: float) -> dict:
    result = {}
    for value in sorted(set(scored[key])):
        selected = scored[key] == value
        result[str(value)] = _metrics(
            scored["labels"][selected],
            scored["scores"][selected],
            threshold,
        )
    return result


def _pair_metrics(scored: dict) -> dict:
    grouped = defaultdict(dict)
    for pair_id, label, score in zip(
        scored["pair_ids"],
        scored["labels"],
        scored["scores"],
        strict=True,
    ):
        if pair_id is not None:
            grouped[pair_id][int(label)] = float(score)
    complete = [pair for pair in grouped.values() if set(pair) == {0, 1}]
    return {
        "pairs": len(complete),
        "attack_scores_higher": (
            float(np.mean([pair[1] > pair[0] for pair in complete]))
            if complete
            else None
        ),
        "mean_attack_minus_benign": (
            float(np.mean([pair[1] - pair[0] for pair in complete]))
            if complete
            else None
        ),
    }


def _by_subtype(scored: dict, threshold: float) -> dict:
    result = {}
    for tag in INSTRUCTION_SUBVERSION_TAGS:
        selected = np.asarray([tag in tags for tags in scored["tags"]])
        if selected.any():
            result[tag] = _metrics(
                scored["labels"][selected],
                scored["scores"][selected],
                threshold,
            )
    return result


def evaluate(
    run: Path,
    *,
    data_dir: Path,
    external_dir: Path,
    output: Path,
    batch_size: int,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    result, encoder, tokenizer, head = _load_run(run)
    views = routing_views(data_dir)
    external, _ = external_rows(external_dir)
    validation_path, validation_spec = views["validation"]
    calibration = _score(
        (
            row
            for row in canonical_rows(
                validation_path,
                validation_spec,
                split="validation",
            )
            if not is_checkpoint_group(row["group_id"])
        ),
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    threshold = choose_threshold(
        calibration["labels"],
        calibration["scores"],
        0.01,
    )
    dev_path, dev_spec = views["dev_test"]
    dev = _score(
        canonical_rows(dev_path, dev_spec, split="dev_test"),
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    promptshield = _score(
        external["promptshield_test"],
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )
    sep = _score(
        external["sep"],
        encoder,
        tokenizer,
        head,
        batch_size=batch_size,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=".mmbert-eval-"))
    try:
        scored_sets = {
            "calibration": calibration,
            "dev_test": dev,
            "promptshield": promptshield,
            "sep": sep,
        }
        score_slices = {}
        arrays = []
        offset = 0
        for name, scored in scored_sets.items():
            stop = offset + len(scored["labels"])
            score_slices[name] = [offset, stop]
            arrays.append(np.column_stack((scored["labels"], scored["scores"])))
            offset = stop
        arrays_path = temporary / "scores.npy"
        np.save(arrays_path, np.concatenate(arrays), allow_pickle=False)
        report = {
            "schema_version": 1,
            "purpose": "advisory mmBERT development evaluation",
            "advisory_only": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "adaptation": result["adaptation"],
            "run_result_sha256": file_sha256(run / "result.json"),
            "inputs": {
                "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
                "external_manifest_sha256": file_sha256(external_dir / "manifest.json"),
                "routing_views": {
                    split: {
                        "sha256": spec["sha256"],
                        "rows": spec["rows"],
                    }
                    for split, (_, spec) in views.items()
                },
            },
            "provenance": source_provenance(
                Path(__file__),
                Path(__file__).with_name("core.py"),
                Path(__file__).with_name("data.py"),
                Path(__file__).resolve().parents[1] / "detector.py",
                Path(__file__).resolve().parents[2] / "data.py",
                Path(__file__).resolve().parents[2] / "normalization.py",
                Path(__file__).resolve().parents[2] / "overlap.py",
            ),
            "calibration": {
                "selection": "canonical validation groups excluded from checkpoint selection",
                "target_fpr": 0.01,
                "metrics": _metrics(
                    calibration["labels"],
                    calibration["scores"],
                    threshold,
                ),
            },
            "canonical_dev_test": {
                "metrics": _metrics(dev["labels"], dev["scores"], threshold),
                "by_source": _by_value(dev, "sources", threshold),
                "by_channel": _by_value(dev, "channels", threshold),
                "by_instruction_subtype": _by_subtype(dev, threshold),
            },
            "promptshield_test": _metrics(
                promptshield["labels"],
                promptshield["scores"],
                threshold,
            ),
            "sep": {
                "metrics": _metrics(sep["labels"], sep["scores"], threshold),
                "pairs": _pair_metrics(sep),
            },
            "scores": {
                "path": "scores.npy",
                "sha256": file_sha256(arrays_path),
                "columns": ["label", "score"],
                "slices": score_slices,
            },
            "limitations": [
                "PromptShield test and SEP are already-open development evidence.",
                "This is not a prospective final test or a source-held-out retrain.",
                "The threshold is descriptive and is not approved for blocking.",
            ],
        }
        (temporary / "evaluation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return output
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("artifacts/mmbert/data"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.batch_size):
        raise ValueError("batch size must be positive")
    output = args.output or args.run / "evaluation"
    print(
        evaluate(
            args.run,
            data_dir=args.data_dir,
            external_dir=args.external_dir,
            output=output,
            batch_size=args.batch_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
