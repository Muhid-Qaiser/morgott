"""Train the update-matched M1 plus PromptShield mmBERT LoRA gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from strict_normalize import strict_normalize
from train_combined_generic_head import (
    DEFAULT_SELECTION,
    REPO_ROOT,
    TARGET,
    VALIDATION_FEATURE_RECORD_CHUNK,
    VALIDATION_PREDICTION_BATCH_SIZE,
    _artifact_path,
    _bce_from_logits,
    _binary_metrics,
    _label_counts,
    _pool,
    _save_head,
    _scores,
    _verify_source_hashes,
    extract_features,
    file_sha256,
    load_records,
    new_head,
    predict_logits,
    resolve_model_revision,
    validate_populations,
    validate_selection_report,
)

DEFAULT_OUTPUT = REPO_ROOT / "artifacts/combined_generic/lora_gate/lora_runs"
MODEL_ID = "jhu-clsp/mmBERT-base"
MODEL_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
TARGET_MODULES = r"layers\.\d+\.attn\.(Wqkv|Wo)"
ROWS_PER_HALF = 18_197
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_MODULES = 44
LORA_PARAMETERS = 811_008
EPOCHS = 3
MICROBATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 256
ADAPTER_LEARNING_RATE = 1e-4
HEAD_LEARNING_RATE = 3e-4
MAX_TOKENS = 512
TOKEN_BUDGET = 4096


def domain_microbatch_weights(
    domain_rows: int,
    microbatch_size: int,
) -> list[float]:
    return [
        0.5 * min(microbatch_size, domain_rows - start) / domain_rows
        for start in range(0, domain_rows, microbatch_size)
    ]


def select_checkpoint_epoch(curve: list[dict]) -> int:
    return min(
        curve,
        key=lambda row: (row["validation_macro_bce"], row["epoch"]),
    )["epoch"]


def stable_validation_bces(
    labels: dict[str, np.ndarray],
    logits: dict[str, np.ndarray],
) -> dict[str, float]:
    bces = {
        name: _bce_from_logits(labels[name], logits[name])
        for name in ("morgott", "promptshield")
    }
    return {**bces, "macro": 0.5 * sum(bces.values())}


def lora_run_directory_name(model_id: str, seed: int) -> str:
    model_tag = re.sub(r"[^a-z0-9]+", "-", model_id.casefold()).strip("-")
    return f"{model_tag}_combined_lora-r8_s{seed}"


def quantize_training_features(pooled):
    import torch

    return pooled.to(dtype=torch.bfloat16)


def combined_lora_schedule(
    *,
    rows_per_half: int,
    epochs: int,
    microbatch_size: int,
    effective_batch_size: int,
) -> dict[str, int]:
    half_batch_size = effective_batch_size // 2
    updates_per_epoch = math.ceil(rows_per_half / half_batch_size)
    microsteps_per_epoch = sum(
        2 * math.ceil(min(half_batch_size, rows_per_half - start) / microbatch_size)
        for start in range(0, rows_per_half, half_batch_size)
    )
    return {
        "half_batch_size": half_batch_size,
        "updates_per_epoch": updates_per_epoch,
        "updates": updates_per_epoch * epochs,
        "forward_backward_microsteps": microsteps_per_epoch * epochs,
    }


def publish_if_unchanged(
    temporary: Path,
    output: Path,
    paths: dict[str, Path],
    expected_hashes: dict[str, str],
) -> None:
    _verify_source_hashes(paths, expected_hashes)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    os.replace(temporary, output)


def _tokenize(tokenizer, records: list[dict], max_tokens: int) -> list[list[int]]:
    return tokenizer(
        [strict_normalize(record["text"]) for record in records],
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False,
    )["input_ids"]


def _collate(
    token_ids: list[list[int]],
    indices: list[int],
    pad_token_id: int,
):
    import torch

    width = max(len(token_ids[index]) for index in indices)
    inputs = torch.full(
        (len(indices), width),
        pad_token_id,
        dtype=torch.long,
        device="cuda",
    )
    mask = torch.zeros_like(inputs)
    for slot, index in enumerate(indices):
        values = token_ids[index]
        inputs[slot, : len(values)] = torch.tensor(
            values,
            dtype=torch.long,
            device="cuda",
        )
        mask[slot, : len(values)] = 1
    return inputs, mask


def _validation_logits(
    model,
    tokenizer,
    head,
    records: list[dict],
    *,
    max_tokens: int,
    token_budget: int,
) -> np.ndarray:
    model.eval()
    head.eval()
    features = extract_features(
        model,
        tokenizer,
        records,
        max_tokens=max_tokens,
        token_budget=token_budget,
        record_chunk=VALIDATION_FEATURE_RECORD_CHUNK,
    )
    return predict_logits(
        head,
        features,
        batch_size=VALIDATION_PREDICTION_BATCH_SIZE,
    )


def _state_dict_on_cpu(module) -> dict:
    return {
        name: value.detach().contiguous().cpu().clone()
        for name, value in module.state_dict().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-dir",
        default=str(DEFAULT_SELECTION.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--microbatch-size", type=int, default=MICROBATCH_SIZE)
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=EFFECTIVE_BATCH_SIZE,
    )
    parser.add_argument(
        "--adapter-learning-rate",
        type=float,
        default=ADAPTER_LEARNING_RATE,
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=HEAD_LEARNING_RATE,
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    try:
        revision = resolve_model_revision(args.model_id, args.model_revision)
    except ValueError as error:
        parser.error(str(error))
    fixed_recipe = (
        args.model_id == MODEL_ID
        and revision == MODEL_REVISION
        and args.epochs == EPOCHS
        and args.microbatch_size == MICROBATCH_SIZE
        and args.effective_batch_size == EFFECTIVE_BATCH_SIZE
        and args.adapter_learning_rate == ADAPTER_LEARNING_RATE
        and args.head_learning_rate == HEAD_LEARNING_RATE
        and args.max_tokens == MAX_TOKENS
        and args.token_budget == TOKEN_BUDGET
    )
    if not fixed_recipe:
        parser.error("the LoRA gate recipe is fixed; only seed and paths may vary")
    if args.seed < 0:
        parser.error("seed must be non-negative")

    selection_dir = (REPO_ROOT / args.selection_dir).resolve()
    selection_report_path = selection_dir / "selection_report.json"
    selection_report_bytes = selection_report_path.read_bytes()
    selection_report_sha256 = hashlib.sha256(selection_report_bytes).hexdigest()
    selection_report = json.loads(selection_report_bytes)
    validate_selection_report(selection_report, selection_dir=selection_dir)
    specs = selection_report["outputs"]
    population_names = (
        "m1",
        "m2",
        "promptshield",
        "validation_morgott_selection",
        "validation_promptshield",
    )
    paths = {
        name: _artifact_path(selection_dir, specs[name]) for name in population_names
    }
    input_paths = {"selection_report": selection_report_path, **paths}
    input_hashes = {
        "selection_report": selection_report_sha256,
        **{name: specs[name]["sha256"] for name in population_names},
    }
    records = {
        name: load_records(paths[name], specs[name]["sha256"])
        for name in population_names
    }
    for name, values in records.items():
        labels = {
            str(label): count for label, count in sorted(_label_counts(values).items())
        }
        if len(values) != specs[name]["rows"] or labels != specs[name]["labels"]:
            raise ValueError(f"{name} row or label count mismatch")
    validate_populations(
        records["m1"],
        records["m2"],
        records["promptshield"],
        records["validation_morgott_selection"],
        records["validation_promptshield"],
    )
    if len(records["m1"]) != ROWS_PER_HALF:
        raise ValueError(
            f"LoRA gate requires {ROWS_PER_HALF} rows per half, "
            f"found {len(records['m1'])}"
        )

    output_root = Path(args.output_root).resolve()
    if not output_root.is_relative_to((REPO_ROOT / "artifacts").resolve()):
        parser.error("--output-root must be inside the artifacts directory")
    output = output_root / lora_run_directory_name(args.model_id, args.seed)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")

    source_paths = {
        "runner_sha256": Path(__file__).resolve(),
        "head_helper_sha256": REPO_ROOT / "experiments/train_combined_generic_head.py",
        "strict_normalizer_sha256": REPO_ROOT / "experiments/strict_normalize.py",
    }
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}

    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        raise ValueError("pinned tokenizer has no pad token")
    training_token_ids = {
        name: _tokenize(tokenizer, records[name], args.max_tokens)
        for name in ("m1", "promptshield")
    }
    base_model = AutoModel.from_pretrained(
        args.model_id,
        revision=revision,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base_model.enable_input_require_grads()
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            target_modules=TARGET_MODULES,
            task_type="FEATURE_EXTRACTION",
        ),
    ).to("cuda")
    targeted_modules = sorted(
        name for name, module in model.named_modules() if hasattr(module, "lora_A")
    )
    if len(targeted_modules) != LORA_MODULES:
        raise ValueError(
            f"expected {LORA_MODULES} attention LoRA modules, "
            f"found {len(targeted_modules)}"
        )
    adapter_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    adapter_parameter_count = sum(parameter.numel() for parameter in adapter_parameters)
    if adapter_parameter_count != LORA_PARAMETERS:
        raise ValueError(
            f"expected {LORA_PARAMETERS} LoRA parameters, "
            f"found {adapter_parameter_count}"
        )

    head = new_head(model.config.hidden_size, args.seed).to("cuda")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapter_parameters,
                "lr": args.adapter_learning_rate,
            },
            {
                "params": head.parameters(),
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=0.01,
    )
    labels = {
        name: np.asarray(
            [record["generic_label"] for record in values],
            dtype=np.int64,
        )
        for name, values in records.items()
        if name != "m2"
    }
    schedule = combined_lora_schedule(
        rows_per_half=len(records["m1"]),
        epochs=args.epochs,
        microbatch_size=args.microbatch_size,
        effective_batch_size=args.effective_batch_size,
    )
    generator = torch.Generator().manual_seed(args.seed)
    curve = []
    best = None
    updates = 0
    microsteps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        orders = {
            domain: torch.randperm(
                len(records[domain]),
                generator=generator,
            ).tolist()
            for domain in ("m1", "promptshield")
        }
        update_losses = []
        model.train()
        head.train()
        for group_start in range(
            0,
            len(records["m1"]),
            schedule["half_batch_size"],
        ):
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for domain in ("m1", "promptshield"):
                indices = orders[domain][
                    group_start : group_start + schedule["half_batch_size"]
                ]
                indices.sort(key=lambda index: len(training_token_ids[domain][index]))
                for start, coefficient in zip(
                    range(0, len(indices), args.microbatch_size),
                    domain_microbatch_weights(len(indices), args.microbatch_size),
                    strict=True,
                ):
                    batch_indices = indices[start : start + args.microbatch_size]
                    inputs, mask = _collate(
                        training_token_ids[domain],
                        batch_indices,
                        tokenizer.pad_token_id,
                    )
                    targets = torch.from_numpy(labels[domain][batch_indices]).to(
                        device="cuda",
                        dtype=torch.float32,
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hidden = model(
                            input_ids=inputs,
                            attention_mask=mask,
                        ).last_hidden_state
                        pooled = quantize_training_features(_pool(hidden, mask))
                        logits = head(pooled)[:, 0]
                        loss = (
                            torch.nn.functional.binary_cross_entropy_with_logits(
                                logits,
                                targets,
                            )
                            * coefficient
                        )
                    loss.backward()
                    update_loss += float(loss.detach().cpu())
                    microsteps += 1
            optimizer.step()
            update_losses.append(update_loss)
            updates += 1

        morgott_logits = _validation_logits(
            model,
            tokenizer,
            head,
            records["validation_morgott_selection"],
            max_tokens=args.max_tokens,
            token_budget=args.token_budget,
        )
        promptshield_logits = _validation_logits(
            model,
            tokenizer,
            head,
            records["validation_promptshield"],
            max_tokens=args.max_tokens,
            token_budget=args.token_budget,
        )
        validation_bces = stable_validation_bces(
            {
                "morgott": labels["validation_morgott_selection"],
                "promptshield": labels["validation_promptshield"],
            },
            {
                "morgott": morgott_logits,
                "promptshield": promptshield_logits,
            },
        )
        curve.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": float(np.mean(update_losses)),
                "validation_morgott_bce": validation_bces["morgott"],
                "validation_promptshield_bce": validation_bces["promptshield"],
                "validation_macro_bce": validation_bces["macro"],
            }
        )
        key = (validation_bces["macro"], epoch + 1)
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "epoch": epoch + 1,
                "adapter": {
                    name: value.detach().contiguous().cpu().clone()
                    for name, value in get_peft_model_state_dict(model).items()
                },
                "head": _state_dict_on_cpu(head),
            }

    if (
        updates != schedule["updates"]
        or microsteps != schedule["forward_backward_microsteps"]
    ):
        raise ValueError(
            f"LoRA schedule mismatch: {updates} updates and {microsteps} microsteps"
        )
    if best["epoch"] != select_checkpoint_epoch(curve):
        raise ValueError("selected checkpoint is not the minimum macro-BCE epoch")
    set_peft_model_state_dict(model, best["adapter"])
    restored_adapter = get_peft_model_state_dict(model)
    if restored_adapter.keys() != best["adapter"].keys() or any(
        not torch.equal(restored_adapter[name].cpu(), best["adapter"][name])
        for name in restored_adapter
    ):
        raise ValueError("restored adapter state does not match selected checkpoint")
    head.load_state_dict(best["head"])

    selected_logits = {
        "morgott": _validation_logits(
            model,
            tokenizer,
            head,
            records["validation_morgott_selection"],
            max_tokens=args.max_tokens,
            token_budget=args.token_budget,
        ),
        "promptshield": _validation_logits(
            model,
            tokenizer,
            head,
            records["validation_promptshield"],
            max_tokens=args.max_tokens,
            token_budget=args.token_budget,
        ),
    }
    selected_scores = {
        name: _scores(values) for name, values in selected_logits.items()
    }
    selected_validation_bces = stable_validation_bces(
        {
            "morgott": labels["validation_morgott_selection"],
            "promptshield": labels["validation_promptshield"],
        },
        selected_logits,
    )
    selected_macro_bce = selected_validation_bces["macro"]
    if (
        abs(selected_macro_bce - curve[best["epoch"] - 1]["validation_macro_bce"])
        > 1e-7
    ):
        raise ValueError("selected checkpoint validation BCE changed after restore")

    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    training_seconds = time.perf_counter() - started
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output_root))
    try:
        adapter_dir = temporary / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        head_path = temporary / "head.safetensors"
        head_sha256 = _save_head(head, head_path)
        arrays = {
            "validation_morgott_selection_logits.npy": selected_logits["morgott"],
            "validation_morgott_selection_scores.npy": selected_scores["morgott"],
            "validation_morgott_selection_labels.npy": labels[
                "validation_morgott_selection"
            ],
            "validation_promptshield_logits.npy": selected_logits["promptshield"],
            "validation_promptshield_scores.npy": selected_scores["promptshield"],
            "validation_promptshield_labels.npy": labels["validation_promptshield"],
        }
        for name, values in arrays.items():
            np.save(temporary / name, values)

        probe_records = [
            *records["validation_morgott_selection"][:32],
            *records["validation_promptshield"][:32],
        ]
        probe_scores = _scores(
            _validation_logits(
                model,
                tokenizer,
                head,
                probe_records,
                max_tokens=args.max_tokens,
                token_budget=args.token_budget,
            )
        )
        del model, head, base_model
        torch.cuda.empty_cache()
        reloaded_base = AutoModel.from_pretrained(
            args.model_id,
            revision=revision,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        reloaded_model = PeftModel.from_pretrained(
            reloaded_base,
            adapter_dir,
            is_trainable=False,
        ).to("cuda")
        reloaded_head = new_head(
            reloaded_model.config.hidden_size,
            args.seed,
        ).to("cuda")
        reloaded_head.load_state_dict(load_file(str(head_path)), strict=True)
        reloaded_scores = _scores(
            _validation_logits(
                reloaded_model,
                tokenizer,
                reloaded_head,
                probe_records,
                max_tokens=args.max_tokens,
                token_budget=args.token_budget,
            )
        )
        roundtrip_delta = float(np.max(np.abs(probe_scores - reloaded_scores)))
        if roundtrip_delta > 1e-5:
            raise ValueError(f"adapter and head roundtrip mismatch: {roundtrip_delta}")

        published_adapter = output / "adapter"
        published_head = output / head_path.name
        adapter_files = {
            path.name: file_sha256(path)
            for path in sorted(adapter_dir.iterdir())
            if path.is_file()
        }
        validation_metrics = {
            "morgott_selection": _binary_metrics(
                labels["validation_morgott_selection"],
                selected_scores["morgott"],
            ),
            "promptshield": _binary_metrics(
                labels["validation_promptshield"],
                selected_scores["promptshield"],
            ),
        }
        validation_metrics["morgott_selection"]["selection_bce"] = (
            selected_validation_bces["morgott"]
        )
        validation_metrics["promptshield"]["selection_bce"] = selected_validation_bces[
            "promptshield"
        ]
        result = {
            "schema_version": 1,
            "purpose": (
                "artifact-only update-matched generic instruction-subversion "
                "LoRA gate experiment"
            ),
            "condition": "combined",
            "adaptation": "lora",
            "generic_target": TARGET,
            "model_id": args.model_id,
            "model_revision": revision,
            "attention_implementation": "sdpa",
            "normalization": "strict",
            "max_tokens": args.max_tokens,
            "token_budget": args.token_budget,
            "validation_feature_record_chunk": VALIDATION_FEATURE_RECORD_CHUNK,
            "validation_prediction_batch_size": (VALIDATION_PREDICTION_BATCH_SIZE),
            "seed": args.seed,
            "lora": {
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "bias": "none",
                "target_modules_regex": TARGET_MODULES,
                "targeted_modules": targeted_modules,
                "adapter_parameters": adapter_parameter_count,
            },
            "training": {
                "epochs": args.epochs,
                "microbatch_size": args.microbatch_size,
                "effective_batch_size": args.effective_batch_size,
                **schedule,
                "adapter_learning_rate": args.adapter_learning_rate,
                "head_learning_rate": args.head_learning_rate,
                "scheduler": "constant",
                "updates": updates,
                "forward_backward_microsteps": microsteps,
                "selected_epoch": best["epoch"],
                "checkpoint_selection": (
                    "minimum equal-domain mean of matched Morgott and "
                    "PromptShield validation BCE"
                ),
                "curve": curve,
                "first_half": "m1",
                "second_half": "promptshield",
                "rows_per_half": len(records["m1"]),
                "labels_per_half": {
                    str(label): count
                    for label, count in sorted(_label_counts(records["m1"]).items())
                },
                "loss": ("0.5 * mean_BCE(first_half) + 0.5 * mean_BCE(second_half)"),
                "base_encoder_frozen": True,
                "adapter_trainable": True,
            },
            "validation": {
                "checkpoint_selection_rows": {
                    "morgott": len(records["validation_morgott_selection"]),
                    "promptshield": len(records["validation_promptshield"]),
                },
                **validation_metrics,
                "macro_bce": selected_macro_bce,
            },
            "runtime": {
                "training_seconds": training_seconds,
                "peak_reserved_bytes": peak_reserved_bytes,
            },
            "artifact": {
                "adapter": str(published_adapter.relative_to(REPO_ROOT)),
                "adapter_files": adapter_files,
                "head": str(published_head.relative_to(REPO_ROOT)),
                "head_sha256": head_sha256,
                "roundtrip_probe_rows": len(probe_records),
                "roundtrip_max_abs_score_delta": roundtrip_delta,
                "arrays": {name: file_sha256(temporary / name) for name in arrays},
            },
            "provenance": {
                "selection_report": str(selection_report_path.relative_to(REPO_ROOT)),
                "selection_report_sha256": selection_report_sha256,
                "selection_inputs": {
                    name: {
                        "path": specs[name]["path"],
                        "sha256": specs[name]["sha256"],
                        "rows": specs[name]["rows"],
                        "labels": specs[name]["labels"],
                    }
                    for name in specs
                },
                **source_hashes,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "numpy",
                        "peft",
                        "safetensors",
                        "scikit-learn",
                        "torch",
                        "transformers",
                    )
                },
            },
            "limitations": [
                "This gate isolates LoRA on M1 plus PromptShield and does not fit "
                "the generated pairs.",
                "Only checkpoint-selection validation is scored by this trainer.",
                "No held-out test or operating threshold is scored by this trainer.",
                "The learned score is advisory and is not approved for blocking.",
            ],
        }
        (temporary / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        publish_if_unchanged(
            temporary,
            output,
            {
                **source_paths,
                **{f"input_{name}": path for name, path in input_paths.items()},
            },
            {
                **source_hashes,
                **{f"input_{name}": digest for name, digest in input_hashes.items()},
            },
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"selected epoch {best['epoch']}; validation macro BCE {selected_macro_bce:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
