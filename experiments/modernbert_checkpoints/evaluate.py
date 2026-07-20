"""Evaluate pinned ModernBERT injection checkpoints on the locked local corpus."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import (
    __version__ as transformers_version,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from morgott.data import manifest_output_hashes, read_verified_jsonl
from morgott.detector import (
    DIRECT_PRECISION_FLOORS,
    DIRECT_REVIEW_PRECISION_FLOOR,
    _rates,
    choose_threshold,
    choose_threshold_for_precision,
    validation_mask,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
HERE = Path(__file__).resolve().parent
OPERATING_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)

MODELS = {
    "siberiancat": {
        "id": "siberiancat/modernbert-prompt-injection",
        "revision": "fd9d17421e2e6bbe2eeea1874269fddc64e95e03",
        "license": "Apache-2.0",
        "base_model": "answerdotai/ModernBERT-base",
        "parameters": 149_606_402,
        "weights_sha256": "0e70e2e0c02c802c98b0a1dd423aed988b4bba4a925cf0927d0dc835d4d4858f",
        "weights_bytes": 598_439_784,
        "finetune_max_length": 256,
        "default_batch_size": 64,
        "published_threshold": 0.03,
    },
    "wolf_small": {
        "id": "patronus-studio/wolf-defender-prompt-injection-small",
        "revision": "9cf7dc2febf057238138ec256f16a0dbeda0d806",
        "license": "Apache-2.0",
        "base_model": "jhu-clsp/mmBERT-small-shaped checkpoint",
        "parameters": 140_642_306,
        "weights_sha256": "0ddb75f2dd31dbb18279a5a7c1ff41948fd147e3f28189ea346e4e2cbe638ad0",
        "weights_bytes": 562_583_392,
        "finetune_max_length": 2048,
        "default_batch_size": 4,
        "published_threshold": 0.5,
    },
}

INPUT_SHA256 = manifest_output_hashes(ROOT / "reports/data_manifest.json")
INPUT_SHA256.pop("nemotron_agentic_ipi")

DEFAULT_DATASETS = tuple(
    name for name in INPUT_SHA256 if name not in {"train", "indirect_train"}
)
INDIRECT_DATASETS = {
    "bipia_payload",
    "bipia_context",
    "bipia_clean_context",
    "tensor_trust_context",
}
HARD_NEGATIVE_DATASETS = (
    "xstest",
    "notinject",
    "oasst1_chat",
    "oasst1_position_stress",
    "do_not_answer",
    "harmbench",
)

# This is provenance, not an assertion that an undisclosed overlap is impossible.
CHECKPOINT_OVERLAP = {
    "siberiancat": {
        **{name: "no_disclosed_source_overlap" for name in DEFAULT_DATASETS},
        "tensor_trust_attack": "no_disclosed_source_overlap;public_benchmark_contamination_unknown",
        "tensor_trust_context": "no_disclosed_source_overlap;public_benchmark_contamination_unknown",
    },
    "wolf_small": {
        **{name: "no_disclosed_source_overlap" for name in DEFAULT_DATASETS},
        "prompt_injections": "same_dataset_training_source:disclosed_train_file;exact_test_row_overlap_unestablished",
        "notinject": "explicit_training_source_overlap_including_test_files",
        "xstest": "benchmark_family_informed:WildJailbreak_benign_generation_used_XSTest_categories",
        "jailbreaks_over_time": "source_family_risk:WildJailbreak_training_vs_in_the_wild_jailbreak_holdout",
        "tensor_trust_attack": "no_disclosed_source_overlap;public_benchmark_contamination_unknown",
        "tensor_trust_context": "no_disclosed_source_overlap;public_benchmark_contamination_unknown",
    },
}

LOCAL_CALIBRATION_OVERLAP = {
    "toxic_chat": "same_source_official_split",
    "prompt_injections": "same_source_official_split",
    "oasst1_position_stress": "same_source_group_heldout",
    "bipia_payload": "same_benchmark_official_train_test_split",
    "bipia_context": "same_benchmark_official_train_test_split",
    "bipia_clean_context": "same_benchmark_official_train_test_split",
}


def read_rows(name: str) -> list[dict]:
    return read_verified_jsonl(DATA / f"{name}.jsonl", INPUT_SHA256[name])


class Sensor:
    def __init__(
        self, model_name: str, device: str, max_length: int, batch_size: int
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.spec = MODELS[model_name]
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        config = AutoConfig.from_pretrained(
            self.spec["id"],
            revision=self.spec["revision"],
            trust_remote_code=False,
        )
        if (
            config.model_type != "modernbert"
            or config.architectures != ["ModernBertForSequenceClassification"]
            or config.num_labels != 2
            or config.classifier_pooling != "mean"
        ):
            raise RuntimeError(f"unexpected checkpoint config: {config}")
        if max_length > config.max_position_embeddings:
            raise ValueError(
                f"max length {max_length} exceeds {config.max_position_embeddings}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec["id"],
            revision=self.spec["revision"],
            trust_remote_code=False,
            use_fast=True,
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.spec["id"],
            revision=self.spec["revision"],
            trust_remote_code=False,
            use_safetensors=True,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, texts: list[str]) -> np.ndarray:
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        output = np.empty(len(texts), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [texts[index] for index in indices],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits.float()
            output[indices] = logits.softmax(-1)[:, 1].cpu().numpy()
        return output


def cached_scores(
    name: str, rows: list[dict], sensor: Sensor, model_name: str
) -> tuple[np.ndarray, bool]:
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    path = cache / (
        f"{model_name}-{MODELS[model_name]['revision'][:8]}-{sensor.device.type}-"
        f"{sensor.max_length}-"
        f"{name}-{INPUT_SHA256[name][:12]}.npy"
    )
    if path.exists():
        return np.load(path), True
    scores = sensor.score([row["text"] for row in rows])
    np.save(path, scores)
    return scores, False


def category_rates(
    rows: list[dict], scores: np.ndarray, threshold: float
) -> dict[str, dict]:
    categories = sorted({str(row.get("category")) for row in rows})
    return {
        category: _rates(
            [row for row in rows if str(row.get("category")) == category],
            np.asarray(
                [
                    score
                    for row, score in zip(rows, scores, strict=True)
                    if str(row.get("category")) == category
                ]
            ),
            threshold,
        )
        for category in categories
    }


def calibrate(
    name: str, sensor: Sensor, model_name: str, max_fpr: float
) -> tuple[float, dict]:
    rows = read_rows(name)
    scores, cache_hit = cached_scores(name, rows, sensor, model_name)
    selected = validation_mask(rows)
    selected_rows = [row for row, use in zip(rows, selected) if use]
    selected_scores = scores[selected]
    threshold = choose_threshold(
        [row["label"] for row in selected_rows], selected_scores, max_fpr=max_fpr
    )
    return threshold, {
        "source": f"deterministic grouped 20% of local {name} corpus",
        "max_fpr": max_fpr,
        "threshold": threshold,
        "metrics": _rates(selected_rows, selected_scores, threshold),
        "cache_hit": cache_hit,
    }


def self_check() -> None:
    rows = [
        {"group_id": "a", "split_group_id": "same"},
        {"group_id": "b", "split_group_id": "same"},
    ]
    mask = validation_mask(rows)
    assert mask.shape == (2,) and mask[0] == mask[1]
    assert set(CHECKPOINT_OVERLAP) == set(MODELS)
    assert set(HARD_NEGATIVE_DATASETS) <= set(DEFAULT_DATASETS)
    assert all(len(digest) == 64 for digest in INPUT_SHA256.values())
    print("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-fpr", type=float, default=0.001)
    parser.add_argument("--indirect-max-fpr", type=float, default=0.0)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_check()
        return
    if not args.model:
        parser.error("--model is required unless --self-test is used")

    spec = MODELS[args.model]
    max_length = args.max_length or spec["finetune_max_length"]
    batch_size = args.batch_size or spec["default_batch_size"]
    if (
        max_length <= 0
        or batch_size <= 0
        or not 0 <= args.max_fpr < 1
        or not 0 <= args.indirect_max_fpr < 1
    ):
        parser.error(
            "max length and batch size must be positive; max FPR must be [0, 1)"
        )
    datasets = args.datasets or list(DEFAULT_DATASETS)
    output = args.output or HERE / f"{args.model}_{args.device}_results.json"

    started = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sensor = Sensor(args.model, args.device, max_length, batch_size)

    direct_threshold, direct_calibration = calibrate(
        "train", sensor, args.model, args.max_fpr
    )
    indirect_threshold, indirect_calibration = calibrate(
        "indirect_train", sensor, args.model, args.indirect_max_fpr
    )
    thresholds = {"direct": direct_threshold, "indirect": indirect_threshold}

    sets = {}
    timing = {}
    dataset_rows = {}
    dataset_scores = {}
    for name in datasets:
        rows = read_rows(name)
        channel = "indirect" if name in INDIRECT_DATASETS else "direct"
        threshold = thresholds[channel]
        step = time.perf_counter()
        scores, cache_hit = cached_scores(name, rows, sensor, args.model)
        dataset_rows[name] = rows
        dataset_scores[name] = scores
        seconds = time.perf_counter() - step
        sets[name] = {
            "metrics": _rates(rows, scores, threshold),
            "published_threshold_metrics": _rates(
                rows, scores, spec["published_threshold"]
            ),
            "checkpoint_overlap": CHECKPOINT_OVERLAP[args.model][name],
            "local_calibration_overlap": LOCAL_CALIBRATION_OVERLAP.get(name, "none"),
            "threshold_channel": channel,
        }
        if name in {
            "bipia_context",
            "jailbreaks_over_time",
            "tensor_trust_attack",
            "tensor_trust_context",
        }:
            sets[name]["by_category"] = category_rates(rows, scores, threshold)
        timing[name] = {
            "seconds": seconds,
            "ms_per_text": seconds * 1_000 / len(rows),
            "cache_hit": cache_hit,
        }

    if set(HARD_NEGATIVE_DATASETS) <= set(datasets):
        hard_rows = [
            row for name in HARD_NEGATIVE_DATASETS for row in dataset_rows[name]
        ]
        hard_scores = np.concatenate(
            [dataset_scores[name] for name in HARD_NEGATIVE_DATASETS]
        )
        dataset_rows["hard_negative_aggregate"] = hard_rows
        dataset_scores["hard_negative_aggregate"] = hard_scores
        sets["hard_negative_aggregate"] = {
            "metrics": _rates(hard_rows, hard_scores, direct_threshold),
            "published_threshold_metrics": _rates(
                hard_rows, hard_scores, spec["published_threshold"]
            ),
            "checkpoint_overlap": "mixed; see component datasets",
            "local_calibration_overlap": "none",
            "threshold_channel": "direct",
        }

    train_rows = read_rows("train")
    train_scores, _ = cached_scores("train", train_rows, sensor, args.model)
    validation = validation_mask(train_rows)
    validation_rows = [row for row, selected in zip(train_rows, validation) if selected]
    validation_scores = train_scores[validation]
    validation_labels = [row["label"] for row in validation_rows]
    direct_operating_points = []
    for budget in OPERATING_FPR_BUDGETS:
        threshold = choose_threshold(
            validation_labels, validation_scores, max_fpr=budget
        )
        direct_operating_points.append(
            {
                "validation_fpr_budget": budget,
                "threshold": threshold,
                "validation": _rates(validation_rows, validation_scores, threshold),
                "sets": {
                    name: _rates(dataset_rows[name], dataset_scores[name], threshold)
                    for name in dataset_rows
                    if name not in INDIRECT_DATASETS
                },
            }
        )
    direct_precision_profiles = []
    for floor in DIRECT_PRECISION_FLOORS:
        try:
            threshold = choose_threshold_for_precision(
                validation_labels, validation_scores, floor
            )
        except ValueError:
            direct_precision_profiles.append(
                {
                    "min_validation_precision": floor,
                    "attained": False,
                    "threshold": None,
                    "validation": None,
                    "sets": {},
                }
            )
            continue
        direct_precision_profiles.append(
            {
                "min_validation_precision": floor,
                "attained": True,
                "threshold": threshold,
                "validation": _rates(validation_rows, validation_scores, threshold),
                "sets": {
                    name: _rates(dataset_rows[name], dataset_scores[name], threshold)
                    for name in dataset_rows
                    if name not in INDIRECT_DATASETS
                },
            }
        )

    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_sha256": {
            name: INPUT_SHA256[name] for name in ["train", "indirect_train", *datasets]
        },
        "model": {
            **spec,
            "architecture": "ModernBertForSequenceClassification; mean pooling and linear head",
            "max_length": max_length,
            "scoring": "single whole-text sequence with tail truncation; no chunk-max ensemble",
            "dtype": "float16" if args.device == "cuda" else "float32",
            "trust_remote_code": False,
            "weights_format": "safetensors",
        },
        "calibration": {
            "direct_user": direct_calibration,
            "untrusted_content": indirect_calibration,
        },
        "direct_operating_points": direct_operating_points,
        "direct_precision_profiles": direct_precision_profiles,
        "default_precision_floor": DIRECT_REVIEW_PRECISION_FLOOR,
        "threshold_protocol": "grouped-validation diagnostics; not production calibration",
        "sets": sets,
        "timing": timing,
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if args.device == "cuda"
                else None
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers_version,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
