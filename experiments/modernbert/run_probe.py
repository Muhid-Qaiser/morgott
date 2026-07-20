"""Frozen ModernBERT linear probe on the locked guard datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from morgott.data import (
    deduplicate,
    manifest_output_hashes,
    read_verified_jsonl,
)
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
MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MODEL_LICENSE = "Apache-2.0"
OPERATING_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
INPUT_SHA256 = manifest_output_hashes(ROOT / "reports/data_manifest.json")
INPUT_SHA256.pop("nemotron_agentic_ipi")
DIRECT_SETS = (
    "toxic_chat",
    "prompt_injections",
    "xstest",
    "notinject",
    "oasst1_chat",
    "oasst1_position_stress",
    "do_not_answer",
    "harmbench",
    "multi_turn",
    "jailbreaks_over_time",
    "tensor_trust_attack",
)
INDIRECT_SETS = (
    "bipia_payload",
    "bipia_context",
    "bipia_clean_context",
    "tensor_trust_context",
)
EXTERNAL_NEGATIVES = (
    "oasst1_chat",
    "oasst1_position_stress",
    "xstest",
    "harmbench",
    "do_not_answer",
    "notinject",
)
E5_COMMON_NEGATIVES = (
    "xstest",
    "notinject",
    "oasst1_chat",
    "do_not_answer",
    "harmbench",
)


def read_rows(name: str) -> list[dict]:
    return read_verified_jsonl(DATA / f"{name}.jsonl", INPUT_SHA256[name])


class Encoder:
    def __init__(self, batch_size: int, max_length: int, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=self.dtype,
            attn_implementation="sdpa",
            trust_remote_code=False,
            use_safetensors=True,
        ).to(self.device)
        if self.model.config.model_type != "modernbert":
            raise RuntimeError(f"unexpected model type: {self.model.config.model_type}")
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        output = {"masked_mean": [], "cls": []}
        batch_size = batch_size or self.batch_size
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            output["masked_mean"].append(
                F.normalize(mean, p=2, dim=1).float().cpu().numpy()
            )
            output["cls"].append(
                F.normalize(hidden[:, 0], p=2, dim=1).float().cpu().numpy()
            )
        return {name: np.concatenate(parts) for name, parts in output.items()}


def cached_embeddings(
    name: str, texts: list[str], input_digest: str, encoder: Encoder
) -> dict[str, np.ndarray]:
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    precision = "fp16" if encoder.device.type == "cuda" else "fp32"
    path = cache / (
        f"modernbert-{MODEL_REVISION[:8]}-{precision}-{encoder.max_length}-"
        f"{name}-{input_digest[:12]}.npz"
    )
    if path.exists():
        with np.load(path) as cached:
            return {name: cached[name] for name in ("masked_mean", "cls")}
    matrices = encoder.encode(texts)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **matrices)
    temporary.replace(path)
    return matrices


def fit_head(rows: list[dict], embeddings: np.ndarray) -> tuple:
    validation = validation_mask(rows)
    head = LogisticRegression(
        class_weight="balanced", max_iter=2_000, random_state=42, solver="liblinear"
    )
    started = time.perf_counter()
    head.fit(
        embeddings[~validation],
        [row["label"] for row, selected in zip(rows, ~validation) if selected],
    )
    return head, validation, time.perf_counter() - started


def chunked_scores(
    name: str,
    rows: list[dict],
    encoder: Encoder,
    head: LogisticRegression,
    pooling: str,
) -> np.ndarray:
    chunks: list[str] = []
    spans = []
    for row in rows:
        text = row["text"]
        parts = [text] + [
            part
            for part in re.split(r"\n\s*\n", text)
            if len(part.strip()) >= 8 and part != text
        ]
        start = len(chunks)
        chunks.extend(parts)
        spans.append((start, len(chunks)))
    subset = hashlib.sha256(
        (
            INPUT_SHA256[name]
            + "\0max-paragraph-v1\0"
            + "\0".join(row["id"] for row in rows)
        ).encode()
    ).hexdigest()
    embeddings = cached_embeddings(name + "-chunks", chunks, subset, encoder)[pooling]
    scores = head.predict_proba(embeddings)[:, 1]
    return np.asarray([scores[start:end].max() for start, end in spans])


def aggregate_metrics(
    names: tuple[str, ...],
    rows: dict[str, list[dict]],
    scores: dict[str, np.ndarray],
    threshold: float,
) -> dict:
    pairs = [
        (row, float(score))
        for name in names
        for row, score in zip(rows[name], scores[name], strict=True)
    ]
    kept, stats = deduplicate(row for row, _ in pairs)
    by_object = {id(row): score for row, score in pairs}
    result = _rates(kept, np.asarray([by_object[id(row)] for row in kept]), threshold)
    result["cross_dataset_duplicates_removed"] = stats["duplicates"]
    return result


def measured_latency(
    encoder: Encoder, head: LogisticRegression, texts: list[str], pooling: str
) -> dict:
    sample = texts[:100]
    for text in sample[:5]:
        head.predict_proba(encoder.encode([text], batch_size=1)[pooling])
    if encoder.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for text in sample:
        head.predict_proba(encoder.encode([text], batch_size=1)[pooling])
    if encoder.device.type == "cuda":
        torch.cuda.synchronize()
    single_ms = (time.perf_counter() - started) * 1_000 / len(sample)

    if encoder.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    head.predict_proba(encoder.encode(sample)[pooling])
    if encoder.device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - started) * 1_000 / len(sample)
    return {
        "batch_1_ms_per_text": single_ms,
        f"batch_{encoder.batch_size}_ms_per_text": batch_ms,
    }


def compact(metrics: dict) -> dict:
    return {
        key: metrics.get(key)
        for key in ("rows", "true_positive", "false_positive", "recall", "fpr")
    }


def comparison() -> dict:
    baseline = json.loads((ROOT / "reports" / "baseline.json").read_text())
    e5 = json.loads(
        (ROOT / "experiments" / "gpu_baselines" / "embedding_results.json").read_text()
    )
    char = baseline["detectors"]["char_ngram_logreg"]
    indirect_char = baseline["detectors"]["indirect_char_ngram_logreg"]
    return {
        "char_ngram_logreg": {
            "source": "reports/baseline.json",
            "direct_validation": compact(baseline["training"]["validation"]),
            "direct_sets": {
                name: compact(char["sets"][name])
                for name in DIRECT_SETS
                if name in char["sets"]
            },
            "indirect_sets": {
                name: compact(indirect_char["sets"][name])
                for name in INDIRECT_SETS
                if name in indirect_char["sets"]
            },
            "latency_us_per_sample": char["latency_us_per_sample"],
        },
        "multilingual_e5_small_frozen": {
            "source": "experiments/gpu_baselines/embedding_results.json",
            "max_length": e5["model"]["max_length"],
            "direct_validation": compact(e5["direct"]["validation"]),
            "direct_sets": {
                name: compact(metrics) for name, metrics in e5["direct"]["sets"].items()
            },
            "indirect_sets": {
                name: compact(metrics)
                for name, metrics in e5["indirect"]["sets"].items()
            },
            "latency": e5["latency"],
        },
    }


def evaluate_pooling(
    pooling: str,
    rows: dict[str, list[dict]],
    matrices: dict[str, dict[str, np.ndarray]],
    encoder: Encoder,
) -> tuple[dict, LogisticRegression, dict[str, float]]:
    direct, direct_validation_mask, direct_fit_seconds = fit_head(
        rows["train"], matrices["train"][pooling]
    )
    direct_validation_rows = [
        row for row, selected in zip(rows["train"], direct_validation_mask) if selected
    ]
    direct_validation_scores = direct.predict_proba(
        matrices["train"][pooling][direct_validation_mask]
    )[:, 1]
    direct_threshold = choose_threshold(
        [row["label"] for row in direct_validation_rows],
        direct_validation_scores,
        max_fpr=0.001,
    )
    direct_scores = {
        name: direct.predict_proba(matrices[name][pooling])[:, 1]
        for name in DIRECT_SETS
    }
    direct_sets = {
        name: _rates(rows[name], direct_scores[name], direct_threshold)
        for name in DIRECT_SETS
    }
    direct_sets["external_hard_negatives"] = aggregate_metrics(
        EXTERNAL_NEGATIVES, rows, direct_scores, direct_threshold
    )
    direct_sets["e5_common_hard_negatives"] = aggregate_metrics(
        E5_COMMON_NEGATIVES, rows, direct_scores, direct_threshold
    )
    direct_operating_points = []
    direct_validation_labels = [row["label"] for row in direct_validation_rows]
    for budget in OPERATING_FPR_BUDGETS:
        threshold = choose_threshold(
            direct_validation_labels, direct_validation_scores, max_fpr=budget
        )
        sets = {
            name: _rates(rows[name], direct_scores[name], threshold)
            for name in DIRECT_SETS
        }
        sets["external_hard_negatives"] = aggregate_metrics(
            EXTERNAL_NEGATIVES, rows, direct_scores, threshold
        )
        sets["e5_common_hard_negatives"] = aggregate_metrics(
            E5_COMMON_NEGATIVES, rows, direct_scores, threshold
        )
        direct_operating_points.append(
            {
                "validation_fpr_budget": budget,
                "threshold": threshold,
                "validation": _rates(
                    direct_validation_rows, direct_validation_scores, threshold
                ),
                "sets": sets,
            }
        )
    direct_precision_profiles = []
    for floor in DIRECT_PRECISION_FLOORS:
        try:
            threshold = choose_threshold_for_precision(
                direct_validation_labels, direct_validation_scores, floor
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
        sets = {
            name: _rates(rows[name], direct_scores[name], threshold)
            for name in DIRECT_SETS
        }
        sets["external_hard_negatives"] = aggregate_metrics(
            EXTERNAL_NEGATIVES, rows, direct_scores, threshold
        )
        sets["e5_common_hard_negatives"] = aggregate_metrics(
            E5_COMMON_NEGATIVES, rows, direct_scores, threshold
        )
        direct_precision_profiles.append(
            {
                "min_validation_precision": floor,
                "attained": True,
                "threshold": threshold,
                "validation": _rates(
                    direct_validation_rows, direct_validation_scores, threshold
                ),
                "sets": sets,
            }
        )

    indirect, indirect_validation_mask, indirect_fit_seconds = fit_head(
        rows["indirect_train"], matrices["indirect_train"][pooling]
    )
    indirect_validation_rows = [
        row
        for row, selected in zip(rows["indirect_train"], indirect_validation_mask)
        if selected
    ]
    chunk_seconds = {}
    step = time.perf_counter()
    indirect_validation_scores = chunked_scores(
        "indirect_train", indirect_validation_rows, encoder, indirect, pooling
    )
    chunk_seconds["indirect_validation_chunked"] = time.perf_counter() - step
    indirect_threshold = choose_threshold(
        [row["label"] for row in indirect_validation_rows],
        indirect_validation_scores,
        max_fpr=0.0,
    )
    indirect_sets = {
        "bipia_payload": _rates(
            rows["bipia_payload"],
            indirect.predict_proba(matrices["bipia_payload"][pooling])[:, 1],
            indirect_threshold,
        )
    }
    for name in INDIRECT_SETS[1:]:
        step = time.perf_counter()
        scores = chunked_scores(name, rows[name], encoder, indirect, pooling)
        chunk_seconds[name + "_chunked"] = time.perf_counter() - step
        indirect_sets[name] = _rates(rows[name], scores, indirect_threshold)

    return (
        {
            "direct": {
                "threshold_target_fpr": 0.001,
                "threshold": direct_threshold,
                "fit_seconds": direct_fit_seconds,
                "validation": _rates(
                    direct_validation_rows,
                    direct_validation_scores,
                    direct_threshold,
                ),
                "sets": direct_sets,
                "operating_points": direct_operating_points,
                "precision_profiles": direct_precision_profiles,
                "default_precision_floor": DIRECT_REVIEW_PRECISION_FLOOR,
                "threshold_protocol": "grouped-validation diagnostics; not production calibration",
            },
            "indirect": {
                "threshold_target_fpr": 0.0,
                "threshold": indirect_threshold,
                "fit_seconds": indirect_fit_seconds,
                "validation": _rates(
                    indirect_validation_rows,
                    indirect_validation_scores,
                    indirect_threshold,
                ),
                "scoring": "maximum score over whole document and blank-line paragraphs",
                "sets": indirect_sets,
            },
        },
        direct,
        chunk_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    started = time.perf_counter()
    rows = {name: read_rows(name) for name in INPUT_SHA256}
    torch.manual_seed(42)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    encoder = Encoder(args.batch_size, args.max_length, args.device)

    matrices = {}
    embedding_seconds = {}
    for name in ("train", *DIRECT_SETS, "indirect_train", "bipia_payload"):
        print(f"embedding {name}...", flush=True)
        step = time.perf_counter()
        matrices[name] = cached_embeddings(
            name, [row["text"] for row in rows[name]], INPUT_SHA256[name], encoder
        )
        embedding_seconds[name] = time.perf_counter() - step
        print(f"embedded {name} in {embedding_seconds[name]:.1f}s", flush=True)

    pooling_results = {}
    direct_heads = {}
    for pooling in ("masked_mean", "cls"):
        print(f"evaluating {pooling} pooling...", flush=True)
        pooling_results[pooling], direct_heads[pooling], chunk_seconds = (
            evaluate_pooling(pooling, rows, matrices, encoder)
        )
        embedding_seconds.update(
            {f"{pooling}/{name}": seconds for name, seconds in chunk_seconds.items()}
        )
    selected = pooling_results["masked_mean"]

    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_sha256": INPUT_SHA256,
        "evaluation_channels": {
            "direct_user": list(DIRECT_SETS),
            "untrusted_content": list(INDIRECT_SETS),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "architecture": "frozen 149M-parameter ModernBERT-base plus balanced logistic head",
            "pooling": "masked mean and CLS extracted in the same forward, L2 normalized",
            "max_length": args.max_length,
            "dtype": "float16 encoder; float32 cached embeddings/head"
            if args.device == "cuda"
            else "float32 encoder/embeddings/head",
            "attention": "PyTorch SDPA",
            "supply_chain": "pinned safetensors; trust_remote_code=False; no repository Python/auto_map",
        },
        "hardware": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
            if args.device == "cuda"
            else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
            if args.device == "cuda"
            else None,
        },
        "selected_pooling": "masked_mean",
        "direct": selected["direct"],
        "indirect": selected["indirect"],
        "poolings": pooling_results,
        "comparison": comparison(),
        "latency": measured_latency(
            encoder,
            direct_heads["masked_mean"],
            [row["text"] for row in rows["oasst1_chat"]],
            "masked_mean",
        ),
        "embedding_seconds": embedding_seconds,
        "wall_seconds": time.perf_counter() - started,
        "caveats": [
            "Frozen-feature performance does not measure end-to-end ModernBERT fine-tuning.",
            "JailbreaksOverTime is held out, but source, attack style, label process, and time are confounded; its score is not pure temporal robustness.",
            "ModernBERT-base is primarily pretrained on English and code, not multilingual text.",
        ],
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    output = HERE / f"results_{args.max_length}_{args.device}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    print(
        json.dumps(
            {
                pooling: {
                    "direct_validation": compact(metrics["direct"]["validation"]),
                    "multi_turn": compact(metrics["direct"]["sets"]["multi_turn"]),
                    "indirect": {
                        name: compact(value)
                        for name, value in metrics["indirect"]["sets"].items()
                    },
                }
                for pooling, metrics in pooling_results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
