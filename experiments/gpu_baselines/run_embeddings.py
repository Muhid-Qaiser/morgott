"""Frozen multilingual embedding baselines for direct and indirect injection."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

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
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_LICENSE = "MIT"
MAX_LENGTH = 384
OPERATING_FPR_BUDGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
INPUT_SHA256 = manifest_output_hashes(ROOT / "reports/data_manifest.json")
INPUT_SHA256.pop("nemotron_agentic_ipi")


def read_rows(name: str) -> list[dict]:
    return read_verified_jsonl(DATA / f"{name}.jsonl", INPUT_SHA256[name])


class Embedder:
    def __init__(self, batch_size: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this experiment")
        self.batch_size = batch_size
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        output = []
        batch_size = batch_size or self.batch_size
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                ["query: " + text for text in texts[start : start + batch_size]],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            output.append(F.normalize(pooled, p=2, dim=1).float().cpu().numpy())
        return np.concatenate(output)


def cached_embeddings(name: str, rows: list[dict], embedder: Embedder) -> np.ndarray:
    cache = HERE / "cache"
    cache.mkdir(exist_ok=True)
    path = (
        cache
        / f"e5-{MODEL_REVISION[:8]}-{MAX_LENGTH}-{name}-{INPUT_SHA256[name][:12]}.npy"
    )
    if path.exists():
        return np.load(path)
    matrix = embedder.encode([row["text"] for row in rows])
    np.save(path, matrix)
    return matrix


def fit_classifier(
    rows: list[dict], embeddings: np.ndarray, max_fpr: float
) -> tuple[LogisticRegression, float, dict, float]:
    validation = validation_mask(rows)
    classifier = LogisticRegression(
        class_weight="balanced", max_iter=2_000, random_state=42, solver="liblinear"
    )
    started = time.perf_counter()
    classifier.fit(
        embeddings[~validation],
        [row["label"] for row, use in zip(rows, ~validation) if use],
    )
    fit_seconds = time.perf_counter() - started
    scores = classifier.predict_proba(embeddings[validation])[:, 1]
    labels = [row["label"] for row, use in zip(rows, validation) if use]
    threshold = choose_threshold(labels, scores, max_fpr=max_fpr)
    metrics = _rates(
        [row for row, use in zip(rows, validation) if use], scores, threshold
    )
    return classifier, threshold, metrics, fit_seconds


def operating_points(
    validation_rows: list[dict],
    validation_scores: np.ndarray,
    evaluations: dict[str, tuple[list[dict], np.ndarray]],
) -> list[dict]:
    labels = [row["label"] for row in validation_rows]
    points = []
    for budget in OPERATING_FPR_BUDGETS:
        threshold = choose_threshold(labels, validation_scores, budget)
        points.append(
            {
                "validation_fpr_budget": budget,
                "threshold": threshold,
                "validation": _rates(validation_rows, validation_scores, threshold),
                "sets": {
                    name: _rates(rows, scores, threshold)
                    for name, (rows, scores) in evaluations.items()
                },
            }
        )
    return points


def precision_profiles(
    validation_rows: list[dict],
    validation_scores: np.ndarray,
    evaluations: dict[str, tuple[list[dict], np.ndarray]],
) -> list[dict]:
    labels = [row["label"] for row in validation_rows]
    profiles = []
    for floor in DIRECT_PRECISION_FLOORS:
        try:
            threshold = choose_threshold_for_precision(labels, validation_scores, floor)
        except ValueError:
            profiles.append(
                {
                    "min_validation_precision": floor,
                    "attained": False,
                    "threshold": None,
                    "validation": None,
                    "sets": {},
                }
            )
            continue
        profiles.append(
            {
                "min_validation_precision": floor,
                "attained": True,
                "threshold": threshold,
                "validation": _rates(validation_rows, validation_scores, threshold),
                "sets": {
                    name: _rates(rows, scores, threshold)
                    for name, (rows, scores) in evaluations.items()
                },
            }
        )
    return profiles


def score_chunks(
    rows: list[dict], embedder: Embedder, classifier: LogisticRegression
) -> np.ndarray:
    chunks: list[str] = []
    spans = []
    for row in rows:
        parts = [row["text"]] + [
            part
            for part in row["text"].split("\n\n")
            if len(part.strip()) >= 8 and part != row["text"]
        ]
        start = len(chunks)
        chunks.extend(parts)
        spans.append((start, len(chunks)))
    scores = classifier.predict_proba(embedder.encode(chunks))[:, 1]
    return np.asarray([scores[start:end].max() for start, end in spans])


def measured_latency(
    embedder: Embedder, classifier: LogisticRegression, texts: list[str]
) -> dict[str, float]:
    sample = texts[:100]
    for text in sample[:5]:
        classifier.predict_proba(embedder.encode([text], batch_size=1))
    torch.cuda.synchronize()
    started = time.perf_counter()
    for text in sample:
        classifier.predict_proba(embedder.encode([text], batch_size=1))
    torch.cuda.synchronize()
    single_ms = (time.perf_counter() - started) * 1_000 / len(sample)

    torch.cuda.synchronize()
    started = time.perf_counter()
    classifier.predict_proba(embedder.encode(sample, batch_size=64))
    torch.cuda.synchronize()
    batch_ms = (time.perf_counter() - started) * 1_000 / len(sample)
    return {"batch_1_ms_per_text": single_ms, "batch_64_ms_per_text": batch_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = {name: read_rows(name) for name in INPUT_SHA256}

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    embedder = Embedder(args.batch_size)
    embedding_seconds = {}
    matrices = {}
    for name in (
        "train",
        "toxic_chat",
        "prompt_injections",
        "tensor_trust_attack",
        "xstest",
        "notinject",
        "oasst1_chat",
        "do_not_answer",
        "harmbench",
        "multi_turn",
        "jailbreaks_over_time",
        "oasst1_position_stress",
        "indirect_train",
        "bipia_payload",
    ):
        step = time.perf_counter()
        matrices[name] = cached_embeddings(name, rows[name], embedder)
        embedding_seconds[name] = time.perf_counter() - step

    direct, direct_threshold, direct_validation, direct_fit_seconds = fit_classifier(
        rows["train"], matrices["train"], max_fpr=0.001
    )
    direct_sets = {}
    direct_scores = {}
    for name in (
        "toxic_chat",
        "prompt_injections",
        "tensor_trust_attack",
        "xstest",
        "notinject",
        "oasst1_chat",
        "do_not_answer",
        "harmbench",
        "multi_turn",
        "jailbreaks_over_time",
        "oasst1_position_stress",
    ):
        direct_scores[name] = direct.predict_proba(matrices[name])[:, 1]
        direct_sets[name] = _rates(rows[name], direct_scores[name], direct_threshold)
    hard_names = (
        "xstest",
        "notinject",
        "oasst1_chat",
        "oasst1_position_stress",
        "do_not_answer",
        "harmbench",
    )
    direct_sets["hard_negative_aggregate"] = _rates(
        [row for name in hard_names for row in rows[name]],
        np.concatenate([direct_scores[name] for name in hard_names]),
        direct_threshold,
    )
    validation = validation_mask(rows["train"])
    direct_validation_rows = [
        row for row, selected in zip(rows["train"], validation) if selected
    ]
    direct_evaluations = {
        name: (rows[name], direct_scores[name]) for name in direct_scores
    }
    direct_evaluations["hard_negative_aggregate"] = (
        [row for name in hard_names for row in rows[name]],
        np.concatenate([direct_scores[name] for name in hard_names]),
    )
    direct_operating_points = operating_points(
        direct_validation_rows,
        direct.predict_proba(matrices["train"][validation])[:, 1],
        direct_evaluations,
    )
    direct_precision_profiles = precision_profiles(
        direct_validation_rows,
        direct.predict_proba(matrices["train"][validation])[:, 1],
        direct_evaluations,
    )

    indirect, indirect_threshold, indirect_validation, indirect_fit_seconds = (
        fit_classifier(rows["indirect_train"], matrices["indirect_train"], max_fpr=0.0)
    )
    indirect_sets = {
        "bipia_payload": _rates(
            rows["bipia_payload"],
            indirect.predict_proba(matrices["bipia_payload"])[:, 1],
            indirect_threshold,
        )
    }
    for name in ("bipia_context", "bipia_clean_context", "tensor_trust_context"):
        step = time.perf_counter()
        scores = score_chunks(rows[name], embedder, indirect)
        embedding_seconds[name + "_chunked"] = time.perf_counter() - step
        indirect_sets[name] = _rates(rows[name], scores, indirect_threshold)

    latency = measured_latency(
        embedder, direct, [row["text"] for row in rows["oasst1_chat"]]
    )
    result = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_sha256": INPUT_SHA256,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "pooling": "attention-mask mean pooling, L2 normalized",
            "prefix": "query: ",
            "max_length": MAX_LENGTH,
            "dtype": "float16 encoder; float32 embeddings",
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        },
        "direct": {
            "threshold_target_fpr": 0.001,
            "threshold": direct_threshold,
            "fit_seconds": direct_fit_seconds,
            "validation": direct_validation,
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
            "validation": indirect_validation,
            "scoring": "maximum score over whole document and blank-line paragraphs",
            "sets": indirect_sets,
        },
        "latency": latency,
        "embedding_seconds": embedding_seconds,
        "wall_seconds": time.perf_counter() - started,
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    output = HERE / "embedding_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
