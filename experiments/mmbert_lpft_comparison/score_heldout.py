"""Score the repository-held-out SWE-rebench V2 pair slices for both candidates.

Window-max scoring mirrors the maintained cascade: strict normalization, then
ordered 512-token windows with 128-token overlap, scored with each run's own
verified artifacts. Outputs metrics only; no source text is persisted.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from morgott.models.mmbert.core import MAX_TOKENS, file_sha256
from morgott.models.mmbert.evaluate import _load_run
from morgott.models.mmbert.serving import WINDOW_OVERLAP
from morgott.normalization import strict_normalize

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "mmbert-base-full-lpft-s42": ROOT / "artifacts/models/mmbert-base-full-lpft-s42",
    "mmbert-lora-full-s42": ROOT / "artifacts/models/mmbert-lora-full-s42",
}
SPLITS = {
    "validation": ROOT / "artifacts/mmbert_lpft_new_data/validation/pairs.jsonl.gz",
    "dev_test": ROOT / "artifacts/mmbert_lpft_new_data/dev_test/pairs.jsonl.gz",
}
HIGH_GATE = 0.99999
LENGTH_EDGES = (2_048, 4_096, 8_192)
OUTPUT = ROOT / "artifacts/mmbert_lpft_comparison/heldout_summary.json"
MICROBATCH = 16
DESCRIPTIVE_FPR_BUDGET = "1.0000%"


def _descriptive_threshold(run: Path) -> float:
    evaluation = json.loads(
        (run / "evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    value = evaluation["thresholds"]["selected"][DESCRIPTIVE_FPR_BUDGET]
    if not isinstance(value, float) or not 0 < value < 1:
        raise ValueError(f"invalid descriptive threshold for {run.name}")
    return value


def _pairs(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["benign"] or not row["attack"]:
                raise ValueError("pair rows must contain non-empty texts")
            rows.append(row)
    if not rows:
        raise ValueError(f"no pairs in {path}")
    return rows


def _window_max_scores(encoder, tokenizer, head, texts: list[str]) -> np.ndarray:
    import torch

    from morgott.models.mmbert.core import pool

    encoder.eval()
    head.eval()
    scores = np.empty(len(texts), dtype=np.float64)
    with torch.no_grad():
        for index, text in enumerate(texts):
            encoded = tokenizer(
                strict_normalize(text),
                add_special_tokens=True,
                max_length=MAX_TOKENS,
                stride=WINDOW_OVERLAP,
                truncation=True,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            best = -math.inf
            for start in range(0, input_ids.shape[0], MICROBATCH):
                batch = {
                    "input_ids": input_ids[start : start + MICROBATCH].to("cuda"),
                    "attention_mask": attention_mask[start : start + MICROBATCH].to(
                        "cuda"
                    ),
                }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = encoder(**batch).last_hidden_state
                    features = pool(hidden, batch["attention_mask"])
                    logits = head(features)[:, 0]
                best = max(best, float(logits.float().max()))
            scores[index] = 1.0 / (1.0 + math.exp(-best))
            if (index + 1) % 500 == 0:
                print(f"scored {index + 1}/{len(texts)} texts", flush=True)
    return scores


def _auroc(benign: np.ndarray, attack: np.ndarray) -> float:
    labels = np.concatenate([np.zeros(len(benign)), np.ones(len(attack))])
    values = np.concatenate([benign, attack])
    return float(roc_auc_score(labels, values))


def _slice_metrics(
    rows: list[dict],
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    threshold: float,
    selector,
) -> dict:
    indexes = [index for index, row in enumerate(rows) if selector(row)]
    if not indexes:
        return {"pairs": 0}
    benign = benign_scores[indexes]
    attack = attack_scores[indexes]
    return {
        "pairs": len(indexes),
        "clean_flag_rate": float((benign >= threshold).mean()),
        "attack_recall": float((attack >= threshold).mean()),
        "pair_ordering": float((attack > benign).mean()),
        "clean_high_gate_rate": float((benign >= HIGH_GATE).mean()),
        "attack_high_gate_rate": float((attack >= HIGH_GATE).mean()),
        "auroc": _auroc(benign, attack),
    }


def _length_bucket(row: dict) -> str:
    length = len(strict_normalize(row["benign"]))
    for edge in LENGTH_EDGES:
        if length < edge:
            return f"under_{edge}"
    return f"at_least_{LENGTH_EDGES[-1]}"


def _repository_macro(
    rows: list[dict],
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    threshold: float,
) -> dict:
    by_repository = defaultdict(list)
    for index, row in enumerate(rows):
        by_repository[row["repository"]].append(index)
    clean_rates = []
    recalls = []
    for indexes in by_repository.values():
        clean_rates.append(float((benign_scores[indexes] >= threshold).mean()))
        recalls.append(float((attack_scores[indexes] >= threshold).mean()))
    return {
        "repositories": len(by_repository),
        "macro_clean_flag_rate": float(np.mean(clean_rates)),
        "macro_attack_recall": float(np.mean(recalls)),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once output already exists: {OUTPUT}")
    report = {
        "purpose": "repository-held-out SWE-rebench V2 pair comparison",
        "advisory_only": True,
        "scoring": "window-max over 512-token windows with 128-token overlap",
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for name, path in SPLITS.items()
        },
        "models": {},
    }
    split_rows = {name: _pairs(path) for name, path in SPLITS.items()}
    for run_name, run in RUNS.items():
        import torch

        threshold = _descriptive_threshold(run)
        result, encoder, tokenizer, head = _load_run(run)
        model_report = {
            "descriptive_1pct_threshold": threshold,
            "high_gate": HIGH_GATE,
            "splits": {},
        }
        for split_name, rows in split_rows.items():
            print(f"{run_name}: scoring {split_name} benign", flush=True)
            benign = _window_max_scores(
                encoder, tokenizer, head, [row["benign"] for row in rows]
            )
            print(f"{run_name}: scoring {split_name} attack", flush=True)
            attack = _window_max_scores(
                encoder, tokenizer, head, [row["attack"] for row in rows]
            )
            buckets = sorted({_length_bucket(row) for row in rows})
            model_report["splits"][split_name] = {
                "overall": _slice_metrics(
                    rows, benign, attack, threshold, lambda row: True
                ),
                "by_length": {
                    bucket: _slice_metrics(
                        rows,
                        benign,
                        attack,
                        threshold,
                        lambda row, bucket=bucket: _length_bucket(row) == bucket,
                    )
                    for bucket in buckets
                },
                "repository_macro": _repository_macro(rows, benign, attack, threshold),
            }
        report["models"][run_name] = model_report
        del encoder, head
        torch.cuda.empty_cache()
    report["thresholded_rate_context"] = (
        "descriptive at each model's validation-calibrated "
        f"{DESCRIPTIVE_FPR_BUDGET} component-FPR coordinate; not a selection target"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
