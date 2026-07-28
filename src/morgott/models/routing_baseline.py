from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..data import iter_verified_jsonl, manifest_output_path, normalize_text

SEED = 42
DEFAULT_EPOCHS = 3
DEFAULT_MAX_PER_SOURCE_LABEL = 20_000
DEFAULT_THRESHOLD = 0.5
BATCH_SIZE = 2_048
MAX_BATCH_CHARACTERS = 2_000_000
FEATURES = 2**20
FAILED_ATTEMPT_SOURCES = {"hackaprompt", "tensor_trust_raw"}


def _direct_strength(row: dict) -> str:
    origins = row.get("origins") or [row]

    def weak(origin: dict) -> bool:
        basis = origin.get("label_basis", "")
        return "weak" in basis or (
            origin.get("routing_label", row.get("routing_label")) == 0
            and "generated" in basis
        )

    if all(weak(origin) for origin in origins):
        return "weak_label"
    if origins and all(
        origin.get("source") in FAILED_ATTEMPT_SOURCES
        and origin.get("source_attack_success") is not True
        for origin in origins
    ):
        return "failed_attack_attempt"
    return "strong"


def _rows(data_dir: Path, manifest: dict, split: str):
    output = manifest["routing_views"][split]
    for row in iter_verified_jsonl(
        manifest_output_path(data_dir, output), output["sha256"]
    ):
        strength = _direct_strength(row)
        if row["input_channel"] != "direct_user" or strength == "weak_label":
            continue
        yield {
            "text": normalize_text(row["text"]),
            "label": row["routing_label"],
            "source": row["source"],
            "group": row["split_group_id"],
            "hash": row["normalized_text_sha256"],
            "strength": strength,
        }


def _cap_rows(rows, maximum: int) -> tuple[list[dict], dict]:
    heaps = defaultdict(list)
    input_counts = Counter()
    sequence = 0
    for row in rows:
        key = (row["source"], row["label"])
        input_counts[key] += 1
        rank = int(row["hash"], 16)
        item = (-rank, sequence, row)
        sequence += 1
        heap = heaps[key]
        if len(heap) < maximum:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = sorted(
        (item[2] for heap in heaps.values() for item in heap),
        key=lambda row: row["hash"],
    )
    if {row["label"] for row in selected} != {0, 1}:
        raise ValueError("routing baseline requires both classes")
    selected_counts = Counter((row["source"], row["label"]) for row in selected)

    def serialize(counts: Counter) -> dict:
        return {
            f"{source}:{label}": count
            for (source, label), count in sorted(counts.items())
        }

    return selected, {
        "input_rows": sum(input_counts.values()),
        "selected_rows": len(selected),
        "max_per_source_label": maximum,
        "input_by_source_label": serialize(input_counts),
        "selected_by_source_label": serialize(selected_counts),
    }


def _row_batches(rows):
    batch = []
    characters = 0
    for row in rows:
        length = len(row["text"])
        if batch and (
            len(batch) == BATCH_SIZE or characters + length > MAX_BATCH_CHARACTERS
        ):
            yield batch
            batch = []
            characters = 0
        batch.append(row)
        characters += length
    if batch:
        yield batch


def _fit(rows: list[dict], epochs: int):
    vectorizer = HashingVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        n_features=FEATURES,
        alternate_sign=False,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        average=True,
        random_state=SEED,
    )
    for epoch in range(epochs):
        order = np.random.default_rng(SEED + epoch).permutation(len(rows))
        shuffled = (rows[int(index)] for index in order)
        for batch in _row_batches(shuffled):
            classifier.partial_fit(
                vectorizer.transform([row["text"] for row in batch]),
                np.asarray([row["label"] for row in batch], dtype=np.int8),
                classes=np.asarray([0, 1]),
            )
    return vectorizer, classifier


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = scores >= DEFAULT_THRESHOLD
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    true_positive = int(np.sum(predictions & (labels == 1)))
    false_positive = int(np.sum(predictions & (labels == 0)))
    result = {
        "rows": len(labels),
        "positive": positives,
        "negative": negatives,
        "recall": true_positive / positives if positives else None,
        "fpr": false_positive / negatives if negatives else None,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "false_signals_per_10k": (
            false_positive / negatives * 10_000 if negatives else None
        ),
        "pr_auc": None,
        "roc_auc": None,
        "brier": float(brier_score_loss(labels, scores)) if len(labels) else None,
    }
    if positives and negatives:
        result["pr_auc"] = float(average_precision_score(labels, scores))
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    return result


def _evaluate(rows, vectorizer, classifier) -> dict:
    labels = []
    sources = []
    scores = []
    for batch in _row_batches(rows):
        labels.extend(row["label"] for row in batch)
        sources.extend(row["source"] for row in batch)
        scores.extend(
            classifier.predict_proba(
                vectorizer.transform([row["text"] for row in batch])
            )[:, 1]
        )
    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores)
    by_source = {}
    source_recalls = []
    source_fprs = []
    for source in sorted(set(sources)):
        mask = np.asarray([value == source for value in sources])
        source_metrics = _metrics(labels_array[mask], scores_array[mask])
        by_source[source] = source_metrics
        if source_metrics["recall"] is not None:
            source_recalls.append(source_metrics["recall"])
        if source_metrics["fpr"] is not None:
            source_fprs.append(source_metrics["fpr"])
    return {
        "all": _metrics(labels_array, scores_array),
        "by_source": by_source,
        "macro_source_recall": (
            float(np.mean(source_recalls)) if source_recalls else None
        ),
        "macro_source_fpr": float(np.mean(source_fprs)) if source_fprs else None,
    }


def _format(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_report(path: Path, result: dict) -> None:
    lines = [
        "# Routing word n-gram baseline",
        "",
        "This is a cheap research control, not a blocking model or a production estimate.",
        "It uses source-supported direct-user rows, unweighted training, and the untouched 0.5 cutoff.",
        "",
        "| Split | Recall | FPR | Precision | PR-AUC | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in result["evaluation"].items():
        values = metrics["all"]
        lines.append(
            f"| {split} | {_format(values['recall'])} | {_format(values['fpr'])} | "
            f"{_format(values['precision'])} | {_format(values['pr_auc'])} | "
            f"{_format(values['roc_auc'])} |"
        )
    lines.extend(
        [
            "",
            "Per-source metrics and exact recipe metadata are in `routing-baseline.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_routing_baseline(
    data_dir: Path = Path("data"),
    artifacts_dir: Path = Path("artifacts"),
    reports_dir: Path = Path("reports"),
    *,
    epochs: int = DEFAULT_EPOCHS,
    max_per_source_label: int = DEFAULT_MAX_PER_SOURCE_LABEL,
) -> dict:
    if epochs < 1 or max_per_source_label < 1:
        raise ValueError("epochs and max_per_source_label must be positive")
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 5:
        raise ValueError("routing baseline requires canonical data schema 5")
    train_rows, selection = _cap_rows(
        _rows(data_dir, manifest, "train"), max_per_source_label
    )
    vectorizer, classifier = _fit(train_rows, epochs)
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "recipe": {
            "model": "word 1-2 gram hashing vectorizer with unweighted SGD logistic regression",
            "seed": SEED,
            "features": FEATURES,
            "epochs": epochs,
            "threshold": DEFAULT_THRESHOLD,
            "threshold_selection": "untouched default; no validation tuning",
        },
        "selection": selection,
        "training_sources": sorted({row["source"] for row in train_rows}),
        "evaluation": {
            split: _evaluate(_rows(data_dir, manifest, split), vectorizer, classifier)
            for split in ("validation", "dev_test")
        },
        "limitations": [
            "public development labels are not production labels",
            "source and label remain confounded",
            "single seed",
            "scores are uncalibrated",
            "classifier output is advisory and never grants authority",
        ],
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "operating_mode": "research_only",
            "data_manifest_sha256": result["data_manifest_sha256"],
            "recipe": result["recipe"],
            "vectorizer": vectorizer,
            "classifier": classifier,
        },
        artifacts_dir / "routing_baseline.joblib",
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "routing-baseline.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(reports_dir / "routing-baseline.md", result)
    return result
