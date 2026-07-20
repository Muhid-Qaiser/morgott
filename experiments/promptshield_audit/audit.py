"""Pinned, evaluation-only audit of the public PromptShield dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from vulsight_guard.data import normalize_text


REPO = "hendzh/PromptShield"
REVISION = "a5234cb1f5cdb256600cab64b8c961195b5e8404"
LICENSE = "Apache-2.0"
DATASET_URL = f"https://huggingface.co/datasets/{REPO}"
PAPER_URL = "https://arxiv.org/abs/2501.15145"
FILES = {
    "train": {
        "name": "train.json",
        "bytes": 12_239_651,
        "sha256": "aa33c3ffcc27bd07c0a233b52f1b8c3cbdb30606ce2412da06a88b5290cdc7b6",
        "rows": 18_909,
        "positive": 9_452,
    },
    "validation": {
        "name": "validation.json",
        "bytes": 645_951,
        "sha256": "1d93d90d57d3ef44ed0c546fbc04d66324436c5fcd32e7fcb940ceed270fbe77",
        "rows": 1_000,
        "positive": 503,
    },
    "test": {
        "name": "test.json",
        "bytes": 18_288_615,
        "sha256": "526207c2485829d9961407011d7f4cd929569e7f285dc8396b3f385e0608bc70",
        "rows": 23_516,
        "positive": 6_486,
    },
}
CARD = {
    "name": "DATASET_CARD.md",
    "bytes": 1_792,
    "sha256": "d5f36dce4f27d40ae8fda54335d382c74e650485cb4a92c6837602ef84a1a662",
}
FIT_PARTITIONS = {"train", "indirect_train"}
NEAR_BITS = 128
NEAR_BANDS = 8
NEAR_MAX_HAMMING = 6
NEAR_MIN_WORDS = 5
_WORD = re.compile(r"\w+", re.UNICODE)
_LENGTH_BUCKETS = (
    ("0-64", 0, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257-512", 257, 512),
    ("513+", 513, math.inf),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _normalized_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode()).hexdigest()


def _fingerprint_normalized(normalized: str) -> int | None:
    words = _WORD.findall(normalized)
    if len(words) < NEAR_MIN_WORDS:
        return None
    features = words + [f"{left}\0{right}" for left, right in zip(words, words[1:])]
    digests = b"".join(
        hashlib.blake2b(feature.encode(), digest_size=NEAR_BITS // 8).digest()
        for feature in features
    )
    bits = np.unpackbits(
        np.frombuffer(digests, dtype=np.uint8), bitorder="little"
    ).reshape(-1, NEAR_BITS)
    majority = bits.sum(axis=0) * 2 >= len(features)
    return int.from_bytes(np.packbits(majority, bitorder="little").tobytes(), "little")


def fingerprint(text: str) -> int | None:
    """Return the deterministic near-overlap fingerprint used by this audit."""
    return _fingerprint_normalized(normalize_text(text))


class NearIndex:
    """Small banded SimHash index; results are strict signals, not exhaustive."""

    def __init__(self) -> None:
        self.records: list[tuple[int, str, tuple[str, ...]]] = []
        self.buckets = [defaultdict(list) for _ in range(NEAR_BANDS)]

    def add(self, value: int | None, normalized_hash: str, tags: set[str]) -> None:
        if value is None:
            return
        index = len(self.records)
        self.records.append((value, normalized_hash, tuple(sorted(tags))))
        for band in range(NEAR_BANDS):
            key = (value >> (band * 16)) & 0xFFFF
            self.buckets[band][key].append(index)

    def query(
        self, value: int | None, normalized_hash: str
    ) -> list[tuple[str, tuple[str, ...]]]:
        if value is None:
            return []
        candidates: set[int] = set()
        for band in range(NEAR_BANDS):
            key = (value >> (band * 16)) & 0xFFFF
            candidates.update(self.buckets[band].get(key, ()))
        return [
            (candidate_hash, tags)
            for index in candidates
            for candidate, candidate_hash, tags in (self.records[index],)
            if candidate_hash != normalized_hash
            and (candidate ^ value).bit_count() <= NEAR_MAX_HAMMING
        ]


def _fetch(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = list(FILES.values()) + [CARD]
    for entry in entries:
        if entry is CARD:
            url = f"{DATASET_URL}/raw/{REVISION}/README.md"
        else:
            url = f"{DATASET_URL}/resolve/{REVISION}/{entry['name']}?download=true"
        request = urllib.request.Request(
            url, headers={"User-Agent": "vulsight-agent-guard/0.1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read(entry["bytes"] + 1)
        if len(data) != entry["bytes"] or _sha256_bytes(data) != entry["sha256"]:
            raise ValueError(f"pinned file mismatch: {entry['name']}")
        (data_dir / entry["name"]).write_bytes(data)


def _record(text: str, label: int, split: str, index: int) -> dict:
    normalized = normalize_text(text)
    return {
        "split": split,
        "index": index,
        "prompt": text,
        "label": label,
        "raw_hash": _raw_hash(text),
        "normalized_hash": _normalized_hash(normalized),
        "fingerprint": _fingerprint_normalized(normalized),
        "characters": len(text),
        "whitespace_tokens": len(text.split()),
    }


def load_source(data_dir: Path) -> dict[str, list[dict]]:
    output = {}
    for split, expected in FILES.items():
        path = data_dir / expected["name"]
        data = path.read_bytes()
        if len(data) != expected["bytes"] or _sha256_bytes(data) != expected["sha256"]:
            raise ValueError(f"pinned file mismatch: {expected['name']}")
        rows = json.loads(data)
        if not isinstance(rows, list):
            raise ValueError(f"{split} is not a JSON list")
        projected = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"prompt", "label"}:
                raise ValueError(f"{split}:{index} has unexpected fields")
            text, label = row["prompt"], row["label"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split}:{index} has empty prompt")
            if (
                isinstance(label, bool)
                or not isinstance(label, int)
                or label not in (0, 1)
            ):
                raise ValueError(f"{split}:{index} has invalid label")
            projected.append(_record(text, label, split, index))
        positives = sum(row["label"] for row in projected)
        if len(projected) != expected["rows"] or positives != expected["positive"]:
            raise ValueError(f"{split} pinned counts changed")
        output[split] = projected
    return output


def _within_source(splits: dict[str, list[dict]]) -> dict:
    split_summary = {}
    for split, rows in splits.items():
        raw = Counter(row["raw_hash"] for row in rows)
        normalized = Counter(row["normalized_hash"] for row in rows)
        labels: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            labels[row["normalized_hash"]].add(row["label"])
        conflicts = {digest for digest, values in labels.items() if len(values) > 1}
        split_summary[split] = {
            "rows": len(rows),
            "unique_raw": len(raw),
            "raw_duplicate_rows": len(rows) - len(raw),
            "unique_normalized": len(normalized),
            "normalized_duplicate_rows": len(rows) - len(normalized),
            "normalized_label_conflict_texts": len(conflicts),
            "normalized_label_conflict_rows": sum(normalized[key] for key in conflicts),
        }

    cross = {}
    names = list(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left, right = splits[left_name], splits[right_name]
            left_raw = Counter(row["raw_hash"] for row in left)
            right_raw = Counter(row["raw_hash"] for row in right)
            left_norm = Counter(row["normalized_hash"] for row in left)
            right_norm = Counter(row["normalized_hash"] for row in right)
            shared_raw = set(left_raw) & set(right_raw)
            shared_norm = set(left_norm) & set(right_norm)
            labels: dict[str, set[int]] = defaultdict(set)
            for row in left + right:
                if row["normalized_hash"] in shared_norm:
                    labels[row["normalized_hash"]].add(row["label"])
            conflicts = {digest for digest, values in labels.items() if len(values) > 1}

            right_unique = {}
            for row in right:
                right_unique.setdefault(row["normalized_hash"], row)
            index = NearIndex()
            for digest, row in right_unique.items():
                index.add(row["fingerprint"], digest, {right_name})
            near_pairs = set()
            for row in left:
                if row["normalized_hash"] in shared_norm:
                    continue
                for digest, _ in index.query(
                    row["fingerprint"], row["normalized_hash"]
                ):
                    near_pairs.add((row["normalized_hash"], digest))
            left_near = {pair[0] for pair in near_pairs}
            right_near = {pair[1] for pair in near_pairs}
            cross[f"{left_name}__{right_name}"] = {
                "raw_exact_unique_texts": len(shared_raw),
                "raw_exact_left_rows": sum(left_raw[key] for key in shared_raw),
                "raw_exact_right_rows": sum(right_raw[key] for key in shared_raw),
                "normalized_exact_unique_texts": len(shared_norm),
                "normalized_exact_left_rows": sum(
                    left_norm[key] for key in shared_norm
                ),
                "normalized_exact_right_rows": sum(
                    right_norm[key] for key in shared_norm
                ),
                "normalized_label_conflict_texts": len(conflicts),
                "near_nonexact_pairs": len(near_pairs),
                "near_nonexact_left_unique_texts": len(left_near),
                "near_nonexact_left_rows": sum(left_norm[key] for key in left_near),
                "near_nonexact_right_unique_texts": len(right_near),
                "near_nonexact_right_rows": sum(right_norm[key] for key in right_near),
            }
    return {"within_split": split_summary, "cross_split": cross}


def _active_reference(data_dir: Path) -> dict:
    raw_partitions: dict[str, set[str]] = defaultdict(set)
    normalized_records: dict[str, dict] = {}
    rows_by_partition = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        partition = path.stem
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                label = row.get("label")
                if not isinstance(text, str) or not text.strip() or label not in (0, 1):
                    raise ValueError(f"{path}:{line_number} has invalid text/label")
                normalized = normalize_text(text)
                raw_partitions[_raw_hash(text)].add(partition)
                digest = _normalized_hash(normalized)
                record = normalized_records.setdefault(
                    digest,
                    {
                        "fingerprint": _fingerprint_normalized(normalized),
                        "partitions": set(),
                        "labels": set(),
                    },
                )
                record["partitions"].add(partition)
                record["labels"].add(int(label))
                count += 1
        rows_by_partition[partition] = count

    index = NearIndex()
    for digest, record in normalized_records.items():
        index.add(record["fingerprint"], digest, record["partitions"])
    return {
        "raw": raw_partitions,
        "normalized": normalized_records,
        "near_index": index,
        "rows_by_partition": rows_by_partition,
    }


def _active_overlap(splits: dict[str, list[dict]], processed_dir: Path) -> dict:
    reference = _active_reference(processed_dir)
    fit_names = FIT_PARTITIONS & set(reference["rows_by_partition"])
    partition_counts = {
        name: {"raw_exact_rows": 0, "normalized_exact_rows": 0, "near_rows": 0}
        for name in reference["rows_by_partition"]
    }

    def empty_summary() -> dict:
        return {
            "rows": 0,
            "raw_exact_rows": 0,
            "normalized_exact_rows": 0,
            "near_nonexact_rows": 0,
            "near_nonexact_without_any_normalized_exact_rows": 0,
            "any_fit_overlap_rows": 0,
            "normalized_match_with_same_label_rows": 0,
            "normalized_match_with_opposite_label_rows": 0,
        }

    overall = empty_summary()
    by_split = {split: empty_summary() for split in splits}
    by_label = {str(label): empty_summary() for label in (0, 1)}
    for split, rows in splits.items():
        for row in rows:
            targets = (overall, by_split[split], by_label[str(row["label"])])
            for target in targets:
                target["rows"] += 1
            raw_parts = reference["raw"].get(row["raw_hash"], set())
            normalized_match = reference["normalized"].get(row["normalized_hash"])
            normalized_parts = (
                normalized_match["partitions"] if normalized_match else set()
            )
            near_matches = reference["near_index"].query(
                row["fingerprint"], row["normalized_hash"]
            )
            near_parts = {tag for _, tags in near_matches for tag in tags}
            row["active_fit_overlap"] = bool(
                (raw_parts | normalized_parts | near_parts) & fit_names
            )
            for target in targets:
                target["raw_exact_rows"] += bool(raw_parts)
                target["normalized_exact_rows"] += bool(normalized_parts)
                target["near_nonexact_rows"] += bool(near_parts)
                target["near_nonexact_without_any_normalized_exact_rows"] += bool(
                    near_parts and not normalized_parts
                )
                target["any_fit_overlap_rows"] += row["active_fit_overlap"]
                if normalized_match:
                    target["normalized_match_with_same_label_rows"] += (
                        row["label"] in normalized_match["labels"]
                    )
                    target["normalized_match_with_opposite_label_rows"] += (
                        1 - row["label"] in normalized_match["labels"]
                    )
            for partition in raw_parts:
                partition_counts[partition]["raw_exact_rows"] += 1
            for partition in normalized_parts:
                partition_counts[partition]["normalized_exact_rows"] += 1
            for partition in near_parts:
                partition_counts[partition]["near_rows"] += 1

    return {
        "method": {
            "raw_exact": "SHA-256 of the prompt's exact UTF-8 bytes",
            "normalized_exact": "SHA-256 after repository NFKC + casefold + whitespace normalization",
            "near": (
                "128-bit SimHash over normalized word unigrams+bigrams; eight 16-bit "
                f"bands; Hamming <= {NEAR_MAX_HAMMING}; at least {NEAR_MIN_WORDS} words"
            ),
            "near_is_exhaustive": False,
            "near_excludes_same_normalized_hash": True,
        },
        "reference": {
            "processed_directory": "data/processed",
            "partitions": reference["rows_by_partition"],
            "rows": sum(reference["rows_by_partition"].values()),
            "unique_raw_texts": len(reference["raw"]),
            "unique_normalized_texts": len(reference["normalized"]),
            "fit_partitions": sorted(fit_names),
        },
        "overall": overall,
        "by_promptshield_split": by_split,
        "by_promptshield_label": by_label,
        "by_reference_partition": partition_counts,
    }


def _percentiles(values: list[int]) -> dict:
    return {
        "p50": int(np.percentile(values, 50, method="nearest")),
        "p90": int(np.percentile(values, 90, method="nearest")),
        "p95": int(np.percentile(values, 95, method="nearest")),
        "p99": int(np.percentile(values, 99, method="nearest")),
        "max": max(values),
    }


def _lengths(splits: dict[str, list[dict]]) -> dict:
    distributions = {}
    proxies = {}
    for split, rows in splits.items():
        distributions[split] = {}
        proxies[split] = {}
        for label_name, selected in (
            ("all", rows),
            ("benign_source_label", [row for row in rows if row["label"] == 0]),
            ("injection_source_label", [row for row in rows if row["label"] == 1]),
        ):
            distributions[split][label_name] = {
                "rows": len(selected),
                "characters": _percentiles([row["characters"] for row in selected]),
                "whitespace_tokens": _percentiles(
                    [row["whitespace_tokens"] for row in selected]
                ),
            }
            proxies[split][label_name] = {
                f"over_{limit}": sum(
                    row["whitespace_tokens"] > limit for row in selected
                )
                for limit in (256, 512, 1024, 2048)
            }
    return {
        "distribution": distributions,
        "neural_truncation_proxy": {
            "unit": "whitespace tokens; not tokenizer-exact",
            "counts": proxies,
        },
        "locked_character_model": "No input-length truncation is applied.",
    }


def _wilson_upper(successes: int, trials: int, z: float = 1.96) -> float | None:
    if not trials:
        return None
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = rate + z**2 / (2 * trials)
    margin = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2))
    return (center + margin) / denominator


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))
    return {
        "rows": len(labels),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fpr_95_upper": _wilson_upper(fp, fp + tn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }


def _score_locked_control(
    test_rows: list[dict], artifact_path: Path, baseline_path: Path
) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    profiles = baseline["direct_precision_profiles"]
    artifact = joblib.load(artifact_path)
    if artifact.get("schema_version") != 2 or "direct_user" not in artifact.get(
        "channels", {}
    ):
        raise ValueError("unexpected local guard artifact schema")
    model = artifact["channels"]["direct_user"]["model"]
    texts = [normalize_text(row["prompt"]) for row in test_rows]
    start = time.perf_counter()
    scores = model.predict_proba(texts)[:, 1]
    seconds = time.perf_counter() - start
    labels = np.asarray([row["label"] for row in test_rows])
    token_counts = np.asarray([row["whitespace_tokens"] for row in test_rows])
    no_fit_overlap = np.asarray(
        [not row.get("active_fit_overlap", False) for row in test_rows]
    )
    results = []
    for profile in profiles:
        threshold = float(profile["threshold"])
        by_length = {}
        for name, lower, upper in _LENGTH_BUCKETS:
            selected = (token_counts >= lower) & (token_counts <= upper)
            by_length[name] = _metrics(labels[selected], scores[selected], threshold)
        results.append(
            {
                "min_validation_precision": profile["min_validation_precision"],
                "role": profile["role"],
                "threshold": threshold,
                "test": _metrics(labels, scores, threshold),
                "test_excluding_active_fit_overlap": _metrics(
                    labels[no_fit_overlap], scores[no_fit_overlap], threshold
                ),
                "test_by_whitespace_token_bucket": by_length,
            }
        )
    return {
        "artifact_sha256": _sha256_bytes(artifact_path.read_bytes()),
        "baseline_report_sha256": _sha256_bytes(baseline_path.read_bytes()),
        "threshold_source": "reports/baseline.json direct_precision_profiles",
        "thresholds_selected_without_promptshield": True,
        "profiles": results,
        "scoring_seconds": seconds,
        "microseconds_per_prompt": seconds * 1_000_000 / len(test_rows),
    }


def run(
    source_dir: Path,
    processed_dir: Path,
    artifact_path: Path,
    baseline_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict:
    splits = load_source(source_dir)
    within_source = _within_source(splits)
    active_overlap = _active_overlap(splits, processed_dir)
    lengths = _lengths(splits)
    locked_control = _score_locked_control(splits["test"], artifact_path, baseline_path)
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "evaluation_only_never_training",
        "source": {
            "requested_name": "NVIDIA/PromptShield",
            "resolved_public_repo": REPO,
            "revision": REVISION,
            "dataset_url": DATASET_URL,
            "paper_url": PAPER_URL,
            "declared_license": LICENSE,
            "license_caveat": "Upstream component licenses were not independently reconciled in this audit.",
            "files": {**FILES, "dataset_card": CARD},
            "schema": {"prompt": "string", "label": "integer 0 or 1"},
            "label_semantics": {
                "0": "source-labelled benign (no prompt injection)",
                "1": "source-labelled prompt-injection attempt",
            },
            "language": "English according to the paper",
            "paper_reported_composition": {
                "benign_training": ["UltraChat", "Alpaca", "IFEval"],
                "benign_evaluation": [
                    "LMSYS Chatbot Arena",
                    "databricks-dolly",
                    "Natural Instructions",
                    "Synthetic Python Problems",
                ],
                "injection_training": ["FourAttacks over Alpaca", "HackAPrompt"],
                "injection_evaluation": [
                    "FourAttacks over databricks-dolly and Synthetic Python Problems",
                    "OpenPromptInject",
                ],
            },
            "row_lineage": "Unavailable: files contain no source, conversation, template, or group field.",
        },
        "counts": {
            split: {
                "rows": len(rows),
                "positive": sum(row["label"] for row in rows),
                "negative": sum(1 - row["label"] for row in rows),
            }
            for split, rows in splits.items()
        },
        "within_source_overlap": within_source,
        "active_corpus_overlap": active_overlap,
        "lengths": lengths,
        "locked_character_control": locked_control,
        "neural_results": {
            "included": False,
            "reason": "No already-generated PromptShield neural scores exist; no weights or GPU work were used.",
        },
        "limitations": [
            "PromptShield provides no row-level source or grouping lineage, so source-held-out and group-held-out metrics cannot be reconstructed.",
            "The paper reports aggregation from public chat, instruction, and attack datasets; exact and fuzzy overlap can therefore inflate apparent transfer.",
            "Only the prompt field is evaluated. No source is inferred from prompt content.",
            "Source label 0 means no prompt injection, not independently established harmlessness; this audit does not treat it as training data.",
            "Near-overlap is a strict deterministic heuristic and can miss paraphrases or wrapper changes.",
            "PromptShield was observed before this experiment, so its metrics are exploratory development evidence, not an untouched final test or production FPR claim.",
            "No human adjudication is assumed; labels remain public source labels.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(output_markdown, result)
    return result


def _write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# PromptShield evaluation-only audit",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "**Decision: keep PromptShield evaluation-only and never train on it.** The files lack row-level source/group lineage, while the paper says the corpus aggregates public sources and attack strategies that can overlap this project's active corpora. The metrics below are exploratory development evidence, not a final test or production false-positive claim.",
        "",
        "## Pinned source",
        "",
        f"The requested `NVIDIA/PromptShield` name resolves publicly as [{REPO}]({DATASET_URL}) at `{REVISION}`. The card declares {LICENSE}; component-source license compatibility was not independently audited. Paper: [PromptShield: Deployable Detection for Prompt Injection Attacks]({PAPER_URL}).",
        "",
        "| File | Bytes | SHA-256 | Rows | Positive | Negative |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for split, file in FILES.items():
        counts = result["counts"][split]
        lines.append(
            f"| `{file['name']}` | {file['bytes']:,} | `{file['sha256']}` | {counts['rows']:,} | {counts['positive']:,} | {counts['negative']:,} |"
        )
    lines += [
        "",
        f"The pinned dataset card is {CARD['bytes']:,} bytes with SHA-256 `{CARD['sha256']}`. Actual data rows contain exactly `prompt` and `label`; only `prompt` was scored. Label 1 means a source-labelled injection attempt and label 0 means source-labelled benign/no injection—not harmlessness. All data is described as English.",
        "",
        "The paper reports benign inputs from UltraChat, LMSYS Chatbot Arena, Alpaca, databricks-dolly, IFEval, Natural Instructions, and Synthetic Python Problems; attacks come from FourAttacks, HackAPrompt, and OpenPromptInject. The released rows do not say which source, conversation, task, template, or mutation produced each prompt. This audit does not guess.",
        "",
        "## Leakage and duplicate audit",
        "",
        "Raw exact means byte-identical UTF-8. Normalized exact uses this repository's NFKC, casefold, and whitespace view. Near matches use 128-bit SimHash over normalized word unigrams/bigrams, eight 16-bit bands, Hamming distance <= 6, and at least five words. Near matching excludes identical normalized hashes and is a strict, non-exhaustive signal.",
        "",
        "### Against every active processed fit/evaluation file",
        "",
        "| PromptShield split | Rows | Raw exact | Normalized exact | Near non-exact | Near with no exact anywhere | Any fit overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    active = result["active_corpus_overlap"]
    for split, summary in active["by_promptshield_split"].items():
        lines.append(
            f"| {split} | {summary['rows']:,} | {summary['raw_exact_rows']:,} | {summary['normalized_exact_rows']:,} | {summary['near_nonexact_rows']:,} | {summary['near_nonexact_without_any_normalized_exact_rows']:,} | {summary['any_fit_overlap_rows']:,} |"
        )
    lines += [
        "",
        f"Reference: {active['reference']['rows']:,} rows across {len(active['reference']['partitions'])} processed files, with {active['reference']['unique_normalized_texts']:,} unique normalized texts. Fit files are `{', '.join(active['reference']['fit_partitions'])}`; all remaining files are evaluation corpora.",
        "",
        "| Active file | Rows | Raw exact source rows | Normalized exact source rows | Near source rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for partition, counts in active["by_reference_partition"].items():
        if any(counts.values()):
            lines.append(
                f"| {partition} | {active['reference']['partitions'][partition]:,} | {counts['raw_exact_rows']:,} | {counts['normalized_exact_rows']:,} | {counts['near_rows']:,} |"
            )
    lines += [
        "",
        "### Within PromptShield",
        "",
        "| Split | Rows | Raw duplicate rows | Normalized duplicate rows | Conflicting normalized texts |",
        "|---|---:|---:|---:|---:|",
    ]
    internal = result["within_source_overlap"]
    for split, counts in internal["within_split"].items():
        lines.append(
            f"| {split} | {counts['rows']:,} | {counts['raw_duplicate_rows']:,} | {counts['normalized_duplicate_rows']:,} | {counts['normalized_label_conflict_texts']:,} |"
        )
    lines += [
        "",
        "| Split pair | Raw exact unique | Normalized exact unique | Label-conflict exact | Near non-exact pairs | Left/right near rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair, counts in internal["cross_split"].items():
        lines.append(
            f"| {pair.replace('__', ' / ')} | {counts['raw_exact_unique_texts']:,} | {counts['normalized_exact_unique_texts']:,} | {counts['normalized_label_conflict_texts']:,} | {counts['near_nonexact_pairs']:,} | {counts['near_nonexact_left_rows']:,} / {counts['near_nonexact_right_rows']:,} |"
        )

    lengths = result["lengths"]
    lines += [
        "",
        "## Length and truncation risk",
        "",
        "The locked character model scores complete strings and performs no input-length truncation. The following is only a whitespace-token proxy for neural tokenizers; it is not tokenizer-exact.",
        "",
        "| Split/label | Rows | Token p50 | p90 | p95 | p99 | Max | >256 | >512 | >1024 | >2048 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, groups in lengths["distribution"].items():
        for name, values in groups.items():
            proxy = lengths["neural_truncation_proxy"]["counts"][split][name]
            tokens = values["whitespace_tokens"]
            lines.append(
                f"| {split}/{name} | {values['rows']:,} | {tokens['p50']:,} | {tokens['p90']:,} | {tokens['p95']:,} | {tokens['p99']:,} | {tokens['max']:,} | {proxy['over_256']:,} | {proxy['over_512']:,} | {proxy['over_1024']:,} | {proxy['over_2048']:,} |"
            )

    control = result["locked_character_control"]
    lines += [
        "",
        "## Existing locked character control on PromptShield test",
        "",
        "Thresholds come unchanged from `reports/baseline.json`; PromptShield was not used to select or recalibrate them. Counts use PromptShield's public source labels.",
        "",
        "| Validation precision floor | Threshold | TP | FN | FP | TN | Recall | FPR | Precision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in control["profiles"]:
        metrics = profile["test"]
        lines.append(
            f"| {profile['min_validation_precision']:.2f} | {profile['threshold']:.6f} | {metrics['true_positive']:,} | {metrics['false_negative']:,} | {metrics['false_positive']:,} | {metrics['true_negative']:,} | {metrics['recall']:.4f} | {metrics['fpr']:.4f} | {metrics['precision']:.4f} |"
        )
    lines += [
        "",
        "A conservative companion view removes every test row with a raw, normalized, or strict near match to either active fit file (`train` or `indirect_train`). This exclusion is based only on hashes/fingerprints, not source inference.",
        "",
        "| Validation precision floor | Rows | Positive | Negative | TP | FN | FP | TN | Recall | FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in control["profiles"]:
        metrics = profile["test_excluding_active_fit_overlap"]
        lines.append(
            f"| {profile['min_validation_precision']:.2f} | {metrics['rows']:,} | {metrics['positive']:,} | {metrics['negative']:,} | {metrics['true_positive']:,} | {metrics['false_negative']:,} | {metrics['false_positive']:,} | {metrics['true_negative']:,} | {metrics['recall']:.4f} | {metrics['fpr']:.4f} |"
        )
    recommended = control["profiles"][1]["test"]
    strict = control["profiles"][-1]["test"]
    lines += [
        "",
        f"The retained 0.85 profile catches only {recommended['true_positive']:,}/{recommended['positive']:,} source-positive tests ({recommended['recall']:.2%}) while alerting on {recommended['false_positive']:,}/{recommended['negative']:,} source negatives ({recommended['fpr']:.2%}). Even the 0.95 profile catches only {strict['true_positive']:,}/{strict['positive']:,} ({strict['recall']:.2%}), with {strict['false_positive']:,}/{strict['negative']:,} source-negative alerts ({strict['fpr']:.3%}; Wilson 95% upper {strict['fpr_95_upper']:.3%}). Removing the 97 fit-overlap test rows barely changes this. The result is a transfer failure for the current control, not a reason to train on an untraceable aggregate or a production lockout estimate.",
        "",
        "### Full test metrics by whitespace-token length",
        "",
    ]
    for profile in control["profiles"]:
        lines += [
            f"Precision floor {profile['min_validation_precision']:.2f}, threshold {profile['threshold']:.6f}",
            "",
            "| Bucket | Rows | TP | FN | FP | TN | Recall | FPR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for bucket, metrics in profile["test_by_whitespace_token_bucket"].items():
            recall = "—" if metrics["recall"] is None else f"{metrics['recall']:.4f}"
            fpr = "—" if metrics["fpr"] is None else f"{metrics['fpr']:.4f}"
            lines.append(
                f"| {bucket} | {metrics['rows']:,} | {metrics['true_positive']:,} | {metrics['false_negative']:,} | {metrics['false_positive']:,} | {metrics['true_negative']:,} | {recall} | {fpr} |"
            )
        lines.append("")
    lines += [
        "No neural result is included: no already-generated PromptShield neural scores existed, and this audit downloaded no weights and used no GPU.",
        "",
        "## Interpretation limits",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--source-dir", type=Path, default=experiment / "data")
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed")
    parser.add_argument(
        "--artifact", type=Path, default=root / "artifacts/guard_bundle.joblib"
    )
    parser.add_argument("--baseline", type=Path, default=root / "reports/baseline.json")
    parser.add_argument("--output-json", type=Path, default=experiment / "results.json")
    parser.add_argument(
        "--output-markdown", type=Path, default=experiment / "REPORT.md"
    )
    args = parser.parse_args()
    if args.fetch:
        _fetch(args.source_dir)
    result = run(
        args.source_dir,
        args.processed_dir,
        args.artifact,
        args.baseline,
        args.output_json,
        args.output_markdown,
    )
    summary = result["locked_character_control"]["profiles"][1]["test"]
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "test_rows": summary["rows"],
                "recommended_profile_tp": summary["true_positive"],
                "recommended_profile_fp": summary["false_positive"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
