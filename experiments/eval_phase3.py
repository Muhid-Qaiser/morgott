"""Evaluate Phase 3 heads with validation-selected operating points.

The threshold is selected once on the archived recipe's validation population,
then applied unchanged to dev-test, PromptShield, and SEP.
Same-test ROC operating points are retained as descriptive diagnostics only.

Run:
  PYTHONPATH=src:experiments/_archived uv run python experiments/eval_phase3.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import fpr_and_recall, load_records, threshold_at_fpr  # noqa: E402
from encoder_infer import (  # noqa: E402
    DIRECT_MAX_TOKENS,
    Member,
    direct_head_probability,
    load_member,
    route_probability,
    score_texts,
)
from strict_normalize import strict_normalize  # noqa: E402

from morgott.data import text_hash  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "artifacts" / "phase3_archived"
PROMPTSHIELD = REPO_ROOT / "artifacts/external_eval_data/promptshield/test.jsonl"
SEP = REPO_ROOT / "artifacts/external_eval_data/sep/sep.jsonl"
EVALUATION_SCHEMA_VERSION = 3
EVALUATION_DIRECTORY = "evaluation_v3"
OUT = RUNS_DIR / "phase3_summary_v3.json"
PAIR_ARCHIVE = REPO_ROOT / "artifacts/matched_pairs/pairs_20260726T105034Z.jsonl"
TARGET_FPR = 0.001
OPERATING_FPRS = (0.001, 0.01)
EVAL_TOKEN_BUDGET = 8_192
PROMPTSHIELD_SHA256 = "c763dcde8cc9921613476887b43f12917229d1e5e6cfa29c07ee5dc36311abf6"
SEP_SHA256 = "0ddcfa5a7963f65f9fc8fdf63af10b9052685f87f0142c243a42a394d6e31a89"
PAIR_SHA256 = "8ec5c1c77b378688b190722f7d1fc51e9bef819ee9670948d2658f4a37082158"
ROUTE_NEGATIVE_LABELS = {"benign", "harmful_non_injection"}
ROUTE_POSITIVE_LABELS = {
    "direct_jailbreak",
    "direct_prompt_injection",
    "indirect_prompt_injection",
}
SCORE_DEFINITION = {
    "direct_head": "sigmoid(direct_instruction_subversion_logit)",
    "route": "max(sigmoid(direct_instruction_subversion_logit), "
    "sigmoid(jailbreak_logit))",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_input_sha256(include_pairs: bool) -> dict[str, str | None]:
    experiments = Path(__file__).resolve().parent
    return {
        "data_manifest": file_sha256(REPO_ROOT / "data/manifest.json"),
        "promptshield": PROMPTSHIELD_SHA256,
        "sep": SEP_SHA256,
        "matched_pairs": PAIR_SHA256 if include_pairs else None,
        "evaluator": file_sha256(Path(__file__)),
        "metrics_helper": file_sha256(experiments / "common.py"),
        "inference_helper": file_sha256(experiments / "encoder_infer.py"),
        "generated_shim": file_sha256(experiments / "_archived/routing_encoder.py"),
        "threshold_helper": file_sha256(REPO_ROOT / "src/morgott/detector.py"),
        "canonical_text_helper": file_sha256(REPO_ROOT / "src/morgott/data.py"),
        "strict_normalizer": file_sha256(experiments / "strict_normalize.py"),
    }


def legacy_condition(directory_name: str) -> str:
    match = re.fullmatch(
        r"(raw|strict)"
        r"(_pairs(?:\d+)?)?"
        r"(_rank[\d.]+)?"
        r"(_mmbert)?"
        r"(?:_s(\d+))?",
        directory_name.removeprefix("archived_"),
    )
    if not match:
        raise ValueError(f"cannot derive condition from {directory_name}")
    condition = match.group(1)
    if match.group(2):
        condition += "+pairs" + (match.group(2).replace("_pairs", "") or "")
    if match.group(3):
        condition += match.group(3).replace("_", "+")
    if match.group(4):
        condition += "+mmbert"
    return condition


def discover(requested: set[str] | None = None) -> list[dict]:
    """Find complete runs and verify all provenance available at training time."""
    manifest_sha256 = file_sha256(REPO_ROOT / "data/manifest.json")
    found = []
    seen = set()
    for directory in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        if requested and directory.name not in requested:
            continue
        summary_path = directory / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing run summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if not summary.get("model_id") or not summary.get("model_revision"):
            raise ValueError(f"incomplete model provenance: {summary_path}")
        if not summary.get("attention"):
            raise ValueError(f"missing attention provenance: {summary_path}")
        heads = list(
            directory.glob(
                "direct_failure_repair/*/wildguard_weak_transfer/*/head.safetensors"
            )
        )
        reports = list((directory / "reports").glob("*.json"))
        if len(heads) != 1 or len(reports) != 1:
            raise ValueError(f"{directory.name} is not one complete weak-transfer run")
        head = heads[0]
        result_path = head.parent / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing result: {result_path}")
        result = json.loads(result_path.read_text())
        report = json.loads(reports[0].read_text())
        if report["data_manifest_sha256"] != manifest_sha256:
            raise ValueError(f"manifest mismatch: {reports[0]}")
        report_runtime = report["runtime"]
        for key in ("model_id", "model_revision"):
            if summary[key] != report_runtime[key]:
                raise ValueError(f"{key} mismatch: {directory}")
        if summary["attention"] != report["attention_implementation"]:
            raise ValueError(f"attention implementation mismatch: {directory}")
        updates = result["training"]["updates"]
        if summary.get("training_updates") not in {None, updates}:
            raise ValueError(f"training update mismatch: {directory}")
        seed_match = re.fullmatch(r"seed_(\d+)", head.parent.name)
        if not seed_match:
            raise ValueError(f"cannot derive seed from {head}")
        seed = int(seed_match.group(1))
        if summary.get("seed") not in {None, seed}:
            raise ValueError(f"training seed mismatch: {directory}")
        result_head_sha256 = (result.get("artifact") or {}).get("head_sha256")
        recorded_head_sha256 = summary.get("head_sha256") or result_head_sha256
        actual_head_sha256 = file_sha256(head)
        if recorded_head_sha256 and recorded_head_sha256 != actual_head_sha256:
            raise ValueError(f"head hash mismatch: {head}")
        if result_head_sha256 and result_head_sha256 != actual_head_sha256:
            raise ValueError(f"result head hash mismatch: {head}")
        report_candidate = report["candidates"]["wildguard_weak_transfer"]
        if report_candidate["artifact"]["head_sha256"] != actual_head_sha256:
            raise ValueError(f"report head hash mismatch: {head}")
        if report_candidate["training"]["updates"] != updates:
            raise ValueError(f"report training update mismatch: {directory}")
        found.append(
            {
                "condition": summary.get("condition")
                or legacy_condition(directory.name),
                "seed": seed,
                "strict": summary["normalize"] == "strict",
                "model_id": summary["model_id"],
                "model_revision": summary["model_revision"],
                "attention": summary["attention"],
                "head": head,
                "head_sha256": actual_head_sha256,
                "head_hash_recorded": bool(recorded_head_sha256),
                "updates": updates,
                "directory": directory,
                "summary": summary,
            }
        )
        seen.add(directory.name)
    if requested and seen != requested:
        raise FileNotFoundError(f"requested runs not found: {sorted(requested - seen)}")
    return found


def read_external(
    path: Path,
    *,
    expected_sha256: str,
    text_field: str,
    expected_rows: int,
    expected_positive: int,
) -> tuple[list[dict], np.ndarray, list[str], list[str]]:
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"input hash mismatch for {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if len(rows) != expected_rows or int(labels.sum()) != expected_positive:
        raise ValueError(
            f"input population mismatch for {path}: "
            f"{len(rows)} rows, {int(labels.sum())} positive"
        )
    raw = [row[text_field] for row in rows]
    return rows, labels, raw, [strict_normalize(text) for text in raw]


def applied_point(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    negative = labels == 0
    positive = labels == 1
    return {
        "threshold": float(threshold),
        "fpr": float(predicted[negative].mean()),
        "tpr": float(predicted[positive].mean()),
        "false_positive": int(predicted[negative].sum()),
        "true_positive": int(predicted[positive].sum()),
        "negative": int(negative.sum()),
        "positive": int(positive.sum()),
    }


def metrics(
    scores: np.ndarray, labels: np.ndarray, thresholds: dict[str, float]
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    empirical_0_1 = threshold_at_fpr(scores, labels, TARGET_FPR)
    empirical_1 = threshold_at_fpr(scores, labels, 0.01)
    return {
        "rows": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "validation_threshold_applied": {
            name: applied_point(scores, labels, threshold)
            for name, threshold in thresholds.items()
        },
        "descriptive_same_test_at_0_1pct_fpr": {
            "target_fpr": TARGET_FPR,
            **applied_point(scores, labels, empirical_0_1),
        },
        "descriptive_same_test_at_1pct_fpr": {
            "target_fpr": 0.01,
            **applied_point(scores, labels, empirical_1),
        },
    }


def score(
    loaded,
    texts: list[str],
    labels: np.ndarray,
    direct_thresholds: dict[str, float],
    route_thresholds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    direct, route = score_values(loaded, texts)
    return (
        direct,
        route,
        metrics(direct, labels, direct_thresholds),
        metrics(route, labels, route_thresholds),
    )


def score_values(loaded, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    logits = score_texts(
        loaded,
        texts,
        channel="direct_user",
        token_budget=EVAL_TOKEN_BUDGET,
    )
    return direct_head_probability(logits), route_probability(logits)


def direct_selection(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        index
        for index, record in enumerate(records)
        if record["targets"]["direct_instruction_subversion"] is not None
    ]
    labels = [
        records[index]["targets"]["direct_instruction_subversion"] for index in selected
    ]
    return np.asarray(selected, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def route_selection(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    labels = []
    for index, record in enumerate(records):
        security_label = record["security_label"]
        if security_label in ROUTE_NEGATIVE_LABELS:
            label = 0
        elif security_label in ROUTE_POSITIVE_LABELS:
            label = 1
        else:
            continue
        selected.append(index)
        labels.append(label)
    return np.asarray(selected, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def select_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[dict, dict]:
    selected = {}
    thresholds = {}
    for target_fpr in OPERATING_FPRS:
        name = f"{target_fpr:.4%}"
        achieved_fpr, tpr, threshold = fpr_and_recall(scores, labels, target_fpr)
        thresholds[name] = threshold
        selected[name] = {
            "target_fpr": target_fpr,
            "threshold": threshold,
            "fpr": achieved_fpr,
            "tpr": tpr,
        }
    return thresholds, selected


def identity_sha256(rows: list[dict], text_field: str | None = None) -> str:
    digest = hashlib.sha256()
    for row in rows:
        identity = row.get("hash")
        if identity is None:
            identity = hashlib.sha256(row[text_field].encode()).hexdigest()
        digest.update(str(identity).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def save_array(directory: Path, name: str, values: np.ndarray) -> dict:
    path = directory / f"{name}.npy"
    np.save(path, values)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
    }


def seen_hashes(records: list[dict], include_pairs: bool) -> tuple[set[str], set[str]]:
    texts = [record["text"] for record in records]
    if include_pairs:
        if file_sha256(PAIR_ARCHIVE) != PAIR_SHA256:
            raise ValueError(f"matched-pair hash mismatch: {PAIR_ARCHIVE}")
        with PAIR_ARCHIVE.open(encoding="utf-8") as handle:
            pairs = [json.loads(line) for line in handle if line.strip()]
        texts.extend(
            pair[key]
            for pair in pairs
            if pair.get("channel") == "direct_user"
            for key in ("benign", "attack")
        )
    return (
        {text_hash(text) for text in texts},
        {hashlib.sha256(strict_normalize(text).encode()).hexdigest() for text in texts},
    )


def exact_seen_mask(
    texts: list[str], normalized: set[str], strict: set[str]
) -> np.ndarray:
    return np.asarray(
        [
            text_hash(text) in normalized
            or hashlib.sha256(strict_normalize(text).encode()).hexdigest() in strict
            for text in texts
        ],
        dtype=bool,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="exact run directory name; repeat to evaluate several runs",
    )
    args = parser.parse_args()
    runs = discover(set(args.run) or None)
    if not runs:
        raise FileNotFoundError("no complete runs found")
    evaluation_inputs = {
        include_pairs: evaluation_input_sha256(include_pairs)
        for include_pairs in {bool(run["summary"].get("matched_pairs")) for run in runs}
    }

    training = load_records("train", require_known_direct_target=False)
    validation = load_records("validation", require_known_direct_target=False)
    dev = load_records("dev_test", require_known_direct_target=False)
    validation_direct_indices, validation_direct_labels = direct_selection(validation)
    validation_route_indices, validation_route_labels = route_selection(validation)
    dev_direct_indices, dev_direct_labels = direct_selection(dev)
    dev_route_indices, dev_route_labels = route_selection(dev)
    validation_raw = [row["text"] for row in validation]
    dev_raw = [row["text"] for row in dev]
    validation_strict = [strict_normalize(text) for text in validation_raw]
    dev_strict = [strict_normalize(text) for text in dev_raw]
    ps_rows, ps_labels, ps_raw, ps_strict = read_external(
        PROMPTSHIELD,
        expected_sha256=PROMPTSHIELD_SHA256,
        text_field="prompt",
        expected_rows=23_516,
        expected_positive=6_486,
    )
    sep_rows, sep_labels, sep_raw, sep_strict = read_external(
        SEP,
        expected_sha256=SEP_SHA256,
        text_field="text",
        expected_rows=18_320,
        expected_positive=9_160,
    )

    results: defaultdict[str, list[dict]] = defaultdict(list)
    seen_cache = {}
    print(
        f"verified {len(runs)} runs; validation {len(validation)}; "
        f"dev-test {len(dev)}; route-labelled dev-test {len(dev_route_indices)}; "
        f"PromptShield {len(ps_rows)}; SEP {len(sep_rows)}"
    )
    for run in runs:
        include_pairs = bool(run["summary"].get("matched_pairs"))
        if include_pairs not in seen_cache:
            seen_cache[include_pairs] = seen_hashes(
                [*training, *validation], include_pairs
            )
        normalized_seen, strict_seen = seen_cache[include_pairs]
        ps_seen = exact_seen_mask(ps_raw, normalized_seen, strict_seen)
        sep_seen = exact_seen_mask(sep_raw, normalized_seen, strict_seen)
        member = Member(
            name=f"{run['condition']}_s{run['seed']}",
            model_id=run["model_id"],
            model_revision=run["model_revision"],
            head_path=run["head"],
            head_sha256=run["head_sha256"],
        )
        loaded = load_member(
            member,
            attention_implementation=run["attention"],
        )
        strict = run["strict"]
        validation_direct_scores, validation_route_scores = score_values(
            loaded, validation_strict if strict else validation_raw
        )
        direct_thresholds, validation_direct_selected = select_thresholds(
            validation_direct_scores[validation_direct_indices],
            validation_direct_labels,
        )
        route_thresholds, validation_route_selected = select_thresholds(
            validation_route_scores[validation_route_indices],
            validation_route_labels,
        )
        dev_direct_scores, dev_route_scores = score_values(
            loaded, dev_strict if strict else dev_raw
        )
        dev_direct_metrics = metrics(
            dev_direct_scores[dev_direct_indices],
            dev_direct_labels,
            direct_thresholds,
        )
        dev_route_metrics = metrics(
            dev_route_scores[dev_route_indices],
            dev_route_labels,
            route_thresholds,
        )
        ps_direct_scores, ps_route_scores, ps_direct_metrics, ps_route_metrics = score(
            loaded,
            ps_strict if strict else ps_raw,
            ps_labels,
            direct_thresholds,
            route_thresholds,
        )
        (
            sep_direct_scores,
            sep_route_scores,
            sep_direct_metrics,
            sep_route_metrics,
        ) = score(
            loaded,
            sep_strict if strict else sep_raw,
            sep_labels,
            direct_thresholds,
            route_thresholds,
        )
        ps_metrics = {
            "exact_seen_rows": int(ps_seen.sum()),
            "direct_head": ps_direct_metrics,
            "route": ps_route_metrics,
            "exact_unseen": {
                "direct_head": metrics(
                    ps_direct_scores[~ps_seen],
                    ps_labels[~ps_seen],
                    direct_thresholds,
                ),
                "route": metrics(
                    ps_route_scores[~ps_seen],
                    ps_labels[~ps_seen],
                    route_thresholds,
                ),
            },
        }
        sep_metrics = {
            "exact_seen_rows": int(sep_seen.sum()),
            "direct_head": sep_direct_metrics,
            "route": sep_route_metrics,
            "exact_unseen": {
                "direct_head": metrics(
                    sep_direct_scores[~sep_seen],
                    sep_labels[~sep_seen],
                    direct_thresholds,
                ),
                "route": metrics(
                    sep_route_scores[~sep_seen],
                    sep_labels[~sep_seen],
                    route_thresholds,
                ),
            },
        }

        output_dir = run["directory"] / EVALUATION_DIRECTORY
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to replace existing evaluation: {output_dir}"
            )
        output_dir.mkdir(exist_ok=True)
        arrays = {
            "validation_direct_scores": save_array(
                output_dir, "validation_direct_scores", validation_direct_scores
            ),
            "validation_route_scores": save_array(
                output_dir, "validation_route_scores", validation_route_scores
            ),
            "validation_direct_labels": save_array(
                output_dir, "validation_direct_labels", validation_direct_labels
            ),
            "validation_direct_indices": save_array(
                output_dir, "validation_direct_indices", validation_direct_indices
            ),
            "validation_route_labels": save_array(
                output_dir, "validation_route_labels", validation_route_labels
            ),
            "validation_route_indices": save_array(
                output_dir, "validation_route_indices", validation_route_indices
            ),
            "dev_test_direct_scores": save_array(
                output_dir, "dev_test_direct_scores", dev_direct_scores
            ),
            "dev_test_route_scores": save_array(
                output_dir, "dev_test_route_scores", dev_route_scores
            ),
            "dev_test_direct_labels": save_array(
                output_dir, "dev_test_direct_labels", dev_direct_labels
            ),
            "dev_test_direct_indices": save_array(
                output_dir, "dev_test_direct_indices", dev_direct_indices
            ),
            "dev_test_route_labels": save_array(
                output_dir, "dev_test_route_labels", dev_route_labels
            ),
            "dev_test_route_indices": save_array(
                output_dir, "dev_test_route_indices", dev_route_indices
            ),
            "promptshield_direct_scores": save_array(
                output_dir, "promptshield_direct_scores", ps_direct_scores
            ),
            "promptshield_route_scores": save_array(
                output_dir, "promptshield_route_scores", ps_route_scores
            ),
            "promptshield_labels": save_array(
                output_dir, "promptshield_labels", ps_labels
            ),
            "promptshield_exact_seen": save_array(
                output_dir, "promptshield_exact_seen", ps_seen
            ),
            "sep_direct_scores": save_array(
                output_dir, "sep_direct_scores", sep_direct_scores
            ),
            "sep_route_scores": save_array(
                output_dir, "sep_route_scores", sep_route_scores
            ),
            "sep_labels": save_array(output_dir, "sep_labels", sep_labels),
            "sep_exact_seen": save_array(output_dir, "sep_exact_seen", sep_seen),
        }
        entry = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "condition": run["condition"],
            "seed": run["seed"],
            "run_directory": str(run["directory"].relative_to(REPO_ROOT)),
            "model_id": run["model_id"],
            "model_revision": run["model_revision"],
            "attention_implementation": run["attention"],
            "normalization": "strict" if run["strict"] else "raw",
            "head_sha256": run["head_sha256"],
            "head_hash_recorded_during_training": run["head_hash_recorded"],
            "training_updates": run["updates"],
            "direct_max_tokens": DIRECT_MAX_TOKENS,
            "evaluation_token_budget": EVAL_TOKEN_BUDGET,
            "score_definition": SCORE_DEFINITION,
            "operating_fprs": list(OPERATING_FPRS),
            "validation": {
                "rows": len(validation),
                "direct_head": {
                    "rows": len(validation_direct_indices),
                    "selected": validation_direct_selected,
                },
                "route": {
                    "rows": len(validation_route_indices),
                    "selected": validation_route_selected,
                },
            },
            "dev_test_direct_head": dev_direct_metrics,
            "dev_test_route": dev_route_metrics,
            "promptshield": ps_metrics,
            "sep": sep_metrics,
            "arrays": arrays,
            "ordered_population_sha256": {
                "validation": identity_sha256(validation),
                "dev_test": identity_sha256(dev),
                "promptshield": identity_sha256(ps_rows, "prompt"),
                "sep": identity_sha256(sep_rows, "text"),
            },
            "input_sha256": evaluation_inputs[include_pairs],
        }
        (output_dir / "evaluation.json").write_text(json.dumps(entry, indent=2))
        results[run["condition"]].append(entry)
        print(
            f"{run['condition']} seed {run['seed']}: "
            f"route TPR@0.1% "
            f"{dev_route_metrics['validation_threshold_applied']['0.1000%']['tpr']:.4%}, "
            f"PromptShield direct/route AUC "
            f"{ps_direct_metrics['roc_auc']:.4f}/{ps_route_metrics['roc_auc']:.4f}, "
            f"direct descriptive TPR@1% "
            f"{ps_direct_metrics['descriptive_same_test_at_1pct_fpr']['tpr']:.4%}"
        )
        del loaded
        import torch

        torch.cuda.empty_cache()

    summary = {}
    for condition, entries in sorted(results.items()):
        dev_route_tpr = {
            name: np.asarray(
                [
                    entry["dev_test_route"]["validation_threshold_applied"][name]["tpr"]
                    for entry in entries
                ]
            )
            for name in ("0.1000%", "1.0000%")
        }
        dev_direct_tpr = {
            name: np.asarray(
                [
                    entry["dev_test_direct_head"]["validation_threshold_applied"][name][
                        "tpr"
                    ]
                    for entry in entries
                ]
            )
            for name in ("0.1000%", "1.0000%")
        }
        ps_direct_auc = np.asarray(
            [entry["promptshield"]["direct_head"]["roc_auc"] for entry in entries]
        )
        ps_route_auc = np.asarray(
            [entry["promptshield"]["route"]["roc_auc"] for entry in entries]
        )
        ps_tpr_1 = np.asarray(
            [
                entry["promptshield"]["direct_head"][
                    "descriptive_same_test_at_1pct_fpr"
                ]["tpr"]
                for entry in entries
            ]
        )
        summary[condition] = {
            "n": len(entries),
            "dev_route_validation_threshold_tpr": {
                name: {
                    "mean": float(values.mean()),
                    "range": [float(values.min()), float(values.max())],
                }
                for name, values in dev_route_tpr.items()
            },
            "dev_direct_validation_threshold_tpr": {
                name: {
                    "mean": float(values.mean()),
                    "range": [float(values.min()), float(values.max())],
                }
                for name, values in dev_direct_tpr.items()
            },
            "promptshield_direct_roc_auc_mean": float(ps_direct_auc.mean()),
            "promptshield_route_roc_auc_mean": float(ps_route_auc.mean()),
            "promptshield_direct_descriptive_tpr_at_1pct_fpr_mean": float(
                ps_tpr_1.mean()
            ),
            "seeds": entries,
        }
    if not args.run:
        OUT.write_text(json.dumps(summary, indent=2))
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    else:
        print("wrote per-run evaluations only; global summary was not replaced")
    print(
        "Only validation_threshold_applied uses a threshold independent of each "
        "test population. Same-test operating points are descriptive."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
