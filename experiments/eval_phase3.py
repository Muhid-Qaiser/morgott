"""Score every Phase 3 condition and report spreads across seeds.

Discovers each `artifacts/phase3_archived/archived_*` run, scores its head on
the frozen dev-test direct suite and on the external PromptShield split, and
groups by condition so raw / strict / strict+pairs can be compared with a seed
spread rather than a single point.

Text is preprocessed to match how each head was trained: a strict-normalised
head is scored on strict-normalised text. Scoring it on raw text would measure
a train/test mismatch rather than the model.

Run: PYTHONPATH=src:experiments/_archived python3 experiments/eval_phase3.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import fpr_and_recall, load_records  # noqa: E402
from encoder_infer import (  # noqa: E402
    Member,
    direct_route_probability,
    load_audit_members,
    load_member,
    score_texts,
)
from strict_normalize import strict_normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "artifacts" / "phase3_archived"
EXTERNAL = REPO_ROOT / "artifacts/external_eval_data/promptshield/test.jsonl"
SEP = REPO_ROOT / "artifacts/external_eval_data/sep/sep.jsonl"
OUT = RUNS_DIR / "phase3_summary.json"
TARGET_FPR = 0.001


def discover() -> list[tuple[str, int, bool, str, Path]]:
    """Return (condition, seed, is_strict, model_id, head_path) per run.

    Two traps this guards against:
    * A crashed run leaves no head.safetensors, so it is skipped rather than
      counted -- always compare the discovered count against what was queued.
    * mmBERT and ModernBERT both have hidden_size 768, so an mmBERT head loads
      cleanly into a ModernBERT encoder and produces meaningless scores with no
      error. The encoder must be chosen from the run directory, not assumed.
    """
    modernbert = "answerdotai/ModernBERT-base"
    mmbert = "jhu-clsp/mmBERT-base"
    found = []
    for directory in sorted(RUNS_DIR.glob("archived_*")):
        head = next(
            directory.glob(
                "direct_failure_repair/*/wildguard_weak_transfer/*/head.safetensors"
            ),
            None,
        )
        if head is None:
            continue
        name = directory.name.replace("archived_", "")
        match = re.match(
            r"(raw|strict)"           # normalisation
            r"(_pairs(?:\d+)?)?"      # optional pairs, optional percentage
            r"(_rank[\d.]+)?"         # optional ranking weight
            r"(_mmbert)?"             # optional backbone
            r"(?:_s(\d+))?$",         # optional seed
            name,
        )
        if not match:
            print(f"  WARNING: unparsed run directory {directory.name}")
            continue
        condition = match.group(1)
        if match.group(2):
            condition += "+pairs" + (match.group(2).replace("_pairs", "") or "")
        if match.group(3):
            condition += match.group(3).replace("_", "+")
        if match.group(4):
            condition += "+mmbert"
        seed = int(match.group(5) or 42)
        model_id = mmbert if match.group(4) else modernbert
        found.append((condition, seed, match.group(1) == "strict", model_id, head))
    return found


def main() -> int:
    from sklearn.metrics import roc_auc_score

    dev = load_records("dev_test")
    dev_labels = np.array(
        [r["targets"]["direct_instruction_subversion"] for r in dev], dtype=np.int64
    )
    dev_raw = [r["text"] for r in dev]
    dev_strict = [strict_normalize(t) for t in dev_raw]

    with EXTERNAL.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ext_labels = np.array([r["label"] for r in rows], dtype=np.int64)
    ext_raw = [r["prompt"][:24000] for r in rows]
    ext_strict = [strict_normalize(t) for t in ext_raw]

    # SEP: matched pairs by construction (same content, probe is the only
    # difference), so it isolates whether the model reads intent or shortcut.
    # Its positives are BENIGN-INTENT imperatives -- it measures
    # instruction-in-data separation, not harmfulness. Low TPR here is a
    # property of the benchmark, not necessarily a failure.
    sep_labels = sep_raw = sep_strict = None
    if SEP.exists():
        with SEP.open(encoding="utf-8") as handle:
            sep_rows = [json.loads(line) for line in handle if line.strip()]
        sep_labels = np.array([r["label"] for r in sep_rows], dtype=np.int64)
        sep_raw = [r["text"][:24000] for r in sep_rows]
        sep_strict = [strict_normalize(t) for t in sep_raw]
        print(f"SEP transfer set: {len(sep_rows)} rows "
              f"({int(sep_labels.sum())} injection)")

    template = next(m for m in load_audit_members() if m.name == "english_modernbert")
    results = defaultdict(list)

    runs = discover()
    print(f"discovered {len(runs)} completed runs\n")
    for condition, seed, is_strict, model_id, head in runs:
        member = Member(
            name=f"{condition}_s{seed}",
            model_id=model_id,
            model_revision="" if model_id != template.model_id
                           else template.model_revision,
            head_path=head,
            head_sha256="",
        )
        loaded = load_member(member)
        dev_scores = direct_route_probability(
            score_texts(loaded, dev_strict if is_strict else dev_raw,
                        channel="direct_user")
        )
        fpr, recall, _ = fpr_and_recall(dev_scores, dev_labels)
        ext_scores = direct_route_probability(
            score_texts(loaded, ext_strict if is_strict else ext_raw,
                        channel="direct_user")
        )
        ext_auc = float(roc_auc_score(ext_labels, ext_scores))
        _, ext_tpr, _ = fpr_and_recall(ext_scores, ext_labels, 0.01)
        entry = {"seed": seed, "dev_fpr": fpr, "dev_recall": recall,
                 "external_auc": ext_auc, "external_tpr_at_1pct": ext_tpr}
        if sep_labels is not None:
            sep_scores = direct_route_probability(
                score_texts(loaded, sep_strict if is_strict else sep_raw,
                            channel="direct_user")
            )
            entry["sep_auc"] = float(roc_auc_score(sep_labels, sep_scores))
            _, sep_tpr, _ = fpr_and_recall(sep_scores, sep_labels, 0.01)
            entry["sep_tpr_at_1pct"] = sep_tpr
        results[condition].append(entry)
        sep_note = (f"  SEP AUC {entry['sep_auc']:.4f}"
                    if "sep_auc" in entry else "")
        print(f"  {condition:14} seed {seed}  dev recall {recall*100:6.2f}%  "
              f"ext AUC {ext_auc:.4f}  ext TPR@1% {ext_tpr*100:5.2f}%{sep_note}")
        del loaded
        import torch

        torch.cuda.empty_cache()

    print(f"\n{'condition':16}{'n':>3}{'dev recall':>22}{'external AUC':>20}"
          f"{'ext TPR@1%':>13}")
    summary = {}
    for condition, entries in sorted(results.items()):
        recalls = np.array([e["dev_recall"] for e in entries])
        aucs = np.array([e["external_auc"] for e in entries])
        tprs = np.array([e["external_tpr_at_1pct"] for e in entries])
        summary[condition] = {
            "n": len(entries),
            "dev_recall_mean": float(recalls.mean()),
            "dev_recall_min": float(recalls.min()),
            "dev_recall_max": float(recalls.max()),
            "external_auc_mean": float(aucs.mean()),
            "external_tpr_at_1pct_mean": float(tprs.mean()),
            "sep_auc_mean": (float(np.mean([e["sep_auc"] for e in entries]))
                             if "sep_auc" in entries[0] else None),
            "sep_tpr_at_1pct_mean": (
                float(np.mean([e["sep_tpr_at_1pct"] for e in entries]))
                if "sep_tpr_at_1pct" in entries[0] else None),
            "seeds": entries,
        }
        spread = (f"{recalls.mean()*100:6.2f}% "
                  f"[{recalls.min()*100:.2f}-{recalls.max()*100:.2f}]")
        auc_spread = (f"{aucs.mean():.4f} "
                      f"[{aucs.min():.4f}-{aucs.max():.4f}]")
        print(f"{condition:16}{len(entries):>3}{spread:>22}{auc_spread:>20}"
              f"{tprs.mean()*100:12.2f}%")

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    print("\nNote: dev recall is measured at a threshold calibrated on dev's own "
          "negatives,\nso it is generous. External TPR@1% FPR is the deployable "
          "number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
