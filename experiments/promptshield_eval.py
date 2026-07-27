"""Score the retained ensemble on the external PromptShield evaluation split.

PromptShield (arXiv 2501.15145; Jacob, Alzahrani, Hu, Alomair, Wagner;
UC Berkeley + KACST; CODASPY 2025) publishes its ~24k-row held-out split at
`hendzh/PromptShield`. This is the first benign distribution morgott's
checkpoints have ever been scored against that morgott did not choose.

The paper's deployment argument is that a usable detector needs an extremely
low FPR because benign traffic vastly outnumbers attacks, so this script
reports TPR at fixed low FPR operating points rather than accuracy.

Contamination: 323 of the 23,516 rows (1.374%) also appear in morgott's
training or validation views, all label 0 (benign). Every metric is reported
twice, with and without those rows.

Run: PYTHONPATH=src:experiments/_archived python3 experiments/promptshield_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import fpr_and_recall  # noqa: E402
from encoder_infer import (  # noqa: E402
    direct_route_probability,
    load_audit_members,
    load_member,
    score_texts,
)

from morgott.data import text_hash  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EXTERNAL = REPO_ROOT / "artifacts" / "external_eval_data" / "promptshield" / "test.jsonl"
OUT_DIR = REPO_ROOT / "artifacts" / "phase0_external_eval"

# Reported in PromptShield Table 4 on this same split.
PAPER_BASELINES = {
    "Meta PromptGuard v1": {"auc": 0.874, "1%": 0.1278, "0.5%": 0.1243, "0.1%": 0.0939},
    "ProtectAI v1": {"auc": 0.646, "1%": 0.0705, "0.5%": 0.0336, "0.1%": 0.0000},
    "ProtectAI v2": {"auc": 0.705, "1%": 0.0197, "0.5%": 0.0134, "0.1%": 0.0000},
    "InjecGuard": {"auc": 0.765, "1%": 0.2037, "0.5%": 0.1630, "0.1%": 0.0661},
    "Fmops (DistilBERT)": {"auc": 0.754, "1%": 0.1300, "0.5%": 0.0839, "0.1%": 0.0210},
    "PromptShield (DeBERTa)": {"auc": 0.976, "1%": 0.4322, "0.5%": 0.4050, "0.1%": 0.3145},
    "PromptShield (Llama-3.1-8B)": {"auc": 0.998, "1%": 0.9480, "0.5%": 0.8780, "0.1%": 0.6533},
}


def training_hashes() -> set[str]:
    hashes: set[str] = set()
    views = [
        "routing/train",
        "routing/validation",
        "injection/direct_train",
        "injection/direct_validation",
    ]
    for view in views:
        path = DATA_DIR / "views" / f"{view}.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                digest = row.get("normalized_text_sha256")
                if not digest:
                    digest = text_hash(row["text"])
                hashes.add(digest)
    return hashes


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((1 - labels).sum()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "tpr_at_fpr": {},
    }
    for target in (0.01, 0.005, 0.001, 0.0005):
        achieved, tpr, _ = fpr_and_recall(scores, labels, target)
        out["tpr_at_fpr"][f"{target:.4%}"] = {"tpr": tpr, "achieved_fpr": achieved}
    return out


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # NOTE: str.splitlines() also breaks on U+2028, U+2085 and friends, which
    # occur inside these prompts. Iterate the handle so only "\n" separates rows.
    with EXTERNAL.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    texts = [row["prompt"] for row in rows]
    labels = np.array([row["label"] for row in rows], dtype=np.int64)
    print(f"PromptShield test: {len(rows)} rows "
          f"({int(labels.sum())} injection / {int((1-labels).sum())} benign)")

    print("computing contamination mask ...", flush=True)
    train = training_hashes()
    contaminated = np.array([text_hash(text) in train for text in texts])
    print(f"  rows also in morgott train/validation views: {int(contaminated.sum())} "
          f"({contaminated.mean()*100:.3f}%)")
    print(f"  of which positive: {int(labels[contaminated].sum())}")

    per_member = {}
    for member in load_audit_members():
        print(f"scoring with {member.name} ...", flush=True)
        loaded = load_member(member)
        logits = score_texts(loaded, texts, channel="direct_user")
        per_member[member.name] = direct_route_probability(logits)
        del loaded
        import torch

        torch.cuda.empty_cache()
    fused = np.mean(list(per_member.values()), axis=0)

    results = {
        "dataset": {
            "id": "hendzh/PromptShield",
            "split": "test",
            "rows": len(rows),
            "path": str(EXTERNAL.relative_to(REPO_ROOT)),
        },
        "contamination": {
            "rows_in_morgott_training_views": int(contaminated.sum()),
            "fraction": float(contaminated.mean()),
            "positive_among_contaminated": int(labels[contaminated].sum()),
        },
        "ensemble_all_rows": evaluate(labels, fused),
        "ensemble_decontaminated": evaluate(labels[~contaminated], fused[~contaminated]),
        "per_member_all_rows": {
            name: evaluate(labels, scores) for name, scores in per_member.items()
        },
        "paper_baselines_same_split": PAPER_BASELINES,
    }

    print("\n=== morgott ensemble on PromptShield test ===")
    for key in ("ensemble_all_rows", "ensemble_decontaminated"):
        block = results[key]
        print(f"\n{key}  (rows={block['rows']}, pos={block['positive']})")
        print(f"  ROC AUC {block['roc_auc']:.4f}   PR AUC {block['pr_auc']:.4f}")
        for target, values in block["tpr_at_fpr"].items():
            print(f"  TPR @ {target:>9} FPR : {values['tpr']*100:6.2f}%  "
                  f"(achieved {values['achieved_fpr']*100:.4f}%)")

    print("\n=== comparison, same split, from PromptShield Table 4 ===")
    print(f"{'detector':30}{'AUC':>8}{'@1%':>9}{'@0.5%':>9}{'@0.1%':>9}")
    block = results["ensemble_decontaminated"]
    print(f"{'morgott ensemble':30}{block['roc_auc']:8.3f}"
          f"{block['tpr_at_fpr']['1.0000%']['tpr']*100:8.2f}%"
          f"{block['tpr_at_fpr']['0.5000%']['tpr']*100:8.2f}%"
          f"{block['tpr_at_fpr']['0.1000%']['tpr']*100:8.2f}%")
    for name, values in PAPER_BASELINES.items():
        print(f"{name:30}{values['auc']:8.3f}{values['1%']*100:8.2f}%"
              f"{values['0.5%']*100:8.2f}%{values['0.1%']*100:8.2f}%")

    (OUT_DIR / "promptshield_eval.json").write_text(json.dumps(results, indent=2))
    np.save(OUT_DIR / "promptshield_fused_scores.npy", fused)
    print(f"\nwrote {OUT_DIR/'promptshield_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
