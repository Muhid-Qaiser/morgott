"""Run the archived direct-failure-repair recipe, changing only preprocessing.

Why this exists instead of `train_head.py`
------------------------------------------
`train_head.py` reimplemented the recipe and could not reproduce it: 38.02%
dev-test recall against the original single head's 63.04%. The diagnosis is not
hyperparameters. The archived runner batches against a 4,096-token budget
(`_example_batches`, routing_encoder.py:1097), which for ~250-token direct rows
is about 17 rows per batch and roughly 10,900 optimizer updates over six epochs.
The reimplementation used fixed 256-row batches: about 720 updates. A 15x
optimization deficit is not something a cosine schedule or a longer epoch count
can repair, and the reimplementation also differed in boundary rows, partially
labelled rows, validation composition, loss normalisation and training
precision.

So this script does not reimplement anything. It calls the archived
`run_direct_failure_repair` unchanged, and interposes exactly one thing: the
text normalisation applied at encode time. That makes the comparison a genuine
single-variable experiment against a baseline that reproduces by construction.

Usage:
  # 1. Reproduce the baseline. Should match the retained numbers.
  PYTHONPATH=src:experiments/_archived python3 experiments/run_archived_recipe.py \
      --normalize raw

  # 2. The actual experiment.
  PYTHONPATH=src:experiments/_archived python3 experiments/run_archived_recipe.py \
      --normalize strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import routing_encoder as archived  # noqa: E402
from strict_normalize import strict_normalize  # noqa: E402

from morgott.data import text_hash  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MATCHED_PAIRS_PATH = REPO_ROOT / "artifacts/matched_pairs/pairs_20260726T105034Z.jsonl"
MATCHED_PAIRS_SHA256 = (
    "8ec5c1c77b378688b190722f7d1fc51e9bef819ee9670948d2658f4a37082158"
)
MATCHED_PAIRS_COUNT = 11_046
PINNED_MODEL_REVISIONS = {
    "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    "jhu-clsp/mmBERT-base": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_normalisation_hook(mode: str) -> None:
    """Wrap `_encode_records` so record text is normalised before tokenisation.

    This is the ONLY behavioural change. Everything downstream -- batching,
    loss, schedule, checkpoint selection, evaluation -- is the archived code
    path untouched.

    A known-payload span is a character offset into the raw text, so it cannot
    survive a transform that deletes and substitutes characters. Rows carrying
    one are left unnormalised rather than silently corrupted; the direct recipe
    does not use them, and a run that hits many of these should be treated as
    out of scope for this hook rather than quietly half-normalised.
    """
    if mode == "raw":
        return

    original = archived._encode_records
    stats = {"normalised": 0, "skipped_known_span": 0}

    def hooked(tokenizer, records, **kwargs):
        rewritten = []
        for record in records:
            if record.get("payload_char_span") is not None:
                stats["skipped_known_span"] += 1
                rewritten.append(record)
                continue
            stats["normalised"] += 1
            rewritten.append({**record, "text": strict_normalize(record["text"])})
        return original(tokenizer, rewritten, **kwargs)

    archived._encode_records = hooked
    archived._normalisation_stats = stats
    print(f"normalisation hook installed (mode={mode})")


def install_matched_pairs_hook(
    *,
    fraction: float = 1.0,
    ranking: bool = False,
) -> dict:
    """Append generated matched pairs to the training records only.

    Interposes on `load_capped_fold_records` so the pairs reach training and
    nothing else -- they must never enter validation or dev-test, because they
    are `label_basis: model_generated` weak supervision and would otherwise
    contaminate the very metrics they are meant to improve.
    """
    if file_sha256(MATCHED_PAIRS_PATH) != MATCHED_PAIRS_SHA256:
        raise ValueError("matched-pair artifact hash mismatch")
    rows = []
    artifact_rows = 0
    with MATCHED_PAIRS_PATH.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            pair = json.loads(line)
            artifact_rows += 1
            if pair.get("channel") != "direct_user":
                continue
            group = f"genpair:{MATCHED_PAIRS_PATH.stem}:{index}"
            for half, label in (("benign", 0), ("attack", 1)):
                text = pair[half]
                rows.append(
                    {
                        "id": f"{group}:{half}",
                        "text": text,
                        "hash": text_hash(text),
                        "channel": "direct_user",
                        # data_role must be present and must not be
                        # research_failure_train, which the weak-transfer recipe
                        # filters out at routing_encoder.py:4999.
                        "data_role": "train",
                        "source": "matched_pairs_generated",
                        "sources": ["matched_pairs_generated"],
                        "group": group,
                        # The runner's own value for generated rows. Keeps these
                        # visibly separate in by_label_strength slices rather
                        # than passing them off as source-supported.
                        "label_strength": "synthetic_recipe",
                        "security_label": "benign"
                        if label == 0
                        else "direct_prompt_injection",
                        "finance": pair.get("category") == "finance_agent",
                        "known_payload_span": False,
                        "payload_char_span": None,
                        "targets": {
                            "direct_instruction_subversion": label,
                            "indirect_instruction_subversion": None,
                            "jailbreak": None,
                            "harmful_intent": None,
                        },
                        # Pair fields left None deliberately. Populating them
                        # would activate aligned_pair_ranking_loss and confound
                        # this experiment, which tests the DATA alone. The
                        # ranking loss is the next experiment, run separately.
                        # Populated only for the ranking experiment. When set,
                        # aligned_pair_ranking_loss ties each benign to its own
                        # attack, which is the relation Phase 0 showed broken.
                        "pair_id": group if ranking else None,
                        # An INDEX into HEADS, not the name: the runner does
                        # HEADS[example["pair_head"]] at
                        # routing_encoder.py:2778.
                        "pair_head": (0 if ranking else None),
                        "pair_label": label if ranking else None,
                        "pair_family": "matched_pairs" if ranking else None,
                    }
                )
    if artifact_rows != MATCHED_PAIRS_COUNT:
        raise ValueError(
            f"matched-pair row count mismatch: {artifact_rows} != {MATCHED_PAIRS_COUNT}"
        )
    if fraction < 1.0:
        # Subsample by PAIR, not by row, so a benign half never survives
        # without its matched attack -- that would reintroduce the imbalance
        # the pairs exist to remove.
        import random as _random

        groups = sorted({r["group"] for r in rows})
        keep = set(
            _random.Random(42).sample(groups, max(1, int(len(groups) * fraction)))
        )
        rows = [r for r in rows if r["group"] in keep]

    original = archived.load_capped_fold_records

    def hooked(data_dir, manifest, split, **kwargs):
        records, crossing = original(data_dir, manifest, split, **kwargs)
        if split == "train":
            records = [*records, *rows]
        return records, crossing

    archived.load_capped_fold_records = hooked
    print(
        f"matched-pairs hook installed: +{len(rows)} training rows "
        f"({len(rows) // 2} pairs), train split only"
    )
    return {
        "artifact_path": str(MATCHED_PAIRS_PATH.relative_to(REPO_ROOT)),
        "artifact_sha256": MATCHED_PAIRS_SHA256,
        "artifact_pairs": artifact_rows,
        "direct_user_pairs_selected": len(rows) // 2,
        "training_rows_added": len(rows),
        "fraction": fraction,
        "ranking": ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", choices=["raw", "strict"], default="raw")
    parser.add_argument(
        "--attention",
        default="sdpa",
        help="sdpa is portable; the original English run pinned flash-attn2, "
        "which shifted dev ROC AUC by ~0.0004 in reproduce_check.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs-fraction",
        type=float,
        default=1.0,
        help="fraction of generated pairs to use (mixing-ratio "
        "sweep: separates shortcut-removal from label noise)",
    )
    parser.add_argument(
        "--pair-ranking",
        type=float,
        default=0.0,
        help="weight on aligned_pair_ranking_loss; populates "
        "pair fields on generated rows so the loss applies",
    )
    parser.add_argument(
        "--only-weak-transfer",
        action="store_true",
        help="train only wildguard_weak_transfer, the recipe "
        "that feeds the retained ensemble (~2x faster)",
    )
    parser.add_argument(
        "--matched-pairs",
        action="store_true",
        help="fold generated matched pairs into training as "
        "model_generated weak supervision",
    )
    parser.add_argument("--model-id", default=archived.MODEL_ID)
    parser.add_argument("--model-revision")
    args = parser.parse_args()

    if not 0 < args.pairs_fraction <= 1:
        parser.error("--pairs-fraction must be in (0, 1]")
    if args.pair_ranking < 0:
        parser.error("--pair-ranking must be nonnegative")
    model_revision = args.model_revision or PINNED_MODEL_REVISIONS.get(args.model_id)
    if not model_revision:
        parser.error("--model-revision is required for an unrecognized model")

    archived.ACTIVE_SEED = args.seed
    if args.only_weak_transfer:
        archived.ACTIVE_RECIPES = ("wildguard_weak_transfer",)
    archived.ACTIVE_RANKING_WEIGHT = args.pair_ranking
    pair_stats = None
    if args.matched_pairs:
        pair_stats = install_matched_pairs_hook(
            fraction=args.pairs_fraction,
            ranking=args.pair_ranking > 0,
        )
    suffix = ""
    if args.matched_pairs:
        suffix = "_pairs"
        if args.pairs_fraction < 1.0:
            suffix += f"{int(args.pairs_fraction * 100)}"
        if args.pair_ranking > 0:
            suffix += f"_rank{args.pair_ranking}"
    model_tag = "" if "ModernBERT" in args.model_id else "_mmbert"
    tag = f"archived_{args.normalize}{suffix}{model_tag}_s{args.seed}"
    artifacts_dir = REPO_ROOT / "artifacts" / "phase3_archived" / tag
    reports_dir = artifacts_dir / "reports"
    if artifacts_dir.exists() and any(artifacts_dir.iterdir()):
        raise FileExistsError(
            f"refusing to reuse nonempty run directory: {artifacts_dir}"
        )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    install_normalisation_hook(args.normalize)

    print(f"model      : {args.model_id} @ {model_revision[:12]}")
    print(f"attention  : {args.attention}")
    print(f"normalize  : {args.normalize}")
    print(f"artifacts  : {artifacts_dir.relative_to(REPO_ROOT)}")
    print("running archived run_direct_failure_repair ...", flush=True)

    result = archived.run_direct_failure_repair(
        DATA_DIR,
        artifacts_dir,
        reports_dir,
        attention_implementation=args.attention,
        model_id=args.model_id,
        model_revision=model_revision,
    )

    report_path = next(reports_dir.glob("*.json"))
    report = json.loads(report_path.read_text())
    candidate = report["candidates"]["wildguard_weak_transfer"]
    head_path = Path(candidate["artifact"]["head"])
    summary = {
        "schema_version": 2,
        "condition": tag.removeprefix("archived_"),
        "normalize": args.normalize,
        "model_id": args.model_id,
        "model_revision": model_revision,
        "attention": args.attention,
        "normalisation_stats": getattr(archived, "_normalisation_stats", None),
        "seed": args.seed,
        "matched_pairs": args.matched_pairs,
        "matched_pair_details": pair_stats,
        "pair_ranking_weight": args.pair_ranking,
        "training_updates": candidate["training"]["updates"],
        "selected_epoch": candidate["training"]["selected_epoch"],
        "head_path": str(head_path.relative_to(REPO_ROOT)),
        "head_sha256": file_sha256(head_path),
        "data_manifest_sha256": file_sha256(DATA_DIR / "manifest.json"),
        "wrapper_sha256": file_sha256(Path(__file__)),
        "normalizer_sha256": file_sha256(REPO_ROOT / "experiments/strict_normalize.py"),
        "shim_builder_sha256": file_sha256(
            REPO_ROOT / "experiments/_archived/build_shim.py"
        ),
        "generated_shim_sha256": file_sha256(
            REPO_ROOT / "experiments/_archived/routing_encoder.py"
        ),
        "archived_source_sha256": (
            "eea89a5e211057869d13dbf302fd63435d6c2a1cacbb8f71fcb693f1e03243cc"
        ),
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # Surface the headline numbers without needing to dig through the report.
    for recipe, block in (result.get("recipes") or {}).items():
        print(f"\n=== {recipe} ===")
        print(json.dumps(block, indent=1)[:1200])
    print(f"\nwrote {(artifacts_dir / 'run_summary.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
