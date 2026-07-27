"""Regenerate the import-shimmed copy of the archived routing-encoder runner.

`experiments/_archived/routing_encoder.py` is 5,500+ lines of DERIVED code and
is deliberately NOT committed (see .gitignore). This script rebuilds it, so the
repository carries ~90 lines instead of ~5,500 while every Phase 3 result stays
reproducible.

Source: artifacts/research-source-archive/2026-07-26/src/morgott/routing_encoder.py

  WARNING: that path is under `artifacts/`, which is gitignored, and the file is
  NOT in git history either -- `git log --diff-filter=D` finds no deletion of
  `src/morgott/routing_encoder.py`. It exists only on this machine. If the
  archive directory is lost, the Phase 3 baseline cannot be reproduced and the
  strict-normalisation / matched-pair results become unverifiable.
  Preserving that directory is a prerequisite for continuing this work.

Three edits are applied, all mechanical and all reversible:

1. Two relative imports (`from .corpus`, `from .data`) become absolute so the
   module resolves against the live `morgott` package.
2. Three `seed=42` / `seed_42` literals inside `run_direct_failure_repair` are
   routed through `ACTIVE_SEED`, which defaults to 42. Unset, behaviour is
   identical to the archive; this exists because every retained result was
   single-seed and no seed band could be measured otherwise.
3. The two-recipe loop and its `ranking_weight=0.0` become `ACTIVE_RECIPES` and
   `ACTIVE_RANKING_WEIGHT`, both defaulting to the archive values. Restricting
   to `wildguard_weak_transfer` roughly halves wall-clock; the ranking weight is
   needed to exercise `aligned_pair_ranking_loss`.

Run:  python3 experiments/_archived/build_shim.py
Check only, no write:  python3 experiments/_archived/build_shim.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO_ROOT
    / "artifacts/research-source-archive/2026-07-26/src/morgott/routing_encoder.py"
)
TARGET = REPO_ROOT / "experiments/_archived/routing_encoder.py"

# sha256 of the pristine archive file this shim was built and validated against.
SOURCE_SHA256 = "eea89a5e211057869d13dbf302fd63435d6c2a1cacbb8f71fcb693f1e03243cc"

HEADER = '''"""Archived routing-encoder runner, import-shimmed for audit and experiment use.

GENERATED FILE -- do not edit. Regenerate with:
    python3 experiments/_archived/build_shim.py

Derived from the 2026-07-26 research source archive. Every change is listed in
build_shim.py; with ACTIVE_SEED / ACTIVE_RECIPES / ACTIVE_RANKING_WEIGHT left at
their defaults, behaviour is identical to the archive.

Per AGENTS.md, old experiment runners must not become the default trainer. This
module is not wired into the CLI and is imported only by scripts under
experiments/.
"""

'''

OVERRIDES = '''
# --- shim overrides (see build_shim.py) -------------------------------------
# Seed used by run_direct_failure_repair. Defaults to the archive's 42.
ACTIVE_SEED = 42
# Which direct-failure-repair recipes to train. Defaults to both, as the archive
# does. Only wildguard_weak_transfer feeds the retained ensemble.
ACTIVE_RECIPES = ("wildguard_weak_transfer", "open_failure_upper_bound")
# Weight on aligned_pair_ranking_loss inside run_direct_failure_repair. Defaults
# to the archive's 0.0. Only meaningful when rows carry pair_id/pair_head.
ACTIVE_RANKING_WEIGHT = 0.0
# ----------------------------------------------------------------------------
'''


def build(text: str) -> str:
    original = text

    text = text.replace("from .corpus import", "from morgott.corpus import")
    text = text.replace("from .data import", "from morgott.data import")
    if text == original:
        raise SystemExit("relative imports not found -- archive layout changed")

    marker = 'MODEL_ID = "answerdotai/ModernBERT-base"'
    if marker not in text:
        raise SystemExit("MODEL_ID anchor not found -- archive layout changed")
    text = text.replace(marker, marker + "\n" + OVERRIDES, 1)

    recipe_loop = """    for recipe, upper_bound in (
        ("wildguard_weak_transfer", False),
        ("open_failure_upper_bound", True),
    ):"""
    if recipe_loop not in text:
        raise SystemExit("recipe loop not found -- archive layout changed")
    text = text.replace(
        recipe_loop,
        """    for recipe, upper_bound in [
        pair
        for pair in (
            ("wildguard_weak_transfer", False),
            ("open_failure_upper_bound", True),
        )
        if pair[0] in ACTIVE_RECIPES
    ]:""",
    )

    # The three seed literals and the ranking weight all live inside
    # run_direct_failure_repair, which begins at this def. Scoping the
    # replacements to that slice avoids touching the other entrypoints.
    start = text.index("def run_direct_failure_repair(")
    head, tail = text[:start], text[start:]
    for old, new, count in (
        ('/ "seed_42"', '/ f"seed_{ACTIVE_SEED}"', 1),
        ("head = _new_head(hidden_size, seed=42)",
         "head = _new_head(hidden_size, seed=ACTIVE_SEED)", 1),
        ("ranking_weight=0.0,", "ranking_weight=ACTIVE_RANKING_WEIGHT,", 1),
    ):
        if tail.count(old) < count:
            raise SystemExit(f"expected {count}x {old!r} in run_direct_failure_repair")
        tail = tail.replace(old, new, count)
    # The training-call seed is the remaining bare `seed=42,` in this function.
    tail = tail.replace("            seed=42,", "            seed=ACTIVE_SEED,", 1)

    return HEADER + head + tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify only; do not write")
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"MISSING ARCHIVE SOURCE: {SOURCE}", file=sys.stderr)
        print("The archive is gitignored and absent from git history. Without it "
              "the Phase 3 baseline cannot be reproduced.", file=sys.stderr)
        return 2

    raw = SOURCE.read_text()
    digest = hashlib.sha256(raw.encode()).hexdigest()
    if digest != SOURCE_SHA256:
        print(f"ARCHIVE HASH MISMATCH\n  expected {SOURCE_SHA256}\n  found    {digest}",
              file=sys.stderr)
        print("The shim was validated against the expected file. Refusing to "
              "build from a different one.", file=sys.stderr)
        return 3

    built = build(raw)
    if args.check:
        if TARGET.exists() and TARGET.read_text() == built:
            print("shim is up to date")
            return 0
        print("shim is MISSING or STALE -- run without --check", file=sys.stderr)
        return 1

    TARGET.write_text(built)
    print(f"wrote {TARGET.relative_to(REPO_ROOT)} ({len(built.splitlines())} lines)")
    print(f"from {SOURCE.relative_to(REPO_ROOT)} @ {digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
