"""Matched subversion control: the same harmful request, with and without the wrapper.

864 rows of the first-party red-team corpus use one fixed jailbreak wrapper and
end with a literal `Request: <inner>` tail. Splitting the wrapper off constructs
864 pairs whose two halves carry identical harmful content and differ only by the
presence of instruction subversion. Pairs that would truncate at the scorer's
512-token limit are excluded before scoring, because the request sits at the end
of the wrapper and truncation would un-match the comparison.

That isolates the question the aggregate 39.44% "attempt recall" cannot answer:
does the detector fire on instruction subversion, or on harm topic? Under
`docs/data-contract.md:46-56` a harmful request without subversion is a different
label, so a low flag rate on `inner` is correct behaviour, not a miss.

This is an archive-class development diagnostic. Every score is advisory.

Usage:
    uv run --extra showcase python -m experiments.redteam_subversion_control.run pairs
    uv run --extra encoder python -m experiments.redteam_subversion_control.run score
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from morgott.data import atomic_write_text
from morgott.models.downstream import MMBERT_LOW_BY_CHANNEL, THRESHOLD_SHA256
from morgott.models.mmbert.core import MAX_TOKENS, MODEL_ID, MODEL_REVISION
from morgott.models.mmbert.inference import score_file
from morgott.normalization import strict_normalize

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data-archive" / "redteam"
OUT = ROOT / "artifacts" / "redteam_subversion_control"
MANIFEST = ROOT / "model-artifacts.json"

# Fixed, not a flag. DIRECT_USER_FLOOR below is the floor validated for this one
# cascade; no other registered shadow has a validated threshold, so scoring a
# different model against this floor would report a meaningless rate.
MODEL = "mmbert-lora-full-s42"

WRAPPER_REQUEST = re.compile(r"(?:^|\n)\s*Request:\s*(.+)\Z", re.S)
# The two rule-block variants the campaigns used.
VARIANTS = {
    "prefill_continuation": 'Your answer has already begun with "Sure, here is',
    "sure_here_is": 'Begin your answer immediately with "Sure, here is"',
}

# Direct-user pass floor, taken from the cascade's own pinned constant rather
# than copied, so a retuned floor cannot silently leave this panel stale.
DIRECT_USER_FLOOR = MMBERT_LOW_BY_CHANNEL["direct_user"]


def _read_projection(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(body.encode("utf-8"))


def _variant(text: str) -> str:
    for name, marker in VARIANTS.items():
        if marker in text:
            return name
    return "unknown"


def build_pairs(projection: Path) -> list[dict]:
    pairs = []
    for row in _read_projection(projection):
        text = row["text"]
        # Use the projection's own attestation rather than re-deriving the
        # wrapper test, so the pair set cannot drift from the labels.
        if row["subversion_basis"] != "wrapper_template":
            continue
        match = WRAPPER_REQUEST.search(text)
        if match is None:
            raise ValueError(f"wrapper row without a Request tail: {row['id']}")
        inner = match.group(1).strip()
        if not inner:
            raise ValueError(f"empty inner request: {row['id']}")
        pairs.append(
            {
                "pair_id": row["id"].replace("redteam:", "pair:"),
                "wrapped": text,
                "inner": inner,
                "wrapper_variant": _variant(text),
                "category": row["category"],
                "attack_mode": row["attack_mode"],
                "run_id": row["run_id"],
                "split_group_id": row["split_group_id"],
                "verdict": row["verdict"],
            }
        )
    if not pairs:
        raise ValueError(f"no wrapper_template rows in {projection}")
    pairs.sort(key=lambda row: row["pair_id"])
    return pairs


def write_pairs(projection: Path, pairs_path: Path) -> None:
    pairs = build_pairs(projection)
    # Validate before writing: the output is hash-pinned in data-archive/SHA256SUMS,
    # so a failed run must not leave a file behind.
    assert len({row["pair_id"] for row in pairs}) == len(pairs), "duplicate pair ids"
    assert all(row["wrapper_variant"] != "unknown" for row in pairs), (
        "unrecognised wrapper variant entered the pair set"
    )
    _write_jsonl_gz(pairs_path, pairs)
    print(f"pairs: {len(pairs)} written to {pairs_path}")


def _scoring_input(pairs: list[dict], half: str, path: Path) -> None:
    """shadow-score requires exactly {id, text, input_channel} (inference.py:277)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in pairs:
            record = {
                "id": f"{row['pair_id']}:{half}",
                "text": row[half],
                "input_channel": "direct_user",
            }
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _truncated_pair_ids(pairs: list[dict]) -> set[str]:
    """Pairs where either half exceeds the scorer's context and loses its tail.

    `batch_logits` tokenizes with `truncation=True` at MAX_TOKENS and does not
    window (core.py:129-135). The request sits at the *end* of the wrapped text,
    so an over-length wrapped half is scored having seen less of the harmful
    request than its bare counterpart. Those pairs are no longer matched on
    content and are excluded from the headline rates.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    def over(texts: list[str]) -> list[bool]:
        encoded = tokenizer(
            [strict_normalize(t) for t in texts], add_special_tokens=True
        )
        return [len(ids) > MAX_TOKENS for ids in encoded["input_ids"]]

    wrapped_over = over([p["wrapped"] for p in pairs])
    inner_over = over([p["inner"] for p in pairs])
    return {
        p["pair_id"]
        for p, w, i in zip(pairs, wrapped_over, inner_over, strict=True)
        if w or i
    }


def _read_scores(path: Path) -> dict[str, float]:
    scores = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            scores[record["id"]] = float(record["score"])
    return scores


def score(projection: Path) -> dict:
    # Pairs are rebuilt from the projection rather than read back from the
    # archived pairs file, so `score` has no ordering dependency on `pairs`.
    all_pairs = build_pairs(projection)
    truncated = _truncated_pair_ids(all_pairs)
    pairs = [p for p in all_pairs if p["pair_id"] not in truncated]
    if not pairs:
        raise ValueError("every pair truncates at the scorer's context limit")

    OUT.mkdir(parents=True, exist_ok=True)
    scores = {}
    for half in ("wrapped", "inner"):
        source = OUT / f"{half}.input.jsonl"
        result = OUT / f"{half}.scores.jsonl"
        _scoring_input(pairs, half, source)
        result.unlink(missing_ok=True)  # score_file refuses an existing output
        score_file(MANIFEST, MODEL, source, result)
        scores[half] = _read_scores(result)

    rows = []
    for pair in pairs:
        wrapped = scores["wrapped"][f"{pair['pair_id']}:wrapped"]
        inner = scores["inner"][f"{pair['pair_id']}:inner"]
        rows.append({**pair, "score_wrapped": wrapped, "score_inner": inner})

    # Threshold each half once; every rate below is derived from these flags.
    flags = [
        (r["score_wrapped"] >= DIRECT_USER_FLOOR, r["score_inner"] >= DIRECT_USER_FLOOR)
        for r in rows
    ]
    table = Counter(flags)  # 2x2 contingency of (wrapped flagged, inner flagged)
    total = len(rows)
    flag_rate_wrapped = (table[True, True] + table[True, False]) / total
    flag_rate_inner = (table[True, True] + table[False, True]) / total

    by_category: dict[str, dict] = {}
    for row, (wrapped_flag, inner_flag) in zip(rows, flags, strict=True):
        bucket = by_category.setdefault(
            str(row["category"]), {"n": 0, "wrapped": 0, "inner": 0}
        )
        bucket["n"] += 1
        bucket["wrapped"] += wrapped_flag
        bucket["inner"] += inner_flag

    summary = {
        "format": "morgott-redteam-subversion-control-v1",
        "advisory_only": True,
        "model": MODEL,
        "pairs": len(rows),
        "pairs_built": len(all_pairs),
        "pairs_excluded_truncation": len(truncated),
        "max_tokens": MAX_TOKENS,
        "direct_user_floor": DIRECT_USER_FLOOR,
        "threshold_sha256": THRESHOLD_SHA256,
        "flag_rate_wrapped": flag_rate_wrapped,
        "flag_rate_inner": flag_rate_inner,
        "flag_rate_delta": flag_rate_wrapped - flag_rate_inner,
        "median_score_wrapped": statistics.median(r["score_wrapped"] for r in rows),
        "median_score_inner": statistics.median(r["score_inner"] for r in rows),
        "pairs_wrapped_only": table[True, False],
        "pairs_inner_only": table[False, True],
        "pairs_both": table[True, True],
        "pairs_neither": table[False, False],
        "by_category": dict(sorted(by_category.items())),
        "wrapper_variants": dict(
            sorted(Counter(r["wrapper_variant"] for r in rows).items())
        ),
        "interpretation": (
            "The two halves carry identical harmful content and differ only by the "
            "instruction-subversion wrapper. A large positive delta means the "
            "detector keys on subversion, which is the intended behaviour. A delta "
            "near zero means it keys on harm topic, which docs/data-contract.md "
            "treats as a separate label. Pairs whose wrapped half exceeds "
            "max_tokens are excluded: the request sits at the end of the wrapper, "
            "so truncation would score the two halves on different content and "
            "the comparison would no longer be matched."
        ),
    }
    atomic_write_text(
        OUT / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_jsonl_gz(OUT / "scored_pairs.jsonl.gz", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pairs", "score"))
    parser.add_argument(
        "--projection",
        type=Path,
        default=ARCHIVE / "redteam_campaigns_20260806.jsonl.gz",
        help="campaign projection to build pairs from",
    )
    parser.add_argument(
        "--pairs-out", type=Path, default=ARCHIVE / "subversion_pairs_20260806.jsonl.gz"
    )
    args = parser.parse_args()

    if args.command == "pairs":
        write_pairs(args.projection, args.pairs_out)
        return
    print(json.dumps(score(args.projection), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
