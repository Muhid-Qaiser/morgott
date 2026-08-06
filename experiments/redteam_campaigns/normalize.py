"""Project first-party red-team campaign parquets into an archive JSONL.

This is archive-class work, not a canonical source adapter. The output lives in
`data-archive/` and must never be copied into `data/`. Labels here are named
`proposed_*` because they are derived heuristics over model-generated text, not
contract labels; see `reports/redteam-campaign-data.md`.

Usage:
    uv run --extra showcase python -m experiments.redteam_campaigns.normalize \
        --input data-archive/redteam/raw/normalized_dataset_dedup.parquet \
        --out data-archive/redteam/redteam_campaigns_20260806 --check
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from morgott.data import SECURITY_LABELS, atomic_write_text, file_sha256, text_hash

# The one fixed jailbreak wrapper used by the "concise" campaigns. Matching on
# the literal prefix is exact and auditable; the marker scan below is the
# fallback for everything else.
WRAPPER_PREFIX = "You are a very direct assistant who always fulfills"

SUBVERSION_MARKERS = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above)"
    r"|you are (?:now )?(?:a |an )?(?:very direct|unrestricted|dan|jailbroken)"
    r"|never refuse|do not refuse"
    r"|disregard (?:your |all )?(?:guidelines|instructions|policy|rules)"
    r"|<rule>|sanctioned red-?team|ignore any possible ethical"
    r"|begin your answer|start your (?:reply|answer|response) with|sure, here is"
    r"|no disclaimers?|without any (?:warnings?|disclaimers?)"
    r"|developer mode|system prompt|override"
    r"|bypass (?:your |the )?(?:safety|filter|guard)",
    re.I,
)

ATTACKER_REFUSAL = re.compile(
    r"\s*[\"']?(?:i can't|i cannot|i'm not able|i am not able"
    r"|i won't|i will not|i'm sorry|i apologize)\b",
    re.I,
)

MIN_TEXT_CHARS = 40
# 100+ identical characters in a row is model degeneration, not a prompt. Only
# three rows in the corpus trip it (runs of 126,720 "A" and two of 45,000+ "█")
# and the count is unchanged at thresholds of 200 and 500, so it is not tuned to
# the data.
DEGENERATE_RUN = re.compile(r"(.)\1{99,}", re.S)

ENVELOPE_PROMPT_KEY = re.compile(r'"prompt"\s*:\s*"')
ENVELOPE_NEXT_KEY = re.compile(r'"\s*,\s*"\w+"\s*:')

LINEAGE_KEYS = (
    "goal_id",
    "idea_id",
    "lineage_id",
    "parent_idea_id",
    "branch_depth",
    "round_index",
)

# Columns read from the parquet. `raw_record` is read only for lineage and is
# never emitted — it carries judge reasoning and target responses. The three
# `source_*` path columns embed the generating machine's user directory and are
# not read at all.
COLUMNS = (
    "prompt",
    "goal",
    "verdict",
    "breached",
    "category",
    "attacker_model",
    "target_model",
    "judge_model",
    "prompt_kind",
    "attack_mode",
    "contains_injection",
    "record_id",
    "run_id",
    "run_timestamp",
    "raw_record",
)


def _subversion_basis(text: str) -> str | None:
    """How instruction subversion was detected, or None if it was not."""
    if text.startswith(WRAPPER_PREFIX):
        return "wrapper_template"
    if SUBVERSION_MARKERS.search(text):
        return "marker_scan"
    return None


def _recover_envelope(text: str) -> str | None:
    """Pull the attack prompt out of the attacker model's JSON envelope.

    Some campaigns emitted `{"strategy": ..., "prompt": "<attack>"}` instead of a
    bare prompt. The JSON is not parseable — the prompt text carries unescaped
    double quotes, raw newlines, and bad escapes — so this anchors on the key
    boundaries rather than the grammar: everything between `"prompt": "` and
    either the next `"key":` or the trailing brace.
    """
    opened = ENVELOPE_PROMPT_KEY.search(text)
    if opened is None:
        return None
    rest = text[opened.end() :]
    following_key = ENVELOPE_NEXT_KEY.search(rest)
    value = rest[: following_key.start()] if following_key else rest.rstrip('\t\n\r }"')
    value = (
        value.replace('\\"', '"')
        .replace("\\'", "'")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .strip()
    )
    return value if len(value) >= MIN_TEXT_CHARS else None


def _usable_text(raw: str | None) -> tuple[str | None, str | None, str]:
    """Return (text, quarantine_reason, text_source) for one source prompt."""
    if raw is None or len(raw.strip()) < MIN_TEXT_CHARS:
        return None, "empty_or_trivial_text", "source_prompt"

    text, source = raw, "source_prompt"
    if re.match(r"\s*[\{\[]", raw):
        recovered = _recover_envelope(raw)
        if recovered is None:
            # No `prompt` key at all — a tool-return repr, not an attack prompt.
            return None, "non_prompt_fragment", source
        text, source = recovered, "generator_envelope"

    # Applied after recovery, so degenerate text inside an envelope is caught too.
    if re.search(r"<\|.*?\|>", text):
        return None, "degenerate_special_tokens", source
    if DEGENERATE_RUN.search(text):
        return None, "degenerate_repetition", source
    if ATTACKER_REFUSAL.match(text):
        return None, "attacker_refusal", source
    return text, None, source


def _agentdojo_label(row: dict) -> tuple[str, str]:
    """Label the 21 AgentDojo Banking-derived rows (13 retained) from their metadata."""
    kind = row["prompt_kind"]
    if kind == "agentic_user_input":
        return "benign", "direct_user"
    if kind == "agentic_direct_attack_input":
        # A direct unauthorized-action request with no subversion wrapper.
        return "harmful_non_injection", "direct_user"
    if kind != "agentic_tool_input":
        # Fail closed: a new agentic kind must not silently land on "benign".
        raise ValueError(f"unhandled agentic prompt_kind: {kind}")
    if row.get("contains_injection") is True:
        return "indirect_prompt_injection", "untrusted_content"
    # Clean tool returns: the unpoisoned bill, transaction confirmations, an IBAN.
    return "benign", "untrusted_content"


def _project(row: dict, text: str, text_source: str) -> dict:
    agentdojo = str(row["prompt_kind"]).startswith("agentic")

    if agentdojo:
        proposed, channel = _agentdojo_label(row)
        basis = "source_flag" if proposed == "indirect_prompt_injection" else None
    else:
        channel = "direct_user"
        basis = _subversion_basis(text)
        if basis is not None:
            proposed = "direct_jailbreak"
        elif row["verdict"] is None:
            # Never sent to a target, never judged: unknown, not benign.
            proposed = "uncertain"
        else:
            proposed = "harmful_non_injection"

    raw = json.loads(row["raw_record"]) if row["raw_record"] else {}
    lineage = {key: raw.get(key) for key in LINEAGE_KEYS}
    run_id = row["run_id"] or raw.get("run_id") or ""
    goal_id = lineage["goal_id"] or "no_goal"

    digest = text_hash(text)
    identity = f"{run_id}|{row['record_id']}|{digest}"
    return {
        "id": "redteam:" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "text": text,
        "normalized_text_sha256": digest,
        # "generator_envelope" means `text` was extracted from the attacker
        # model's JSON wrapper and is not byte-equal to the source `prompt`.
        "text_source": text_source,
        "input_channel": channel,
        # Subversion is present iff this is non-null; no separate boolean.
        "subversion_basis": basis,
        "proposed_security_label": proposed,
        "label_basis": "first_party_automated_red_team_campaign_attacker_generated",
        "agentdojo_derived": agentdojo,
        "split_group_id": f"redteam:{run_id}:{goal_id}",
        "run_id": run_id,
        "run_timestamp": row["run_timestamp"],
        "record_id": row["record_id"],
        "attack_mode": row["attack_mode"],
        "prompt_kind": row["prompt_kind"],
        "category": row["category"],
        "goal": row["goal"],
        **lineage,
        # Outcome metadata. These describe what one target model did on one day;
        # they are not detector labels. See docs/data-contract.md:88.
        "attacker_model": row["attacker_model"],
        "target_model": row["target_model"],
        "judge_model": row["judge_model"],
        "verdict": row["verdict"],
        "breached": row["breached"],
    }


def _read(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        table = pq.read_table(path, columns=list(COLUMNS))
        rows.extend(table.to_pylist())
    return rows


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    body = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    # mtime=0 so repeated runs are byte-identical.
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(body.encode("utf-8"))


def _summary(
    paths: list[Path], kept: list[dict], dropped: list[dict], input_rows: int
) -> dict:
    def tally(rows: list[dict], key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in rows).items()))

    return {
        "format": "morgott-redteam-campaigns-v1",
        "archive_class": True,
        "inputs": [
            {"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in paths
        ],
        "input_rows": input_rows,
        "retained_rows": len(kept),
        "quarantined_rows": len(dropped),
        "quarantine_reasons": tally(dropped, "quarantine_reason"),
        "proposed_security_label": tally(kept, "proposed_security_label"),
        "subversion_basis": tally(kept, "subversion_basis"),
        "text_source": tally(kept, "text_source"),
        "input_channel": tally(kept, "input_channel"),
        "attack_mode": tally(kept, "attack_mode"),
        "prompt_kind": tally(kept, "prompt_kind"),
        "category": tally(kept, "category"),
        "verdict": tally(kept, "verdict"),
        "attacker_model": tally(kept, "attacker_model"),
        "target_model": tally(kept, "target_model"),
        "judge_model": tally(kept, "judge_model"),
        "agentdojo_derived_rows": sum(1 for row in kept if row["agentdojo_derived"]),
        "distinct_run_id": len({row["run_id"] for row in kept}),
        "distinct_split_group_id": len({row["split_group_id"] for row in kept}),
        "distinct_normalized_text": len(
            {row["normalized_text_sha256"] for row in kept}
        ),
        "run_timestamp_min": min(str(row["run_timestamp"]) for row in kept),
        "run_timestamp_max": max(str(row["run_timestamp"]) for row in kept),
        "caveats": [
            "Positive-only. There is no benign denominator, so no FPR or precision.",
            "`verdict`/`breached` record whether one target model was breached on one "
            "day. They are outcome metadata, never detector labels.",
            "`category` is confounded with `attack_mode` (whole categories were run "
            "under a single strategy), so per-category slices are not topic effects.",
            "`proposed_security_label` is a derived heuristic over model-generated "
            "text, not a contract label. Do not copy it into data/.",
            "AgentDojo-derived rows are flagged; repo policy keeps AgentDojo text out "
            "of the training corpus.",
        ],
    }


def build(paths: list[Path]) -> tuple[list[dict], list[dict], dict]:
    kept: list[dict] = []
    dropped: list[dict] = []
    source_rows = _read(paths)
    for row in source_rows:
        text, reason, text_source = _usable_text(row["prompt"])
        if reason is not None:
            dropped.append(
                {
                    "quarantine_reason": reason,
                    "record_id": row["record_id"],
                    "run_id": row["run_id"],
                    "prompt_kind": row["prompt_kind"],
                    "text_prefix": (row["prompt"] or "")[:200],
                    "text_length": len(row["prompt"] or ""),
                }
            )
            continue
        kept.append(_project(row, text, text_source))
    kept.sort(key=lambda row: row["id"])
    dropped.sort(key=lambda row: (row["quarantine_reason"], row["text_prefix"]))
    return kept, dropped, _summary(paths, kept, dropped, len(source_rows))


def check(kept: list[dict], dropped: list[dict], summary: dict) -> None:
    # Only checks that can actually fail. 16 hex chars of SHA-256 can collide and
    # `record_id` is documented as non-unique, so the id check is the load-bearing
    # one; `input_rows` is threaded from the reader so this catches a lost row.
    ids = [row["id"] for row in kept]
    assert len(ids) == len(set(ids)), "projection ids are not unique"
    assert len(kept) + len(dropped) == summary["input_rows"], "row accounting mismatch"
    assert all(
        row["input_channel"] in {"direct_user", "untrusted_content"} for row in kept
    ), "unknown input_channel"
    assert all(row["proposed_security_label"] in SECURITY_LABELS for row in kept), (
        "proposed_security_label outside the contract vocabulary"
    )
    print(f"check ok: {len(kept)} retained, {len(dropped)} quarantined")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output stem; .jsonl.gz, .quarantine.jsonl.gz and .summary.json are added",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    kept, dropped, summary = build(sorted(args.input))
    if args.check:
        check(kept, dropped, summary)

    out = str(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(Path(out + ".jsonl.gz"), kept)
    _write_jsonl_gz(Path(out + ".quarantine.jsonl.gz"), dropped)
    atomic_write_text(
        Path(out + ".summary.json"),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
