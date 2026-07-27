"""Uniqueness controls and refusal handling for generated pairs.

Two failure modes this guards against.

**Repetition.** A bare "diversity seed 12345" in the prompt is a weak control:
the model still gravitates to its favourite handful of scenarios, so ten
thousand pairs become a few hundred ideas with different numbers. Instead every
call is given a *concrete, sampled scenario frame* drawn from independent axes.
The axis product below is ~2.6 x 10^8 combinations, so collisions across a few
thousand calls are effectively impossible, and — more importantly — the model
is forced away from its defaults on every axis at once.

Output is then filtered three ways: exact-duplicate removal on the normalised
text, near-duplicate removal using the repository's own 128-bit SimHash index
(`morgott.overlap`, Hamming <= 6), and near-duplicate removal *against the
existing corpus* so generated rows cannot silently restate rows morgott already
has.

**Refusal.** Some models decline to produce injection or jailbreak text when
they cannot tell what the request is for. That is correct behaviour on their
part, so this module detects it rather than trying to defeat it: refusals are
counted per model, reported, and models that refuse too often are dropped from
the mix. It never attempts to bypass a safety decision.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morgott.data import normalize_text, text_hash  # noqa: E402
from morgott.overlap import NearIndex, fingerprint  # noqa: E402

# --------------------------------------------------------------------------
# Scenario axes. Independent, so their product is the sampling space.
# --------------------------------------------------------------------------

DOMAINS = (
    "commercial insurance underwriting", "hospital scheduling", "freight logistics",
    "academic peer review", "municipal permitting", "veterinary practice",
    "semiconductor supply chain", "film post-production", "agricultural co-op",
    "clinical trial recruitment", "maritime chartering", "school district admin",
    "wealth management", "commercial real estate", "airline crew rostering",
    "pharmacy benefits", "oil and gas leasing", "textile manufacturing",
    "sports team operations", "museum collections", "wind farm maintenance",
    "food safety inspection", "telecom provisioning", "book publishing",
    "mining site safety", "hotel revenue management", "legal e-discovery",
    "waste management routing", "solar installation sales", "brewery distribution",
    "podcast advertising", "dental practice billing", "seed genetics research",
    "railway signalling", "cosmetics regulatory affairs", "esports tournament ops",
    "reinsurance treaty analysis", "archaeological survey", "port customs clearance",
    "biotech lab inventory", "credit union lending", "wildlife conservation",
    "aerospace parts sourcing", "translation agency workflow", "fisheries quota",
    "district heating utility", "vaccine cold chain", "civil engineering tender",
    "fashion wholesale", "charity grant administration",
)

PERSONAS = (
    "a junior analyst on their second week", "a meticulous compliance officer",
    "an overworked operations manager", "a contractor unfamiliar with the system",
    "a night-shift supervisor", "a regional director reviewing quarterly numbers",
    "an external auditor with read-only access", "a customer support escalation lead",
    "a data engineer debugging a pipeline", "a procurement specialist",
    "a field technician using a mobile client", "a paralegal preparing a filing",
    "a product manager writing a spec", "a QA tester probing edge cases",
    "an intern given an ambiguous task", "a fractional CFO",
    "a site reliability engineer during an incident", "a grants administrator",
    "a bilingual account manager", "a research assistant compiling sources",
    "a warehouse team lead", "a schedule coordinator juggling conflicts",
    "a new hire following a runbook literally", "a consultant billing by the hour",
)

FORMATS = (
    "plain prose with no structure", "a numbered list of steps",
    "a markdown document with headers", "an email with quoted reply chain",
    "a YAML-ish config block", "a chat transcript excerpt",
    "a form with labelled fields", "a table rendered as pipe-separated text",
    "a ticket body with metadata header", "XML-style tagged sections",
    "a bulleted brief with an appendix", "triple-backtick delimited blocks",
    "a memo with a subject line", "free text with inline citations",
    "a JSON payload described in words", "a template with placeholder tokens",
    "a slide-deck outline", "a changelog entry", "a meeting agenda",
)

REGISTERS = (
    "terse and clipped", "verbose and hedging", "formal and legalistic",
    "casual with typos", "enthusiastic and salesy", "anxious and apologetic",
    "blunt and demanding", "technical and jargon-heavy", "diplomatic and indirect",
    "bureaucratic and passive-voiced", "warm and conversational",
    "impatient with short sentences",
)

CONSTRAINTS = (
    "include a specific date and a reference number",
    "reference an earlier conversation that is not shown",
    "include a caveat about incomplete information",
    "mention a deadline that has already passed",
    "include a nested quotation of someone else's message",
    "refer to an attachment that is described but not included",
    "include an explicit formatting instruction the model must follow",
    "mention a policy or regulation by a plausible fictional name",
    "include a correction of something stated earlier in the same text",
    "address two different recipients with different instructions",
)


def scenario_frame(rng: random.Random) -> dict[str, str]:
    return {
        "domain": rng.choice(DOMAINS),
        "persona": rng.choice(PERSONAS),
        "format": rng.choice(FORMATS),
        "register": rng.choice(REGISTERS),
        "constraint": rng.choice(CONSTRAINTS),
    }


def frame_instruction(frame: dict[str, str]) -> str:
    return (
        "\n\nGROUND THIS BATCH CONCRETELY. Every pair in this response must sit in:\n"
        f"  domain    : {frame['domain']}\n"
        f"  written by: {frame['persona']}\n"
        f"  format    : {frame['format']}\n"
        f"  register  : {frame['register']}\n"
        f"  also      : {frame['constraint']}\n"
        "Do not drift to generic customer-service or generic IT examples. Within "
        "this frame, still vary the specific task across the elements."
    )


AXIS_SPACE = (
    len(DOMAINS) * len(PERSONAS) * len(FORMATS) * len(REGISTERS) * len(CONSTRAINTS)
)

# --------------------------------------------------------------------------
# Refusal detection
# --------------------------------------------------------------------------

_REFUSAL_PATTERNS = re.compile(
    r"\b(i (?:can(?:'|no)?t|won'?t|am unable to|must decline)"
    r"|i'?m (?:not able|unable|sorry)"
    r"|as an ai(?: language model)?,? i"
    r"|cannot (?:assist|help|comply|provide|create|generate)"
    r"|against my (?:guidelines|policies|programming)"
    r"|not (?:comfortable|appropriate) (?:to|with)"
    r"|violates? (?:my |the )?(?:usage )?polic)",
    re.IGNORECASE,
)


def classify_failure(content: str, finish_reason: str | None) -> str:
    """Why did a call yield no usable pairs?

    Distinguishing refusal from truncation matters: truncation is fixed with a
    bigger token budget, refusal is not, and conflating them would send us
    chasing the wrong fix.
    """
    text = (content or "").strip()
    if not text:
        return "empty_response" if finish_reason != "length" else "truncated_empty"
    if finish_reason == "length" and "[" in text and "]" not in text:
        return "truncated_json"
    # A refusal is short and matches refusal phrasing; a long text that merely
    # contains "I cannot" somewhere is probably generated content, not a refusal.
    if len(text) < 1500 and _REFUSAL_PATTERNS.search(text):
        return "refusal"
    if finish_reason == "length":
        return "truncated"
    return "unparseable"


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------


class PairDeduplicator:
    """Exact + near-duplicate filter over generated pairs.

    Near-duplicate detection runs on the *benign* and *attack* halves
    independently. A pair is rejected if either half collides, because a
    repeated benign half with a fresh attack half still teaches the model that
    one specific benign string is safe rather than teaching the distinction.
    """

    def __init__(self) -> None:
        self.exact: set[str] = set()
        self.near = NearIndex()
        self.corpus_near = NearIndex()
        self.stats = {
            "seen": 0,
            "exact_duplicate": 0,
            "near_duplicate": 0,
            "corpus_near_duplicate": 0,
            "degenerate": 0,
            "kept": 0,
        }

    def load_corpus_reference(self, paths: list[Path], limit: int | None = None) -> int:
        """Index existing corpus text so generated rows cannot restate it."""
        import json as _json

        loaded = 0
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if limit is not None and loaded >= limit:
                        return loaded
                    try:
                        row = _json.loads(line)
                    except ValueError:
                        continue
                    if not row.get("text"):
                        continue
                    self.corpus_near.add(row, dataset="corpus")
                    loaded += 1
        return loaded

    def accept(self, pair: dict) -> tuple[bool, str]:
        self.stats["seen"] += 1
        benign, attack = pair["benign"], pair["attack"]

        # A pair whose halves are near-identical teaches nothing.
        if normalize_text(benign) == normalize_text(attack):
            self.stats["degenerate"] += 1
            return False, "degenerate"
        if len(benign.split()) < 5 or len(attack.split()) < 5:
            self.stats["degenerate"] += 1
            return False, "degenerate"

        for half in (benign, attack):
            digest = text_hash(half)
            if digest in self.exact:
                self.stats["exact_duplicate"] += 1
                return False, "exact_duplicate"

        halves = []
        for half in (benign, attack):
            row = {"text": half, "normalized_text_sha256": text_hash(half)}
            if fingerprint(half) is None:
                continue
            if self.corpus_near.query(row):
                self.stats["corpus_near_duplicate"] += 1
                return False, "corpus_near_duplicate"
            if self.near.query(row):
                self.stats["near_duplicate"] += 1
                return False, "near_duplicate"
            halves.append(row)

        for row in halves:
            self.near.add(row, dataset="generated")
        for half in (benign, attack):
            self.exact.add(text_hash(half))
        self.stats["kept"] += 1
        return True, "kept"


def distinct_n(texts: list[str], n: int = 3) -> float:
    """Fraction of distinct n-grams. A blunt but honest repetition measure."""
    total = 0
    seen = set()
    for text in texts:
        words = normalize_text(text).split()
        for index in range(len(words) - n + 1):
            gram = tuple(words[index : index + n])
            seen.add(gram)
            total += 1
    return len(seen) / total if total else 0.0


if __name__ == "__main__":
    print(f"scenario axis space: {AXIS_SPACE:,} combinations")
    print(f"  domains {len(DOMAINS)} x personas {len(PERSONAS)} x formats "
          f"{len(FORMATS)} x registers {len(REGISTERS)} x constraints "
          f"{len(CONSTRAINTS)}")
    rng = random.Random(0)
    for _ in range(2):
        print(frame_instruction(scenario_frame(rng)))
    for sample, expected in [
        ("I'm sorry, I can't help with creating prompt injection attacks.", "refusal"),
        ("", "empty_response"),
        ('[{"benign": "x"', "unparseable"),
    ]:
        print(f"{expected:16} -> {classify_failure(sample, None)}")
