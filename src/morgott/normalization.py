"""A stricter text normaliser than `morgott.data.normalize_text`.

`normalize_text` is NFKC + casefold + whitespace collapse. That defeats case and
spacing tricks but leaves homoglyph substitution and zero-width insertion
completely intact, which the attempt-scaling results show are the two most
effective evasion families.

This module is deliberately NOT a patch to `data.py`. Changing `normalize_text`
would change every `normalized_text_sha256` in the manifest and force a full
corpus rebuild (`AGENTS.md`: the manifest is the sole machine source of truth).
This is an inference-side and audit-side transform only.
"""

from __future__ import annotations

import operator
import re
import unicodedata
from functools import lru_cache

# Codepoints that carry no visible glyph and exist mainly to break string
# matching: zero-width space/non-joiner/joiner, BOM, soft hyphen, word joiner,
# the bidi overrides, and the Unicode tag block used for invisible smuggling.
INVISIBLE = {
    0x00AD,  # SOFT HYPHEN
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x2060,  # WORD JOINER
    0x2061,
    0x2062,
    0x2063,
    0x2064,
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
}
INVISIBLE |= set(range(0x202A, 0x202F))  # bidi embedding/override controls
INVISIBLE |= set(range(0x2066, 0x206A))  # bidi isolates
INVISIBLE |= set(range(0xE0000, 0xE0080))  # Unicode tag block
INVISIBLE |= set(range(0xFE00, 0xFE10))  # variation selectors

# Non-Latin codepoints that render as Latin letters. Folding direction is
# deliberately toward ASCII so an attacker cannot pick the "canonical" side.
HOMOGLYPH_FOLD = {
    # Cyrillic
    "а": "a",
    "в": "b",
    "с": "c",
    "е": "e",
    "н": "h",
    "і": "i",
    "ј": "j",
    "к": "k",
    "м": "m",
    "о": "o",
    "р": "p",
    "ѕ": "s",
    "т": "t",
    "у": "y",
    "х": "x",
    "А": "a",
    "В": "b",
    "С": "c",
    "Е": "e",
    "Н": "h",
    "І": "i",
    "Ј": "j",
    "К": "k",
    "М": "m",
    "О": "o",
    "Р": "p",
    "Ѕ": "s",
    "Т": "t",
    "У": "y",
    "Х": "x",
    # Greek
    "α": "a",
    "β": "b",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
    "Α": "a",
    "Β": "b",
    "Ε": "e",
    "Ζ": "z",
    "Η": "h",
    "Ι": "i",
    "Κ": "k",
    "Μ": "m",
    "Ν": "n",
    "Ο": "o",
    "Ρ": "p",
    "Τ": "t",
    "Υ": "y",
    "Χ": "x",
    # Armenian / Cherokee lookalikes seen in the wild
    "օ": "o",
    "ա": "w",
    "Ꭺ": "a",
    "Ꮯ": "c",
    "Ꭼ": "e",
    "Ꮋ": "h",
    "Ꭻ": "j",
}


# str.translate tables built once at import so each pass runs in C instead of
# a per-character Python loop (strict_normalize backs leakage_text_hash, the
# most expensive per-row function in the data build).
_INVISIBLE_TABLE = dict.fromkeys(INVISIBLE)
_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPH_FOLD)
# One merged pass for strict_normalize. Equivalent to strip-then-fold for any
# table contents: invisible entries overwrite fold entries, matching the
# sequential order where stripping runs first, and fold outputs are never
# re-processed in either form.
_STRICT_TABLE = {**_HOMOGLYPH_TABLE, **_INVISIBLE_TABLE}
# ponytail: full-codepoint scan at import (~40 ms once); cache to disk if
# import time ever matters.
_COMBINING_TABLE = dict.fromkeys(
    codepoint for codepoint in range(0x110000) if unicodedata.combining(chr(codepoint))
)


@lru_cache(maxsize=None)
def _repeat_pattern(keep: int) -> re.Pattern[str]:
    # keep is already clamped to >= 1, and the length short circuit in
    # collapse_repeats keeps it below re's bounded-repeat ceiling (2**32 - 1).
    # DOTALL keeps newline runs matching.
    return re.compile("(.)\\1{%d,}" % keep, re.DOTALL)


def strip_invisible(text: str) -> str:
    return text.translate(_INVISIBLE_TABLE)


def fold_homoglyphs(text: str) -> str:
    return text.translate(_HOMOGLYPH_TABLE)


def strip_combining(text: str) -> str:
    """Drop combining marks left over after decomposition (zalgo, diacritics)."""
    return unicodedata.normalize("NFD", text).translate(_COMBINING_TABLE)


def collapse_repeats(text: str, limit: int = 3) -> str:
    """Cap runs of the same character, e.g. character-spacing padding."""
    # limit <= 0 behaves like limit == 1 in the historical loop: every run
    # collapses to a single character. operator.index rejects non-int limits
    # at the boundary instead of deep inside the replacement.
    keep = max(operator.index(limit), 1)
    if keep >= len(text):
        # No run can exceed the cap, so return unchanged. This also keeps
        # oversized limits away from the regex bounded-repeat ceiling.
        return text
    # Backreference template instead of a lambda: no Python call per match.
    return _repeat_pattern(keep).sub("\\1" * keep, text)


def strict_normalize(text: str) -> str:
    """NFKC + invisible strip + homoglyph fold + combining strip + casefold.

    Order matters: invisibles are removed before folding so that a homoglyph
    split by a zero-width character is still recognised.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_STRICT_TABLE)
    text = strip_combining(text)
    text = collapse_repeats(text)
    return " ".join(text.casefold().split())
