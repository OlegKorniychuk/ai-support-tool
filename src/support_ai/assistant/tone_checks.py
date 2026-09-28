"""Deterministic checks over the 3 drafts (SPEC_MVP2.md's tone rules): word count for
`short`, contractions for `formal`, and how distinct the three tones are from each other.

Pure and cheap on purpose: these run on every eval ticket (and could run on every live
request) without an LLM call, so drafts that clearly break a tone rule are caught before
they ever reach the judge.
"""

import re

from pydantic import BaseModel

from support_ai.assistant.schema import Draft, Tone

# Above this pairwise word overlap, two drafts read as the same reply reworded rather than
# genuinely distinct tones.
DISTINCT_MAX_OVERLAP = 0.8

# A word containing one or more apostrophes (straight or curly): candidate contractions
# and possessives alike, sorted out by `find_contractions` below.
_APOSTROPHE_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)+")

# Common English contractions, apart from "n't" forms (matched generically below).
# Deliberately excludes possessives ("customer's", "the app's") — those aren't in this set
# and don't end in "n't", so they never match.
_CONTRACTIONS = {
    "i'm",
    "you're",
    "we're",
    "they're",
    "it's",
    "that's",
    "there's",
    "here's",
    "what's",
    "let's",
    "i'll",
    "you'll",
    "we'll",
    "they'll",
    "i've",
    "you've",
    "we've",
    "i'd",
    "you'd",
    "we'd",
}

_WORD = re.compile(r"[a-z0-9]+")


class ToneCheck(BaseModel):
    """The result of `check_drafts` for one ticket's 3 drafts."""

    short_words: int
    short_ok: bool
    formal_contractions: list[str]
    formal_ok: bool
    max_pair_overlap: float
    distinct_ok: bool


def word_count(text: str) -> int:
    """Word count by whitespace splitting — matches `SHORT_MAX_WORDS`'s definition."""
    return len(text.split())


def find_contractions(text: str) -> list[str]:
    """Every contraction in `text`, in the order they appear (original spelling kept).

    Flags "n't" forms (don't, can't, isn't, ... — any word ending in "n't"/"n’t") and the
    fixed list of common contractions in `_CONTRACTIONS`. Both straight (') and curly (’)
    apostrophes count, and matching is case-insensitive. Possessives ("customer's", "the
    app's") are never flagged: they're neither an "n't" form nor in `_CONTRACTIONS`.
    """
    found = []
    for match in _APOSTROPHE_WORD.finditer(text):
        word = match.group()
        normalized = word.replace("’", "'").lower()
        if normalized.endswith("n't") or normalized in _CONTRACTIONS:
            found.append(word)
    return found


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity of `a` and `b`'s lowercase word sets, punctuation stripped.

    1.0 when the two (non-empty) word sets are identical, 0.0 when they share nothing.
    Also 0.0 when both are empty: there's no shared vocabulary to speak of, so treating
    "nothing vs nothing" as similar would be misleading for a distinctness check.
    """
    words_a = set(_WORD.findall(a.lower()))
    words_b = set(_WORD.findall(b.lower()))
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def check_drafts(drafts: list[Draft], *, short_max_words: int) -> ToneCheck:
    """Run every deterministic tone check over one ticket's 3 drafts.

    Raises `ValueError` unless `drafts` has exactly one draft per `Tone` (formal,
    empathetic, short) — any other shape means the caller passed something other than a
    clean, unflagged reply.
    """
    by_tone = {draft.tone: draft for draft in drafts}
    if set(by_tone) != set(Tone):
        raise ValueError(f"check_drafts expects exactly {set(Tone)}, got {set(by_tone)}")

    short_text = by_tone[Tone.SHORT].text
    formal_text = by_tone[Tone.FORMAL].text
    empathetic_text = by_tone[Tone.EMPATHETIC].text

    short_words = word_count(short_text)
    formal_contractions = find_contractions(formal_text)

    texts = [formal_text, empathetic_text, short_text]
    max_pair_overlap = max(
        token_overlap(texts[i], texts[j])
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    )

    return ToneCheck(
        short_words=short_words,
        short_ok=short_words <= short_max_words,
        formal_contractions=formal_contractions,
        formal_ok=not formal_contractions,
        max_pair_overlap=max_pair_overlap,
        distinct_ok=max_pair_overlap < DISTINCT_MAX_OVERLAP,
    )
