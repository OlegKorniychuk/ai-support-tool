"""Contractions, word count, overlap, and `check_drafts`'s pass/fail cases."""

import pytest

from support_ai.assistant.schema import Draft, Tone
from support_ai.assistant.tone_checks import (
    DISTINCT_MAX_OVERLAP,
    ToneCheck,
    check_drafts,
    find_contractions,
    token_overlap,
    word_count,
)


def _drafts(*, formal: str, empathetic: str, short: str) -> list[Draft]:
    return [
        Draft(tone=Tone.FORMAL, text=formal),
        Draft(tone=Tone.EMPATHETIC, text=empathetic),
        Draft(tone=Tone.SHORT, text=short),
    ]


# --- word_count --------------------------------------------------------------------------


def test_word_count_splits_on_whitespace():
    assert word_count("Open Profile then tap Save") == 5


def test_word_count_empty_string():
    assert word_count("") == 0


# --- find_contractions --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I don't know", ["don't"]),
        ("You can't do that", ["can't"]),
        ("It isn't working and wasn't before either", ["isn't", "wasn't"]),
        ("I'm sure you're right and we're both wrong", ["I'm", "you're", "we're"]),
        ("They're on their way", ["They're"]),
        (
            "It's fine, that's clear, there's no doubt, here's why",
            ["It's", "that's", "there's", "here's"],
        ),
        ("What's up? Let's go.", ["What's", "Let's"]),
        ("I'll try, you'll see, we'll know, they'll agree", ["I'll", "you'll", "we'll", "they'll"]),
        ("I've done it, you've seen it, we've tried it", ["I've", "you've", "we've"]),
        ("I'd like that, you'd love it, we'd agree", ["I'd", "you'd", "we'd"]),
    ],
)
def test_find_contractions_common_forms(text, expected):
    assert find_contractions(text) == expected


def test_find_contractions_curly_apostrophes():
    text = "I’m sure you’re right, this doesn’t work"
    assert find_contractions(text) == ["I’m", "you’re", "doesn’t"]


def test_find_contractions_is_case_insensitive():
    assert find_contractions("DON'T WORRY") == ["DON'T"]


@pytest.mark.parametrize(
    "text",
    [
        "The customer's account was charged twice.",
        "Check the app's settings menu.",
        "James's session credit was returned.",
        "The experts' schedule is full.",
    ],
)
def test_find_contractions_never_flags_possessives(text):
    assert find_contractions(text) == []


def test_find_contractions_no_apostrophes():
    assert find_contractions("Open Profile then tap Save.") == []


# --- token_overlap ------------------------------------------------------------------------


def test_token_overlap_identical_text_is_one():
    assert token_overlap("Open Profile then tap Save", "open profile then tap save") == 1.0


def test_token_overlap_disjoint_text_is_zero():
    assert token_overlap("Open Profile then tap Save", "Contact support for more help") == 0.0


def test_token_overlap_ignores_punctuation():
    assert token_overlap("Open Profile, then tap Save!", "Open Profile then tap Save") == 1.0


def test_token_overlap_partial():
    # {open, profile} vs {open, settings}: intersection 1, union 3.
    assert token_overlap("Open Profile", "Open Settings") == pytest.approx(1 / 3)


def test_token_overlap_both_empty_is_zero():
    assert token_overlap("", "") == 0.0


# --- check_drafts -------------------------------------------------------------------------


def test_check_drafts_all_pass():
    drafts = _drafts(
        formal="Dear customer, open Profile then tap Birth details to update your time.",
        empathetic="I understand you want to fix your birth time — head to Profile and tap "
        "Birth details.",
        short="Open Profile, then Birth details, then Save.",
    )
    result = check_drafts(drafts, short_max_words=50)

    assert isinstance(result, ToneCheck)
    assert result.short_ok is True
    assert result.formal_ok is True
    assert result.formal_contractions == []
    assert result.distinct_ok is True
    assert result.max_pair_overlap < DISTINCT_MAX_OVERLAP


def test_check_drafts_short_too_long_fails():
    long_short = " ".join(["word"] * 51)
    drafts = _drafts(
        formal="Dear customer, please open your profile.",
        empathetic="I hear you.",
        short=long_short,
    )
    result = check_drafts(drafts, short_max_words=50)

    assert result.short_words == 51
    assert result.short_ok is False


def test_check_drafts_formal_with_contraction_fails():
    drafts = _drafts(
        formal="We're sorry, but that's not something we can change here.",
        empathetic="I understand this is frustrating.",
        short="Contact support for help.",
    )
    result = check_drafts(drafts, short_max_words=50)

    assert result.formal_ok is False
    assert result.formal_contractions == ["We're", "that's"]


def test_check_drafts_not_distinct_fails():
    same_text = "Open Profile then tap Birth details to change your birth time."
    drafts = _drafts(formal=same_text, empathetic=same_text, short=same_text)
    result = check_drafts(drafts, short_max_words=50)

    assert result.max_pair_overlap == 1.0
    assert result.distinct_ok is False


def test_check_drafts_wrong_tone_set_raises():
    drafts = [
        Draft(tone=Tone.FORMAL, text="a"),
        Draft(tone=Tone.EMPATHETIC, text="b"),
    ]
    with pytest.raises(ValueError):
        check_drafts(drafts, short_max_words=50)


def test_check_drafts_duplicate_tone_raises():
    drafts = [
        Draft(tone=Tone.FORMAL, text="a"),
        Draft(tone=Tone.FORMAL, text="b"),
        Draft(tone=Tone.SHORT, text="c"),
    ]
    with pytest.raises(ValueError):
        check_drafts(drafts, short_max_words=50)
