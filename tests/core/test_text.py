import pytest

from support_ai.core.text import normalize, quote_in_text, wrap_ticket

TEXT = "I can't log in at all since the update. Please fix it!"


def test_wrap_ticket_format():
    assert wrap_ticket("Hello there") == "<<<TICKET>>>\nHello there\n<<<END TICKET>>>"


def test_wrap_ticket_preserves_ticket_body_verbatim():
    ticket = "line one\nline two"
    wrapped = wrap_ticket(ticket)
    assert wrapped.startswith("<<<TICKET>>>\n")
    assert wrapped.endswith("\n<<<END TICKET>>>")
    assert ticket in wrapped


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello   World", "hello world"),
        ("  leading and trailing  ", "leading and trailing"),
        ("Mixed\tWhitespace\nHere", "mixed whitespace here"),
        ("UPPER CASE", "upper case"),
        ("Already normal", "already normal"),
    ],
)
def test_normalize_collapses_whitespace_and_casefolds(text, expected):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "quote",
    [
        "I can't log in at all",
        "i CAN'T   log in at all",
        '"I can\'t log in at all."',
        "“I can't log in at all since the update…”",
    ],
)
def test_quote_in_text_tolerates_case_whitespace_and_quote_marks(quote):
    assert quote_in_text(quote, TEXT)


@pytest.mark.parametrize("quote", ["I cannot log in", "the app is broken", "", "  ...  "])
def test_quote_in_text_rejects_paraphrases_and_empty_quotes(quote):
    assert not quote_in_text(quote, TEXT)


def test_quote_in_text_is_generic_beyond_tickets():
    """MVP 2 reuses this to verify a KB quote against the article body, not just evidence
    against a ticket — the name and signature (`quote`, `text`) are generic on purpose."""
    article = "Open Profile → Birth details and tap the time field to edit it."
    assert quote_in_text("Open Profile → Birth details and tap the time field", article)
    assert not quote_in_text("Open Settings and edit birth time", article)
