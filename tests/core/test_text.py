import pytest

from support_ai.core.text import normalize, quote_in_text, strip_markers, wrap_ticket

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


def test_strip_markers_drops_whole_wrapped_ticket_block():
    whole_block = wrap_ticket("My Premium access just stopped working yesterday.")
    assert strip_markers(whole_block) == "My Premium access just stopped working yesterday."


def test_strip_markers_drops_kb_block_markers():
    kb_block = "<<<KB id=edit-birth-data title=Edit birth data>>>\nBody text.\n<<<END KB>>>"
    assert strip_markers(kb_block) == "Body text."


def test_strip_markers_leaves_plain_text_untouched():
    assert strip_markers("Just a normal quote, no markers here.") == (
        "Just a normal quote, no markers here."
    )


def test_strip_markers_removes_inline_marker_not_on_its_own_line():
    assert strip_markers("before <<<TICKET>>> after") == "before  after"


def test_strip_markers_result_still_verifies_with_quote_in_text():
    evidence = wrap_ticket("I was charged twice on March 3rd for my subscription.")
    stripped = strip_markers(evidence)
    assert quote_in_text("charged twice on March 3rd", stripped)
