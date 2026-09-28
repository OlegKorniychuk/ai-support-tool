"""Shared text helpers: ticket wrapping and verbatim-quote matching.

The classifier uses these to wrap the ticket for the prompt and to verify a
`priority_raise_evidence` quote is really present in the ticket. MVP 2 reuses the same two
needs against different text: wrapping the ticket for the reply prompt, and verifying an
LLM-cited quote (a KB quote against the article, or evidence against the ticket) is really
present in its source rather than paraphrased — hence the generic `quote_in_text` name.
MVP 2 also reuses `strip_markers`: an LLM field can accidentally echo a whole wrapped block
(prompt markers included), which would otherwise never verify against the real content.
"""

import re

# Characters typically added around a quoted sentence that `quote_in_text` should ignore:
# whitespace, straight/curly quote marks, guillemets, ellipsis and trailing punctuation.
_QUOTE_EDGES = " \t\n\"'“”‘’«»….,;:!?"

# A `<<<...>>>` prompt delimiter: `<<<TICKET>>>`, `<<<END TICKET>>>`, `<<<KB id=... title=...>>>`,
# `<<<END KB>>>`. No literal `<` or `>` is ever used inside one, so this is unambiguous.
_MARKER_LINE = re.compile(r"^[ \t]*<<<[^\n<>]*>>>[ \t]*$")
_MARKER_TOKEN = re.compile(r"<<<[^\n<>]*>>>")


def wrap_ticket(ticket: str) -> str:
    """Wrap raw ticket text in a delimited block, as a guard against prompt injection."""
    return f"<<<TICKET>>>\n{ticket}\n<<<END TICKET>>>"


def strip_markers(text: str) -> str:
    """Remove `<<<...>>>` prompt-delimiter markers from `text`.

    A line that's just a marker (whitespace aside) is dropped entirely, not left behind as
    a blank line; a marker embedded inline is cut out in place. Meant for an LLM-reported
    field that accidentally echoes a whole wrapped block (e.g. the entire
    `<<<TICKET>>>...<<<END TICKET>>>` ticket as "evidence") so the real content underneath
    still verifies with `quote_in_text`.
    """
    lines = [line for line in text.splitlines() if not _MARKER_LINE.match(line)]
    return _MARKER_TOKEN.sub("", "\n".join(lines)).strip()


def normalize(text: str) -> str:
    """Collapse whitespace runs and casefold, for lenient text comparison."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """True if `quote` appears verbatim in `text`.

    Lenient only about what quoting typically changes: case, runs of whitespace, and
    surrounding quote marks, ellipses or punctuation. Paraphrases do not match.
    """
    stripped = normalize(quote).strip(_QUOTE_EDGES)
    return bool(stripped) and stripped in normalize(text)
