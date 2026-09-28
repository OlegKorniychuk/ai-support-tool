"""Shared text helpers: ticket wrapping and verbatim-quote matching.

The classifier uses these to wrap the ticket for the prompt and to verify a
`priority_raise_evidence` quote is really present in the ticket. MVP 2 reuses the same two
needs against different text: wrapping the ticket for the reply prompt, and verifying an
LLM-cited quote (a KB quote against the article, or evidence against the ticket) is really
present in its source rather than paraphrased — hence the generic `quote_in_text` name.
"""

import re

# Characters typically added around a quoted sentence that `quote_in_text` should ignore:
# whitespace, straight/curly quote marks, guillemets, ellipsis and trailing punctuation.
_QUOTE_EDGES = " \t\n\"'“”‘’«»….,;:!?"


def wrap_ticket(ticket: str) -> str:
    """Wrap raw ticket text in a delimited block, as a guard against prompt injection."""
    return f"<<<TICKET>>>\n{ticket}\n<<<END TICKET>>>"


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
