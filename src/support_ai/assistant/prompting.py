"""Builds the reply-generation prompt: loading the static system prompt and assembling the
user message (wrapped ticket + retrieved KB articles) the model actually sees.

Kept pure — no LLM calls, the only I/O is reading a prompt file — so it's cheaply unit
tested and reusable between `assist.py` and `scripts/assist_one.py`.
"""

from pathlib import Path

from support_ai.core.text import wrap_ticket
from support_ai.kb.base import KBHit

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, version: str) -> str:
    """Read `prompts/{name}_{version}.md`. Raises `FileNotFoundError` naming the path that
    doesn't exist, e.g. `load_prompt("reply", "v99")`."""
    path = PROMPTS_DIR / f"{name}_{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file at {path}")
    return path.read_text()


def build_user_message(ticket: str, hits: list[KBHit]) -> str:
    """The wrapped ticket, followed by one `<<<KB id=... title=...>>>` block per hit.

    No KB blocks at all when `hits` is empty (e.g. retrieval found nothing, or a
    summary-only call that never looked up KB candidates).
    """
    parts = [wrap_ticket(ticket)]
    for hit in hits:
        article = hit.article
        parts.append(
            f"<<<KB id={article.id} title={article.title}>>>\n{article.text}\n<<<END KB>>>"
        )
    return "\n\n".join(parts)
