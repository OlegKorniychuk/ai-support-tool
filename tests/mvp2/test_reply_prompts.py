"""Prompt <-> code contract tests for the current reply prompt.

The response shape is enforced by structured output (`LLMReply` / `LLMSummary`), so the
prompt carries no JSON block. What must not drift: every `Tone` and every `LLMReply` field
is mentioned, and the `short` tone's own word limit matches `config.SHORT_MAX_WORDS`.

Older prompt versions stay in `prompts/` as change history only; they are not tested
against the current schema.
"""

import re

import pytest

from support_ai.assistant.prompting import PROMPTS_DIR, build_user_message, load_prompt
from support_ai.assistant.schema import LLMReply, Tone
from support_ai.core.config import (
    DEFAULT_REPLY_PROMPT_VERSION,
    SHORT_MAX_WORDS,
    SUMMARY_PROMPT_VERSION,
)
from support_ai.core.text import wrap_ticket
from support_ai.kb.base import Article, KBHit

REPLY_PROMPT = load_prompt("reply", DEFAULT_REPLY_PROMPT_VERSION)
# A "section" heading: a markdown heading whose whole text is one backtick-optional word
# (a field name or a tone), e.g. "### `short`" or "### summary". Multi-name headings like
# "### `formal`, `empathetic`, `short`" don't match and aren't sections in this sense.
_HEADING_RE = re.compile(r"^#{1,6}\s*`?(\w+)`?\s*$", re.MULTILINE)


def _sections(text: str) -> dict[str, str]:
    """Map each single-word heading to the prose between it and the next such heading."""
    matches = list(_HEADING_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[start:end]
    return sections


REPLY_SECTIONS = _sections(REPLY_PROMPT)


def test_default_reply_prompt_is_the_latest_version():
    versions = [int(p.stem.removeprefix("reply_v")) for p in PROMPTS_DIR.glob("reply_v*.md")]
    assert DEFAULT_REPLY_PROMPT_VERSION == f"v{max(versions)}"


def test_summary_prompt_version_is_the_latest_version():
    versions = [int(p.stem.removeprefix("summary_v")) for p in PROMPTS_DIR.glob("summary_v*.md")]
    assert SUMMARY_PROMPT_VERSION == f"v{max(versions)}"


@pytest.mark.parametrize("tone", list(Tone))
def test_every_tone_is_a_heading_in_the_reply_prompt(tone):
    assert tone.value in REPLY_SECTIONS


def test_short_tone_word_limit_matches_config():
    match = re.search(r"(\d+)\s+words", REPLY_SECTIONS[Tone.SHORT.value])
    assert match is not None, "no '<number> words' phrase found under the short heading"
    assert int(match.group(1)) == SHORT_MAX_WORDS


@pytest.mark.parametrize("field", list(LLMReply.model_fields))
def test_reply_prompt_mentions_every_llmreply_field(field):
    assert f"`{field}`" in REPLY_PROMPT


def test_load_prompt_reads_the_versioned_file():
    assert load_prompt("reply", DEFAULT_REPLY_PROMPT_VERSION) == REPLY_PROMPT


def test_load_prompt_missing_file_raises_a_clear_error():
    with pytest.raises(FileNotFoundError, match="reply_v999"):
        load_prompt("reply", "v999")


def test_build_user_message_wraps_the_ticket_and_adds_no_kb_blocks_when_hits_is_empty():
    message = build_user_message("Hello there", [])

    assert message == wrap_ticket("Hello there")
    assert "<<<KB" not in message
    assert "<<<END KB>>>" not in message


def test_build_user_message_one_block_per_hit_with_id_and_title_in_order():
    hits = [
        KBHit(article=Article(id="a1", title="Alpha", text="alpha body"), score=0.9),
        KBHit(article=Article(id="a2", title="Beta", text="beta body"), score=0.5),
    ]

    message = build_user_message("Hello", hits)

    assert message.count("<<<KB id=") == 2
    assert message.count("<<<END KB>>>") == 2
    assert "<<<KB id=a1 title=Alpha>>>\nalpha body\n<<<END KB>>>" in message
    assert "<<<KB id=a2 title=Beta>>>\nbeta body\n<<<END KB>>>" in message
    assert message.index("<<<TICKET>>>") < message.index("<<<KB id=a1")
    assert message.index("<<<KB id=a1") < message.index("<<<KB id=a2")
