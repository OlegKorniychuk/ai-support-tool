"""Prompt <-> code contract tests for the current prompt (v5).

The response shape is enforced by structured output (`LLMClassification`), so the prompt
carries no JSON block. What must not drift is the priority table: the markdown table in
v5.md is parsed and compared with `rules.PRIORITY_TABLE`.

v1–v4 stay in `prompts/` as change history only; they target the pre-v5 schema and are
not tested against it.
"""

import re

import pytest

from support_ai.classifier.classify import PROMPTS_DIR
from support_ai.classifier.rules import PRIORITY_TABLE
from support_ai.classifier.schema import Category, NextStep, Priority
from support_ai.core.config import DEFAULT_PROMPT_VERSION

PROMPT = (PROMPTS_DIR / f"{DEFAULT_PROMPT_VERSION}.md").read_text()
# Tolerates column padding, so a markdown formatter re-aligning the table doesn't break it.
TABLE_ROW = re.compile(
    r"^\|\s*`(?P<category>\w+)`\s*\|\s*`(?P<next_step>\w+)`\s*"
    r"\|\s*(?P<base>P\d)\s*\|\s*(?P<raise>P\d|—)\s*\|"
)


def _parse_priority_table(text: str) -> dict[tuple[Category, NextStep], tuple[Priority, Priority]]:
    table = {}
    for line in text.splitlines():
        match = TABLE_ROW.match(line)
        if not match:
            continue
        base = Priority(match["base"])
        highest = base if match["raise"] == "—" else Priority(match["raise"])
        table[(Category(match["category"]), NextStep(match["next_step"]))] = (base, highest)
    return table


def test_default_prompt_is_v5():
    assert DEFAULT_PROMPT_VERSION == "v5"


def test_default_prompt_is_the_latest_version():
    versions = [int(p.stem.removeprefix("v")) for p in PROMPTS_DIR.glob("v*.md")]
    assert DEFAULT_PROMPT_VERSION == f"v{max(versions)}"


def test_prompt_priority_table_matches_rules():
    assert _parse_priority_table(PROMPT) == PRIORITY_TABLE


@pytest.mark.parametrize("value", [*Category, *NextStep], ids=str)
def test_prompt_mentions_every_enum_value(value):
    assert f"`{value.value}`" in PROMPT


def test_prompt_mentions_every_free_text_field():
    for field in ("next_step_note", "language", "tone", "confidence", "rationale"):
        assert f"`{field}`" in PROMPT


@pytest.mark.parametrize(
    "forbidden", ["## Response format", "requests_human", "secondary_categories"]
)
def test_prompt_has_no_v4_leftovers(forbidden):
    assert forbidden not in PROMPT
