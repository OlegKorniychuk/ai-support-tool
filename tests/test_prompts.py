"""Prompt <-> schema contract test.

`prompts/v2.md` ends with a "Response format" section listing the exact JSON shape the
LLM must return. This test parses that block and checks its keys against
`LLMClassification.model_fields`, so the prompt and the schema can never silently drift
apart.
"""

import json
import re

from support_ai.classifier.classify import PROMPTS_DIR
from support_ai.classifier.schema import Category, LLMClassification, NextStep

RESPONSE_FORMAT_BLOCK = re.compile(r"## Response format.*?```json\n(?P<shape>.*?)\n```", re.DOTALL)


def _response_shape_keys(prompt_version: str) -> set[str]:
    text = (PROMPTS_DIR / f"{prompt_version}.md").read_text()
    match = RESPONSE_FORMAT_BLOCK.search(text)
    assert match, f"no '## Response format' fenced JSON block found in {prompt_version}.md"
    shape = json.loads(match.group("shape"))
    return set(shape.keys())


def test_v2_response_shape_keys_match_llm_classification_fields():
    assert _response_shape_keys("v2") == set(LLMClassification.model_fields.keys())


def test_v3_response_shape_keys_match_llm_classification_fields():
    assert _response_shape_keys("v3") == set(LLMClassification.model_fields.keys())


def test_v3_response_shape_lists_every_next_step():
    text = (PROMPTS_DIR / "v3.md").read_text()
    shape = json.loads(RESPONSE_FORMAT_BLOCK.search(text).group("shape"))
    assert all(step.value in shape["next_step"] for step in NextStep)


def test_v3_response_shape_lists_every_category():
    text = (PROMPTS_DIR / "v3.md").read_text()
    shape = json.loads(RESPONSE_FORMAT_BLOCK.search(text).group("shape"))
    assert all(category.value in shape["category"] for category in Category)


def test_v4_response_shape_matches_schema_and_enums():
    assert _response_shape_keys("v4") == set(LLMClassification.model_fields.keys())
    text = (PROMPTS_DIR / "v4.md").read_text()
    shape = json.loads(RESPONSE_FORMAT_BLOCK.search(text).group("shape"))
    assert all(step.value in shape["next_step"] for step in NextStep)
    assert all(category.value in shape["category"] for category in Category)


def test_v4_priority_is_a_first_match_cascade():
    text = (PROMPTS_DIR / "v4.md").read_text()
    assert "stop at the first match" in text
    assert (
        text.index("1. `P1`")
        < text.index("2. `P2`")
        < text.index("3. `P3`")
        < text.index("4. `P4`")
    )
