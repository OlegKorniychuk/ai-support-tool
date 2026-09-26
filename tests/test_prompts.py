"""Prompt <-> schema contract test.

`prompts/v2.md` ends with a "Response format" section listing the exact JSON shape the
LLM must return. This test parses that block and checks its keys against
`LLMClassification.model_fields`, so the prompt and the schema can never silently drift
apart.
"""

import json
import re

from support_ai.classifier.classify import PROMPTS_DIR
from support_ai.classifier.schema import LLMClassification

RESPONSE_FORMAT_BLOCK = re.compile(r"## Response format.*?```json\n(?P<shape>.*?)\n```", re.DOTALL)


def _response_shape_keys(prompt_version: str) -> set[str]:
    text = (PROMPTS_DIR / f"{prompt_version}.md").read_text()
    match = RESPONSE_FORMAT_BLOCK.search(text)
    assert match, f"no '## Response format' fenced JSON block found in {prompt_version}.md"
    shape = json.loads(match.group("shape"))
    return set(shape.keys())


def test_v2_response_shape_keys_match_llm_classification_fields():
    assert _response_shape_keys("v2") == set(LLMClassification.model_fields.keys())
