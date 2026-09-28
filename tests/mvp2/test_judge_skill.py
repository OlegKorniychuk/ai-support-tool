"""Contract test between the committed `judge-replies` skill and the code it drives:
the rubric must name every `Tone` and state `SHORT_MAX_WORDS`, and the skill's frontmatter
must actually be runnable as a Sonnet skill named `judge-replies`.
"""

import re
from pathlib import Path

from support_ai.assistant.schema import Tone
from support_ai.core.config import SHORT_MAX_WORDS

SKILL_DIR = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "judge-replies"


def _frontmatter(text: str) -> str:
    """The `---`-delimited YAML frontmatter block at the top of a skill/markdown file."""
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "expected a --- frontmatter block at the top of the file"
    return match.group(1)


def test_skill_md_exists():
    assert (SKILL_DIR / "SKILL.md").exists()


def test_rubric_md_exists():
    assert (SKILL_DIR / "rubric.md").exists()


def test_skill_frontmatter_has_name_and_model():
    frontmatter = _frontmatter((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))
    assert re.search(r"^name:\s*judge-replies\s*$", frontmatter, re.MULTILINE)
    assert re.search(r"^model:\s*sonnet\s*$", frontmatter, re.MULTILINE)


def test_skill_frontmatter_has_argument_hint_and_allowed_tools():
    frontmatter = _frontmatter((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))
    assert re.search(r"^argument-hint:", frontmatter, re.MULTILINE)
    assert "allowed-tools:" in frontmatter
    assert "Read" in frontmatter
    assert "Write" in frontmatter
    assert "scripts/record_judgement.py" in frontmatter


def test_skill_references_the_recorder_and_the_rubric():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert "record_judgement.py" in text
    assert "rubric.md" in text


def test_rubric_states_rubric_version_v1():
    text = (SKILL_DIR / "rubric.md").read_text(encoding="utf-8")
    assert "rubric_version: v1" in text


def test_rubric_has_a_heading_for_every_tone():
    text = (SKILL_DIR / "rubric.md").read_text(encoding="utf-8")
    for tone in Tone:
        assert re.search(rf"^#+\s*`?{tone.value}`?\s*$", text, re.MULTILINE | re.IGNORECASE), (
            f"rubric.md has no heading for tone {tone.value!r}"
        )


def test_rubric_states_short_max_words():
    text = (SKILL_DIR / "rubric.md").read_text(encoding="utf-8")
    assert str(SHORT_MAX_WORDS) in text


def test_rubric_defines_faithful_addresses_request_summary_accurate_and_distinct():
    text = (SKILL_DIR / "rubric.md").read_text(encoding="utf-8")
    for term in ["faithful", "addresses_request", "summary_accurate", "distinct"]:
        assert term in text, f"rubric.md never defines {term!r}"


def test_rubric_stresses_judging_drafts_not_retrieval():
    text = (SKILL_DIR / "rubric.md").read_text(encoding="utf-8")
    assert "not whether retrieval" in text.lower()
