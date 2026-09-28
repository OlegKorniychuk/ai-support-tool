"""Checks on the reply-assistant test set (`data/reply_tickets.jsonl`): composition,
label consistency, and that every cited KB id actually exists in `data/kb/`.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from support_ai.assistant.dataset import DEFAULT_REPLY_DATASET_PATH, ReplyCase, load_reply_cases
from support_ai.kb.loader import load_articles

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"
KB_IDS = {article.id for article in load_articles(KB_DIR)}
CASES = load_reply_cases()

# The two planted contradictions (SPEC_MVP2.md); a `conflicting_kb` case must name both
# articles of (at least) one of these pairs.
PLANTED_PAIRS = [
    {"subscription-plans", "cancel-subscription"},
    {"booking-expert-session", "missed-expert-session"},
]


def _cases_tagged(tag: str) -> list[ReplyCase]:
    return [case for case in CASES if tag in case.tags]


def test_loads_thirty_cases():
    assert len(CASES) == 30


def test_ids_are_unique():
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))


def test_every_expected_kb_id_exists_in_kb():
    for case in CASES:
        for kb_id in case.expected_kb_ids:
            assert kb_id in KB_IDS, f"{case.id}: unknown KB id {kb_id!r}"


@pytest.mark.parametrize(
    "tag,count",
    [
        ("answerable", 20),
        ("kb_gap", 3),
        ("account_specific", 3),
        ("conflicting_kb", 3),
        ("injection", 1),
        ("non_english", 3),
        ("near_dup", 4),  # "at least 4"; checked as a lower bound below instead
    ],
)
def test_tag_composition(tag, count):
    actual = len(_cases_tagged(tag))
    if tag == "near_dup":
        assert actual >= count
    else:
        assert actual == count


def test_kb_gap_cases_have_no_kb_ids_and_only_kb_not_found():
    for case in _cases_tagged("kb_gap"):
        assert case.expected_kb_ids == []
        assert case.expected_reasons == ["kb_not_found"]
        assert case.expected_needs_judgment is True


def test_account_specific_cases_are_flagged_with_only_that_reason():
    for case in _cases_tagged("account_specific"):
        assert case.expected_reasons == ["account_specific"]
        assert case.expected_needs_judgment is True
        msg = f"{case.id}: account_specific case should still cite an article"
        assert case.expected_kb_ids, msg


def test_conflicting_kb_cases_name_both_articles_of_a_planted_pair():
    for case in _cases_tagged("conflicting_kb"):
        assert case.expected_reasons == ["conflicting_kb"]
        assert case.expected_needs_judgment is True
        assert len(case.expected_kb_ids) >= 2
        ids = set(case.expected_kb_ids)
        msg = f"{case.id}: {ids} matches no planted pair"
        assert any(pair <= ids for pair in PLANTED_PAIRS), msg


def test_injection_case_is_labeled_answerable_not_flagged():
    injection_cases = _cases_tagged("injection")
    assert len(injection_cases) == 1
    case = injection_cases[0]
    assert case.expected_needs_judgment is False
    assert case.expected_reasons == []
    assert case.expected_kb_ids


def test_answerable_cases_are_not_flagged():
    for case in _cases_tagged("answerable"):
        assert case.expected_needs_judgment is False
        assert case.expected_reasons == []
        assert case.expected_kb_ids


def test_default_path_points_at_the_committed_dataset():
    assert DEFAULT_REPLY_DATASET_PATH.name == "reply_tickets.jsonl"
    assert DEFAULT_REPLY_DATASET_PATH.exists()


def test_validator_rejects_inconsistent_case():
    with pytest.raises(ValidationError):
        ReplyCase.model_validate(
            {
                "id": "bad",
                "text": "anything",
                "expected_kb_ids": [],
                "expected_needs_judgment": True,
                "expected_reasons": [],  # inconsistent: flagged but no reasons
                "tags": [],
            }
        )
