import pytest

from support_ai.classifier.dataset import load_tickets
from support_ai.classifier.rules import PRIORITY_TABLE, clamp_priority
from support_ai.classifier.schema import NextStep

CASES = load_tickets()


def _tagged(tag: str):
    return [c for c in CASES if tag in c.tags]


def test_loader_parses_all_tickets_with_unique_ids():
    assert len(CASES) >= 40
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids)), "ticket ids must be unique"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_every_label_follows_the_priority_table(case):
    bounds = PRIORITY_TABLE.get((case.expected_category, case.expected_next_step))
    assert bounds, f"{case.id}: invalid (category, next_step) pair"
    assert clamp_priority(case.expected_priority, *bounds) is case.expected_priority, (
        f"{case.id}: {case.expected_priority} is outside {bounds}"
    )


def test_every_category_response_pair_is_covered_at_base_priority():
    base_labels = {
        (c.expected_category, c.expected_next_step)
        for c in _tagged("base")
        if c.expected_priority is PRIORITY_TABLE[(c.expected_category, c.expected_next_step)][0]
    }
    assert base_labels == set(PRIORITY_TABLE)


def test_every_raisable_pair_has_a_raised_ticket():
    raisable = {pair for pair, (base, highest) in PRIORITY_TABLE.items() if base != highest}
    raised = {
        (c.expected_category, c.expected_next_step)
        for c in CASES
        if c.expected_priority is PRIORITY_TABLE[(c.expected_category, c.expected_next_step)][1]
        and c.expected_priority
        is not PRIORITY_TABLE[(c.expected_category, c.expected_next_step)][0]
    }
    assert raised == raisable


@pytest.mark.parametrize(
    "tag,minimum",
    [
        ("multilingual", 2),
        ("mixed", 2),
        ("injection", 2),
        ("too_short", 2),
        ("human_request", 1),
        ("aggressive", 1),
        ("violence", 1),
        ("self_harm", 1),
    ],
)
def test_edge_case_groups_are_present(tag, minimum):
    assert len(_tagged(tag)) >= minimum


def test_multilingual_tickets_cover_more_than_one_language():
    languages = {t for c in _tagged("multilingual") for t in c.tags if len(t) == 2}
    assert {"uk", "es"} <= languages


def test_too_short_tickets_are_other_no_reply_and_not_reviewed():
    for case in _tagged("too_short"):
        assert case.expected_next_step is NextStep.NO_REPLY
        assert not case.expected_needs_review


def test_a_pure_injection_ticket_is_ignored_and_not_reviewed():
    pure = [c for c in _tagged("injection") if c.expected_next_step is NextStep.NO_REPLY]
    assert pure
    assert all(not c.expected_needs_review for c in pure)


def test_a_human_request_alone_does_not_escalate():
    assert all(not c.expected_needs_review for c in _tagged("human_request"))


def test_expected_needs_review_is_derived_from_escalate_human():
    for case in CASES:
        assert case.expected_needs_review is (case.expected_next_step is NextStep.ESCALATE_HUMAN)
