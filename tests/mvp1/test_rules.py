import pytest

from support_ai.classifier.rules import (
    INVALID_PAIR_PRIORITY,
    PRIORITY_TABLE,
    apply_rules,
    compute_priority,
    evidence_in_ticket,
)
from support_ai.classifier.schema import Category, LLMClassification, NextStep, Priority

TICKET = "I can't log in at all since the update. Please fix it!"

BASE_KWARGS = {
    "category": Category.QUALITY_COMPLAINT,
    "next_step": NextStep.CREATE_BUG_TICKET,
    "priority_raise_evidence": None,
    "next_step_note": "Open a bug ticket for the login failure.",
    "language": "en",
    "tone": "neutral",
    "confidence": 0.9,
    "rationale": "A clear bug report.",
}


def _classification(**overrides) -> LLMClassification:
    return LLMClassification.model_validate({**BASE_KWARGS, **overrides})


def test_priority_table_covers_the_16_v5_pairs():
    assert len(PRIORITY_TABLE) == 16
    for base, raised_to in PRIORITY_TABLE.values():
        # `raised_to` is at least as urgent as `base`, and at most one level above it
        assert raised_to <= base
        assert int(base.value[1]) - int(raised_to.value[1]) <= 1


def test_every_category_except_general_question_can_escalate():
    escalating = {c for c, s in PRIORITY_TABLE if s is NextStep.ESCALATE_HUMAN}
    assert escalating == set(Category) - {Category.GENERAL_QUESTION}


@pytest.mark.parametrize("pair,bounds", list(PRIORITY_TABLE.items()))
def test_compute_priority_follows_the_table(pair, bounds):
    assert compute_priority(*pair, raised=False) is bounds[0]
    assert compute_priority(*pair, raised=True) is bounds[1]


def test_compute_priority_for_an_invalid_pair():
    priority = compute_priority(Category.THREAT, NextStep.SEND_KB_ANSWER, raised=True)
    assert priority is INVALID_PAIR_PRIORITY


@pytest.mark.parametrize(
    "evidence",
    [
        "I can't log in at all",
        "i CAN'T   log in at all",
        '"I can\'t log in at all."',
        "“I can't log in at all since the update…”",
    ],
)
def test_evidence_matcher_tolerates_case_whitespace_and_quote_marks(evidence):
    assert evidence_in_ticket(evidence, TICKET)


@pytest.mark.parametrize("evidence", ["I cannot log in", "the app is broken", "", "  ...  "])
def test_evidence_matcher_rejects_paraphrases_and_empty_quotes(evidence):
    assert not evidence_in_ticket(evidence, TICKET)


def test_no_evidence_gives_base_priority():
    result = apply_rules(_classification(), ticket=TICKET)
    assert result.priority is Priority.P3
    assert result.priority_raise_evidence is None
    assert result.needs_human_review is False
    assert result.review_reasons == []


def test_verified_evidence_raises_priority():
    classification = _classification(priority_raise_evidence="I can't log in at all")
    result = apply_rules(classification, ticket=TICKET)
    assert result.priority is Priority.P2
    assert result.priority_raise_evidence == "I can't log in at all"


def test_unverified_evidence_is_dropped_and_priority_stays_base():
    classification = _classification(priority_raise_evidence="all my data was lost")
    result = apply_rules(classification, ticket=TICKET)
    assert result.priority is Priority.P3
    assert result.priority_raise_evidence is None


def test_evidence_on_a_non_raisable_pair_is_dropped():
    classification = _classification(
        category=Category.GENERAL_QUESTION,
        next_step=NextStep.SEND_KB_ANSWER,
        priority_raise_evidence="I can't log in at all",
    )
    result = apply_rules(classification, ticket=TICKET)
    assert result.priority is Priority.P4
    assert result.priority_raise_evidence is None


def test_escalate_human_is_flagged():
    ticket = "My lawyer has already reviewed my case against Nebula."
    classification = _classification(
        category=Category.THREAT,
        next_step=NextStep.ESCALATE_HUMAN,
        priority_raise_evidence="My lawyer has already reviewed my case",
    )
    result = apply_rules(classification, ticket=ticket)
    assert result.priority is Priority.P1
    assert result.needs_human_review is True
    assert result.review_reasons == ["escalate_human"]


def test_invalid_pair_is_flagged_with_a_fixed_priority():
    classification = _classification(
        category=Category.THREAT,
        next_step=NextStep.SEND_KB_ANSWER,
        priority_raise_evidence="I can't log in at all",
    )
    result = apply_rules(classification, ticket=TICKET)
    assert result.needs_human_review is True
    assert result.review_reasons == ["invalid_next_step"]
    assert result.next_step is NextStep.SEND_KB_ANSWER
    assert result.priority is INVALID_PAIR_PRIORITY
    assert result.priority_raise_evidence is None


def test_failed_classification_is_flagged():
    result = apply_rules(_classification(), ticket="", failed=True)
    assert result.needs_human_review is True
    assert result.review_reasons == ["classification_failed"]


def test_reasons_combine_in_order():
    classification = _classification(
        category=Category.GENERAL_QUESTION, next_step=NextStep.ESCALATE_HUMAN
    )
    result = apply_rules(classification, ticket=TICKET, failed=True)
    assert result.review_reasons == [
        "invalid_next_step",
        "escalate_human",
        "classification_failed",
    ]


@pytest.mark.parametrize("confidence", [0.0, 0.1])
def test_low_confidence_does_not_flag(confidence):
    result = apply_rules(_classification(confidence=confidence), ticket=TICKET)
    assert result.needs_human_review is False


def test_apply_rules_preserves_llm_fields():
    classification = _classification()
    result = apply_rules(classification, ticket=TICKET)
    assert result.category is classification.category
    assert result.next_step is classification.next_step
    assert result.next_step_note == classification.next_step_note
    assert result.rationale == classification.rationale
    assert result.confidence == classification.confidence
