import pytest

from support_ai.classifier.rules import PRIORITY_TABLE, apply_rules
from support_ai.classifier.schema import Category, LLMClassification, NextStep, Priority

BASE_KWARGS = {
    "category": Category.QUALITY_COMPLAINT,
    "next_step": NextStep.CREATE_BUG_TICKET,
    "priority": Priority.P3,
    "next_step_note": "Open a bug ticket for the crash.",
    "language": "en",
    "tone": "neutral",
    "confidence": 0.9,
    "rationale": "A clear bug report.",
}


def _classification(**overrides) -> LLMClassification:
    return LLMClassification.model_validate({**BASE_KWARGS, **overrides})


def test_priority_table_covers_the_16_v5_pairs():
    assert len(PRIORITY_TABLE) == 16
    for base, highest in PRIORITY_TABLE.values():
        # `highest` is at least as urgent as `base`, and at most one level above it
        assert highest <= base
        assert int(base.value[1]) - int(highest.value[1]) <= 1


def test_every_category_except_general_question_can_escalate():
    escalating = {c for c, s in PRIORITY_TABLE if s is NextStep.ESCALATE_HUMAN}
    assert escalating == set(Category) - {Category.GENERAL_QUESTION}


def test_valid_non_escalating_ticket_is_unflagged():
    result = apply_rules(_classification())
    assert result.needs_human_review is False
    assert result.review_reasons == []


@pytest.mark.parametrize(
    "llm_priority,expected",
    [
        (Priority.P4, Priority.P3),  # below base → base
        (Priority.P3, Priority.P3),  # base
        (Priority.P2, Priority.P2),  # raised one level
        (Priority.P1, Priority.P2),  # above the raise limit → limit
    ],
)
def test_priority_is_clamped_into_the_pair_range(llm_priority, expected):
    result = apply_rules(_classification(priority=llm_priority))
    assert result.priority is expected


@pytest.mark.parametrize("llm_priority", list(Priority))
def test_non_raisable_pair_always_gets_its_base(llm_priority):
    classification = _classification(
        category=Category.GENERAL_QUESTION,
        next_step=NextStep.SEND_KB_ANSWER,
        priority=llm_priority,
    )
    assert apply_rules(classification).priority is Priority.P4


@pytest.mark.parametrize("pair,bounds", list(PRIORITY_TABLE.items()))
def test_every_pair_keeps_an_in_range_priority_unchanged(pair, bounds):
    category, next_step = pair
    for priority in set(bounds):
        classification = _classification(category=category, next_step=next_step, priority=priority)
        assert apply_rules(classification).priority is priority


def test_escalate_human_is_flagged():
    classification = _classification(
        category=Category.THREAT, next_step=NextStep.ESCALATE_HUMAN, priority=Priority.P1
    )
    result = apply_rules(classification)
    assert result.needs_human_review is True
    assert result.review_reasons == ["escalate_human"]


def test_p1_alone_adds_no_reason_beyond_escalation():
    classification = _classification(
        category=Category.PAYMENT_ISSUE, next_step=NextStep.ESCALATE_HUMAN, priority=Priority.P1
    )
    result = apply_rules(classification)
    assert result.priority is Priority.P1
    assert result.review_reasons == ["escalate_human"]


def test_invalid_pair_is_flagged_and_left_untouched():
    classification = _classification(
        category=Category.THREAT, next_step=NextStep.SEND_KB_ANSWER, priority=Priority.P1
    )
    result = apply_rules(classification)
    assert result.needs_human_review is True
    assert result.review_reasons == ["invalid_next_step"]
    assert result.next_step is NextStep.SEND_KB_ANSWER
    assert result.priority is Priority.P1


def test_failed_classification_is_flagged():
    result = apply_rules(_classification(), failed=True)
    assert result.needs_human_review is True
    assert result.review_reasons == ["classification_failed"]


def test_reasons_combine_in_order():
    classification = _classification(
        category=Category.GENERAL_QUESTION, next_step=NextStep.ESCALATE_HUMAN
    )
    result = apply_rules(classification, failed=True)
    assert result.review_reasons == [
        "invalid_next_step",
        "escalate_human",
        "classification_failed",
    ]


@pytest.mark.parametrize("confidence", [0.0, 0.1])
def test_low_confidence_does_not_flag(confidence):
    result = apply_rules(_classification(confidence=confidence))
    assert result.needs_human_review is False


def test_apply_rules_preserves_llm_fields():
    classification = _classification()
    result = apply_rules(classification)
    assert result.category is classification.category
    assert result.next_step is classification.next_step
    assert result.next_step_note == classification.next_step_note
    assert result.rationale == classification.rationale
    assert result.confidence == classification.confidence
