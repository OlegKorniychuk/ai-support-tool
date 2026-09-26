import pytest

from support_ai.classifier.rules import apply_rules
from support_ai.classifier.schema import Category, LLMClassification, NextStep, Priority

BASE_KWARGS = {
    "category": Category.TECHNICAL_BUG,
    "secondary_categories": [],
    "priority": Priority.P3,
    "next_step": NextStep.ROUTE_TECH_SUPPORT,
    "next_step_note": "Investigate the crash report.",
    "language": "en",
    "tone": "neutral",
    "confidence": 0.9,
    "rationale": "A clear technical bug report.",
}


def _classification(**overrides) -> LLMClassification:
    return LLMClassification.model_validate({**BASE_KWARGS, **overrides})


def test_no_rules_firing_leaves_ticket_unflagged():
    result = apply_rules(_classification())
    assert result.needs_human_review is False
    assert result.review_reasons == []


@pytest.mark.parametrize(
    "overrides,failed,expected_reason",
    [
        ({"confidence": 0.5}, False, "low_confidence"),
        ({"confidence": 0.69}, False, "low_confidence"),
        ({"category": Category.REFUND_REQUEST}, False, "refund_request"),
        ({"category": Category.EXPERT_COMPLAINT}, False, "expert_complaint"),
        ({"priority": Priority.P1}, False, "p1_priority"),
        ({"secondary_categories": [Category.BILLING_SUBSCRIPTION]}, False, "mixed_topics"),
        ({}, True, "classification_failed"),
    ],
)
def test_each_rule_fires_on_its_own(overrides, failed, expected_reason):
    result = apply_rules(_classification(**overrides), failed=failed)
    assert result.needs_human_review is True
    assert result.review_reasons == [expected_reason]


def test_confidence_at_threshold_does_not_flag():
    # threshold is exclusive: confidence == 0.7 should not trigger low_confidence
    result = apply_rules(_classification(confidence=0.7))
    assert result.needs_human_review is False
    assert result.review_reasons == []


def test_confidence_threshold_is_configurable():
    result = apply_rules(_classification(confidence=0.8), confidence_threshold=0.9)
    assert result.needs_human_review is True
    assert result.review_reasons == ["low_confidence"]


def test_multiple_rules_combine():
    classification = _classification(
        category=Category.REFUND_REQUEST,
        priority=Priority.P1,
        secondary_categories=[Category.EXPERT_COMPLAINT],
        confidence=0.4,
    )
    result = apply_rules(classification)
    assert result.needs_human_review is True
    assert result.review_reasons == [
        "low_confidence",
        "refund_request",
        "p1_priority",
        "mixed_topics",
    ]


def test_failed_combines_with_other_firing_rules():
    # the SPEC.md fallback result has confidence=0, which also trips low_confidence
    classification = _classification(category=Category.OTHER, priority=Priority.P3, confidence=0.0)
    result = apply_rules(classification, failed=True)
    assert result.needs_human_review is True
    assert result.review_reasons == ["low_confidence", "classification_failed"]


def test_apply_rules_preserves_llm_fields():
    classification = _classification()
    result = apply_rules(classification)
    assert result.category is classification.category
    assert result.priority is classification.priority
    assert result.next_step is classification.next_step
    assert result.rationale == classification.rationale
