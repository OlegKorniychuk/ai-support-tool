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
    "requests_human": False,
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
        ({"priority": Priority.P1}, False, "p1_priority"),
        ({"requests_human": True}, False, "human_requested"),
        ({}, True, "classification_failed"),
    ],
)
def test_each_rule_fires_on_its_own(overrides, failed, expected_reason):
    result = apply_rules(_classification(**overrides), failed=failed)
    assert result.needs_human_review is True
    assert result.review_reasons == [expected_reason]


def test_multiple_rules_combine():
    classification = _classification(priority=Priority.P1, requests_human=True)
    result = apply_rules(classification, failed=True)
    assert result.needs_human_review is True
    assert result.review_reasons == ["p1_priority", "human_requested", "classification_failed"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 0.1},
        {"confidence": 0.0},
        {"category": Category.REFUND_REQUEST},
        {"category": Category.EXPERT_COMPLAINT},
        {"secondary_categories": [Category.BILLING_SUBSCRIPTION]},
    ],
)
def test_removed_rules_no_longer_fire(overrides):
    """Low confidence, refund/expert-complaint category and mixed topics used to flag a
    ticket on their own. None of them do anymore — only P1, requests_human and a failed
    classification do."""
    result = apply_rules(_classification(**overrides))
    assert result.needs_human_review is False
    assert result.review_reasons == []


def test_apply_rules_preserves_llm_fields():
    classification = _classification()
    result = apply_rules(classification)
    assert result.category is classification.category
    assert result.priority is classification.priority
    assert result.rationale == classification.rationale
    assert result.requests_human == classification.requests_human
    assert result.confidence == classification.confidence
    assert result.secondary_categories == classification.secondary_categories


@pytest.mark.parametrize(
    "category,expected_next_step",
    [
        (Category.USAGE_HELP, NextStep.SEND_KB_ARTICLE),
        (Category.UNCLEAR, NextStep.REQUEST_MORE_INFO),
        (Category.GENERAL_FEEDBACK, NextStep.SEND_GENERIC_REPLY),
    ],
)
def test_next_step_is_enforced_for_mapped_categories(category, expected_next_step):
    # the LLM picked the "wrong" next step; rules.py must override it regardless
    classification = _classification(category=category, next_step=NextStep.ROUTE_BILLING)
    result = apply_rules(classification)
    assert result.next_step is expected_next_step


def test_next_step_is_untouched_for_other_categories():
    classification = _classification(
        category=Category.TECHNICAL_BUG, next_step=NextStep.ROUTE_TECH_SUPPORT
    )
    result = apply_rules(classification)
    assert result.next_step is NextStep.ROUTE_TECH_SUPPORT


def test_general_feedback_is_not_flagged_for_review():
    classification = _classification(
        category=Category.GENERAL_FEEDBACK,
        priority=Priority.P4,
        next_step=NextStep.ROUTE_TECH_SUPPORT,
    )
    result = apply_rules(classification)
    assert result.needs_human_review is False
    assert result.next_step is NextStep.SEND_GENERIC_REPLY


def test_general_feedback_with_explicit_human_request_is_still_flagged():
    classification = _classification(category=Category.GENERAL_FEEDBACK, requests_human=True)
    result = apply_rules(classification)
    assert result.needs_human_review is True
    assert result.review_reasons == ["human_requested"]
    assert result.next_step is NextStep.SEND_GENERIC_REPLY
