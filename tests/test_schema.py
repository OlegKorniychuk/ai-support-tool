import pytest
from pydantic import ValidationError

from support_ai.classifier.schema import (
    Category,
    Classification,
    LLMClassification,
    NextStep,
    Priority,
)

VALID_KWARGS = {
    "category": Category.REFUND_REQUEST,
    "secondary_categories": [Category.EXPERT_COMPLAINT],
    "priority": Priority.P2,
    "next_step": NextStep.ROUTE_REFUNDS,
    "next_step_note": "Verify last charge.",
    "language": "uk",
    "tone": "aggressive",
    "confidence": 0.82,
    "requests_human": False,
    "rationale": "Customer asks for a refund and complains about an expert.",
}


def test_llm_classification_accepts_valid_data():
    parsed = LLMClassification.model_validate(VALID_KWARGS)
    assert parsed.category is Category.REFUND_REQUEST
    assert parsed.priority is Priority.P2
    assert parsed.next_step is NextStep.ROUTE_REFUNDS
    assert parsed.requests_human is False


def test_llm_classification_rejects_invalid_category():
    bad = {**VALID_KWARGS, "category": "not_a_category"}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


def test_llm_classification_rejects_invalid_priority():
    bad = {**VALID_KWARGS, "priority": "P9"}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


def test_llm_classification_rejects_invalid_next_step():
    bad = {**VALID_KWARGS, "next_step": "do_something_else"}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
def test_llm_classification_rejects_confidence_out_of_range(confidence):
    bad = {**VALID_KWARGS, "confidence": confidence}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 0.5])
def test_llm_classification_accepts_confidence_boundaries(confidence):
    ok = {**VALID_KWARGS, "confidence": confidence}
    parsed = LLMClassification.model_validate(ok)
    assert parsed.confidence == confidence


def test_llm_classification_schema_has_no_review_fields():
    schema = LLMClassification.model_json_schema()
    assert "needs_human_review" not in schema["properties"]
    assert "review_reasons" not in schema["properties"]


def test_category_enum_is_exactly_the_expected_set():
    # An exhaustive equality check (rather than "not in") so the removed category can't
    # reappear without also updating this test.
    values = {c.value for c in Category}
    assert values == {
        "billing_subscription",
        "refund_request",
        "technical_bug",
        "expert_complaint",
        "account_access",
        "feature_request",
        "usage_help",
        "general_feedback",
        "unclear",
    }


def test_llm_classification_accepts_usage_help_and_unclear_categories():
    for category in (Category.USAGE_HELP, Category.UNCLEAR):
        parsed = LLMClassification.model_validate({**VALID_KWARGS, "category": category})
        assert parsed.category is category


@pytest.mark.parametrize("value", [True, False])
def test_llm_classification_accepts_requests_human(value):
    parsed = LLMClassification.model_validate({**VALID_KWARGS, "requests_human": value})
    assert parsed.requests_human is value


def test_llm_classification_requires_requests_human():
    missing = {k: v for k, v in VALID_KWARGS.items() if k != "requests_human"}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(missing)


def test_llm_classification_rejects_extra_fields():
    """`extra="forbid"` so an unexpected key is a validation error, not silently dropped —
    this is what lets the gateway's repair retry kick in on a malformed response."""
    bad = {**VALID_KWARGS, "unexpected_field": "surprise"}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


def test_classification_adds_review_fields():
    classification = Classification.model_validate(
        {**VALID_KWARGS, "needs_human_review": True, "review_reasons": ["p1_priority"]}
    )
    assert classification.needs_human_review is True
    assert classification.review_reasons == ["p1_priority"]


def test_classification_review_fields_default_to_unflagged():
    classification = Classification.model_validate(VALID_KWARGS)
    assert classification.needs_human_review is False
    assert classification.review_reasons == []
