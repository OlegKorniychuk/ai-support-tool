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
    "category": Category.PAYMENT_ISSUE,
    "next_step": NextStep.SEND_REFUND_POLICY,
    "priority": Priority.P2,
    "next_step_note": "Send the refund policy; the customer was charged twice.",
    "language": "uk",
    "tone": "aggressive",
    "confidence": 0.82,
    "rationale": "Customer asks for a refund of a duplicate charge.",
}


def test_llm_classification_accepts_valid_data():
    parsed = LLMClassification.model_validate(VALID_KWARGS)
    assert parsed.category is Category.PAYMENT_ISSUE
    assert parsed.priority is Priority.P2
    assert parsed.next_step is NextStep.SEND_REFUND_POLICY


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


def test_category_enum_is_exactly_the_v5_set():
    # An exhaustive equality check so a removed category can't reappear without also
    # updating this test.
    assert {c.value for c in Category} == {
        "general_question",
        "quality_complaint",
        "expert_complaint",
        "payment_issue",
        "threat",
        "other",
    }


def test_next_step_enum_is_exactly_the_v5_set():
    assert {s.value for s in NextStep} == {
        "send_user_guide",
        "send_kb_answer",
        "generic_reply",
        "record_feature_request",
        "create_bug_ticket",
        "record_expert_complaint",
        "send_refund_policy",
        "escalate_human",
        "no_reply",
    }


def test_llm_classification_generates_next_step_before_priority():
    """Structured output follows field order: the response is picked before priority."""
    fields = list(LLMClassification.model_fields)
    assert fields.index("category") < fields.index("next_step") < fields.index("priority")


@pytest.mark.parametrize("removed_field", ["requests_human", "secondary_categories"])
def test_llm_classification_rejects_removed_v4_fields(removed_field):
    bad = {**VALID_KWARGS, removed_field: False}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


def test_llm_classification_rejects_extra_fields():
    """`extra="forbid"` so an unexpected key is a validation error, not silently dropped —
    this is what lets the gateway's repair retry kick in on a malformed response."""
    bad = {**VALID_KWARGS, "unexpected_field": "surprise"}
    with pytest.raises(ValidationError):
        LLMClassification.model_validate(bad)


def test_classification_adds_review_fields():
    classification = Classification.model_validate(
        {**VALID_KWARGS, "needs_human_review": True, "review_reasons": ["escalate_human"]}
    )
    assert classification.needs_human_review is True
    assert classification.review_reasons == ["escalate_human"]


def test_classification_review_fields_default_to_unflagged():
    classification = Classification.model_validate(VALID_KWARGS)
    assert classification.needs_human_review is False
    assert classification.review_reasons == []
