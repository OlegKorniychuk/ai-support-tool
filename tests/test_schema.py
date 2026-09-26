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
    "rationale": "Customer asks for a refund and complains about an expert.",
}


def test_llm_classification_accepts_valid_data():
    parsed = LLMClassification.model_validate(VALID_KWARGS)
    assert parsed.category is Category.REFUND_REQUEST
    assert parsed.priority is Priority.P2
    assert parsed.next_step is NextStep.ROUTE_REFUNDS


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


def test_classification_adds_review_fields():
    classification = Classification.model_validate(
        {**VALID_KWARGS, "needs_human_review": True, "review_reasons": ["refund_request"]}
    )
    assert classification.needs_human_review is True
    assert classification.review_reasons == ["refund_request"]


def test_classification_review_fields_default_to_unflagged():
    classification = Classification.model_validate(VALID_KWARGS)
    assert classification.needs_human_review is False
    assert classification.review_reasons == []
