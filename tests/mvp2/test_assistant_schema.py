"""Schema-level checks: field order (structured output depends on it), `extra="forbid"`,
and that `AssistResult`'s computed totals actually serialize (the eval runner writes the
result to JSON)."""

import pytest
from pydantic import ValidationError

from support_ai.assistant.schema import AssistResult, LLMReply, StepInfo


def _llm_reply(**overrides) -> dict:
    fields = dict(
        summary="Customer asks how to change their birth time.",
        kb_article_id="edit-birth-data",
        kb_quote="Open Profile → Birth details",
        account_specific_evidence=None,
        conflicting_article_ids=[],
        formal="Dear customer, ...",
        empathetic="I understand ...",
        short="Open Profile → Birth details.",
    )
    fields.update(overrides)
    return fields


def test_llm_reply_field_order_matches_structured_output_contract():
    # Structured output is generated in field-declaration order: the model must commit to
    # source, quote and judgment evidence before writing the drafts. See schema.py.
    assert list(LLMReply.model_fields) == [
        "summary",
        "kb_article_id",
        "kb_quote",
        "account_specific_evidence",
        "conflicting_article_ids",
        "formal",
        "empathetic",
        "short",
    ]


def test_llm_reply_rejects_extra_key():
    with pytest.raises(ValidationError):
        LLMReply.model_validate(_llm_reply(unexpected_field="oops"))


def test_llm_reply_nullable_fields_are_required():
    # No defaults: a missing key is a validation error, not a silently-filled None.
    fields = _llm_reply()
    del fields["kb_article_id"]
    with pytest.raises(ValidationError):
        LLMReply.model_validate(fields)


def test_assist_result_totals_serialize():
    steps = [
        StepInfo(
            step="retrieval", model_used="text-embedding-3-small", latency_ms=100, cost_usd=0.001
        ),
        StepInfo(step="generation", model_used="gpt-5.4-nano", latency_ms=400, cost_usd=0.002),
    ]
    result = AssistResult(
        summary="Customer asks how to change their birth time.",
        needs_human_judgment=False,
        steps=steps,
    )

    assert result.total_latency_ms == 500
    assert result.total_cost_usd == pytest.approx(0.003)

    dumped = result.model_dump()
    assert dumped["total_latency_ms"] == 500
    assert dumped["total_cost_usd"] == pytest.approx(0.003)
    assert "total_latency_ms" in result.model_dump_json()
