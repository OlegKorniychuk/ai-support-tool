"""classify() pipeline tests, using FakeProvider. No network calls."""

import itertools

import pytest
from fakes import FakeProvider, make_result, register_fake

from support_ai.classifier.classify import classify
from support_ai.core.config import ModelConfig
from support_ai.core.llm.errors import LLMProviderError

_name_counter = itertools.count()

VALID_DATA = {
    "category": "refund_request",
    "secondary_categories": [],
    "priority": "P2",
    "next_step": "route_refunds",
    "next_step_note": "Verify the charge.",
    "language": "en",
    "tone": "neutral",
    "confidence": 0.9,
    "rationale": "Explicit refund ask.",
}


def _fake_chain(fake: FakeProvider, monkeypatch, *, model_id: str = "fake-model") -> list[str]:
    """Point a one-model chain at `fake`, without touching the real 'openai' registration."""
    provider_name = f"fake-classify-{next(_name_counter)}"
    register_fake(provider_name, fake)
    model_config = ModelConfig(
        provider=provider_name,
        model_id=model_id,
        input_price_per_1m=0.2,
        output_price_per_1m=1.25,
        timeout_s=1.0,
    )
    monkeypatch.setattr("support_ai.classifier.classify.MODEL_REGISTRY", {model_id: model_config})
    return [model_id]


def test_classify_normal_path(monkeypatch):
    fake = FakeProvider(responses=[make_result(VALID_DATA, input_tokens=100, output_tokens=50)])
    chain = _fake_chain(fake, monkeypatch)

    result = classify("I want a refund", model_chain=chain)

    assert result.classification.category.value == "refund_request"
    assert result.classification.needs_human_review is True  # refund_request always flags
    assert result.model_used == chain[0]
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.cost_usd > 0
    assert result.attempts[-1].outcome == "success"


def test_classify_fallback_path_on_all_models_failed(monkeypatch):
    fake = FakeProvider(
        responses=[LLMProviderError("boom"), LLMProviderError("boom"), LLMProviderError("boom")]
    )
    chain = _fake_chain(fake, monkeypatch)

    result = classify("Some ticket text", model_chain=chain, sleep=lambda _seconds: None)

    assert result.classification.category.value == "other"
    assert result.classification.priority.value == "P3"
    assert result.classification.needs_human_review is True
    assert "classification_failed" in result.classification.review_reasons
    assert result.model_used == "none"
    assert result.cost_usd == 0.0


@pytest.mark.parametrize("ticket_text", ["", "   ", "\n\t"])
def test_classify_empty_ticket_path(ticket_text):
    result = classify(ticket_text)

    assert result.classification.category.value == "other"
    assert result.classification.needs_human_review is True
    assert "classification_failed" in result.classification.review_reasons
    assert result.model_used == "none"
    assert result.cost_usd == 0.0


def test_classify_never_raises_when_api_key_missing(monkeypatch):
    """A missing key fails inside provider construction, outside the gateway's own retry
    loop. classify() must still never raise (SPEC.md: 'It never raises'). No network call
    happens: the OpenAI SDK rejects a missing key before any HTTP request is made."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = classify("A short ticket", sleep=lambda _seconds: None)

    assert result.model_used == "none"
    assert result.classification.needs_human_review is True
    assert "classification_failed" in result.classification.review_reasons
