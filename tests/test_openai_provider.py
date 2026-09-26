"""Unit tests for the OpenAI adapter, with the SDK client mocked. No network calls."""

from types import SimpleNamespace

import httpx2
import openai
import pydantic
import pytest

from support_ai.classifier.schema import LLMClassification
from support_ai.core.llm.errors import (
    LLMInvalidOutput,
    LLMProviderError,
    LLMRateLimited,
    LLMTimeout,
)
from support_ai.core.llm.openai_provider import OpenAIProvider

VALID_KWARGS = {
    "category": "refund_request",
    "secondary_categories": [],
    "priority": "P2",
    "next_step": "route_refunds",
    "next_step_note": "note",
    "language": "en",
    "tone": "neutral",
    "confidence": 0.9,
    "rationale": "why",
}


def _provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.com/v1/responses")


def _call(provider: OpenAIProvider):
    return provider.complete_structured(
        system="sys",
        user="user text",
        schema=LLMClassification,
        model="gpt-5.4-nano",
        timeout_s=20.0,
    )


def _fake_success_response():
    parsed = LLMClassification.model_validate(VALID_KWARGS)
    return SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(input_tokens=42, output_tokens=17),
        model="gpt-5.4-nano",
    )


def test_complete_structured_success(mocker):
    provider = _provider()
    mocker.patch.object(provider._client.responses, "parse", return_value=_fake_success_response())

    result = _call(provider)

    assert result.data["category"] == "refund_request"
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 17
    assert result.model == "gpt-5.4-nano"
    assert result.latency_ms >= 0


def test_complete_structured_maps_timeout(mocker):
    provider = _provider()
    mocker.patch.object(
        provider._client.responses, "parse", side_effect=openai.APITimeoutError(request=_request())
    )
    with pytest.raises(LLMTimeout):
        _call(provider)


def test_complete_structured_maps_rate_limit(mocker):
    provider = _provider()
    resp = httpx2.Response(429, request=_request(), json={"error": {"message": "slow down"}})
    mocker.patch.object(
        provider._client.responses,
        "parse",
        side_effect=openai.RateLimitError(
            "slow down", response=resp, body={"message": "slow down"}
        ),
    )
    with pytest.raises(LLMRateLimited):
        _call(provider)


def test_complete_structured_maps_internal_server_error_to_rate_limited(mocker):
    provider = _provider()
    resp = httpx2.Response(500, request=_request(), json={"error": {"message": "boom"}})
    mocker.patch.object(
        provider._client.responses,
        "parse",
        side_effect=openai.InternalServerError("boom", response=resp, body={"message": "boom"}),
    )
    with pytest.raises(LLMRateLimited):
        _call(provider)


def test_complete_structured_maps_validation_error_to_invalid_output(mocker):
    provider = _provider()
    try:
        LLMClassification.model_validate({})
    except pydantic.ValidationError as exc:
        validation_error = exc
    mocker.patch.object(provider._client.responses, "parse", side_effect=validation_error)
    with pytest.raises(LLMInvalidOutput):
        _call(provider)


def test_complete_structured_maps_refusal_to_invalid_output(mocker):
    provider = _provider()
    refusal_response = SimpleNamespace(
        output_parsed=None,
        incomplete_details=None,
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="policy")])],
        usage=SimpleNamespace(input_tokens=5, output_tokens=1),
        model="gpt-5.4-nano",
    )
    mocker.patch.object(provider._client.responses, "parse", return_value=refusal_response)
    with pytest.raises(LLMInvalidOutput):
        _call(provider)


def test_complete_structured_maps_other_api_error_to_provider_error(mocker):
    provider = _provider()
    resp = httpx2.Response(400, request=_request(), json={"error": {"message": "bad request"}})
    mocker.patch.object(
        provider._client.responses,
        "parse",
        side_effect=openai.BadRequestError(
            "bad request", response=resp, body={"message": "bad request"}
        ),
    )
    with pytest.raises(LLMProviderError):
        _call(provider)
