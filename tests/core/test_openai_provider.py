"""Unit tests for the OpenAI adapter, with the SDK client mocked. No network calls."""

from types import SimpleNamespace

import httpx2
import openai
import pydantic
import pytest
from fakes import FakeProvider

from support_ai.classifier.schema import LLMClassification
from support_ai.core.llm.base import LLMProvider
from support_ai.core.llm.errors import (
    LLMInvalidOutput,
    LLMProviderError,
    LLMRateLimited,
    LLMTimeout,
)
from support_ai.core.llm.openai_provider import OpenAIProvider

VALID_KWARGS = {
    "category": "payment_issue",
    "next_step": "send_refund_policy",
    "priority_raise_evidence": None,
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


def _fake_success_response(input_tokens_details=None):
    parsed = LLMClassification.model_validate(VALID_KWARGS)
    return SimpleNamespace(
        output_parsed=parsed,
        usage=SimpleNamespace(
            input_tokens=42, output_tokens=17, input_tokens_details=input_tokens_details
        ),
        model="gpt-5.4-nano",
    )


def test_complete_structured_success(mocker):
    provider = _provider()
    mocker.patch.object(provider._client.responses, "parse", return_value=_fake_success_response())

    result = _call(provider)

    assert result.data["category"] == "payment_issue"
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 17
    assert result.usage.cached_input_tokens == 0
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


def test_complete_structured_maps_cached_tokens(mocker):
    provider = _provider()
    response = _fake_success_response(input_tokens_details=SimpleNamespace(cached_tokens=30))
    mocker.patch.object(provider._client.responses, "parse", return_value=response)

    assert _call(provider).usage.cached_input_tokens == 30


def test_complete_structured_null_cached_tokens_default_to_zero(mocker):
    provider = _provider()
    response = _fake_success_response(input_tokens_details=SimpleNamespace(cached_tokens=None))
    mocker.patch.object(provider._client.responses, "parse", return_value=response)

    assert _call(provider).usage.cached_input_tokens == 0


def _embed(provider: OpenAIProvider, texts: tuple[str, ...] = ("hello", "world")):
    return provider.embed(texts=list(texts), model="text-embedding-3-small", timeout_s=10.0)


def _fake_embedding_response(vectors, indices=None, model="text-embedding-3-small", usage=True):
    indices = range(len(vectors)) if indices is None else indices
    data = [
        SimpleNamespace(embedding=vec, index=idx) for vec, idx in zip(vectors, indices, strict=True)
    ]
    return SimpleNamespace(
        data=data,
        model=model,
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7) if usage else None,
    )


def test_embed_success_maps_vectors_usage_and_model(mocker):
    provider = _provider()
    response = _fake_embedding_response([[0.1, 0.2], [0.3, 0.4]])
    mocker.patch.object(provider._client.embeddings, "create", return_value=response)

    result = _embed(provider)

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 0
    assert result.model == "text-embedding-3-small"
    assert result.latency_ms >= 0


def test_embed_orders_vectors_by_index_not_response_order(mocker):
    provider = _provider()
    # response.data comes back scrambled; the result must still line up with input order.
    response = _fake_embedding_response([[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]], indices=[2, 0, 1])
    mocker.patch.object(provider._client.embeddings, "create", return_value=response)

    result = _embed(provider, texts=("a", "b", "c"))

    assert result.vectors == [[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]]


def test_embed_empty_texts_makes_no_call(mocker):
    provider = _provider()
    create = mocker.patch.object(provider._client.embeddings, "create")

    result = provider.embed(texts=[], model="text-embedding-3-small", timeout_s=10.0)

    assert result.vectors == []
    assert result.usage.input_tokens == 0
    create.assert_not_called()


def test_embed_missing_usage_defaults_input_tokens_to_zero(mocker):
    provider = _provider()
    response = _fake_embedding_response([[0.1]], usage=False)
    mocker.patch.object(provider._client.embeddings, "create", return_value=response)

    assert _embed(provider, texts=("only",)).usage.input_tokens == 0


def test_embed_maps_timeout(mocker):
    provider = _provider()
    mocker.patch.object(
        provider._client.embeddings,
        "create",
        side_effect=openai.APITimeoutError(request=_request()),
    )
    with pytest.raises(LLMTimeout):
        _embed(provider)


def test_embed_maps_rate_limit(mocker):
    provider = _provider()
    resp = httpx2.Response(429, request=_request(), json={"error": {"message": "slow down"}})
    mocker.patch.object(
        provider._client.embeddings,
        "create",
        side_effect=openai.RateLimitError(
            "slow down", response=resp, body={"message": "slow down"}
        ),
    )
    with pytest.raises(LLMRateLimited):
        _embed(provider)


def test_embed_maps_internal_server_error_to_rate_limited(mocker):
    provider = _provider()
    resp = httpx2.Response(500, request=_request(), json={"error": {"message": "boom"}})
    mocker.patch.object(
        provider._client.embeddings,
        "create",
        side_effect=openai.InternalServerError("boom", response=resp, body={"message": "boom"}),
    )
    with pytest.raises(LLMRateLimited):
        _embed(provider)


def test_embed_maps_other_api_error_to_provider_error(mocker):
    provider = _provider()
    resp = httpx2.Response(400, request=_request(), json={"error": {"message": "bad request"}})
    mocker.patch.object(
        provider._client.embeddings,
        "create",
        side_effect=openai.BadRequestError(
            "bad request", response=resp, body={"message": "bad request"}
        ),
    )
    with pytest.raises(LLMProviderError):
        _embed(provider)


def test_fake_provider_satisfies_the_llm_provider_protocol():
    assert isinstance(FakeProvider(), LLMProvider)
