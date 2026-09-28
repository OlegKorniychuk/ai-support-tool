"""Gateway retry/fallback tests, using FakeProvider. No network calls."""

import itertools

import pytest
from fakes import FakeProvider, make_embedding_result, make_result, register_fake
from pydantic import BaseModel

from support_ai.core.config import ModelConfig
from support_ai.core.llm.errors import (
    LLMInvalidOutput,
    LLMProviderError,
    LLMRateLimited,
    LLMTimeout,
)
from support_ai.core.llm.gateway import (
    AllModelsFailed,
    EmbeddingFailed,
    complete_with_fallback,
    embed_with_retry,
)

_name_counter = itertools.count()


class DummySchema(BaseModel):
    foo: str = ""


def _unique_provider_name() -> str:
    return f"fake-{next(_name_counter)}"


def _model_config(
    provider_name: str, *, model_id: str = "model", timeout_s: float = 1.0
) -> ModelConfig:
    return ModelConfig(
        provider=provider_name,
        model_id=model_id,
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        timeout_s=timeout_s,
    )


def _chain_of_one(fake: FakeProvider, *, model_id: str = "model") -> list[ModelConfig]:
    provider_name = _unique_provider_name()
    register_fake(provider_name, fake)
    return [_model_config(provider_name, model_id=model_id)]


def _embed_model_config(fake: FakeProvider, *, model_id: str = "embed-model") -> ModelConfig:
    return _chain_of_one(fake, model_id=model_id)[0]


def _no_sleep(_seconds: float) -> None:
    return None


def test_success_on_first_try():
    fake = FakeProvider(responses=[make_result({"foo": "bar"})])
    chain = _chain_of_one(fake)

    gateway_result = complete_with_fallback(
        chain, system="sys", user="ticket text", schema=DummySchema, sleep=_no_sleep
    )

    assert gateway_result.model_used == "model"
    assert gateway_result.result.data == {"foo": "bar"}
    assert [a.outcome for a in gateway_result.attempts] == ["success"]


def test_recovers_after_invalid_output_with_repair_retry():
    fake = FakeProvider(
        responses=[LLMInvalidOutput("missing field 'foo'"), make_result({"foo": "repaired"})]
    )
    chain = _chain_of_one(fake)

    gateway_result = complete_with_fallback(
        chain, system="sys", user="ticket text", schema=DummySchema, sleep=_no_sleep
    )

    assert gateway_result.result.data == {"foo": "repaired"}
    assert [a.outcome for a in gateway_result.attempts] == ["invalid_output", "success"]
    # the repair retry must append the validation error to the original prompt
    second_call_user = fake.calls[1]["user"]
    assert "ticket text" in second_call_user
    assert "missing field 'foo'" in second_call_user


def test_recovers_after_rate_limit_with_backoff():
    fake = FakeProvider(responses=[LLMRateLimited("429"), make_result({"foo": "bar"})])
    chain = _chain_of_one(fake)
    sleeps: list[float] = []

    gateway_result = complete_with_fallback(
        chain, system="sys", user="u", schema=DummySchema, sleep=sleeps.append
    )

    assert gateway_result.result.data == {"foo": "bar"}
    assert [a.outcome for a in gateway_result.attempts] == ["rate_limited", "success"]
    assert sleeps == [1]  # 2**0


def test_recovers_after_provider_error_with_backoff():
    fake = FakeProvider(responses=[LLMProviderError("5xx"), make_result({"foo": "bar"})])
    chain = _chain_of_one(fake)
    sleeps: list[float] = []

    gateway_result = complete_with_fallback(
        chain, system="sys", user="u", schema=DummySchema, sleep=sleeps.append
    )

    assert gateway_result.result.data == {"foo": "bar"}
    assert [a.outcome for a in gateway_result.attempts] == ["provider_error", "success"]
    assert sleeps == [1]


def test_timeout_falls_back_to_next_model_after_one_retry():
    failing = FakeProvider(responses=[LLMTimeout("t1"), LLMTimeout("t2")])
    succeeding = FakeProvider(responses=[make_result({"foo": "from-second-model"})])
    failing_name = _unique_provider_name()
    succeeding_name = _unique_provider_name()
    register_fake(failing_name, failing)
    register_fake(succeeding_name, succeeding)
    chain = [
        _model_config(failing_name, model_id="model-a"),
        _model_config(succeeding_name, model_id="model-b"),
    ]

    gateway_result = complete_with_fallback(
        chain, system="sys", user="u", schema=DummySchema, sleep=_no_sleep
    )

    assert gateway_result.model_used == "model-b"
    assert gateway_result.result.data == {"foo": "from-second-model"}
    outcomes = [(a.model, a.outcome) for a in gateway_result.attempts]
    assert outcomes == [
        ("model-a", "timeout"),
        ("model-a", "timeout"),
        ("model-b", "success"),
    ]


def test_all_models_failing_raises_all_models_failed():
    fake = FakeProvider(
        responses=[LLMProviderError("p1"), LLMProviderError("p2"), LLMProviderError("p3")]
    )
    chain = _chain_of_one(fake)

    with pytest.raises(AllModelsFailed) as exc_info:
        complete_with_fallback(chain, system="sys", user="u", schema=DummySchema, sleep=_no_sleep)

    assert len(exc_info.value.attempts) == 3
    assert all(a.outcome == "provider_error" for a in exc_info.value.attempts)


def test_all_models_failing_across_multiple_models():
    fake_a = FakeProvider(responses=[LLMTimeout("t1"), LLMTimeout("t2")])
    fake_b = FakeProvider(responses=[LLMTimeout("t1"), LLMTimeout("t2")])
    name_a = _unique_provider_name()
    name_b = _unique_provider_name()
    register_fake(name_a, fake_a)
    register_fake(name_b, fake_b)
    chain = [_model_config(name_a, model_id="model-a"), _model_config(name_b, model_id="model-b")]

    with pytest.raises(AllModelsFailed) as exc_info:
        complete_with_fallback(chain, system="sys", user="u", schema=DummySchema, sleep=_no_sleep)

    assert [a.model for a in exc_info.value.attempts] == [
        "model-a",
        "model-a",
        "model-b",
        "model-b",
    ]


def test_embed_success_on_first_try():
    fake = FakeProvider(embed_responses=[make_embedding_result([[0.1, 0.2]])])
    model_config = _embed_model_config(fake)

    gateway_result = embed_with_retry(model_config, texts=["a"], sleep=_no_sleep)

    assert gateway_result.result.vectors == [[0.1, 0.2]]
    assert [a.outcome for a in gateway_result.attempts] == ["success"]


def test_embed_recovers_after_timeout_with_one_retry():
    fake = FakeProvider(embed_responses=[LLMTimeout("t1"), make_embedding_result([[0.5]])])
    model_config = _embed_model_config(fake)

    gateway_result = embed_with_retry(model_config, texts=["a"], sleep=_no_sleep)

    assert gateway_result.result.vectors == [[0.5]]
    assert [a.outcome for a in gateway_result.attempts] == ["timeout", "success"]


def test_embed_recovers_after_two_rate_limits_with_backoff():
    fake = FakeProvider(
        embed_responses=[
            LLMRateLimited("429"),
            LLMRateLimited("429"),
            make_embedding_result([[0.9]]),
        ]
    )
    model_config = _embed_model_config(fake)
    sleeps: list[float] = []

    gateway_result = embed_with_retry(model_config, texts=["a"], sleep=sleeps.append)

    assert gateway_result.result.vectors == [[0.9]]
    assert [a.outcome for a in gateway_result.attempts] == [
        "rate_limited",
        "rate_limited",
        "success",
    ]
    assert sleeps == [1, 2]  # 2**0, 2**1


def test_embed_recovers_after_provider_error_with_backoff():
    fake = FakeProvider(embed_responses=[LLMProviderError("5xx"), make_embedding_result([[0.3]])])
    model_config = _embed_model_config(fake)
    sleeps: list[float] = []

    gateway_result = embed_with_retry(model_config, texts=["a"], sleep=sleeps.append)

    assert gateway_result.result.vectors == [[0.3]]
    assert [a.outcome for a in gateway_result.attempts] == ["provider_error", "success"]
    assert sleeps == [1]


def test_embed_budget_exhausted_raises_embedding_failed_with_full_attempt_log():
    fake = FakeProvider(
        embed_responses=[LLMRateLimited("1"), LLMRateLimited("2"), LLMRateLimited("3")]
    )
    model_config = _embed_model_config(fake, model_id="embed-model")

    with pytest.raises(EmbeddingFailed) as exc_info:
        embed_with_retry(model_config, texts=["a"], sleep=_no_sleep)

    assert [a.outcome for a in exc_info.value.attempts] == [
        "rate_limited",
        "rate_limited",
        "rate_limited",
    ]
    assert all(a.model == "embed-model" for a in exc_info.value.attempts)
