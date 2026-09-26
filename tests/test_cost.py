from support_ai.core.config import MODEL_REGISTRY
from support_ai.core.cost import cost_for_usage, forecast
from support_ai.core.llm.base import Usage


def test_cost_for_usage_one_million_tokens_equals_listed_price():
    model_config = MODEL_REGISTRY["gpt-5.4-nano"]
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = cost_for_usage(usage, model_config)
    assert cost == model_config.input_price_per_1m + model_config.output_price_per_1m


def test_cost_for_usage_zero_tokens_is_free():
    model_config = MODEL_REGISTRY["gpt-5.4-nano"]
    usage = Usage(input_tokens=0, output_tokens=0)
    assert cost_for_usage(usage, model_config) == 0.0


def test_cost_for_usage_known_small_call():
    model_config = MODEL_REGISTRY["gpt-5.4-mini"]
    usage = Usage(input_tokens=500, output_tokens=200)
    expected = (
        500 / 1_000_000 * model_config.input_price_per_1m
        + 200 / 1_000_000 * model_config.output_price_per_1m
    )
    assert cost_for_usage(usage, model_config) == expected


def test_cost_for_usage_differs_by_model():
    usage = Usage(input_tokens=1000, output_tokens=1000)
    nano_cost = cost_for_usage(usage, MODEL_REGISTRY["gpt-5.4-nano"])
    gpt5_cost = cost_for_usage(usage, MODEL_REGISTRY["gpt-5"])
    assert nano_cost < gpt5_cost


def test_forecast_scales_linearly():
    assert forecast(0.001, 10_000) == 10.0


def test_forecast_zero_tickets_is_free():
    assert forecast(0.001, 0) == 0.0


def test_forecast_zero_cost_per_ticket():
    assert forecast(0.0, 10_000) == 0.0
