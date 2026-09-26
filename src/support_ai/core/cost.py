"""Token usage → dollar cost, and simple cost forecasting."""

from support_ai.core.config import ModelConfig
from support_ai.core.llm.base import Usage


def cost_for_usage(usage: Usage, model_config: ModelConfig) -> float:
    """Dollar cost of one call, given its token usage and the model's per-1M-token prices.

    Cached input tokens are billed at `cached_input_price_per_1m` when the model has one,
    otherwise at the full input price.
    """
    cached_price = model_config.cached_input_price_per_1m
    if cached_price is None:
        cached_price = model_config.input_price_per_1m
    uncached_tokens = usage.input_tokens - usage.cached_input_tokens
    input_cost = uncached_tokens / 1_000_000 * model_config.input_price_per_1m
    cached_cost = usage.cached_input_tokens / 1_000_000 * cached_price
    output_cost = usage.output_tokens / 1_000_000 * model_config.output_price_per_1m
    return input_cost + cached_cost + output_cost


def forecast(cost_per_ticket: float, n_tickets: int) -> float:
    """Total dollar cost of classifying `n_tickets` tickets at `cost_per_ticket` each."""
    return cost_per_ticket * n_tickets
