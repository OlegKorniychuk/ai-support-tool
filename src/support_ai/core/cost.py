"""Token usage → dollar cost, and simple cost forecasting."""

from support_ai.core.config import ModelConfig
from support_ai.core.llm.base import Usage


def cost_for_usage(usage: Usage, model_config: ModelConfig) -> float:
    """Dollar cost of one call, given its token usage and the model's per-1M-token prices."""
    input_cost = usage.input_tokens / 1_000_000 * model_config.input_price_per_1m
    output_cost = usage.output_tokens / 1_000_000 * model_config.output_price_per_1m
    return input_cost + output_cost


def forecast(cost_per_ticket: float, n_tickets: int) -> float:
    """Total dollar cost of classifying `n_tickets` tickets at `cost_per_ticket` each."""
    return cost_per_ticket * n_tickets
