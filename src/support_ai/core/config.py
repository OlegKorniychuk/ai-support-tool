"""Settings and the model registry.

The registry holds, per model id, the owning provider, and the price and timeout used by
`cost.py` and the gateway. It is the one place to edit when a model id, price or timeout
changes; no other module hardcodes them.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class ModelConfig(BaseModel):
    provider: str
    model_id: str
    input_price_per_1m: float
    output_price_per_1m: float
    # Price for input tokens served from the prompt cache; None means no discount.
    cached_input_price_per_1m: float | None = None
    timeout_s: float
    # Passed through to the provider as a reasoning-effort hint, if it supports one.
    # Kept low for gpt-5 (a slower reasoning model) to reduce the odds of hitting timeout_s.
    reasoning_effort: str | None = None


# Prices and model ids confirmed against https://developers.openai.com/api/docs/models/<id>
# on 2026-09-26. All three models (gpt-5.4-nano, gpt-5.4-mini, gpt-5) exist as of that date;
# no substitution was needed. Cached-input prices confirmed on the same pages on 2026-09-26.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-5.4-nano": ModelConfig(
        provider="openai",
        model_id="gpt-5.4-nano",
        input_price_per_1m=0.20,
        cached_input_price_per_1m=0.02,
        output_price_per_1m=1.25,
        timeout_s=20.0,
    ),
    "gpt-5.4-mini": ModelConfig(
        provider="openai",
        model_id="gpt-5.4-mini",
        input_price_per_1m=0.75,
        cached_input_price_per_1m=0.075,
        output_price_per_1m=4.50,
        timeout_s=25.0,
    ),
    "gpt-5": ModelConfig(
        provider="openai",
        model_id="gpt-5",
        input_price_per_1m=1.25,
        cached_input_price_per_1m=0.125,
        output_price_per_1m=10.00,
        # gpt-5 is a reasoning model and is slower than the 5.4 nano/mini pair.
        timeout_s=45.0,
        reasoning_effort="low",
    ),
}

# Cheapest first: nano is tried first, falling back to mini then gpt-5 on failure.
DEFAULT_MODEL_CHAIN: list[str] = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5"]
DEFAULT_MODEL: str = DEFAULT_MODEL_CHAIN[0]
DEFAULT_PROMPT_VERSION: str = "v5"


def get_api_key() -> str | None:
    """Read OPENAI_API_KEY from the environment (`.env` locally) or `st.secrets` when deployed."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None
