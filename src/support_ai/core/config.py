"""Settings and the model registry.

The registry holds, per model id, the owning provider, and the price and timeout used by
`cost.py` and the gateway. It is the one place to edit when a model id, price or timeout
changes; no other module hardcodes them. It holds both chat and embedding models, told
apart by `ModelConfig.kind`; callers that only want the classifier's/reply generator's chat
models (e.g. an eval script's `--model` choices) filter on it rather than assuming every
registry entry is a chat model.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class ModelConfig(BaseModel):
    provider: str
    model_id: str
    # "chat" models go through `complete_structured`; "embedding" models through `embed`.
    kind: Literal["chat", "embedding"] = "chat"
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
    # Price confirmed against https://developers.openai.com/api/docs/models/text-embedding-3-small
    # and https://developers.openai.com/api/docs/pricing on 2026-09-28: $0.02 / 1M tokens.
    # Embeddings have no output tokens and no prompt-cache discount.
    "text-embedding-3-small": ModelConfig(
        provider="openai",
        model_id="text-embedding-3-small",
        kind="embedding",
        input_price_per_1m=0.02,
        output_price_per_1m=0.0,
        cached_input_price_per_1m=None,
        timeout_s=10.0,
    ),
}

# Cheapest first: nano is tried first, falling back to mini then gpt-5 on failure.
DEFAULT_MODEL_CHAIN: list[str] = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5"]
DEFAULT_MODEL: str = DEFAULT_MODEL_CHAIN[0]
DEFAULT_PROMPT_VERSION: str = "v6"
EMBEDDING_MODEL: str = "text-embedding-3-small"
# Default KB retrieval backend, looked up in kb/base.py's registry (kb/__init__.py wires
# "in_memory" in on import). Swapping backends is changing this one string.
KB_BACKEND: str = "in_memory"
# 4: r007's article ranks 4th; conflict pairs can still fall outside, e.g. r027 at rank 6.
KB_TOP_K: int = 4
# `in_memory`-adapter-only (other backends set their own threshold on their own scale).
# Checked 2026-09-28 with scripts/kb_scores.py on data/reply_tickets.jsonl: answerable best
# scores start at 0.43, but near-topic gaps (gift, family sharing: 0.42-0.49) overlap them, so
# no threshold separates the groups. It is set to never block an answerable ticket and only
# catch clearly off-KB ones (promo codes: 0.24); near-topic gaps rely on the model citing no
# article (the post-generation `kb_not_found` check).
KB_MIN_SCORE: float = 0.35
KB_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "kb"
KB_CACHE_PATH: Path = Path(__file__).resolve().parents[3] / ".cache" / "kb_embeddings.json"

# Cheapest first, as DEFAULT_MODEL_CHAIN; the default is the cheapest model that meets the
# targets after the Task 19 comparison (as in MVP 1).
DEFAULT_REPLY_CHAIN: list[str] = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5"]
DEFAULT_REPLY_PROMPT_VERSION: str = "v4"
SUMMARY_PROMPT_VERSION: str = "v1"
# The `short` tone's own word-count rule; `test_reply_prompts.py` checks the prompt states
# this same number, so changing either alone fails the contract test.
SHORT_MAX_WORDS: int = 50


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
