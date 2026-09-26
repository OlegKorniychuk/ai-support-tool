"""Registers the built-in LLM providers.

Importing this package wires `OpenAIProvider` into `get_provider("openai")` without
constructing it (registration is lazy — see `base.register_provider`), so importing it
never requires `OPENAI_API_KEY` to be set.
"""

from support_ai.core.llm.base import get_provider, register_provider
from support_ai.core.llm.openai_provider import OpenAIProvider

register_provider("openai", OpenAIProvider)

__all__ = ["get_provider", "register_provider"]
