"""Normalized LLM error hierarchy.

Providers map their SDK-specific exceptions onto these, so the gateway and everything
above it can reason about failures without knowing which provider raised them.
"""


class LLMError(Exception):
    """Base class for all LLM call failures."""


class LLMTimeout(LLMError):
    """The call did not complete within the configured timeout."""


class LLMRateLimited(LLMError):
    """The provider rejected the call for rate limiting (HTTP 429) or a transient 5xx."""


class LLMInvalidOutput(LLMError):
    """The response could not be parsed against the schema, or the model refused."""


class LLMProviderError(LLMError):
    """Any other provider-side failure (auth, bad request, connection, etc.)."""
