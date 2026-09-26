"""Provider-agnostic LLM contracts.

Only `openai_provider.py` may import the `openai` SDK. The classifier, the gateway, the
pages, the scripts and the eval code depend only on `LLMProvider`, `LLMResult` and the
normalized errors in `errors.py`. Switching provider means adding one adapter file and
registering it here; nothing else changes.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    # Subset of `input_tokens` served from the provider's prompt cache (billed at a discount).
    cached_input_tokens: int = 0


class LLMResult(BaseModel):
    data: dict
    usage: Usage
    model: str
    latency_ms: int


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model: str,
        timeout_s: float,
    ) -> LLMResult: ...


_FACTORIES: dict[str, Callable[[], LLMProvider]] = {}
_INSTANCES: dict[str, LLMProvider] = {}


def register_provider(name: str, factory: Callable[[], LLMProvider]) -> None:
    """Register a provider factory under `name`. Adapters call this on import.

    Registration is lazy: the factory runs only the first time `get_provider(name)` is
    called, so importing a provider module never requires an API key to be present.
    """
    _FACTORIES[name] = factory


def get_provider(name: str) -> LLMProvider:
    """Return the (lazily constructed, memoized) provider registered under `name`."""
    if name not in _INSTANCES:
        try:
            factory = _FACTORIES[name]
        except KeyError:
            raise KeyError(
                f"Unknown LLM provider {name!r}. Registered: {sorted(_FACTORIES)}"
            ) from None
        _INSTANCES[name] = factory()
    return _INSTANCES[name]
