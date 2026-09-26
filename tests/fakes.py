"""A scripted fake `LLMProvider` for tests. Never touches the network."""

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from support_ai.core.llm.base import LLMResult, Usage, register_provider
from support_ai.core.llm.errors import LLMError


@dataclass
class FakeProvider:
    """Replays `responses` in order.

    Each entry is either an `LLMResult` to return or an `LLMError` instance to raise.
    Every call is recorded in `calls` so tests can assert on prompts and models used.
    """

    responses: list[LLMResult | LLMError]
    name: ClassVar[str] = "fake"
    calls: list[dict] = field(default_factory=list)

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model: str,
        timeout_s: float,
    ) -> LLMResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "model": model,
                "timeout_s": timeout_s,
            }
        )
        if not self.responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        outcome = self.responses.pop(0)
        if isinstance(outcome, LLMError):
            raise outcome
        return outcome


def register_fake(name: str, provider: FakeProvider) -> None:
    """Register `provider` in the real provider registry under `name`, for gateway tests."""
    register_provider(name, lambda: provider)


def make_result(
    data: dict,
    *,
    model: str = "fake-model",
    input_tokens: int = 10,
    output_tokens: int = 10,
    latency_ms: int = 5,
) -> LLMResult:
    """Convenience builder for a scripted successful `LLMResult`."""
    return LLMResult(
        data=data,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        latency_ms=latency_ms,
    )
