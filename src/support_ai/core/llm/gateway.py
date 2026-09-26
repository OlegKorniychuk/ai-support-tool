"""Retry, repair-retry and model fallback.

Provider-agnostic (SPEC.md's Failure Handling, X3): this module only ever sees the
normalized errors in `errors.py` and a resolved chain of `ModelConfig`s, so it behaves
identically no matter which provider backs each model.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from support_ai.core.config import ModelConfig
from support_ai.core.llm.base import LLMProvider, LLMResult, get_provider
from support_ai.core.llm.errors import (
    LLMError,
    LLMInvalidOutput,
    LLMProviderError,
    LLMRateLimited,
    LLMTimeout,
)

MAX_BACKOFF_RETRIES = 2  # rate limit or provider error (5xx)
MAX_TIMEOUT_RETRIES = 1
MAX_REPAIR_RETRIES = 1


@dataclass
class Attempt:
    """The outcome of one call attempt against one model."""

    model: str
    outcome: str  # "success" | "rate_limited" | "provider_error" | "timeout" | "invalid_output"
    detail: str | None = None


@dataclass
class GatewayResult:
    result: LLMResult
    model_used: str
    attempts: list[Attempt]


class AllModelsFailed(LLMError):
    """Raised when every model in the fallback chain exhausted its retries."""

    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        tried = ", ".join(a.model for a in attempts)
        super().__init__(f"All models in the fallback chain failed. Attempts: {tried}")


def complete_with_fallback(
    chain: list[ModelConfig],
    *,
    system: str,
    user: str,
    schema: type[BaseModel],
    sleep: Callable[[float], None] = time.sleep,
) -> GatewayResult:
    """Try each model in `chain` in order, retrying transient failures, until one succeeds.

    Raises `AllModelsFailed` (carrying the full attempt log) if every model in the chain
    exhausts its retries. `sleep` can be injected so backoff waits don't slow down tests.
    """
    attempts: list[Attempt] = []
    for model_config in chain:
        provider = get_provider(model_config.provider)
        result = _try_model(
            provider=provider,
            model_config=model_config,
            system=system,
            user=user,
            schema=schema,
            attempts=attempts,
            sleep=sleep,
        )
        if result is not None:
            return GatewayResult(result=result, model_used=model_config.model_id, attempts=attempts)
    raise AllModelsFailed(attempts)


def _try_model(
    *,
    provider: LLMProvider,
    model_config: ModelConfig,
    system: str,
    user: str,
    schema: type[BaseModel],
    attempts: list[Attempt],
    sleep: Callable[[float], None],
) -> LLMResult | None:
    """Run one model to exhaustion of its retry budgets. `None` means: try the next model."""
    model = model_config.model_id
    current_user = user
    backoff_retries = 0
    timeout_retries = 0
    repair_retries = 0

    while True:
        try:
            result = provider.complete_structured(
                system=system,
                user=current_user,
                schema=schema,
                model=model,
                timeout_s=model_config.timeout_s,
            )
        except LLMTimeout as exc:
            attempts.append(Attempt(model=model, outcome="timeout", detail=str(exc)))
            if timeout_retries >= MAX_TIMEOUT_RETRIES:
                return None
            timeout_retries += 1
            continue
        except LLMInvalidOutput as exc:
            attempts.append(Attempt(model=model, outcome="invalid_output", detail=str(exc)))
            if repair_retries >= MAX_REPAIR_RETRIES:
                return None
            current_user = _append_repair_note(user, str(exc))
            repair_retries += 1
            continue
        except (LLMRateLimited, LLMProviderError) as exc:
            outcome = "rate_limited" if isinstance(exc, LLMRateLimited) else "provider_error"
            attempts.append(Attempt(model=model, outcome=outcome, detail=str(exc)))
            if backoff_retries >= MAX_BACKOFF_RETRIES:
                return None
            sleep(2**backoff_retries)
            backoff_retries += 1
            continue

        attempts.append(Attempt(model=model, outcome="success"))
        return result


def _append_repair_note(original_user: str, validation_error: str) -> str:
    """Append the validation error to the prompt for one repair retry."""
    return (
        f"{original_user}\n\n"
        "<<<PREVIOUS_OUTPUT_INVALID>>>\n"
        f"Your previous response failed schema validation with this error:\n{validation_error}\n"
        "Return output that strictly matches the required schema.\n"
        "<<<END_PREVIOUS_OUTPUT_INVALID>>>"
    )
