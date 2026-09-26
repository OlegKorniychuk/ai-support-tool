"""The OpenAI adapter. This is the only module in the codebase that imports `openai`.

Maps the SDK's exceptions onto our normalized error hierarchy so the gateway and
everything above it never has to know which provider is in use.
"""

import json
import time
from typing import ClassVar

import openai
import pydantic

from support_ai.core.config import MODEL_REGISTRY, get_api_key
from support_ai.core.llm.base import LLMResult, Usage
from support_ai.core.llm.errors import (
    LLMInvalidOutput,
    LLMProviderError,
    LLMRateLimited,
    LLMTimeout,
)


def _describe_missing_output(response: object) -> str:
    """Best-effort explanation for why a successful call produced no parsed output."""
    incomplete = getattr(response, "incomplete_details", None)
    if incomplete is not None:
        return f"Response incomplete: {incomplete}"
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                return f"Model refused: {getattr(content, 'refusal', 'no reason given')}"
    return "Model response contained no parsable structured output"


class OpenAIProvider:
    """`LLMProvider` backed by the OpenAI Responses API structured-output parsing."""

    name: ClassVar[str] = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or get_api_key()
        # max_retries=0: our own gateway.py owns all retry/backoff/fallback decisions.
        self._client = openai.OpenAI(api_key=key, max_retries=0)

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[pydantic.BaseModel],
        model: str,
        timeout_s: float,
    ) -> LLMResult:
        model_config = MODEL_REGISTRY.get(model)
        reasoning = (
            {"effort": model_config.reasoning_effort}
            if model_config and model_config.reasoning_effort
            else openai.omit
        )

        start = time.monotonic()
        try:
            response = self._client.responses.parse(
                model=model,
                instructions=system,
                input=user,
                text_format=schema,
                timeout=timeout_s,
                reasoning=reasoning,
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeout(str(exc)) from exc
        except (openai.RateLimitError, openai.InternalServerError) as exc:
            raise LLMRateLimited(str(exc)) from exc
        except (json.JSONDecodeError, pydantic.ValidationError) as exc:
            raise LLMInvalidOutput(str(exc)) from exc
        except openai.APIError as exc:
            raise LLMProviderError(str(exc)) from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        parsed = response.output_parsed
        if parsed is None:
            raise LLMInvalidOutput(_describe_missing_output(response))

        usage = response.usage
        details = getattr(usage, "input_tokens_details", None) if usage else None
        return LLMResult(
            data=parsed.model_dump(mode="json"),
            usage=Usage(
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                cached_input_tokens=(getattr(details, "cached_tokens", None) or 0),
            ),
            model=response.model,
            latency_ms=latency_ms,
        )
