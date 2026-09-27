"""Build the prompt, call the gateway, validate, apply the rules and compute cost.

This module never imports `openai`; it depends only on the gateway and the normalized
`LLMProvider` contracts, so it works the same no matter which provider backs a model.
"""

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from support_ai.classifier.rules import apply_rules
from support_ai.classifier.schema import (
    Category,
    Classification,
    LLMClassification,
    NextStep,
    Priority,
)
from support_ai.core.config import DEFAULT_MODEL_CHAIN, DEFAULT_PROMPT_VERSION, MODEL_REGISTRY
from support_ai.core.cost import cost_for_usage
from support_ai.core.llm.base import Usage
from support_ai.core.llm.gateway import AllModelsFailed, Attempt, complete_with_fallback

PROMPTS_DIR = Path(__file__).parent / "prompts"


class ClassifyResult(BaseModel):
    """Everything the UI and the eval runner need about one classification call."""

    classification: Classification
    model_used: str
    latency_ms: int
    cost_usd: float
    usage: Usage
    attempts: list[Attempt] = []

    model_config = {"arbitrary_types_allowed": True}


def render_prompt(prompt_version: str) -> str:
    """Load the versioned system prompt.

    Kept static (independent of the ticket) so OpenAI's provider-side prompt caching can
    apply to it automatically; the ticket itself travels in the user message instead.
    """
    path = PROMPTS_DIR / f"{prompt_version}.md"
    return path.read_text()


def _wrap_ticket(ticket: str) -> str:
    """Wrap the raw ticket text in a delimited block, as a guard against prompt injection."""
    return f"<<<TICKET>>>\n{ticket}\n<<<END TICKET>>>"


def _fallback_result(attempts: list[Attempt] | None = None) -> ClassifyResult:
    """The SPEC.md fallback result (Failure Handling, X3): always flagged for review."""
    fallback_llm = LLMClassification(
        category=Category.OTHER,
        next_step=NextStep.ESCALATE_HUMAN,
        priority=Priority.P3,
        next_step_note="Automatic classification failed; a human must review this ticket.",
        language="unknown",
        tone="unknown",
        confidence=0.0,
        rationale="Classification failed.",
    )
    classification = apply_rules(fallback_llm, failed=True)
    return ClassifyResult(
        classification=classification,
        model_used="none",
        latency_ms=0,
        cost_usd=0.0,
        usage=Usage(input_tokens=0, output_tokens=0),
        attempts=attempts or [],
    )


def _empty_result() -> ClassifyResult:
    """Deterministic result for empty or whitespace-only input.

    No LLM call is made, and this is never flagged for review: an empty ticket is
    expected, everyday input, not a classification failure (SPEC.md's `other` /
    `no_reply` path, the same one the prompt uses for too-short tickets).
    """
    empty_llm = LLMClassification(
        category=Category.OTHER,
        next_step=NextStep.NO_REPLY,
        priority=Priority.P4,
        next_step_note="The ticket text was empty; no reply is needed.",
        language="unknown",
        tone="unknown",
        confidence=1.0,
        rationale="The ticket text was empty or whitespace-only.",
    )
    classification = apply_rules(empty_llm)
    return ClassifyResult(
        classification=classification,
        model_used="none",
        latency_ms=0,
        cost_usd=0.0,
        usage=Usage(input_tokens=0, output_tokens=0),
        attempts=[],
    )


def classify(
    ticket: str,
    *,
    model_chain: list[str] | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    sleep: Callable[[float], None] = time.sleep,
) -> ClassifyResult:
    """Classify one ticket. Never raises; on failure returns a fallback flagged for review.

    `sleep` is forwarded to the gateway's backoff and can be replaced with a no-op in
    tests so retries don't slow down the suite.
    """
    if not ticket or not ticket.strip():
        return _empty_result()

    system = render_prompt(prompt_version)
    user = _wrap_ticket(ticket)
    chain = [MODEL_REGISTRY[model_id] for model_id in (model_chain or DEFAULT_MODEL_CHAIN)]

    try:
        gateway_result = complete_with_fallback(
            chain, system=system, user=user, schema=LLMClassification, sleep=sleep
        )
    except AllModelsFailed as exc:
        return _fallback_result(attempts=exc.attempts)
    except Exception as exc:  # noqa: BLE001 - classify() must never raise (SPEC.md X3)
        return _fallback_result(
            attempts=[Attempt(model="unknown", outcome="unexpected_error", detail=str(exc))]
        )

    llm_classification = LLMClassification.model_validate(gateway_result.result.data)
    classification = apply_rules(llm_classification)
    model_config = MODEL_REGISTRY[gateway_result.model_used]
    cost_usd = cost_for_usage(gateway_result.result.usage, model_config)

    return ClassifyResult(
        classification=classification,
        model_used=gateway_result.model_used,
        latency_ms=gateway_result.result.latency_ms,
        cost_usd=cost_usd,
        usage=gateway_result.result.usage,
        attempts=gateway_result.attempts,
    )
