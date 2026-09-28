"""The Reply Assistant pipeline: retrieve -> decide mode -> generate -> judge.

Ticket in, `AssistResult` out. Retrieval (`kb.search`) decides whether generation runs in
full mode (3 drafts) or summary-only mode (already flagged before generation); `rules.judge`
then verifies every fact the model reported and is the only place that decides
`needs_human_judgment` — never the LLM itself. See SPEC_MVP2.md, "Pipeline" and
"Human-judgment rules". `assist()` never raises.
"""

import time
from collections.abc import Callable
from typing import Literal

from support_ai.assistant import rules
from support_ai.assistant.prompting import build_user_message, load_prompt
from support_ai.assistant.schema import AssistResult, LLMReply, LLMSummary, StepInfo
from support_ai.core import config
from support_ai.core.config import MODEL_REGISTRY
from support_ai.core.cost import cost_for_usage
from support_ai.core.llm.base import Usage
from support_ai.core.llm.gateway import AllModelsFailed, Attempt, complete_with_fallback
from support_ai.kb import KnowledgeBase, RetrievalError, SearchResult, get_knowledge_base

_Mode = Literal["full", "summary_only"]


def assist(
    ticket: str,
    *,
    kb: KnowledgeBase | None = None,
    reply_chain: list[str] | None = None,
    prompt_version: str = config.DEFAULT_REPLY_PROMPT_VERSION,
    sleep: Callable[[float], None] = time.sleep,
) -> AssistResult:
    """Summarize, ground and draft replies for one general question. Never raises.

    `kb` defaults to `get_knowledge_base()` (the registry's `config.KB_BACKEND` instance).
    `sleep` is forwarded to the gateway's backoff and can be replaced with a no-op in tests.
    """
    if not ticket or not ticket.strip():
        return _empty_result()

    search, retrieval_step = _retrieve(ticket, kb)
    pre = rules.pre_generation_reasons(search)
    mode: _Mode = "full" if not pre else "summary_only"

    reply, generation_step = _generate(
        ticket, search, mode, reply_chain=reply_chain, prompt_version=prompt_version, sleep=sleep
    )

    judgment = rules.judge(ticket, search, reply, pre)
    summary = (
        reply.summary if reply is not None else "Summary unavailable: reply generation failed."
    )

    return AssistResult(
        summary=summary,
        kb_source=judgment.kb_source,
        kb_candidates=search.hits if search is not None else [],
        drafts=judgment.drafts,
        needs_human_judgment=judgment.needs_human_judgment,
        judgment_reasons=judgment.reasons,
        account_specific_evidence=judgment.account_specific_evidence,
        conflicting_article_ids=judgment.conflicting_article_ids,
        dropped_evidence=judgment.dropped_evidence,
        steps=[retrieval_step, generation_step],
    )


def _empty_result() -> AssistResult:
    """Deterministic result for empty or whitespace-only input: no calls, not flagged."""
    return AssistResult(
        summary="The ticket is empty.",
        kb_candidates=[],
        drafts=[],
        needs_human_judgment=False,
        steps=[],
    )


def _retrieve(ticket: str, kb: KnowledgeBase | None) -> tuple[SearchResult | None, StepInfo]:
    """Resolve the KB backend and search it. Never raises.

    A bad `KB_BACKEND` (or any other failure resolving `kb`) has no backend identity to
    attribute the failure to, so `model_used="unknown"`; once a backend is resolved,
    `model_used` is always its `name`, even if `search()` itself then fails.
    """
    try:
        resolved_kb = kb or get_knowledge_base()
    except Exception as exc:  # noqa: BLE001 - assist() must never raise
        attempts = [Attempt(model="unknown", outcome="unexpected_error", detail=str(exc))]
        return None, StepInfo(
            step="retrieval", model_used="unknown", latency_ms=0, cost_usd=0.0, attempts=attempts
        )

    try:
        search = resolved_kb.search(ticket, k=config.KB_TOP_K)
    except RetrievalError as exc:
        return None, StepInfo(
            step="retrieval",
            model_used=resolved_kb.name,
            latency_ms=0,
            cost_usd=0.0,
            attempts=exc.attempts,
        )
    except Exception as exc:  # noqa: BLE001 - assist() must never raise
        attempts = [Attempt(model="unknown", outcome="unexpected_error", detail=str(exc))]
        return None, StepInfo(
            step="retrieval",
            model_used=resolved_kb.name,
            latency_ms=0,
            cost_usd=0.0,
            attempts=attempts,
        )

    step = StepInfo(
        step="retrieval",
        model_used=resolved_kb.name,
        latency_ms=search.latency_ms,
        usage=search.usage,
        cost_usd=search.cost_usd,
        attempts=search.attempts,
    )
    return search, step


def _generate(
    ticket: str,
    search: SearchResult | None,
    mode: _Mode,
    *,
    reply_chain: list[str] | None,
    prompt_version: str,
    sleep: Callable[[float], None],
) -> tuple[LLMReply | LLMSummary | None, StepInfo]:
    """Run the generation call for `mode`. Never raises."""
    hits = search.hits if search is not None else []
    if mode == "full":
        system = load_prompt("reply", prompt_version)
        user = build_user_message(ticket, hits)
        schema: type[LLMReply | LLMSummary] = LLMReply
    else:
        system = load_prompt("summary", config.SUMMARY_PROMPT_VERSION)
        user = build_user_message(ticket, [])
        schema = LLMSummary

    chain = [MODEL_REGISTRY[model_id] for model_id in (reply_chain or config.DEFAULT_REPLY_CHAIN)]

    try:
        gateway_result = complete_with_fallback(
            chain, system=system, user=user, schema=schema, sleep=sleep
        )
    except AllModelsFailed as exc:
        return None, StepInfo(
            step="generation", model_used="none", latency_ms=0, cost_usd=0.0, attempts=exc.attempts
        )
    except Exception as exc:  # noqa: BLE001 - assist() must never raise
        attempts = [Attempt(model="unknown", outcome="unexpected_error", detail=str(exc))]
        return None, StepInfo(
            step="generation", model_used="none", latency_ms=0, cost_usd=0.0, attempts=attempts
        )

    reply = schema.model_validate(gateway_result.result.data)
    model_config = MODEL_REGISTRY[gateway_result.model_used]
    usage: Usage = gateway_result.result.usage
    step = StepInfo(
        step="generation",
        model_used=gateway_result.model_used,
        latency_ms=gateway_result.result.latency_ms,
        usage=usage,
        cost_usd=cost_for_usage(usage, model_config),
        attempts=gateway_result.attempts,
    )
    return reply, step
