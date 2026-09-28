"""Pydantic models and enums for the Reply Assistant's output.

`LLMReply` (full mode) and `LLMSummary` (summary-only mode) are what the model is asked to
produce; `rules.py` turns either one into a `Judgment` — `needs_human_judgment` and its
reasons are always decided by deterministic code, never by the LLM (SPEC_MVP2.md,
"Human-judgment rules"). `AssistResult` is the top-level output the page and the eval runner
consume, combining a `Judgment` with retrieval candidates and per-step cost/latency.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from support_ai.core.llm.base import Usage
from support_ai.core.llm.gateway import Attempt
from support_ai.kb.base import KBHit


class Tone(StrEnum):
    """Declaration order is the order drafts are shown in: formal, empathetic, short."""

    FORMAL = "formal"
    EMPATHETIC = "empathetic"
    SHORT = "short"


class JudgmentReason(StrEnum):
    """Declaration order is display order, matching SPEC_MVP2.md's rules table."""

    KB_NOT_FOUND = "kb_not_found"
    KB_QUOTE_UNVERIFIED = "kb_quote_unverified"
    ACCOUNT_SPECIFIC = "account_specific"
    CONFLICTING_KB = "conflicting_kb"
    RETRIEVAL_FAILED = "retrieval_failed"
    GENERATION_FAILED = "generation_failed"


class LLMReply(BaseModel):
    """The fields the LLM is asked to fill in full mode (retrieval found a candidate).

    Field order matters: structured output is generated in this order, so the model
    commits to its source, quote and judgment evidence before writing the drafts. `rules.py`
    verifies every reported fact in code; the LLM's job is only to report them accurately.

    `extra="forbid"` makes an unexpected extra key a validation error, so the gateway's
    repair retry (gateway.py) kicks in instead of the extra key silently passing through.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str
    # Required but nullable (no default), as strict structured output expects.
    kb_article_id: str | None = Field(
        description="Id of the retrieved article that answers the question, or null if "
        "none of the retrieved articles answer it."
    )
    kb_quote: str | None = Field(
        description="Shortest verbatim quote from the cited article that backs the reply, or null."
    )
    account_specific_evidence: str | None = Field(
        description="Shortest verbatim quote from the ticket showing the answer depends on "
        "this customer's own account data (their charges, renewal date, session history), "
        "or null."
    )
    conflicting_article_ids: list[str] = Field(
        description="Ids of retrieved articles that give contradictory answers to this "
        "question; empty if none."
    )
    formal: str
    empathetic: str
    short: str


class LLMSummary(BaseModel):
    """The only field the LLM is asked to fill in summary-only mode (already flagged)."""

    model_config = ConfigDict(extra="forbid")

    summary: str


class Draft(BaseModel):
    tone: Tone
    text: str


class KBSource(BaseModel):
    """The article and verified quote a reply is grounded in."""

    article_id: str
    title: str
    quote: str
    text: str


class StepInfo(BaseModel):
    """Cost/latency metadata for one pipeline step (retrieval or generation)."""

    step: Literal["retrieval", "generation"]
    model_used: str
    latency_ms: int
    usage: Usage | None = None
    cost_usd: float
    attempts: list[Attempt] = []

    # `Attempt` is a plain stdlib dataclass (shared with the LLM gateway), not a pydantic
    # model; see `ClassifyResult` (classifier/classify.py) for the same workaround.
    model_config = {"arbitrary_types_allowed": True}


class Judgment(BaseModel):
    """The result of `rules.judge`: what to show, and why — decided entirely by code from
    the LLM's reported facts, `search` and `ticket`. Never built from LLM output directly.
    """

    kb_source: KBSource | None = None
    drafts: list[Draft] = []
    reasons: list[JudgmentReason] = []
    account_specific_evidence: str | None = None
    conflicting_article_ids: list[str] = []
    dropped_evidence: list[str] = []

    @property
    def needs_human_judgment(self) -> bool:
        return bool(self.reasons)


class AssistResult(BaseModel):
    """Everything the Reply Assistant page and the eval runner need about one ticket."""

    summary: str
    kb_source: KBSource | None = None
    kb_candidates: list[KBHit] = []
    drafts: list[Draft] = []
    needs_human_judgment: bool
    judgment_reasons: list[JudgmentReason] = []
    account_specific_evidence: str | None = None
    conflicting_article_ids: list[str] = []
    dropped_evidence: list[str] = []
    steps: list[StepInfo] = []

    @computed_field
    @property
    def total_latency_ms(self) -> int:
        return sum(step.latency_ms for step in self.steps)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return sum(step.cost_usd for step in self.steps)
