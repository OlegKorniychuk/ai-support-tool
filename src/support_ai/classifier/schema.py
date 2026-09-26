"""Pydantic models and enums for the ticket classification output.

`LLMClassification` is what the model is asked to produce. `Classification` adds
`needs_human_review` and `review_reasons`, which are always set by deterministic code in
`rules.py` and must never be delegated to the LLM (see SPEC.md, D1.7).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    BILLING_SUBSCRIPTION = "billing_subscription"
    REFUND_REQUEST = "refund_request"
    TECHNICAL_BUG = "technical_bug"
    EXPERT_COMPLAINT = "expert_complaint"
    ACCOUNT_ACCESS = "account_access"
    FEATURE_REQUEST = "feature_request"
    USAGE_HELP = "usage_help"
    GENERAL_FEEDBACK = "general_feedback"
    UNCLEAR = "unclear"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class NextStep(StrEnum):
    ROUTE_BILLING = "route_billing"
    ROUTE_REFUNDS = "route_refunds"
    ROUTE_TECH_SUPPORT = "route_tech_support"
    ROUTE_EXPERT_QUALITY = "route_expert_quality"
    ROUTE_ACCOUNT_SUPPORT = "route_account_support"
    SEND_KB_ARTICLE = "send_kb_article"
    REQUEST_MORE_INFO = "request_more_info"
    ESCALATE_SENIOR = "escalate_senior"
    SEND_GENERIC_REPLY = "send_generic_reply"


class LLMClassification(BaseModel):
    """The fields the LLM is asked to fill. Excludes the human-review decision.

    `extra="forbid"` makes an unexpected extra key a validation error, so the gateway's
    repair retry (gateway.py) kicks in instead of the extra key silently passing through.
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    secondary_categories: list[Category] = []
    priority: Priority
    next_step: NextStep
    next_step_note: str
    language: str
    tone: str
    confidence: float = Field(ge=0, le=1)
    requests_human: bool = Field(
        description=(
            "True only if the customer explicitly asks for a live support person, "
            "human agent or manager instead of a bot. Deterministic code, not the "
            "LLM, decides whether that flags the ticket for review (see rules.py)."
        )
    )
    rationale: str


class Classification(LLMClassification):
    """The full classification, with the human-review decision applied by `rules.py`."""

    needs_human_review: bool = False
    review_reasons: list[str] = []
