"""Pydantic models and enums for the ticket classification output.

`LLMClassification` is what the model is asked to produce. `Classification` adds
`needs_human_review` and `review_reasons`, which are always set by deterministic code in
`rules.py` and must never be delegated to the LLM (see SPEC.md, D1.7).
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    GENERAL_QUESTION = "general_question"
    QUALITY_COMPLAINT = "quality_complaint"
    EXPERT_COMPLAINT = "expert_complaint"
    PAYMENT_ISSUE = "payment_issue"
    THREAT = "threat"
    OTHER = "other"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class NextStep(StrEnum):
    """The response to a ticket. Which responses are valid for which category is defined
    by `rules.PRIORITY_TABLE`, not by this enum."""

    SEND_USER_GUIDE = "send_user_guide"
    SEND_KB_ANSWER = "send_kb_answer"
    GENERIC_REPLY = "generic_reply"
    RECORD_FEATURE_REQUEST = "record_feature_request"
    CREATE_BUG_TICKET = "create_bug_ticket"
    RECORD_EXPERT_COMPLAINT = "record_expert_complaint"
    SEND_REFUND_POLICY = "send_refund_policy"
    ESCALATE_HUMAN = "escalate_human"
    NO_REPLY = "no_reply"


class LLMClassification(BaseModel):
    """The fields the LLM is asked to fill. Excludes priority and the human-review decision.

    The LLM never picks a priority: it only quotes the ticket's evidence for raising it,
    and `rules.py` computes the priority from `PRIORITY_TABLE`. Field order matters:
    structured output is generated in this order, so the model commits to `category` and
    `next_step` before looking for evidence against that pair's raise condition.

    `extra="forbid"` makes an unexpected extra key a validation error, so the gateway's
    repair retry (gateway.py) kicks in instead of the extra key silently passing through.
    """

    model_config = ConfigDict(extra="forbid")

    category: Category
    next_step: NextStep
    # Required but nullable (no default), as strict structured output expects.
    priority_raise_evidence: str | None = Field(
        description=(
            "Shortest verbatim quote from the ticket stating a fact from the chosen "
            "pair's raise condition, or null."
        )
    )
    next_step_note: str
    language: str
    tone: str
    confidence: float = Field(ge=0, le=1)
    rationale: str


class Classification(LLMClassification):
    """The full classification, with priority and human review applied by `rules.py`."""

    priority: Priority
    needs_human_review: bool = False
    review_reasons: list[str] = []
