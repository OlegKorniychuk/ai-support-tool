"""Deterministic post-LLM rules: the priority table and human-in-the-loop flag.

The LLM never decides `needs_human_review` — this module does, purely from the LLM's
other output fields plus whether the call ultimately failed. See SPEC.md, D1.7.
"""

from support_ai.classifier.schema import (
    Category,
    Classification,
    LLMClassification,
    NextStep,
    Priority,
)

# Every valid (category, response) pair, with its base priority and the highest priority
# a stated fact may raise it to. Mirrors the table in prompts/v5.md; test_prompts.py
# checks the two never drift apart. A pair missing from here is an invalid classification.
PRIORITY_TABLE: dict[tuple[Category, NextStep], tuple[Priority, Priority]] = {
    (Category.GENERAL_QUESTION, NextStep.SEND_USER_GUIDE): (Priority.P4, Priority.P4),
    (Category.GENERAL_QUESTION, NextStep.SEND_KB_ANSWER): (Priority.P4, Priority.P4),
    (Category.QUALITY_COMPLAINT, NextStep.GENERIC_REPLY): (Priority.P4, Priority.P4),
    (Category.QUALITY_COMPLAINT, NextStep.RECORD_FEATURE_REQUEST): (Priority.P4, Priority.P4),
    (Category.QUALITY_COMPLAINT, NextStep.CREATE_BUG_TICKET): (Priority.P3, Priority.P2),
    (Category.QUALITY_COMPLAINT, NextStep.ESCALATE_HUMAN): (Priority.P2, Priority.P2),
    (Category.EXPERT_COMPLAINT, NextStep.GENERIC_REPLY): (Priority.P4, Priority.P4),
    (Category.EXPERT_COMPLAINT, NextStep.RECORD_EXPERT_COMPLAINT): (Priority.P3, Priority.P3),
    (Category.EXPERT_COMPLAINT, NextStep.ESCALATE_HUMAN): (Priority.P2, Priority.P1),
    (Category.PAYMENT_ISSUE, NextStep.GENERIC_REPLY): (Priority.P4, Priority.P4),
    (Category.PAYMENT_ISSUE, NextStep.SEND_REFUND_POLICY): (Priority.P3, Priority.P2),
    (Category.PAYMENT_ISSUE, NextStep.ESCALATE_HUMAN): (Priority.P2, Priority.P1),
    (Category.THREAT, NextStep.GENERIC_REPLY): (Priority.P4, Priority.P4),
    (Category.THREAT, NextStep.ESCALATE_HUMAN): (Priority.P2, Priority.P1),
    (Category.OTHER, NextStep.NO_REPLY): (Priority.P4, Priority.P4),
    (Category.OTHER, NextStep.ESCALATE_HUMAN): (Priority.P3, Priority.P3),
}


def clamp_priority(priority: Priority, base: Priority, highest: Priority) -> Priority:
    """Bound `priority` to [base, highest]. P1 is the most urgent, so it sorts lowest."""
    return min(max(priority, highest), base)


def apply_rules(classification: LLMClassification, *, failed: bool = False) -> Classification:
    """Clamp priority to the table and decide `needs_human_review` and `review_reasons`.

    For a valid (category, next_step) pair, the LLM's priority is clamped into the pair's
    [base, highest] range, so the model can only choose *whether* a raise fact applies,
    never an arbitrary level. An invalid pair is left as returned and flagged instead.

    A ticket is flagged only if one of these holds: the response is `escalate_human`,
    the (category, next_step) pair is not in `PRIORITY_TABLE`, or the classification
    pipeline itself failed and returned a fallback result. `confidence` is informational
    only and never affects the flag.
    """
    reasons: list[str] = []
    data = classification.model_dump()

    bounds = PRIORITY_TABLE.get((classification.category, classification.next_step))
    if bounds is None:
        reasons.append("invalid_next_step")
    else:
        data["priority"] = clamp_priority(classification.priority, *bounds)

    if classification.next_step is NextStep.ESCALATE_HUMAN:
        reasons.append("escalate_human")
    if failed:
        reasons.append("classification_failed")

    return Classification(
        **data,
        needs_human_review=bool(reasons),
        review_reasons=reasons,
    )
