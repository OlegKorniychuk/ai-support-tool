"""Deterministic post-LLM rules: priority from the table, and the human-in-the-loop flag.

The LLM never decides `priority` or `needs_human_review` — this module does, from the
LLM's other output fields, the ticket text, and whether the call ultimately failed.
See SPEC.md, D1.7.
"""

from support_ai.classifier.schema import (
    Category,
    Classification,
    LLMClassification,
    NextStep,
    Priority,
)
from support_ai.core.text import quote_in_text

# Every valid (category, response) pair, with its base priority and the priority a stated
# fact raises it to. Mirrors the table in the current prompt; test_prompts.py checks the
# two never drift apart. A pair missing from here is an invalid classification.
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

# Priority for a (category, next_step) pair the table doesn't know. Such a ticket is also
# flagged for review, so this only orders it in the queue: same as `other` / escalate.
INVALID_PAIR_PRIORITY = Priority.P3


def compute_priority(category: Category, next_step: NextStep, *, raised: bool) -> Priority:
    """The pair's raised priority if `raised`, else its base; see `PRIORITY_TABLE`."""
    bounds = PRIORITY_TABLE.get((category, next_step))
    if bounds is None:
        return INVALID_PAIR_PRIORITY
    base, raised_to = bounds
    return raised_to if raised else base


def apply_rules(
    classification: LLMClassification, *, ticket: str, failed: bool = False
) -> Classification:
    """Compute priority and decide `needs_human_review` and `review_reasons`.

    Priority is raised only when the pair can be raised and the LLM's
    `priority_raise_evidence` is really a quote from `ticket`. Otherwise the evidence is
    dropped, so the output shows evidence exactly when the priority was raised.

    A ticket is flagged only if one of these holds: the response is `escalate_human`,
    the (category, next_step) pair is not in `PRIORITY_TABLE`, or the classification
    pipeline itself failed and returned a fallback result. `confidence` is informational
    only and never affects the flag.
    """
    reasons: list[str] = []
    pair = (classification.category, classification.next_step)
    bounds = PRIORITY_TABLE.get(pair)
    if bounds is None:
        reasons.append("invalid_next_step")

    evidence = classification.priority_raise_evidence
    raised = (
        bounds is not None
        and bounds[0] != bounds[1]
        and evidence is not None
        and quote_in_text(evidence, ticket)
    )

    if classification.next_step is NextStep.ESCALATE_HUMAN:
        reasons.append("escalate_human")
    if failed:
        reasons.append("classification_failed")

    data = classification.model_dump()
    data["priority_raise_evidence"] = evidence if raised else None
    return Classification(
        **data,
        priority=compute_priority(*pair, raised=raised),
        needs_human_review=bool(reasons),
        review_reasons=reasons,
    )
