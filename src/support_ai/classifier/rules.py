"""Deterministic human-in-the-loop rules (SPEC.md's HITL section).

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

# Categories whose next step is fixed by policy, regardless of what the LLM picked.
CATEGORY_NEXT_STEP: dict[Category, NextStep] = {
    Category.USAGE_HELP: NextStep.SEND_KB_ARTICLE,
    Category.UNCLEAR: NextStep.REQUEST_MORE_INFO,
}


def apply_rules(classification: LLMClassification, *, failed: bool = False) -> Classification:
    """Decide `needs_human_review` and `review_reasons` from `classification` and `failed`.

    A ticket is flagged only if one of these holds: P1 priority, the customer explicitly
    requested a live human agent (`requests_human`), or the classification pipeline
    itself failed and returned a fallback result. `confidence` and `secondary_categories`
    remain informational only — they no longer affect the flag.

    Independently of the flag, `next_step` is forced to match `CATEGORY_NEXT_STEP` for
    `usage_help` and `unclear`, so those two categories always route the same way no
    matter what the LLM returned for `next_step`.
    """
    reasons: list[str] = []

    if classification.priority is Priority.P1:
        reasons.append("p1_priority")
    if classification.requests_human:
        reasons.append("human_requested")
    if failed:
        reasons.append("classification_failed")

    data = classification.model_dump()
    if classification.category in CATEGORY_NEXT_STEP:
        data["next_step"] = CATEGORY_NEXT_STEP[classification.category]

    return Classification(
        **data,
        needs_human_review=bool(reasons),
        review_reasons=reasons,
    )
