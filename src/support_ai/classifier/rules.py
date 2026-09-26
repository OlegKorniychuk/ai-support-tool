"""Deterministic human-in-the-loop rules (SPEC.md's HITL section).

The LLM never decides `needs_human_review` — this module does, purely from the LLM's
other output fields plus whether the call ultimately failed. See SPEC.md, D1.7.
"""

from support_ai.classifier.schema import Classification, LLMClassification, Priority
from support_ai.core.config import HITL_CONFIDENCE_THRESHOLD

# Categories that always warrant a human look, regardless of confidence.
REVIEW_CATEGORIES = frozenset({"refund_request", "expert_complaint"})


def apply_rules(
    classification: LLMClassification,
    *,
    failed: bool = False,
    confidence_threshold: float = HITL_CONFIDENCE_THRESHOLD,
) -> Classification:
    """Decide `needs_human_review` and `review_reasons` from `classification` and `failed`.

    A ticket is flagged if any of these holds: low confidence, a refund or expert
    complaint category, P1 priority, two or more topics (secondary categories present),
    or the classification pipeline itself failed and returned a fallback result.
    """
    reasons: list[str] = []

    if classification.confidence < confidence_threshold:
        reasons.append("low_confidence")
    if classification.category in REVIEW_CATEGORIES:
        reasons.append(str(classification.category))
    if classification.priority is Priority.P1:
        reasons.append("p1_priority")
    if len(classification.secondary_categories) >= 1:
        reasons.append("mixed_topics")
    if failed:
        reasons.append("classification_failed")

    return Classification(
        **classification.model_dump(),
        needs_human_review=bool(reasons),
        review_reasons=reasons,
    )
