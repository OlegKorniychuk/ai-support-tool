"""Pure functions that turn per-ticket eval records into summary metrics.

No I/O, no LLM calls: everything here is unit-testable on handmade records.
"""

import math

from pydantic import BaseModel


class EvalRecord(BaseModel):
    """One ticket's expected-vs-actual outcome from one eval run."""

    ticket_id: str
    expected_category: str
    actual_category: str
    expected_priority: str
    actual_priority: str
    expected_needs_review: bool
    actual_needs_review: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cost_usd: float
    error: str | None = None

    @property
    def category_pass(self) -> bool:
        return self.expected_category == self.actual_category

    @property
    def priority_pass(self) -> bool:
        return self.expected_priority == self.actual_priority


def category_accuracy(records: list[EvalRecord]) -> float:
    """Fraction of records whose actual category matches the expected one."""
    if not records:
        return 0.0
    return sum(r.category_pass for r in records) / len(records)


def priority_accuracy(records: list[EvalRecord]) -> float:
    """Fraction of records whose actual priority matches the expected one."""
    if not records:
        return 0.0
    return sum(r.priority_pass for r in records) / len(records)


def human_review_recall(records: list[EvalRecord]) -> float:
    """Of the tickets expected to need review, the fraction actually flagged.

    Tickets that don't expect review are excluded from the denominator: recall measures
    whether the reviews we need happen, not how often we (correctly) skip review.
    An empty expected-review set has perfect (1.0) recall — there is nothing to miss.
    """
    expected_review = [r for r in records if r.expected_needs_review]
    if not expected_review:
        return 1.0
    return sum(r.actual_needs_review for r in expected_review) / len(expected_review)


def human_review_precision(records: list[EvalRecord]) -> float:
    """Of the tickets actually flagged for review, the fraction that were expected to be.

    Recall's complement: recall asks whether the reviews we need happen; precision asks
    whether the reviews we trigger are ones we actually needed, so false-positive
    escalations count against it. Tickets that aren't flagged are excluded from the
    denominator. A ticket set with no flagged tickets has perfect (1.0) precision — there
    are no false escalations to find.
    """
    flagged = [r for r in records if r.actual_needs_review]
    if not flagged:
        return 1.0
    return sum(r.expected_needs_review for r in flagged) / len(flagged)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. `pct` is a fraction in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct * len(ordered)) - 1)
    return ordered[index]


def latency_p50(records: list[EvalRecord]) -> float:
    return _percentile([r.latency_ms for r in records], 0.50)


def latency_p95(records: list[EvalRecord]) -> float:
    return _percentile([r.latency_ms for r in records], 0.95)


def cache_hit_rate(records: list[EvalRecord]) -> float:
    """Fraction of all input tokens that were served from the provider's prompt cache."""
    total_input = sum(r.input_tokens for r in records)
    if not total_input:
        return 0.0
    return sum(r.cached_input_tokens for r in records) / total_input


def cost_per_ticket(records: list[EvalRecord]) -> float:
    """Average dollar cost of one call across `records`."""
    if not records:
        return 0.0
    return sum(r.cost_usd for r in records) / len(records)
