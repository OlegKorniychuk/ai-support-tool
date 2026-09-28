"""Pure functions that turn per-ticket reply-assistant eval records into summary metrics.

No I/O, no LLM calls: everything here is unit-testable on handmade records, mirroring
`eval/metrics.py`'s classifier metrics. Judge aggregates and judge-vs-hand agreement are
added in a later task, once the judge schema (`eval/judge.py`) exists.

Two families of empty-denominator convention, matching `eval/metrics.py`:
- A metric over a derived *subset* (e.g. "of answerable tickets...", "of tickets expected
  to need judgment...") is trivially perfect — 1.0 — when that subset is empty: there is
  nothing in that condition to have gotten wrong.
- A metric that reports how often something happened across the *whole* run (an error, a
  dropped fact, average cost) is 0.0 on an empty run: there is nothing to report.
"""

from collections.abc import Callable

from pydantic import BaseModel

from support_ai.assistant.tone_checks import ToneCheck
from support_ai.eval.metrics import percentile


class ReplyEvalRecord(BaseModel):
    """One ticket's expected-vs-actual outcome from one reply-assistant eval run."""

    ticket_id: str
    tags: list[str]
    expected_kb_ids: list[str]
    retrieved_ids: list[str]  # hit order, best first
    cited_article_id: str | None  # the verified `kb_source` id, or None
    expected_needs_judgment: bool
    actual_needs_judgment: bool
    expected_reasons: list[str]
    actual_reasons: list[str]
    dropped_evidence: list[str] = []
    summary: str
    drafts: dict[str, str] = {}  # tone -> text; empty when drafts weren't shown
    tone_check: ToneCheck | None = None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cost_usd: float
    error: str | None = None  # e.g. "retrieval_failed" / "generation_failed"

    @property
    def is_answerable(self) -> bool:
        """A ticket expected to get drafts rather than the human-judgment banner. Includes
        the prompt-injection ticket: it's expected to answer normally, with faithful
        drafts — the judge scores faithfulness, not this module.
        """
        return not self.expected_needs_judgment


def retrieval_hit_rate(records: list[ReplyEvalRecord]) -> float:
    """Hit@k: of answerable tickets that expect a KB id, the fraction where at least one
    expected id is anywhere among the retrieved hits (order doesn't matter here; citation
    accuracy below checks what was actually cited).
    """
    subset = [r for r in records if r.is_answerable and r.expected_kb_ids]
    if not subset:
        return 1.0
    hits = sum(any(kb_id in r.retrieved_ids for kb_id in r.expected_kb_ids) for r in subset)
    return hits / len(subset)


def citation_accuracy(records: list[ReplyEvalRecord]) -> float:
    """Of answerable tickets, the fraction where the cited article is one of the expected
    ones (a `None` citation never counts, even if `expected_kb_ids` is empty)."""
    subset = [r for r in records if r.is_answerable]
    if not subset:
        return 1.0
    return sum(r.cited_article_id in r.expected_kb_ids for r in subset) / len(subset)


def judgment_recall(records: list[ReplyEvalRecord]) -> float:
    """Of tickets expected to need human judgment, the fraction actually flagged. Mirrors
    `metrics.human_review_recall`."""
    expected_flagged = [r for r in records if r.expected_needs_judgment]
    if not expected_flagged:
        return 1.0
    return sum(r.actual_needs_judgment for r in expected_flagged) / len(expected_flagged)


def judgment_precision(records: list[ReplyEvalRecord]) -> float:
    """Of tickets actually flagged, the fraction that were expected to be. Mirrors
    `metrics.human_review_precision`."""
    actual_flagged = [r for r in records if r.actual_needs_judgment]
    if not actual_flagged:
        return 1.0
    return sum(r.expected_needs_judgment for r in actual_flagged) / len(actual_flagged)


def reason_match_rate(records: list[ReplyEvalRecord]) -> float:
    """Of tickets expected to need judgment, the fraction whose actual reasons are a
    superset of the expected ones. An extra, unexpected reason doesn't count against it —
    only a missing expected one does.
    """
    expected_flagged = [r for r in records if r.expected_needs_judgment]
    if not expected_flagged:
        return 1.0
    matches = sum(set(r.expected_reasons) <= set(r.actual_reasons) for r in expected_flagged)
    return matches / len(expected_flagged)


def drafts_shown_rate(records: list[ReplyEvalRecord]) -> float:
    """Of answerable tickets, the fraction that actually got 3 drafts shown."""
    subset = [r for r in records if r.is_answerable]
    if not subset:
        return 1.0
    return sum(bool(r.drafts) for r in subset) / len(subset)


def short_ok_rate(records: list[ReplyEvalRecord]) -> float:
    """Of records with a `tone_check` (drafts were shown), the fraction whose short draft
    is within `SHORT_MAX_WORDS`."""
    return _tone_check_rate(records, lambda check: check.short_ok)


def formal_ok_rate(records: list[ReplyEvalRecord]) -> float:
    """Of records with a `tone_check`, the fraction whose formal draft has no
    contractions."""
    return _tone_check_rate(records, lambda check: check.formal_ok)


def distinct_ok_rate(records: list[ReplyEvalRecord]) -> float:
    """Of records with a `tone_check`, the fraction where the 3 tones are distinct enough
    (see `tone_checks.DISTINCT_MAX_OVERLAP`)."""
    return _tone_check_rate(records, lambda check: check.distinct_ok)


def _tone_check_rate(
    records: list[ReplyEvalRecord], predicate: Callable[[ToneCheck], bool]
) -> float:
    checks = [r.tone_check for r in records if r.tone_check is not None]
    if not checks:
        return 1.0
    return sum(predicate(check) for check in checks) / len(checks)


def dropped_evidence_rate(records: list[ReplyEvalRecord]) -> float:
    """Fraction of all records where some reported evidence or conflict didn't verify and
    was dropped. 0.0 on an empty run."""
    if not records:
        return 0.0
    return sum(bool(r.dropped_evidence) for r in records) / len(records)


def error_rate(records: list[ReplyEvalRecord]) -> float:
    """Fraction of all records that failed (retrieval or generation). 0.0 on an empty
    run."""
    if not records:
        return 0.0
    return sum(r.error is not None for r in records) / len(records)


def latency_p50(records: list[ReplyEvalRecord]) -> float:
    return percentile([r.latency_ms for r in records], 0.50)


def latency_p95(records: list[ReplyEvalRecord]) -> float:
    return percentile([r.latency_ms for r in records], 0.95)


def cost_per_ticket(records: list[ReplyEvalRecord]) -> float:
    """Average dollar cost of one ticket across `records`. 0.0 on an empty run."""
    if not records:
        return 0.0
    return sum(r.cost_usd for r in records) / len(records)


def cache_hit_rate(records: list[ReplyEvalRecord]) -> float:
    """Fraction of all input tokens served from the provider's prompt cache. 0.0 when
    there were no input tokens at all."""
    total_input = sum(r.input_tokens for r in records)
    if not total_input:
        return 0.0
    return sum(r.cached_input_tokens for r in records) / total_input
