"""Handmade records for every reply-eval metric, including empty-denominator behavior."""

import pytest

from support_ai.assistant.tone_checks import ToneCheck
from support_ai.eval.reply_metrics import (
    ReplyEvalRecord,
    cache_hit_rate,
    citation_accuracy,
    cost_per_ticket,
    distinct_ok_rate,
    drafts_shown_rate,
    dropped_evidence_rate,
    error_rate,
    formal_ok_rate,
    judgment_precision,
    judgment_recall,
    latency_p50,
    latency_p95,
    reason_match_rate,
    retrieval_hit_rate,
    short_ok_rate,
)


def _check(*, short_ok=True, formal_ok=True, distinct_ok=True) -> ToneCheck:
    return ToneCheck(
        short_words=10,
        short_ok=short_ok,
        formal_contractions=[] if formal_ok else ["don't"],
        formal_ok=formal_ok,
        max_pair_overlap=0.1 if distinct_ok else 0.9,
        distinct_ok=distinct_ok,
    )


def _record(**overrides) -> ReplyEvalRecord:
    defaults = dict(
        ticket_id="r001",
        ticket_text="How do I change my birth time in my profile?",
        tags=["answerable"],
        expected_kb_ids=["edit-birth-data"],
        retrieved_ids=["edit-birth-data"],
        cited_article_id="edit-birth-data",
        expected_needs_judgment=False,
        actual_needs_judgment=False,
        expected_reasons=[],
        actual_reasons=[],
        summary="Customer asks how to change their birth time.",
        kb_quote="Open Profile → Birth details",
        latency_ms=500,
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.002,
    )
    defaults.update(overrides)
    return ReplyEvalRecord.model_validate(defaults)


# --- retrieval_hit_rate --------------------------------------------------------------------


def test_retrieval_hit_rate_all_hit():
    records = [_record(), _record(ticket_id="r002")]
    assert retrieval_hit_rate(records) == 1.0


def test_retrieval_hit_rate_partial():
    records = [
        _record(ticket_id="r001", retrieved_ids=["edit-birth-data"]),
        _record(ticket_id="r002", retrieved_ids=["support-hours"]),
    ]
    assert retrieval_hit_rate(records) == 0.5


def test_retrieval_hit_rate_excludes_flagged_and_no_expected_id_records():
    flagged = _record(
        ticket_id="r_flagged",
        expected_needs_judgment=True,
        actual_needs_judgment=True,
        expected_reasons=["kb_not_found"],
        actual_reasons=["kb_not_found"],
        expected_kb_ids=[],
        retrieved_ids=[],
        cited_article_id=None,
    )
    no_expected_id = _record(ticket_id="r_gap_like", expected_kb_ids=[], retrieved_ids=[])
    hit = _record(ticket_id="r_hit")
    assert retrieval_hit_rate([flagged, no_expected_id, hit]) == 1.0


def test_retrieval_hit_rate_empty_denominator_is_perfect():
    flagged = _record(
        expected_needs_judgment=True,
        actual_needs_judgment=True,
        expected_reasons=["kb_not_found"],
        actual_reasons=["kb_not_found"],
        expected_kb_ids=[],
    )
    assert retrieval_hit_rate([flagged]) == 1.0
    assert retrieval_hit_rate([]) == 1.0


# --- citation_accuracy --------------------------------------------------------------------


def test_citation_accuracy_all_correct():
    records = [_record(), _record(ticket_id="r002")]
    assert citation_accuracy(records) == 1.0


def test_citation_accuracy_wrong_citation():
    records = [_record(cited_article_id="support-hours")]
    assert citation_accuracy(records) == 0.0


def test_citation_accuracy_none_citation_fails():
    records = [_record(cited_article_id=None)]
    assert citation_accuracy(records) == 0.0


def test_citation_accuracy_empty_denominator_is_perfect():
    assert citation_accuracy([]) == 1.0


# --- judgment_recall / judgment_precision --------------------------------------------------


def _flagged(**overrides) -> ReplyEvalRecord:
    defaults = dict(
        expected_needs_judgment=True,
        actual_needs_judgment=True,
        expected_reasons=["kb_not_found"],
        actual_reasons=["kb_not_found"],
        expected_kb_ids=[],
        retrieved_ids=[],
        cited_article_id=None,
    )
    defaults.update(overrides)
    return _record(**defaults)


def test_judgment_recall_full_marks():
    records = [_flagged(ticket_id="r1"), _flagged(ticket_id="r2")]
    assert judgment_recall(records) == 1.0


def test_judgment_recall_missed_one():
    records = [
        _flagged(ticket_id="r1"),
        _flagged(ticket_id="r2", actual_needs_judgment=False, actual_reasons=[]),
    ]
    assert judgment_recall(records) == 0.5


def test_judgment_recall_empty_denominator_is_perfect():
    assert judgment_recall([_record()]) == 1.0
    assert judgment_recall([]) == 1.0


def test_judgment_precision_full_marks():
    records = [_flagged(ticket_id="r1"), _flagged(ticket_id="r2")]
    assert judgment_precision(records) == 1.0


def test_judgment_precision_false_positive():
    records = [
        _flagged(ticket_id="r1"),
        _record(ticket_id="r2", actual_needs_judgment=True, actual_reasons=["kb_not_found"]),
    ]
    assert judgment_precision(records) == 0.5


def test_judgment_precision_empty_denominator_is_perfect():
    assert judgment_precision([_record()]) == 1.0
    assert judgment_precision([]) == 1.0


# --- reason_match_rate ----------------------------------------------------------------------


def test_reason_match_rate_exact_match():
    records = [_flagged()]
    assert reason_match_rate(records) == 1.0


def test_reason_match_rate_extra_actual_reason_still_matches():
    records = [_flagged(actual_reasons=["kb_not_found", "account_specific"])]
    assert reason_match_rate(records) == 1.0


def test_reason_match_rate_missing_reason_fails():
    records = [
        _flagged(ticket_id="r1"),
        _flagged(
            ticket_id="r2", expected_reasons=["conflicting_kb"], actual_reasons=["kb_not_found"]
        ),
    ]
    assert reason_match_rate(records) == 0.5


def test_reason_match_rate_empty_denominator_is_perfect():
    assert reason_match_rate([_record()]) == 1.0
    assert reason_match_rate([]) == 1.0


# --- drafts_shown_rate -----------------------------------------------------------------------


def test_drafts_shown_rate_all_shown():
    records = [_record(drafts={"formal": "a", "empathetic": "b", "short": "c"})]
    assert drafts_shown_rate(records) == 1.0


def test_drafts_shown_rate_missing_drafts_counts_against_it():
    records = [
        _record(ticket_id="r1", drafts={"formal": "a", "empathetic": "b", "short": "c"}),
        _record(ticket_id="r2", drafts={}, error="generation_failed"),
    ]
    assert drafts_shown_rate(records) == 0.5


def test_drafts_shown_rate_excludes_flagged_tickets():
    flagged = _flagged(drafts={})
    answerable_with_drafts = _record(drafts={"formal": "a", "empathetic": "b", "short": "c"})
    assert drafts_shown_rate([flagged, answerable_with_drafts]) == 1.0


def test_drafts_shown_rate_empty_denominator_is_perfect():
    assert drafts_shown_rate([_flagged()]) == 1.0
    assert drafts_shown_rate([]) == 1.0


# --- short_ok_rate / formal_ok_rate / distinct_ok_rate -----------------------------------------


def test_tone_rates_all_pass():
    records = [_record(tone_check=_check())]
    assert short_ok_rate(records) == 1.0
    assert formal_ok_rate(records) == 1.0
    assert distinct_ok_rate(records) == 1.0


def test_tone_rates_mixed():
    records = [
        _record(ticket_id="r1", tone_check=_check(short_ok=True, formal_ok=True, distinct_ok=True)),
        _record(
            ticket_id="r2", tone_check=_check(short_ok=False, formal_ok=False, distinct_ok=False)
        ),
    ]
    assert short_ok_rate(records) == 0.5
    assert formal_ok_rate(records) == 0.5
    assert distinct_ok_rate(records) == 0.5


def test_tone_rates_ignore_records_without_tone_check():
    records = [_record(tone_check=None), _record(ticket_id="r2", tone_check=_check())]
    assert short_ok_rate(records) == 1.0


def test_tone_rates_empty_denominator_is_perfect():
    records = [_record(tone_check=None)]
    assert short_ok_rate(records) == 1.0
    assert formal_ok_rate(records) == 1.0
    assert distinct_ok_rate(records) == 1.0
    assert short_ok_rate([]) == 1.0


# --- dropped_evidence_rate / error_rate -------------------------------------------------------


def test_dropped_evidence_rate_counts_records_with_any_dropped_evidence():
    records = [
        _record(ticket_id="r1", dropped_evidence=["evidence not found in ticket: 'x'"]),
        _record(ticket_id="r2", dropped_evidence=[]),
    ]
    assert dropped_evidence_rate(records) == 0.5


def test_dropped_evidence_rate_empty_run_is_zero():
    assert dropped_evidence_rate([]) == 0.0


def test_error_rate_counts_errored_records():
    records = [
        _record(ticket_id="r1", error="retrieval_failed"),
        _record(ticket_id="r2", error=None),
    ]
    assert error_rate(records) == 0.5


def test_error_rate_empty_run_is_zero():
    assert error_rate([]) == 0.0


# --- latency percentiles / cost / cache hit rate ------------------------------------------------


def test_latency_percentiles_on_handmade_records():
    latencies = [100, 200, 300, 400]
    records = [_record(ticket_id=str(i), latency_ms=ms) for i, ms in enumerate(latencies)]
    assert latency_p50(records) == 200
    assert latency_p95(records) == 400


def test_cost_per_ticket_averages():
    records = [_record(cost_usd=0.002), _record(ticket_id="r2", cost_usd=0.004)]
    assert cost_per_ticket(records) == 0.003


def test_cost_per_ticket_empty_run_is_zero():
    assert cost_per_ticket([]) == 0.0


def test_cache_hit_rate_computed_over_total_input_tokens():
    records = [
        _record(input_tokens=100, cached_input_tokens=50),
        _record(ticket_id="r2", input_tokens=100, cached_input_tokens=0),
    ]
    assert cache_hit_rate(records) == 0.25


def test_cache_hit_rate_empty_run_is_zero():
    assert cache_hit_rate([]) == 0.0


def test_cache_hit_rate_zero_input_tokens_is_zero():
    assert cache_hit_rate([_record(input_tokens=0, cached_input_tokens=0)]) == 0.0


# --- ReplyEvalRecord.is_answerable ---------------------------------------------------------------


def test_is_answerable_true_when_not_expected_to_need_judgment():
    assert _record(expected_needs_judgment=False).is_answerable is True


def test_is_answerable_false_when_expected_to_need_judgment():
    assert _flagged().is_answerable is False


@pytest.mark.parametrize("tags", [[], ["injection"]])
def test_is_answerable_ignores_tags(tags):
    # Tags are informational only; `is_answerable` is purely `not expected_needs_judgment`,
    # which is what makes the injection ticket count as answerable.
    assert _record(expected_needs_judgment=False, tags=tags).is_answerable is True
