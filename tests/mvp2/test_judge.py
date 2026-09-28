"""Schema, cross-validation and aggregates for judge verdicts (`eval/judge.py`)."""

import pytest
from pydantic import ValidationError

from support_ai.assistant.schema import Tone
from support_ai.eval.judge import (
    DraftVerdict,
    JudgeRun,
    TicketVerdict,
    addresses_request_rate,
    distinct_rate,
    faithful_rate,
    mean_tone_score,
    summary_accurate_rate,
    validate_against_run,
)
from support_ai.eval.reply_metrics import ReplyEvalRecord


def _record(**overrides) -> ReplyEvalRecord:
    defaults = dict(
        ticket_id="r001",
        ticket_text="How do I change my birth time?",
        tags=["answerable"],
        expected_kb_ids=["edit-birth-data"],
        retrieved_ids=["edit-birth-data"],
        cited_article_id="edit-birth-data",
        expected_needs_judgment=False,
        actual_needs_judgment=False,
        expected_reasons=[],
        actual_reasons=[],
        summary="Customer asks how to change their birth time.",
        drafts={
            "formal": "Dear customer, ...",
            "empathetic": "I understand ...",
            "short": "Open ...",
        },
        latency_ms=500,
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.002,
    )
    defaults.update(overrides)
    return ReplyEvalRecord.model_validate(defaults)


def _flagged_record(**overrides) -> ReplyEvalRecord:
    defaults = dict(
        ticket_id="r002",
        expected_needs_judgment=True,
        actual_needs_judgment=True,
        expected_reasons=["kb_not_found"],
        actual_reasons=["kb_not_found"],
        expected_kb_ids=[],
        retrieved_ids=[],
        cited_article_id=None,
        drafts={},
    )
    defaults.update(overrides)
    return _record(**defaults)


def _draft_verdict(tone: Tone, **overrides) -> DraftVerdict:
    defaults = dict(tone=tone, tone_score=4, faithful=True, addresses_request=True)
    defaults.update(overrides)
    return DraftVerdict.model_validate(defaults)


def _drafted_verdict(ticket_id="r001", **overrides) -> TicketVerdict:
    defaults = dict(
        ticket_id=ticket_id,
        summary_accurate=True,
        distinct=True,
        drafts=[
            _draft_verdict(Tone.FORMAL),
            _draft_verdict(Tone.EMPATHETIC),
            _draft_verdict(Tone.SHORT),
        ],
    )
    defaults.update(overrides)
    return TicketVerdict.model_validate(defaults)


def _flagged_verdict(ticket_id="r002", **overrides) -> TicketVerdict:
    defaults = dict(ticket_id=ticket_id, summary_accurate=True, distinct=None, drafts=[])
    defaults.update(overrides)
    return TicketVerdict.model_validate(defaults)


def _judge_run(verdicts, **overrides) -> JudgeRun:
    defaults = dict(
        run="gpt-5.4-nano_v1_20260928T120000Z",
        judge_model="claude-sonnet-5",
        rubric_version="v1",
        judged_at="2026-09-28T12:30:00Z",
        verdicts=verdicts,
    )
    defaults.update(overrides)
    return JudgeRun.model_validate(defaults)


# --- schema -----------------------------------------------------------------------------------


def test_draft_verdict_rejects_extra_key():
    with pytest.raises(ValidationError):
        DraftVerdict.model_validate(
            {"tone": "formal", "tone_score": 4, "faithful": True, "addresses_request": True, "x": 1}
        )


def test_draft_verdict_tone_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        DraftVerdict.model_validate(
            {"tone": "formal", "tone_score": 6, "faithful": True, "addresses_request": True}
        )
    with pytest.raises(ValidationError):
        DraftVerdict.model_validate(
            {"tone": "formal", "tone_score": 0, "faithful": True, "addresses_request": True}
        )


def test_ticket_verdict_rejects_extra_key():
    with pytest.raises(ValidationError):
        TicketVerdict.model_validate(
            {"ticket_id": "r001", "summary_accurate": True, "distinct": True, "x": 1}
        )


def test_judge_run_rejects_extra_key():
    with pytest.raises(ValidationError):
        JudgeRun.model_validate(
            {
                "run": "x",
                "judge_model": "claude-sonnet-5",
                "rubric_version": "v1",
                "judged_at": "2026-09-28T12:30:00Z",
                "verdicts": [],
                "x": 1,
            }
        )


# --- validate_against_run: the clean path ----------------------------------------------------


def test_validate_against_run_passes_for_matching_run():
    records = [_record(), _flagged_record()]
    judge_run = _judge_run([_drafted_verdict(), _flagged_verdict()])
    validate_against_run(judge_run, records)  # must not raise


# --- validate_against_run: each problem --------------------------------------------------------


def test_validate_against_run_unknown_ticket_id():
    records = [_record()]
    judge_run = _judge_run([_drafted_verdict(), _drafted_verdict(ticket_id="ghost")])
    with pytest.raises(ValueError, match="unknown ticket id"):
        validate_against_run(judge_run, records)


def test_validate_against_run_missing_verdict():
    records = [_record(), _flagged_record()]
    judge_run = _judge_run([_drafted_verdict()])
    with pytest.raises(ValueError, match="missing verdict"):
        validate_against_run(judge_run, records)


def test_validate_against_run_duplicate_verdict():
    records = [_record()]
    judge_run = _judge_run([_drafted_verdict(), _drafted_verdict()])
    with pytest.raises(ValueError, match="duplicate verdicts"):
        validate_against_run(judge_run, records)


def test_validate_against_run_drafted_ticket_missing_a_tone():
    records = [_record()]
    verdict = _drafted_verdict(
        drafts=[_draft_verdict(Tone.FORMAL), _draft_verdict(Tone.EMPATHETIC)]
    )
    judge_run = _judge_run([verdict])
    with pytest.raises(ValueError, match="expected exactly one DraftVerdict per tone"):
        validate_against_run(judge_run, records)


def test_validate_against_run_drafted_ticket_duplicate_tone():
    records = [_record()]
    verdict = _drafted_verdict(
        drafts=[
            _draft_verdict(Tone.FORMAL),
            _draft_verdict(Tone.FORMAL),
            _draft_verdict(Tone.SHORT),
        ]
    )
    judge_run = _judge_run([verdict])
    with pytest.raises(ValueError, match="expected exactly one DraftVerdict per tone"):
        validate_against_run(judge_run, records)


def test_validate_against_run_drafted_ticket_null_distinct():
    records = [_record()]
    judge_run = _judge_run([_drafted_verdict(distinct=None)])
    with pytest.raises(ValueError, match="has drafts but distinct is null"):
        validate_against_run(judge_run, records)


def test_validate_against_run_flagged_ticket_with_draft_verdicts():
    records = [_flagged_record()]
    verdict = _flagged_verdict(drafts=[_draft_verdict(Tone.FORMAL)], distinct=None)
    judge_run = _judge_run([verdict])
    with pytest.raises(ValueError, match="has no drafts but verdict has draft"):
        validate_against_run(judge_run, records)


def test_validate_against_run_flagged_ticket_with_non_null_distinct():
    records = [_flagged_record()]
    judge_run = _judge_run([_flagged_verdict(distinct=False)])
    with pytest.raises(ValueError, match="has no drafts but distinct isn't null"):
        validate_against_run(judge_run, records)


def test_validate_against_run_reports_every_problem_at_once():
    records = [_record(), _flagged_record()]
    judge_run = _judge_run([_drafted_verdict(ticket_id="ghost")])  # unknown + missing r002
    with pytest.raises(ValueError) as excinfo:
        validate_against_run(judge_run, records)
    message = str(excinfo.value)
    assert "unknown ticket id" in message
    assert "missing verdict" in message


# --- aggregates --------------------------------------------------------------------------------


def test_mean_tone_score_per_tone():
    verdicts = [
        _drafted_verdict(
            ticket_id="r1",
            drafts=[
                _draft_verdict(Tone.FORMAL, tone_score=5),
                _draft_verdict(Tone.EMPATHETIC, tone_score=3),
                _draft_verdict(Tone.SHORT, tone_score=4),
            ],
        ),
        _drafted_verdict(
            ticket_id="r2",
            drafts=[
                _draft_verdict(Tone.FORMAL, tone_score=3),
                _draft_verdict(Tone.EMPATHETIC, tone_score=5),
                _draft_verdict(Tone.SHORT, tone_score=4),
            ],
        ),
    ]
    assert mean_tone_score(verdicts, Tone.FORMAL) == 4.0
    assert mean_tone_score(verdicts, Tone.EMPATHETIC) == 4.0
    assert mean_tone_score(verdicts, Tone.SHORT) == 4.0


def test_mean_tone_score_empty_is_zero():
    assert mean_tone_score([], Tone.FORMAL) == 0.0


def test_faithful_rate():
    verdicts = [
        _drafted_verdict(
            drafts=[
                _draft_verdict(Tone.FORMAL, faithful=True),
                _draft_verdict(Tone.EMPATHETIC, faithful=False),
                _draft_verdict(Tone.SHORT, faithful=True),
            ]
        )
    ]
    assert faithful_rate(verdicts) == pytest.approx(2 / 3)


def test_faithful_rate_empty_is_zero():
    assert faithful_rate([]) == 0.0
    assert faithful_rate([_flagged_verdict()]) == 0.0


def test_addresses_request_rate():
    verdicts = [
        _drafted_verdict(
            drafts=[
                _draft_verdict(Tone.FORMAL, addresses_request=True),
                _draft_verdict(Tone.EMPATHETIC, addresses_request=True),
                _draft_verdict(Tone.SHORT, addresses_request=False),
            ]
        )
    ]
    assert addresses_request_rate(verdicts) == pytest.approx(2 / 3)


def test_summary_accurate_rate_counts_all_tickets():
    verdicts = [
        _drafted_verdict(ticket_id="r1", summary_accurate=True),
        _flagged_verdict(ticket_id="r2", summary_accurate=False),
    ]
    assert summary_accurate_rate(verdicts) == 0.5


def test_summary_accurate_rate_empty_is_zero():
    assert summary_accurate_rate([]) == 0.0


def test_distinct_rate_only_over_drafted_tickets():
    verdicts = [
        _drafted_verdict(ticket_id="r1", distinct=True),
        _drafted_verdict(ticket_id="r2", distinct=False),
        _flagged_verdict(ticket_id="r3"),  # distinct=None, excluded from the denominator
    ]
    assert distinct_rate(verdicts) == 0.5


def test_distinct_rate_empty_is_zero():
    assert distinct_rate([]) == 0.0
    assert distinct_rate([_flagged_verdict()]) == 0.0
