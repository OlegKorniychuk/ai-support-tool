"""Pure dataframe-building logic (`eval/reply_report.py`) on handmade data: the join, the
latest-judgment selection, and per-ticket pass/fail."""

import pandas as pd
import pytest

from support_ai.eval.reply_report import (
    add_judge_verdicts,
    add_ticket_status,
    add_tone_check_flags,
    latest_judgment_per_run,
    model_comparison,
)

# --- latest_judgment_per_run --------------------------------------------------------------


def test_latest_judgment_per_run_picks_the_most_recent_row():
    df = pd.DataFrame(
        [
            {"run": "r1", "judged_at": "2026-09-28T10:00:00Z", "distinct_rate": 0.5},
            {"run": "r1", "judged_at": "2026-09-28T14:00:00Z", "distinct_rate": 1.0},
            {"run": "r2", "judged_at": "2026-09-28T09:00:00Z", "distinct_rate": 0.0},
        ]
    )

    result = latest_judgment_per_run(df)

    assert sorted(result["run"]) == ["r1", "r2"]
    r1_row = result[result["run"] == "r1"].iloc[0]
    assert r1_row["distinct_rate"] == 1.0


def test_latest_judgment_per_run_empty_input_stays_empty():
    df = pd.DataFrame(columns=["run", "judged_at"])
    assert latest_judgment_per_run(df).empty


# --- model_comparison ------------------------------------------------------------------------


def _summary_row(**overrides) -> dict:
    defaults = dict(
        timestamp="20260928T120000Z",
        run="gpt-5.4-nano_v1_20260928T120000Z",
        model="gpt-5.4-nano",
        prompt_version="v1",
        n_tickets=30,
        retrieval_hit_rate=0.9,
        citation_accuracy=0.85,
        latency_p50_ms=500.0,
        cost_per_ticket_usd=0.002,
    )
    defaults.update(overrides)
    return defaults


def _judge_row(**overrides) -> dict:
    defaults = dict(
        run="gpt-5.4-nano_v1_20260928T120000Z",
        judged_at="2026-09-28T14:00:00Z",
        judge_model="claude-sonnet-5",
        rubric_version="v1",
        n_tickets=30,
        n_drafted=27,
        tone_score_formal=4.5,
        faithful_rate=0.98,
        distinct_rate=0.93,
    )
    defaults.update(overrides)
    return defaults


def test_model_comparison_joins_matching_run():
    summary_df = pd.DataFrame([_summary_row()])
    judge_df = pd.DataFrame([_judge_row()])

    result = model_comparison(summary_df, judge_df)

    assert len(result) == 1
    assert result.iloc[0]["tone_score_formal"] == 4.5
    assert result.iloc[0]["faithful_rate"] == 0.98
    # the collision on `n_tickets` is resolved by suffixing the judge side
    assert result.iloc[0]["n_tickets"] == 30
    assert result.iloc[0]["n_tickets_judge"] == 30


def test_model_comparison_keeps_unjudged_run_with_blank_judge_columns():
    summary_df = pd.DataFrame([_summary_row(run="unjudged_run")])
    judge_df = pd.DataFrame([_judge_row()])  # judges a different run

    result = model_comparison(summary_df, judge_df)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["tone_score_formal"])
    assert pd.isna(result.iloc[0]["faithful_rate"])


def test_model_comparison_uses_the_latest_judgment_when_run_judged_twice():
    summary_df = pd.DataFrame([_summary_row()])
    judge_df = pd.DataFrame(
        [
            _judge_row(judged_at="2026-09-28T10:00:00Z", distinct_rate=0.5),
            _judge_row(judged_at="2026-09-28T16:00:00Z", distinct_rate=1.0),
        ]
    )

    result = model_comparison(summary_df, judge_df)

    assert result.iloc[0]["distinct_rate"] == 1.0


@pytest.mark.parametrize("judge_df", [None, pd.DataFrame(columns=["run", "judged_at"])])
def test_model_comparison_with_no_judge_summary_at_all(judge_df):
    summary_df = pd.DataFrame([_summary_row()])

    result = model_comparison(summary_df, judge_df)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["tone_score_formal"])
    assert pd.isna(result.iloc[0]["faithful_rate"])
    assert pd.isna(result.iloc[0]["distinct_rate"])


# --- add_ticket_status ------------------------------------------------------------------------


def _ticket_row(**overrides) -> dict:
    defaults = dict(
        ticket_id="r001",
        expected_needs_judgment=False,
        actual_needs_judgment=False,
        expected_reasons=[],
        actual_reasons=[],
        expected_kb_ids=["edit-birth-data"],
        cited_article_id="edit-birth-data",
    )
    defaults.update(overrides)
    return defaults


def test_add_ticket_status_answerable_pass():
    df = pd.DataFrame([_ticket_row()])
    result = add_ticket_status(df)
    assert result.iloc[0]["ticket_pass"]
    assert result.iloc[0]["status"] == "✅"


def test_add_ticket_status_answerable_wrong_citation_fails():
    df = pd.DataFrame([_ticket_row(cited_article_id="support-hours")])
    result = add_ticket_status(df)
    assert not result.iloc[0]["ticket_pass"]
    assert result.iloc[0]["status"] == "❌"
    assert not result.iloc[0]["citation_ok"]


def test_add_ticket_status_flag_mismatch_fails():
    df = pd.DataFrame([_ticket_row(expected_needs_judgment=False, actual_needs_judgment=True)])
    result = add_ticket_status(df)
    assert not result.iloc[0]["ticket_pass"]
    assert not result.iloc[0]["flag_match"]


def test_add_ticket_status_flagged_ticket_ignores_citation():
    # A flagged ticket has no expected/cited kb id; citation_ok is trivially true so it
    # doesn't count against a correctly-flagged ticket.
    df = pd.DataFrame(
        [
            _ticket_row(
                expected_needs_judgment=True,
                actual_needs_judgment=True,
                expected_reasons=["kb_not_found"],
                actual_reasons=["kb_not_found"],
                expected_kb_ids=[],
                cited_article_id=None,
            )
        ]
    )
    result = add_ticket_status(df)
    assert result.iloc[0]["citation_ok"]
    assert result.iloc[0]["ticket_pass"]


def test_add_ticket_status_missing_expected_reason_fails():
    df = pd.DataFrame(
        [
            _ticket_row(
                expected_needs_judgment=True,
                actual_needs_judgment=True,
                expected_reasons=["kb_not_found", "conflicting_kb"],
                actual_reasons=["kb_not_found"],
                expected_kb_ids=[],
                cited_article_id=None,
            )
        ]
    )
    result = add_ticket_status(df)
    assert not result.iloc[0]["reasons_ok"]
    assert not result.iloc[0]["ticket_pass"]


def test_add_ticket_status_extra_actual_reason_still_passes():
    df = pd.DataFrame(
        [
            _ticket_row(
                expected_needs_judgment=True,
                actual_needs_judgment=True,
                expected_reasons=["kb_not_found"],
                actual_reasons=["kb_not_found", "account_specific"],
                expected_kb_ids=[],
                cited_article_id=None,
            )
        ]
    )
    result = add_ticket_status(df)
    assert result.iloc[0]["reasons_ok"]
    assert result.iloc[0]["ticket_pass"]


# --- add_tone_check_flags ----------------------------------------------------------------------


def test_add_tone_check_flags_extracts_from_dict():
    df = pd.DataFrame(
        [
            {
                "ticket_id": "r001",
                "tone_check": {
                    "short_words": 10,
                    "short_ok": True,
                    "formal_contractions": [],
                    "formal_ok": True,
                    "max_pair_overlap": 0.2,
                    "distinct_ok": True,
                },
            }
        ]
    )
    result = add_tone_check_flags(df)
    assert result.iloc[0]["short_ok"]
    assert result.iloc[0]["formal_ok"]
    assert result.iloc[0]["distinct_ok"]


def test_add_tone_check_flags_none_when_no_tone_check():
    df = pd.DataFrame([{"ticket_id": "r002", "tone_check": None}])
    result = add_tone_check_flags(df)
    assert pd.isna(result.iloc[0]["short_ok"])
    assert pd.isna(result.iloc[0]["formal_ok"])
    assert pd.isna(result.iloc[0]["distinct_ok"])


# --- add_judge_verdicts -----------------------------------------------------------------------


def _draft_verdict(tone: str, **overrides) -> dict:
    defaults = dict(tone=tone, tone_score=4, faithful=True, addresses_request=True, note="")
    defaults.update(overrides)
    return defaults


def test_add_judge_verdicts_fills_in_matching_ticket():
    df = pd.DataFrame([{"ticket_id": "r001"}, {"ticket_id": "r002"}])
    verdicts = [
        {
            "ticket_id": "r001",
            "summary_accurate": True,
            "distinct": True,
            "note": "good",
            "drafts": [
                _draft_verdict("formal", tone_score=5),
                _draft_verdict("empathetic", tone_score=4, faithful=False),
                _draft_verdict("short", tone_score=3),
            ],
        }
    ]

    result = add_judge_verdicts(df, verdicts)

    r001 = result[result["ticket_id"] == "r001"].iloc[0]
    assert r001["summary_accurate"] is True
    assert r001["distinct"] is True
    assert r001["judge_note"] == "good"
    assert r001["formal_score"] == 5
    assert r001["empathetic_faithful"] is False
    assert r001["short_score"] == 3

    r002 = result[result["ticket_id"] == "r002"].iloc[0]
    assert pd.isna(r002["summary_accurate"])
    assert pd.isna(r002["formal_score"])


@pytest.mark.parametrize("verdicts", [None, []])
def test_add_judge_verdicts_no_verdicts_leaves_columns_blank(verdicts):
    df = pd.DataFrame([{"ticket_id": "r001"}])

    result = add_judge_verdicts(df, verdicts)

    assert pd.isna(result.iloc[0]["summary_accurate"])
    assert pd.isna(result.iloc[0]["distinct"])
    assert pd.isna(result.iloc[0]["formal_score"])
