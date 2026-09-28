"""Smoke + render tests for pages/4_Reply_Eval.py via `streamlit.testing.v1.AppTest`.

The page's results directory is overridden per test via the `SUPPORT_AI_REPLY_RESULTS_DIR`
env var (see the page's own docstring), pointed at a `tmp_path` so these tests never touch
the real `results/reply/`.
"""

import csv
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = str(REPO_ROOT / "pages" / "4_Reply_Eval.py")

SUMMARY_COLUMNS = [
    "timestamp",
    "run",
    "model",
    "prompt_version",
    "n_tickets",
    "retrieval_hit_rate",
    "citation_accuracy",
    "judgment_recall",
    "judgment_precision",
    "reason_match_rate",
    "drafts_shown_rate",
    "short_ok_rate",
    "formal_ok_rate",
    "distinct_ok_rate",
    "dropped_evidence_rate",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_ticket_usd",
    "cost_per_10k_tickets_usd",
    "cache_hit_rate",
]

JUDGE_SUMMARY_COLUMNS = [
    "run",
    "judged_at",
    "judge_model",
    "rubric_version",
    "n_tickets",
    "n_drafted",
    "tone_score_formal",
    "tone_score_empathetic",
    "tone_score_short",
    "faithful_rate",
    "addresses_request_rate",
    "summary_accurate_rate",
    "distinct_rate",
]

RUN = "gpt-5.4-nano_v1_20260928T120000Z"


def _summary_row(**overrides) -> dict:
    defaults = dict(
        timestamp="20260928T120000Z",
        run=RUN,
        model="gpt-5.4-nano",
        prompt_version="v1",
        n_tickets=2,
        retrieval_hit_rate=1.0,
        citation_accuracy=1.0,
        judgment_recall=1.0,
        judgment_precision=1.0,
        reason_match_rate=1.0,
        drafts_shown_rate=0.5,
        short_ok_rate=1.0,
        formal_ok_rate=1.0,
        distinct_ok_rate=1.0,
        dropped_evidence_rate=0.0,
        error_rate=0.0,
        latency_p50_ms=500.0,
        latency_p95_ms=600.0,
        cost_per_ticket_usd=0.002,
        cost_per_10k_tickets_usd=20.0,
        cache_hit_rate=0.1,
    )
    defaults.update(overrides)
    return defaults


def _judge_summary_row(**overrides) -> dict:
    defaults = dict(
        run=RUN,
        judged_at="2026-09-28T14:00:00Z",
        judge_model="claude-sonnet-5",
        rubric_version="v1",
        n_tickets=2,
        n_drafted=1,
        tone_score_formal=5.0,
        tone_score_empathetic=4.0,
        tone_score_short=5.0,
        faithful_rate=1.0,
        addresses_request_rate=1.0,
        summary_accurate_rate=1.0,
        distinct_rate=1.0,
    )
    defaults.update(overrides)
    return defaults


def _answerable_record() -> dict:
    return dict(
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
        dropped_evidence=[],
        summary="Customer asks how to change their birth time.",
        kb_quote="Open Profile → Birth details",
        drafts={
            "formal": "Dear customer, open Profile then Birth details.",
            "empathetic": "I understand you want to update your birth time.",
            "short": "Open Profile then Birth details.",
        },
        tone_check={
            "short_words": 5,
            "short_ok": True,
            "formal_contractions": [],
            "formal_ok": True,
            "max_pair_overlap": 0.3,
            "distinct_ok": True,
        },
        latency_ms=500,
        input_tokens=200,
        output_tokens=100,
        cached_input_tokens=0,
        cost_usd=0.002,
        error=None,
    )


def _flagged_record() -> dict:
    return dict(
        ticket_id="r021",
        ticket_text="Do you have any promo codes right now?",
        tags=["kb_gap"],
        expected_kb_ids=[],
        retrieved_ids=[],
        cited_article_id=None,
        expected_needs_judgment=True,
        actual_needs_judgment=True,
        expected_reasons=["kb_not_found"],
        actual_reasons=["kb_not_found"],
        dropped_evidence=[],
        summary="Customer asks about promo codes.",
        kb_quote=None,
        drafts={},
        tone_check=None,
        latency_ms=300,
        input_tokens=150,
        output_tokens=0,
        cached_input_tokens=0,
        cost_usd=0.0005,
        error=None,
    )


def _draft_verdict(tone: str, **overrides) -> dict:
    defaults = dict(tone=tone, tone_score=5, faithful=True, addresses_request=True, note="")
    defaults.update(overrides)
    return defaults


def _judge_run() -> dict:
    return {
        "run": RUN,
        "judge_model": "claude-sonnet-5",
        "rubric_version": "v1",
        "judged_at": "2026-09-28T14:00:00Z",
        "verdicts": [
            {
                "ticket_id": "r001",
                "summary_accurate": True,
                "distinct": True,
                "note": "",
                "drafts": [
                    _draft_verdict("formal"),
                    _draft_verdict("empathetic", tone_score=4),
                    _draft_verdict("short"),
                ],
            },
            {
                "ticket_id": "r021",
                "summary_accurate": True,
                "distinct": None,
                "note": "correctly flagged",
                "drafts": [],
            },
        ],
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_empty_results_dir_shows_info_message(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_AI_REPLY_RESULTS_DIR", str(tmp_path))
    at = AppTest.from_file(PAGE_PATH)

    at.run(timeout=15)

    assert not at.exception
    assert len(at.info) >= 1
    assert "run_reply_eval.py" in at.info[0].value


def test_populated_results_dir_with_judgment_renders_without_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_AI_REPLY_RESULTS_DIR", str(tmp_path))
    _write_csv(tmp_path / "summary.csv", SUMMARY_COLUMNS, [_summary_row()])
    _write_csv(tmp_path / "judge_summary.csv", JUDGE_SUMMARY_COLUMNS, [_judge_summary_row()])
    (tmp_path / f"{RUN}.json").write_text(
        json.dumps([_answerable_record(), _flagged_record()]), encoding="utf-8"
    )
    (tmp_path / f"{RUN}.judge.json").write_text(json.dumps(_judge_run()), encoding="utf-8")

    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=15)

    assert not at.exception
    assert len(at.dataframe) >= 2  # comparison table + per-ticket table
    assert len(at.metric) >= 6
    assert len(at.warning) == 0


def test_populated_results_dir_without_judgment_renders_without_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_AI_REPLY_RESULTS_DIR", str(tmp_path))
    _write_csv(tmp_path / "summary.csv", SUMMARY_COLUMNS, [_summary_row()])
    (tmp_path / f"{RUN}.json").write_text(
        json.dumps([_answerable_record(), _flagged_record()]), encoding="utf-8"
    )
    # no judge_summary.csv, no <run>.judge.json

    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=15)

    assert not at.exception
    assert any("judge-replies" in caption.value for caption in at.caption)
