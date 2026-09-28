"""Tests for scripts/record_judgement.py: a valid (run, judge) pair appends one row to
judge_summary.csv; invalid input exits non-zero and writes nothing.

Loaded by file path (not `import scripts.record_judgement`), same as
`tests/mvp1/test_run_eval.py` does for `scripts/run_eval.py`.
"""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "record_judgement", REPO_ROOT / "scripts" / "record_judgement.py"
)
record_judgement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_judgement)


def _run_record(**overrides) -> dict:
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
        kb_quote="Open Profile → Birth details",
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
    return defaults


def _flagged_run_record(**overrides) -> dict:
    defaults = dict(
        ticket_id="r002",
        expected_needs_judgment=True,
        actual_needs_judgment=True,
        expected_reasons=["kb_not_found"],
        actual_reasons=["kb_not_found"],
        expected_kb_ids=[],
        retrieved_ids=[],
        cited_article_id=None,
        kb_quote=None,
        drafts={},
    )
    defaults.update(overrides)
    return _run_record(**defaults)


def _draft_verdict(tone: str, **overrides) -> dict:
    defaults = dict(tone=tone, tone_score=4, faithful=True, addresses_request=True)
    defaults.update(overrides)
    return defaults


def _drafted_verdict(ticket_id="r001", **overrides) -> dict:
    defaults = dict(
        ticket_id=ticket_id,
        summary_accurate=True,
        distinct=True,
        drafts=[
            _draft_verdict("formal"),
            _draft_verdict("empathetic"),
            _draft_verdict("short"),
        ],
    )
    defaults.update(overrides)
    return defaults


def _flagged_verdict(ticket_id="r002", **overrides) -> dict:
    defaults = dict(ticket_id=ticket_id, summary_accurate=True, distinct=None, drafts=[])
    defaults.update(overrides)
    return defaults


def _write_run_and_judge(
    tmp_path: Path, run: str, records: list[dict], verdicts: list[dict]
) -> Path:
    (tmp_path / f"{run}.json").write_text(json.dumps(records), encoding="utf-8")
    judge_run = {
        "run": run,
        "judge_model": "claude-sonnet-5",
        "rubric_version": "v1",
        "judged_at": "2026-09-28T12:30:00Z",
        "verdicts": verdicts,
    }
    judge_path = tmp_path / f"{run}.judge.json"
    judge_path.write_text(json.dumps(judge_run), encoding="utf-8")
    return judge_path


def test_valid_judgement_appends_one_row(tmp_path):
    run = "gpt-5.4-nano_v1_20260928T120000Z"
    judge_path = _write_run_and_judge(
        tmp_path,
        run,
        [_run_record(), _flagged_run_record()],
        [_drafted_verdict(), _flagged_verdict()],
    )

    summary_path = record_judgement.record_judgement(judge_path)

    assert summary_path == tmp_path / "judge_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run"] == run
    assert rows[0]["n_tickets"] == "2"
    assert rows[0]["n_drafted"] == "1"
    assert float(rows[0]["tone_score_formal"]) == 4.0
    assert float(rows[0]["faithful_rate"]) == 1.0
    assert float(rows[0]["summary_accurate_rate"]) == 1.0
    assert float(rows[0]["distinct_rate"]) == 1.0


def test_rejudging_a_run_appends_another_row(tmp_path):
    run = "gpt-5.4-nano_v1_20260928T120000Z"
    judge_path = _write_run_and_judge(
        tmp_path, run, [_run_record()], [_drafted_verdict(distinct=True)]
    )
    record_judgement.record_judgement(judge_path)

    # A second, different judge pass over the same run.
    judge_path.write_text(
        json.dumps(
            {
                "run": run,
                "judge_model": "claude-sonnet-5",
                "rubric_version": "v1",
                "judged_at": "2026-09-28T13:00:00Z",
                "verdicts": [_drafted_verdict(distinct=False)],
            }
        ),
        encoding="utf-8",
    )
    record_judgement.record_judgement(judge_path)

    summary_path = tmp_path / "judge_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["distinct_rate"] == "1.0"
    assert rows[1]["distinct_rate"] == "0.0"


def test_missing_judge_file_exits_nonzero_and_writes_nothing(tmp_path):
    judge_path = tmp_path / "does-not-exist.judge.json"

    with pytest.raises(SystemExit) as excinfo:
        record_judgement.record_judgement(judge_path)

    assert excinfo.value.code != 0
    assert not (tmp_path / "judge_summary.csv").exists()


def test_invalid_judge_json_exits_nonzero_and_writes_nothing(tmp_path):
    judge_path = tmp_path / "bad.judge.json"
    judge_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        record_judgement.record_judgement(judge_path)

    assert excinfo.value.code != 0
    assert not (tmp_path / "judge_summary.csv").exists()


def test_judge_json_failing_schema_exits_nonzero_and_writes_nothing(tmp_path):
    run = "gpt-5.4-nano_v1_20260928T120000Z"
    judge_path = tmp_path / f"{run}.judge.json"
    # missing required keys (judge_model, rubric_version, judged_at)
    judge_path.write_text(json.dumps({"run": run, "verdicts": []}), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        record_judgement.record_judgement(judge_path)

    assert excinfo.value.code != 0
    assert not (tmp_path / "judge_summary.csv").exists()


def test_missing_run_file_exits_nonzero_and_writes_nothing(tmp_path):
    run = "gpt-5.4-nano_v1_20260928T120000Z"
    judge_path = tmp_path / f"{run}.judge.json"
    judge_path.write_text(
        json.dumps(
            {
                "run": run,
                "judge_model": "claude-sonnet-5",
                "rubric_version": "v1",
                "judged_at": "2026-09-28T12:30:00Z",
                "verdicts": [],
            }
        ),
        encoding="utf-8",
    )
    # no <run>.json written next to it

    with pytest.raises(SystemExit) as excinfo:
        record_judgement.record_judgement(judge_path)

    assert excinfo.value.code != 0
    assert not (tmp_path / "judge_summary.csv").exists()


def test_judge_run_not_matching_run_file_exits_nonzero_and_writes_nothing(tmp_path):
    run = "gpt-5.4-nano_v1_20260928T120000Z"
    judge_path = _write_run_and_judge(
        tmp_path,
        run,
        [_run_record(), _flagged_run_record()],
        [_drafted_verdict()],  # missing the verdict for r002
    )

    with pytest.raises(SystemExit) as excinfo:
        record_judgement.record_judgement(judge_path)

    assert excinfo.value.code != 0
    assert not (tmp_path / "judge_summary.csv").exists()


def test_main_wires_argv_to_record_judgement(tmp_path, monkeypatch, capsys):
    run = "gpt-5.4-nano_v1_20260928T120000Z"
    judge_path = _write_run_and_judge(tmp_path, run, [_run_record()], [_drafted_verdict()])
    monkeypatch.setattr("sys.argv", ["record_judgement.py", str(judge_path)])

    record_judgement.main()

    assert (tmp_path / "judge_summary.csv").exists()
    assert "Recorded judgement" in capsys.readouterr().out
