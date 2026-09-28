#!/usr/bin/env python
"""Validate one judge run and append its summary row to judge_summary.csv.

Called by the `.claude/skills/judge-replies` skill after it writes
`results/reply/<run>.judge.json`, but can also be run by hand. Looks for the reply-eval
run file (`<run>.json`, a list of `ReplyEvalRecord`) next to the judge file. Never writes
a row on invalid input: schema and cross-checks both fail loudly with a non-zero exit
before `judge_summary.csv` is touched.

Usage: uv run python scripts/record_judgement.py results/reply/<run>.judge.json
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from support_ai.assistant.schema import Tone
from support_ai.eval.judge import (
    JUDGE_SUMMARY_COLUMNS,
    JudgeRun,
    addresses_request_rate,
    distinct_rate,
    faithful_rate,
    mean_tone_score,
    summary_accurate_rate,
    validate_against_run,
)
from support_ai.eval.reply_metrics import ReplyEvalRecord
from support_ai.eval.summary_csv import append_row


def _fail(message: str) -> None:
    print(f"record_judgement: {message}", file=sys.stderr)
    sys.exit(1)


def _load_judge_run(judge_path: Path) -> JudgeRun:
    if not judge_path.exists():
        _fail(f"{judge_path} does not exist")
    try:
        data = json.loads(judge_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{judge_path} is not valid JSON: {exc}")
    try:
        return JudgeRun.model_validate(data)
    except ValidationError as exc:
        _fail(f"{judge_path} does not match the JudgeRun schema:\n{exc}")


def _load_run_records(run_path: Path) -> list[ReplyEvalRecord]:
    if not run_path.exists():
        _fail(f"run file {run_path} does not exist (expected next to the judge file)")
    try:
        data = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{run_path} is not valid JSON: {exc}")
    try:
        return [ReplyEvalRecord.model_validate(item) for item in data]
    except ValidationError as exc:
        _fail(f"{run_path} does not match the ReplyEvalRecord schema:\n{exc}")


def record_judgement(judge_path: Path) -> Path:
    """Validate `judge_path` against its run file and append a summary row.

    Returns the `judge_summary.csv` path written to. Exits the process (via `_fail`) with
    a clear message on any invalid input; never writes a row in that case.
    """
    judge_run = _load_judge_run(judge_path)
    run_path = judge_path.parent / f"{judge_run.run}.json"
    records = _load_run_records(run_path)

    try:
        validate_against_run(judge_run, records)
    except ValueError as exc:
        _fail(str(exc))

    verdicts = judge_run.verdicts
    n_drafted = sum(1 for verdict in verdicts if verdict.distinct is not None)
    row = {
        "run": judge_run.run,
        "judged_at": judge_run.judged_at,
        "judge_model": judge_run.judge_model,
        "rubric_version": judge_run.rubric_version,
        "n_tickets": len(records),
        "n_drafted": n_drafted,
        "tone_score_formal": round(mean_tone_score(verdicts, Tone.FORMAL), 3),
        "tone_score_empathetic": round(mean_tone_score(verdicts, Tone.EMPATHETIC), 3),
        "tone_score_short": round(mean_tone_score(verdicts, Tone.SHORT), 3),
        "faithful_rate": round(faithful_rate(verdicts), 4),
        "addresses_request_rate": round(addresses_request_rate(verdicts), 4),
        "summary_accurate_rate": round(summary_accurate_rate(verdicts), 4),
        "distinct_rate": round(distinct_rate(verdicts), 4),
    }

    summary_path = judge_path.parent / "judge_summary.csv"
    append_row(summary_path, JUDGE_SUMMARY_COLUMNS, row)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judge_file", type=Path, help="Path to results/reply/<run>.judge.json")
    args = parser.parse_args()

    summary_path = record_judgement(args.judge_file)
    print(f"Recorded judgement for {args.judge_file} to {summary_path}")


if __name__ == "__main__":
    main()
