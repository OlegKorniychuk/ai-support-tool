"""Tests for scripts/run_eval.py's summary.csv writer.

Loaded by file path (not `import scripts.run_eval`) since `scripts/` isn't a package on
`support-ai`'s import path.

Focus: an older summary.csv (written before `human_review_precision` existed) must stay
loadable and its existing rows must never be rewritten — only the header may be widened
to add the new trailing column (see `_upgrade_summary_header`'s docstring).
"""

import csv
import importlib.util
from pathlib import Path

from support_ai.eval.metrics import EvalRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_eval", REPO_ROOT / "scripts" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

OLD_HEADER = [
    "timestamp",
    "model",
    "prompt_version",
    "n_tickets",
    "category_accuracy",
    "priority_accuracy",
    "human_review_recall",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_ticket_usd",
    "cost_per_10k_tickets_usd",
]
OLD_ROW = [
    "20260101T000000Z",
    "gpt-5.4-nano",
    "v1",
    "3",
    "1.0",
    "1.0",
    "1.0",
    "1500",
    "2000",
    "0.0003",
    "3.0",
]


def _record(**overrides) -> EvalRecord:
    defaults = {
        "ticket_id": "t1",
        "expected_category": "unclear",
        "actual_category": "unclear",
        "expected_priority": "P4",
        "actual_priority": "P4",
        "expected_needs_review": False,
        "actual_needs_review": False,
        "latency_ms": 100,
        "input_tokens": 10,
        "output_tokens": 10,
        "cost_usd": 0.001,
    }
    return EvalRecord.model_validate({**defaults, **overrides})


def test_append_summary_row_upgrades_old_header_without_touching_old_rows(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OLD_HEADER)
        writer.writerow(OLD_ROW)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)

    run_eval._append_summary_row(
        "gpt-5.4-nano", "v2", "20260926T150000Z", [_record(expected_needs_review=True)]
    )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == run_eval.SUMMARY_COLUMNS
    assert rows[1] == OLD_ROW  # untouched, byte-for-byte
    precision_idx = run_eval.SUMMARY_COLUMNS.index("human_review_precision")
    assert rows[2][precision_idx] == "1.0"

    # pandas must still load the file cleanly: the old row gets NaN for the new column
    # instead of a parse error or misaligned values.
    import pandas as pd

    df = pd.read_csv(summary_path)
    assert pd.isna(df.loc[0, "human_review_precision"])
    assert df.loc[1, "human_review_precision"] == 1.0
    assert pd.isna(df.loc[0, "cache_hit_rate"])
    assert df.loc[1, "cache_hit_rate"] == 0.0
    # the old row's own columns must be unaffected by the header widening
    assert df.loc[0, "latency_p50_ms"] == 1500.0
    assert df.loc[0, "cost_per_10k_tickets_usd"] == 3.0


def test_append_summary_row_creates_a_fresh_file_with_the_full_header(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.csv"
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)

    run_eval._append_summary_row("gpt-5.4-nano", "v2", "20260926T150000Z", [_record()])

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == run_eval.SUMMARY_COLUMNS
    assert len(rows) == 2


def test_upgrade_summary_header_is_a_noop_when_already_current(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(run_eval.SUMMARY_COLUMNS)
        writer.writerow(OLD_ROW + ["0.9"])

    before = summary_path.read_text(encoding="utf-8")
    run_eval._upgrade_summary_header(summary_path)
    assert summary_path.read_text(encoding="utf-8") == before
