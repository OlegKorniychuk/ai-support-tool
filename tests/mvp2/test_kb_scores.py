"""Tests for `scripts/kb_scores.py`'s pure threshold-suggestion logic.

Loaded by file path, like `tests/mvp1/test_run_eval.py` does, since `scripts/` isn't a
package on `support-ai`'s import path. No KB, no network: `ScoreRow` is built by hand.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("kb_scores", REPO_ROOT / "scripts" / "kb_scores.py")
kb_scores = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kb_scores)


def _row(ticket_id: str, best_score: float, *, is_gap: bool) -> "kb_scores.ScoreRow":
    return kb_scores.ScoreRow(
        ticket_id=ticket_id,
        tags=["gap"] if is_gap else ["answerable"],
        expected_kb_ids=[] if is_gap else ["some-article"],
        top3=[("some-article", best_score)],
        is_gap=is_gap,
    )


def test_suggest_threshold_separable_groups():
    rows = [
        _row("g1", 0.10, is_gap=True),
        _row("g2", 0.20, is_gap=True),
        _row("g3", 0.15, is_gap=True),
        _row("a1", 0.50, is_gap=False),
        _row("a2", 0.60, is_gap=False),
        _row("a3", 0.55, is_gap=False),
    ]

    threshold, details = kb_scores.suggest_threshold(rows)

    assert threshold == 0.35  # midpoint of gap max (0.20) and non-gap min (0.50)
    assert "Separable" in details
    assert "0.2000" in details  # gap max
    assert "0.5000" in details  # non-gap min


def test_suggest_threshold_overlapping_groups_names_the_overlap():
    rows = [
        _row("g1", 0.10, is_gap=True),
        _row("g2", 0.45, is_gap=True),  # overlaps into the non-gap range
        _row("a1", 0.40, is_gap=False),  # overlaps into the gap range
        _row("a2", 0.60, is_gap=False),
    ]

    threshold, details = kb_scores.suggest_threshold(rows)

    assert threshold is None
    assert "Overlap" in details
    assert "g2" in details
    assert "a1" in details
    # non-overlapping tickets aren't named
    assert "g1" not in details.split("Overlapping")[-1]
    assert "a2" not in details.split("Overlapping")[-1]


def test_suggest_threshold_tie_counts_as_overlap():
    """A margin of exactly 0 (gap max == non-gap min) must not be treated as separable: a
    `>=` threshold at that score would also match the tied gap ticket."""
    rows = [_row("g1", 0.30, is_gap=True), _row("a1", 0.30, is_gap=False)]

    threshold, details = kb_scores.suggest_threshold(rows)

    assert threshold is None
    assert "Overlap" in details


def test_suggest_threshold_no_gap_tickets_returns_none():
    rows = [_row("a1", 0.50, is_gap=False), _row("a2", 0.60, is_gap=False)]

    threshold, details = kb_scores.suggest_threshold(rows)

    assert threshold is None
    assert "Need at least one gap" in details


def test_suggest_threshold_no_non_gap_tickets_returns_none():
    rows = [_row("g1", 0.10, is_gap=True), _row("g2", 0.20, is_gap=True)]

    threshold, details = kb_scores.suggest_threshold(rows)

    assert threshold is None
    assert "Need at least one gap" in details


def test_suggest_threshold_empty_rows_returns_none():
    threshold, details = kb_scores.suggest_threshold([])

    assert threshold is None
    assert "Need at least one gap" in details


def test_score_row_hit_checks_top3_for_expected_ids():
    row = kb_scores.ScoreRow(
        ticket_id="a1",
        tags=["answerable"],
        expected_kb_ids=["edit-birth-data"],
        top3=[("other-article", 0.5), ("edit-birth-data", 0.3)],
        is_gap=False,
    )
    assert row.hit is True
    assert row.best_score == 0.5


def test_score_row_hit_false_when_no_expected_id_retrieved():
    row = kb_scores.ScoreRow(
        ticket_id="a1",
        tags=["answerable"],
        expected_kb_ids=["missing-article"],
        top3=[("other-article", 0.5)],
        is_gap=False,
    )
    assert row.hit is False


def test_score_row_best_score_zero_when_no_hits():
    row = kb_scores.ScoreRow(ticket_id="g1", tags=["gap"], expected_kb_ids=[], top3=[], is_gap=True)
    assert row.best_score == 0.0
