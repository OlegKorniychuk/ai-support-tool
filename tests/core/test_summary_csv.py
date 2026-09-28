"""Tests for `eval/summary_csv.py`'s shared summary.csv helpers, used by both eval
scripts and the judge recorder. Behavior mirrors `scripts/run_eval.py`'s original
(now-moved) `_upgrade_summary_header`/append logic — see `tests/mvp1/test_run_eval.py`
for the classifier eval script's own integration tests of the same behavior.
"""

import csv

import pandas as pd
import pytest

from support_ai.eval.summary_csv import append_row, upgrade_header

COLUMNS = ["a", "b", "c"]


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# --- upgrade_header -------------------------------------------------------------------------


def test_upgrade_header_noop_when_file_missing(tmp_path):
    path = tmp_path / "summary.csv"
    upgrade_header(path, COLUMNS)  # must not raise or create the file
    assert not path.exists()


def test_upgrade_header_noop_when_already_current(tmp_path):
    path = tmp_path / "summary.csv"
    _write_csv(path, COLUMNS, [["1", "2", "3"]])
    before = path.read_text(encoding="utf-8")

    upgrade_header(path, COLUMNS)

    assert path.read_text(encoding="utf-8") == before


def test_upgrade_header_widens_old_header_without_touching_data_rows(tmp_path):
    path = tmp_path / "summary.csv"
    _write_csv(path, ["a", "b"], [["1", "2"]])

    upgrade_header(path, COLUMNS)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == COLUMNS
    assert rows[1] == ["1", "2"]  # untouched, byte-for-byte

    df = pd.read_csv(path)
    assert pd.isna(df.loc[0, "c"])


def test_upgrade_header_rejects_unrecognized_header(tmp_path):
    path = tmp_path / "summary.csv"
    _write_csv(path, ["a", "totally_different"], [["1", "2"]])

    with pytest.raises(ValueError):
        upgrade_header(path, COLUMNS)


def test_upgrade_header_noop_on_empty_file(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("", encoding="utf-8")

    upgrade_header(path, COLUMNS)  # must not raise

    assert path.read_text(encoding="utf-8") == ""


# --- append_row -----------------------------------------------------------------------------


def test_append_row_creates_file_with_header(tmp_path):
    path = tmp_path / "summary.csv"

    append_row(path, COLUMNS, {"a": "1", "b": "2", "c": "3"})

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [COLUMNS, ["1", "2", "3"]]


def test_append_row_appends_to_existing_file_without_rewriting_header(tmp_path):
    path = tmp_path / "summary.csv"
    append_row(path, COLUMNS, {"a": "1", "b": "2", "c": "3"})

    append_row(path, COLUMNS, {"a": "4", "b": "5", "c": "6"})

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [COLUMNS, ["1", "2", "3"], ["4", "5", "6"]]


def test_append_row_upgrades_old_header_first_then_appends(tmp_path):
    path = tmp_path / "summary.csv"
    _write_csv(path, ["a", "b"], [["1", "2"]])

    append_row(path, COLUMNS, {"a": "3", "b": "4", "c": "5"})

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == COLUMNS
    assert rows[1] == ["1", "2"]  # old row untouched
    assert rows[2] == ["3", "4", "5"]  # new row uses the widened header

    df = pd.read_csv(path)
    assert pd.isna(df.loc[0, "c"])
    assert df.loc[1, "c"] == 5
