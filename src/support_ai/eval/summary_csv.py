"""Shared summary.csv read/append helpers, used by both eval scripts
(`scripts/run_eval.py` for the classifier, `scripts/run_reply_eval.py` for the reply
assistant, and `scripts/record_judgement.py` for judge verdicts).

Every eval run appends one row to its own `summary.csv`. A script's columns grow over time
(a new metric, a new column), and an older `summary.csv` written before that column existed
must keep loading correctly rather than erroring or misaligning columns. `upgrade_header`
widens an old header to the current column list in place; `append_row` does that
automatically before writing, and creates the file fresh (with a header) if it doesn't
exist yet.
"""

import csv
from pathlib import Path


def upgrade_header(path: Path, columns: list[str]) -> None:
    """Widen an older CSV's header to `columns`, in place, if needed.

    Callers must always append new columns at the *end* of `columns`, so older rows (which
    don't have them) still parse correctly: pandas pads missing trailing fields with NaN
    rather than misaligning the row. This only ever rewrites the header line — every
    existing data row is left byte-for-byte untouched, per CLAUDE.md/SPEC.md's "never...
    rewrite... eval results". A no-op if `path` doesn't exist yet (nothing to upgrade) or
    its header already matches `columns`. Raises `ValueError` if the existing header isn't
    a subset of `columns` — an unrecognized header we shouldn't silently rewrite.
    """
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return
    current_header = next(csv.reader([lines[0]]))
    if current_header == columns:
        return
    if not set(current_header).issubset(columns):
        raise ValueError(
            f"{path} has an unrecognized header {current_header}; refusing to touch it"
        )
    lines[0] = ",".join(columns) + "\n"
    path.write_text("".join(lines), encoding="utf-8")


def append_row(path: Path, columns: list[str], row: dict) -> None:
    """Append one `row` to `path`'s CSV, in `columns` order.

    Upgrades an existing header to `columns` first (see `upgrade_header`), then creates the
    file with a header if it doesn't exist yet, then appends `row`. Never rewrites an
    existing data row.
    """
    upgrade_header(path, columns)
    is_new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
