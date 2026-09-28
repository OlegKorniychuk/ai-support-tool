"""Pure dataframe-building logic for `pages/4_Reply_Eval.py`.

No I/O, no Streamlit: everything here takes dataframes / plain dicts already loaded from
`results/reply/` and returns a new dataframe, so it's unit-testable on handmade data. The
page itself only reads files (`results/reply/summary.csv`, `results/reply/judge_summary.csv`,
`results/reply/<run>.json`, `results/reply/<run>.judge.json`) and calls these functions —
see that page's docstring for the file formats.
"""

import pandas as pd

from support_ai.assistant.schema import Tone
from support_ai.eval.judge import JUDGE_SUMMARY_COLUMNS

# Every non-`run` judge_summary.csv column, added (as NaN) to `model_comparison`'s output
# even when a run hasn't been judged yet, so the page's column selection never KeyErrors.
_JUDGE_COLUMNS = [c for c in JUDGE_SUMMARY_COLUMNS if c != "run"]

# Per-ticket judge columns `add_judge_verdicts` adds, beyond the per-tone ones below.
_JUDGE_TICKET_COLUMNS = ["summary_accurate", "distinct", "judge_note"]


def latest_judgment_per_run(judge_summary_df: pd.DataFrame) -> pd.DataFrame:
    """One row per `run`: the `judge_summary.csv` row with the latest `judged_at`.

    `judge_summary.csv` is append-only (re-judging a run appends another row), so a run can
    have several rows; the page always shows the most recent judgment.
    """
    if judge_summary_df.empty:
        return judge_summary_df
    df = judge_summary_df.copy()
    df["_judged_at_ts"] = pd.to_datetime(df["judged_at"], utc=True, errors="coerce")
    df = df.sort_values("_judged_at_ts").drop_duplicates("run", keep="last")
    return df.drop(columns="_judged_at_ts").reset_index(drop=True)


def model_comparison(
    summary_df: pd.DataFrame, judge_summary_df: pd.DataFrame | None
) -> pd.DataFrame:
    """`summary_df` (one row per eval run) left-joined with the latest judge row per run
    (see `latest_judgment_per_run`), on `run`.

    A run with no judgment yet (or when `judge_summary_df` is `None`/empty — no
    `judge_summary.csv` at all) gets `NaN` in every judge column rather than being dropped
    or raising: the page shows those rows with blanks and a caption pointing at
    `/judge-replies`.
    """
    if judge_summary_df is None or judge_summary_df.empty:
        df = summary_df.copy()
        for column in _JUDGE_COLUMNS:
            df[column] = pd.NA
        return df
    latest = latest_judgment_per_run(judge_summary_df)
    # `suffixes` handles the one real collision (both files have their own `n_tickets`);
    # every other judge column is uniquely named.
    return summary_df.merge(latest, on="run", how="left", suffixes=("", "_judge"))


def _flag_match(row: pd.Series) -> bool:
    return bool(row.get("expected_needs_judgment")) == bool(row.get("actual_needs_judgment"))


def _reasons_ok(row: pd.Series) -> bool:
    expected = set(row.get("expected_reasons") or [])
    actual = set(row.get("actual_reasons") or [])
    return expected <= actual


def _citation_ok(row: pd.Series) -> bool:
    if row.get("expected_needs_judgment"):
        return True  # the citation check only applies to answerable tickets
    expected_ids = row.get("expected_kb_ids") or []
    return row.get("cited_article_id") in expected_ids


def add_ticket_status(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-ticket pass/fail columns to a run's detail dataframe (one row per
    `ReplyEvalRecord`): `flag_match`, `reasons_ok`, `citation_ok`, `ticket_pass`, `status`.

    A ticket passes when the actual human-judgment flag matches the expected one, the
    actual reasons are a superset of the expected ones, and — only for an answerable ticket
    (`expected_needs_judgment` is `False`) — the cited article is one of the expected ids.
    """
    df = detail_df.copy()
    df["flag_match"] = df.apply(_flag_match, axis=1)
    df["reasons_ok"] = df.apply(_reasons_ok, axis=1)
    df["citation_ok"] = df.apply(_citation_ok, axis=1)
    df["ticket_pass"] = df["flag_match"] & df["reasons_ok"] & df["citation_ok"]
    df["status"] = df["ticket_pass"].map({True: "✅", False: "❌"})
    return df


def add_tone_check_flags(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten each row's `tone_check` dict (or `None`, when drafts weren't shown) into
    `short_ok`, `formal_ok`, `distinct_ok` columns."""

    def _flag(row: pd.Series, key: str):
        check = row.get("tone_check")
        return check.get(key) if isinstance(check, dict) else pd.NA

    df = detail_df.copy()
    df["short_ok"] = df.apply(lambda row: _flag(row, "short_ok"), axis=1)
    df["formal_ok"] = df.apply(lambda row: _flag(row, "formal_ok"), axis=1)
    df["distinct_ok"] = df.apply(lambda row: _flag(row, "distinct_ok"), axis=1)
    return df


def add_judge_verdicts(detail_df: pd.DataFrame, verdicts: list[dict] | None) -> pd.DataFrame:
    """Add per-ticket judge columns from a `JudgeRun.verdicts` list (each item a
    `TicketVerdict.model_dump()` dict, keyed by `ticket_id`): `summary_accurate`,
    `distinct`, `judge_note`, and per tone `<tone>_score` / `<tone>_faithful` /
    `<tone>_addresses`.

    `verdicts` is `None` (or empty) when the run hasn't been judged yet — every judge
    column is then `NaN`, so the page's column selection never KeyErrors either way.
    """
    df = detail_df.copy()
    judge_columns = list(_JUDGE_TICKET_COLUMNS)
    for tone in Tone:
        judge_columns += [
            f"{tone.value}_score",
            f"{tone.value}_faithful",
            f"{tone.value}_addresses",
        ]
    for column in judge_columns:
        df[column] = pd.NA

    if not verdicts:
        return df

    by_ticket = {verdict["ticket_id"]: verdict for verdict in verdicts}
    for index, row in df.iterrows():
        verdict = by_ticket.get(row.get("ticket_id"))
        if verdict is None:
            continue
        df.at[index, "summary_accurate"] = verdict.get("summary_accurate")
        df.at[index, "distinct"] = verdict.get("distinct")
        df.at[index, "judge_note"] = verdict.get("note", "")
        for draft in verdict.get("drafts", []):
            tone = draft["tone"]
            df.at[index, f"{tone}_score"] = draft["tone_score"]
            df.at[index, f"{tone}_faithful"] = draft["faithful"]
            df.at[index, f"{tone}_addresses"] = draft["addresses_request"]
    return df
