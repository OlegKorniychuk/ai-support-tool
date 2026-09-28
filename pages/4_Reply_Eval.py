"""Reads `results/reply/` and shows the model comparison and per-run detail for the
Reply Assistant. This page only reads files — it never calls an LLM.

File formats, all written by `scripts/run_reply_eval.py` and
`scripts/record_judgement.py` (Task 14/15):
- `results/reply/<run>.json` — a JSON array of `ReplyEvalRecord` dumps.
- `results/reply/summary.csv` — one row per eval run.
- `results/reply/judge_summary.csv` — one row per judge pass (append-only; a run judged
  twice has two rows, joined to the run's `summary.csv` row by `run`).
- `results/reply/<run>.judge.json` — that run's `JudgeRun` (verdicts per ticket).

The results directory defaults to `results/reply` next to the repo root, but can be
overridden with the `SUPPORT_AI_REPLY_RESULTS_DIR` environment variable — used by
`tests/mvp2/test_reply_eval_page.py` to point the page at a throwaway `tmp_path` instead
of the real `results/reply/`.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from support_ai.assistant.dataset import load_reply_cases
from support_ai.assistant.schema import Tone
from support_ai.eval.judge import JudgeRun
from support_ai.eval.reply_metrics import ReplyEvalRecord
from support_ai.eval.reply_report import (
    add_judge_verdicts,
    add_ticket_status,
    add_tone_check_flags,
    model_comparison,
)

_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "reply"
RESULTS_DIR = Path(os.environ.get("SUPPORT_AI_REPLY_RESULTS_DIR") or _DEFAULT_RESULTS_DIR)
SUMMARY_PATH = RESULTS_DIR / "summary.csv"
JUDGE_SUMMARY_PATH = RESULTS_DIR / "judge_summary.csv"

# Columns shown in the compact model-comparison table, in order; a column missing from a
# given `summary.csv`/`judge_summary.csv` vintage is silently skipped (see `pages/2_Eval.py`).
COMPARISON_COLUMNS = [
    "run",
    "model",
    "prompt_version",
    "retrieval_hit_rate",
    "citation_accuracy",
    "judgment_recall",
    "judgment_precision",
    "drafts_shown_rate",
    "short_ok_rate",
    "formal_ok_rate",
    "distinct_ok_rate",
    "tone_score_formal",
    "tone_score_empathetic",
    "tone_score_short",
    "faithful_rate",
    "addresses_request_rate",
    "summary_accurate_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_ticket_usd",
    "cost_per_10k_tickets_usd",
]

DETAIL_COLUMNS = [
    "ticket_id",
    "tags",
    "ticket_text",
    "expected_kb_ids",
    "retrieved_ids",
    "cited_article_id",
    "expected_needs_judgment",
    "expected_reasons",
    "actual_needs_judgment",
    "actual_reasons",
    "status",
    "short_ok",
    "formal_ok",
    "distinct_ok",
    "summary_accurate",
    "distinct",
    "formal_score",
    "empathetic_score",
    "short_score",
    "formal_faithful",
    "empathetic_faithful",
    "short_faithful",
    "formal_addresses",
    "empathetic_addresses",
    "short_addresses",
    "judge_note",
    "error",
]

st.set_page_config(page_title="Reply Eval — Support AI", page_icon="📈", layout="wide")
st.title("Reply Assistant Evaluation Results")

if not SUMMARY_PATH.exists() or SUMMARY_PATH.stat().st_size == 0:
    st.info(
        "No reply-eval runs yet. Populate this page with:\n\n"
        "`uv run python scripts/run_reply_eval.py --model gpt-5.4-nano [--limit 5]`\n\n"
        "Then judge a run with `/judge-replies results/reply/<run>.json` in Claude Code."
    )
    st.stop()

summary_df = pd.read_csv(SUMMARY_PATH)
_has_judge_summary = JUDGE_SUMMARY_PATH.exists() and JUDGE_SUMMARY_PATH.stat().st_size > 0
judge_summary_df = pd.read_csv(JUDGE_SUMMARY_PATH) if _has_judge_summary else None

st.subheader("Model comparison")
st.caption(
    "One row per eval run, joined with its latest judge verdicts (if any), "
    "from `results/reply/summary.csv` and `results/reply/judge_summary.csv`."
)
comparison_df = model_comparison(summary_df, judge_summary_df)
display_comparison_columns = [c for c in COMPARISON_COLUMNS if c in comparison_df.columns]
st.dataframe(comparison_df[display_comparison_columns], width="stretch", hide_index=True)
if comparison_df["judge_model"].isna().any():
    st.caption(
        "Rows with blank judge columns haven't been judged yet — run "
        "`/judge-replies results/reply/<run>.json` in Claude Code."
    )

run_files = sorted(p for p in RESULTS_DIR.glob("*.json") if not p.name.endswith(".judge.json"))
if not run_files:
    st.info("The summary table has rows, but no per-run JSON files were found in `results/reply/`.")
    st.stop()

run_labels = [f.stem for f in run_files]
selected_label = st.selectbox("Choose a run to inspect", run_labels, index=len(run_labels) - 1)
selected_path = RESULTS_DIR / f"{selected_label}.json"

try:
    raw_records = json.loads(selected_path.read_text())
    records = [ReplyEvalRecord.model_validate(item) for item in raw_records]
except (json.JSONDecodeError, ValueError) as exc:
    st.warning(f"Could not read {selected_path.name}: {exc}")
    st.stop()

if not records:
    st.info("This run has no ticket records.")
    st.stop()

detail_df = pd.DataFrame([record.model_dump() for record in records])

# best-effort join of the original ticket's tags (the dataset is the source of truth;
# ReplyEvalRecord.tags should already match, but this keeps the page usable even if a run
# predates that field).
if "tags" not in detail_df.columns:
    try:
        tags_by_id = {case.id: case.tags for case in load_reply_cases()}
        detail_df["tags"] = detail_df["ticket_id"].map(tags_by_id)
    except (FileNotFoundError, ValueError):
        detail_df["tags"] = None

detail_df = add_ticket_status(detail_df)
detail_df = add_tone_check_flags(detail_df)

judge_path = RESULTS_DIR / f"{selected_label}.judge.json"
verdicts = None
if judge_path.exists():
    try:
        judge_run = JudgeRun.model_validate(json.loads(judge_path.read_text()))
        verdicts = [v.model_dump() for v in judge_run.verdicts]
    except (json.JSONDecodeError, ValueError) as exc:
        st.warning(f"Could not read {judge_path.name}: {exc}")
detail_df = add_judge_verdicts(detail_df, verdicts)

comparison_row = comparison_df[comparison_df["run"] == selected_label]
row = comparison_row.iloc[0] if len(comparison_row) else None

st.subheader(f"Run detail — {selected_label}")

col1, col2, col3 = st.columns(3)
col1.metric("Retrieval hit rate", f"{row['retrieval_hit_rate']:.0%}" if row is not None else "n/a")
col2.metric("Citation accuracy", f"{row['citation_accuracy']:.0%}" if row is not None else "n/a")
col3.metric("Drafts shown rate", f"{row['drafts_shown_rate']:.0%}" if row is not None else "n/a")
col4, col5, col6 = st.columns(3)
col4.metric("Judgment recall", f"{row['judgment_recall']:.0%}" if row is not None else "n/a")
col5.metric("Judgment precision", f"{row['judgment_precision']:.0%}" if row is not None else "n/a")
col6.metric(
    "Cost / 10k tickets",
    f"${row['cost_per_10k_tickets_usd']:.2f}" if row is not None else "n/a",
)

if row is not None and pd.notna(row.get("judge_model")):
    st.caption(
        f"Judged by {row['judge_model']} (rubric {row['rubric_version']}) — "
        f"tone scores: formal {row['tone_score_formal']:.1f}, "
        f"empathetic {row['tone_score_empathetic']:.1f}, short {row['tone_score_short']:.1f} · "
        f"faithful {row['faithful_rate']:.0%} · addresses request "
        f"{row['addresses_request_rate']:.0%} · summary accurate "
        f"{row['summary_accurate_rate']:.0%}"
    )
else:
    st.caption("Not judged yet.")

st.dataframe(
    detail_df[[c for c in DETAIL_COLUMNS if c in detail_df.columns]],
    width="stretch",
    hide_index=True,
)

n_failed = int((~detail_df["ticket_pass"]).sum())
if n_failed:
    st.caption(f"{n_failed} of {len(detail_df)} ticket(s) failed (marked ❌ above).")

with st.expander("View one ticket's summary and drafts"):
    ticket_ids = detail_df["ticket_id"].tolist()
    chosen_id = st.selectbox("Ticket", ticket_ids)
    chosen = next(r for r in records if r.ticket_id == chosen_id)
    st.write(f"**Summary:** {chosen.summary}")
    if chosen.drafts:
        tabs = st.tabs([tone.value.capitalize() for tone in Tone])
        for tab, tone in zip(tabs, Tone, strict=True):
            with tab:
                st.write(chosen.drafts.get(tone.value, "*(no draft for this tone)*"))
    else:
        st.caption("No drafts were shown for this ticket.")
