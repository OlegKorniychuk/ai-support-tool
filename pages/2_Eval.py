"""Reads `results/` and shows the eval summary, model comparison and per-run detail.

This page only reads files written by `scripts/run_eval.py` — it never calls an LLM.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from support_ai.classifier.dataset import load_tickets
from support_ai.core.cost import forecast

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.csv"

st.set_page_config(page_title="Eval — Support AI", page_icon="📊", layout="wide")
st.title("Evaluation Results")

if not SUMMARY_PATH.exists() or SUMMARY_PATH.stat().st_size == 0:
    st.info(
        "No eval runs yet. Populate this page with:\n\n"
        "`uv run python scripts/run_eval.py --model gpt-5.4-nano --prompt v5 --limit 3`"
    )
    st.stop()

summary_df = pd.read_csv(SUMMARY_PATH)

st.subheader("Model comparison")
st.caption("One row per eval run: accuracy, latency and cost, from `results/summary.csv`.")
st.dataframe(summary_df, width="stretch")

run_files = sorted(RESULTS_DIR.glob("*.json"))
if not run_files:
    st.info("The summary table has rows, but no per-run JSON files were found in `results/`.")
    st.stop()

run_labels = [f.stem for f in run_files]
selected_label = st.selectbox("Choose a run to inspect", run_labels, index=len(run_labels) - 1)
selected_path = RESULTS_DIR / f"{selected_label}.json"

try:
    records = json.loads(selected_path.read_text())
except json.JSONDecodeError:
    st.warning(f"Could not read {selected_path.name}: not valid JSON.")
    st.stop()

detail_df = pd.DataFrame(records)

if detail_df.empty:
    st.info("This run has no ticket records.")
else:
    detail_df["category_pass"] = detail_df["expected_category"] == detail_df["actual_category"]
    detail_df["priority_pass"] = detail_df["expected_priority"] == detail_df["actual_priority"]
    # runs before prompt v5 have no next-step columns; they are judged on category + priority
    has_next_step = "expected_next_step" in detail_df.columns
    if has_next_step:
        detail_df["next_step_pass"] = (
            detail_df["expected_next_step"] == detail_df["actual_next_step"]
        )
    else:
        detail_df["next_step_pass"] = True
    detail_df["all_pass"] = (
        detail_df["category_pass"] & detail_df["priority_pass"] & detail_df["next_step_pass"]
    )
    detail_df["status"] = ["✅ pass" if ok else "❌ fail" for ok in detail_df["all_pass"]]

    category_accuracy = detail_df["category_pass"].mean()
    next_step_accuracy = detail_df["next_step_pass"].mean() if has_next_step else None
    priority_accuracy = detail_df["priority_pass"].mean()
    expected_review = detail_df[detail_df["expected_needs_review"]]
    human_review_recall = (
        expected_review["actual_needs_review"].mean() if len(expected_review) else 1.0
    )
    flagged = detail_df[detail_df["actual_needs_review"]]
    human_review_precision = flagged["expected_needs_review"].mean() if len(flagged) else 1.0
    cost_per_ticket = detail_df["cost_usd"].mean()
    # older runs predate cached-token tracking
    if "cached_input_tokens" not in detail_df.columns:
        detail_df["cached_input_tokens"] = 0
    total_input = detail_df["input_tokens"].sum()
    cache_hit_rate = detail_df["cached_input_tokens"].sum() / total_input if total_input else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Category accuracy", f"{category_accuracy:.0%}")
    col2.metric(
        "Response accuracy",
        f"{next_step_accuracy:.0%}" if next_step_accuracy is not None else "n/a",
    )
    col3.metric("Priority accuracy", f"{priority_accuracy:.0%}")
    col4, col5, col6 = st.columns(3)
    col4.metric("Human-review recall", f"{human_review_recall:.0%}")
    col5.metric("Human-review precision", f"{human_review_precision:.0%}")
    col6.metric("Cost / 10k tickets", f"${forecast(cost_per_ticket, 10_000):.2f}")
    st.caption(f"Prompt cache hit rate: {cache_hit_rate:.0%} of input tokens")

    st.subheader(f"Per-ticket results — {selected_label}")

    # best-effort join of the original ticket text, for an input → expected → actual view
    try:
        tickets_by_id = {case.id: case.text for case in load_tickets()}
        detail_df.insert(1, "text", detail_df["ticket_id"].map(tickets_by_id))
    except (FileNotFoundError, ValueError):
        pass

    display_columns = [
        c
        for c in [
            "ticket_id",
            "text",
            "expected_category",
            "actual_category",
            "expected_next_step",
            "actual_next_step",
            "expected_priority",
            "actual_priority",
            "status",
            "expected_needs_review",
            "actual_needs_review",
            "latency_ms",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "cost_usd",
            "error",
        ]
        if c in detail_df.columns
    ]
    st.dataframe(detail_df[display_columns], width="stretch")

    n_failed = int((~detail_df["all_pass"]).sum())
    if n_failed:
        st.caption(f"{n_failed} of {len(detail_df)} ticket(s) failed (marked ❌ above).")
