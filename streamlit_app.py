"""Home page for the Support AI prototype: a ticket classifier (MVP 1) and a reply
assistant (MVP 2)."""

import streamlit as st

st.set_page_config(page_title="Support AI", page_icon="🎫")

st.title("Support AI")
st.markdown(
    """
Welcome. This prototype has two LLM-backed tools for support tickets.

Use the sidebar to navigate:

- **Classifier** — paste a ticket and get its category, priority, next step and
  a needs-human-review flag.
- **Eval** — inspect classifier evaluation runs over the synthetic test set:
  accuracy, latency and cost per model.
- **Reply Assistant** — paste a general-question ticket and get a summary, the
  KB article it's grounded in, and 3 ready-to-edit reply drafts — or a
  "decide yourself" banner when it can't safely draft one.
- **Reply Eval** *(coming soon)* — per-ticket retrieval, grounding and judge
  results, plus a model comparison, for the Reply Assistant.
"""
)
