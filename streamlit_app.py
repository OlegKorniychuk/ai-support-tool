"""Home page for the Support AI prototype (MVP 1: ticket classifier)."""

import streamlit as st

st.set_page_config(page_title="Support AI", page_icon="🎫")

st.title("Support AI — Ticket Classifier")
st.markdown(
    """
Welcome. This prototype classifies support tickets with an LLM and flags the
ones that need a human to review them.

Use the sidebar to navigate:

- **Classifier** — paste a ticket and get its category, priority, next step and
  a needs-human-review flag.
- **Eval** — inspect evaluation runs over the synthetic test set: accuracy,
  latency and cost per model.
"""
)
