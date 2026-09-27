"""Paste a ticket, classify it, and see the result plus the human-review badge.

This page calls `classify()` only — it knows nothing about providers, models, prompts
or retries. Any of that logic belongs in `support_ai.classifier` or `support_ai.core`.
"""

import streamlit as st

from support_ai.classifier.classify import classify
from support_ai.core.config import get_api_key

st.set_page_config(page_title="Classifier — Support AI", page_icon="🎫")
st.title("Ticket Classifier")

if not get_api_key():
    st.error(
        "OPENAI_API_KEY is not set. Add it to a local `.env` file (see `.env.example`) "
        "or to `st.secrets` when deployed."
    )
    st.stop()

ticket_text = st.text_area("Ticket text", height=180, placeholder="Paste a support ticket here…")

if st.button("Classify", type="primary", disabled=not ticket_text.strip()):
    with st.spinner("Classifying…"):
        st.session_state["last_result"] = classify(ticket_text)

result = st.session_state.get("last_result")

if result is None:
    st.info("Paste a ticket above and click Classify to see a result.")
else:
    classification = result.classification

    if classification.needs_human_review:
        reasons = ", ".join(classification.review_reasons) or "unspecified"
        st.error(f"🔴 Needs human review — reasons: {reasons}")

    if result.model_used == "none":
        st.warning(
            "Automatic classification failed after retrying every model in the fallback "
            "chain. Showing a safe default result — please review this ticket by hand."
        )
    elif len(result.attempts) > 1:
        st.info(
            f"The first model(s) tried had trouble, so this ticket was classified by "
            f"**{result.model_used}** instead."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Category", classification.category.value)
    col2.metric("Response", classification.next_step.value)
    col3.metric("Priority", classification.priority.value)
    col4.metric("Confidence", f"{classification.confidence:.0%}")

    st.write(f"**Next step:** {classification.next_step_note}")
    st.write(f"**Language:** {classification.language}  ·  **Tone:** {classification.tone}")
    st.write(f"**Rationale:** {classification.rationale}")

    st.divider()
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.caption(f"Model: {result.model_used}")
    meta_col2.caption(f"Latency: {result.latency_ms} ms")
    meta_col3.caption(f"Cost: ${result.cost_usd:.5f}")
    usage = result.usage
    st.caption(
        f"Tokens: {usage.input_tokens} in ({usage.cached_input_tokens} cached) / "
        f"{usage.output_tokens} out"
    )
