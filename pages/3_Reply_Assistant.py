"""Paste a ticket, get a summary, a grounded reply in 3 tones, or a "decide yourself"
banner when the tool can't safely draft one.

This page calls `assist()` only — it knows nothing about retrieval, providers, models,
prompts or retries. Any of that logic belongs in `support_ai.assistant` or `support_ai.kb`.
"""

import hashlib

import pandas as pd
import streamlit as st

from support_ai.assistant.assist import assist
from support_ai.assistant.schema import JudgmentReason, Tone
from support_ai.assistant.tone_checks import word_count
from support_ai.core.config import get_api_key

REASON_LABELS: dict[JudgmentReason, str] = {
    JudgmentReason.KB_NOT_FOUND: "The knowledge base has no article that answers this.",
    JudgmentReason.KB_QUOTE_UNVERIFIED: (
        "The cited KB quote could not be verified in the article."
    ),
    JudgmentReason.ACCOUNT_SPECIFIC: (
        "The answer depends on this customer's own account — check it yourself."
    ),
    JudgmentReason.CONFLICTING_KB: "KB articles contradict each other on this question.",
    JudgmentReason.RETRIEVAL_FAILED: "The knowledge base could not be searched right now.",
    JudgmentReason.GENERATION_FAILED: "Reply generation failed on every model tried.",
}

st.set_page_config(page_title="Reply Assistant — Support AI", page_icon="💬")
st.title("Reply Assistant")

if not get_api_key():
    st.error(
        "OPENAI_API_KEY is not set. Add it to a local `.env` file (see `.env.example`) "
        "or to `st.secrets` when deployed."
    )
    st.stop()

ticket_text = st.text_area("Ticket text", height=180, placeholder="Paste a support ticket here…")

if st.button("Suggest replies", type="primary", disabled=not ticket_text.strip()):
    with st.spinner("Thinking…"):
        st.session_state["last_assist"] = assist(ticket_text)

result = st.session_state.get("last_assist")

if result is None:
    st.info("Paste a ticket above and click Suggest replies to see a result.")
else:
    if result.needs_human_judgment:
        st.error("🔴 Decide yourself — no AI drafts")
        candidate_titles = {hit.article.id: hit.article.title for hit in result.kb_candidates}
        for reason in result.judgment_reasons:
            if reason is JudgmentReason.CONFLICTING_KB:
                titles = ", ".join(
                    candidate_titles.get(article_id, article_id)
                    for article_id in result.conflicting_article_ids
                )
                st.write(f"- KB articles contradict each other: {titles}.")
            else:
                st.write(f"- {REASON_LABELS.get(reason, reason.value)}")
            if reason is JudgmentReason.ACCOUNT_SPECIFIC and result.account_specific_evidence:
                st.caption(f"Evidence: “{result.account_specific_evidence}”")

    st.subheader("Summary")
    st.write(result.summary)

    st.subheader("KB source")
    if result.kb_source is not None:
        st.write(f"**{result.kb_source.title}**")
        st.markdown(f"> {result.kb_source.quote}")
        with st.expander("Full article"):
            st.write(result.kb_source.text)
    else:
        st.caption("No verified KB source for this ticket.")

    if result.drafts:
        st.subheader("Drafts")
        by_tone = {draft.tone: draft for draft in result.drafts}
        tabs = st.tabs(["Formal", "Empathetic", "Short"])
        for tab, tone in zip(tabs, Tone, strict=True):
            with tab:
                draft = by_tone[tone]
                edited = st.text_area(
                    f"{tone.value} draft",
                    value=draft.text,
                    height=200,
                    # Keyed by the draft text too: a fixed key would keep showing the previous
                    # ticket's (possibly edited) draft after a new result arrives.
                    key=f"draft_{tone.value}_{hashlib.sha1(draft.text.encode()).hexdigest()[:8]}",
                    label_visibility="collapsed",
                )
                st.caption(f"{word_count(edited)} words")

    if result.kb_candidates:
        with st.expander("Retrieved articles"):
            candidates_df = pd.DataFrame(
                [
                    {
                        "id": hit.article.id,
                        "title": hit.article.title,
                        "score": f"{hit.score:.2f}",
                    }
                    for hit in result.kb_candidates
                ]
            )
            st.dataframe(candidates_df, width="stretch", hide_index=True)

    for step in result.steps:
        if step.model_used in ("none", "unknown"):
            what_failed = "no KB source is available" if step.step == "retrieval" else "no drafts"
            st.warning(f"The {step.step} step failed after retrying every option — {what_failed}.")
        elif len(step.attempts) > 1:
            st.info(f"The {step.step} step needed a retry, and used **{step.model_used}**.")

    st.divider()
    for step in result.steps:
        cols = st.columns(4)
        cols[0].caption(f"{step.step.capitalize()} model: {step.model_used}")
        cols[1].caption(f"Latency: {step.latency_ms} ms")
        cols[2].caption(f"Cost: ${step.cost_usd:.5f}")
        if step.usage is not None:
            cols[3].caption(
                f"Tokens: {step.usage.input_tokens} in "
                f"({step.usage.cached_input_tokens} cached) / {step.usage.output_tokens} out"
            )
        else:
            cols[3].caption("Tokens: n/a")
    st.caption(
        f"**Total** — cost: ${result.total_cost_usd:.5f} · latency: {result.total_latency_ms} ms"
    )
