"""Smoke + render tests for the Reply Assistant page via `streamlit.testing.v1.AppTest`.

The plain smoke test needs an API key to get past the page's key check (it doesn't call
the network — `assist()` is never triggered without a button click). The two render tests
put a prebuilt `AssistResult` straight into `st.session_state["last_assist"]` before
`at.run()`, so they exercise the display logic only, with no LLM call either.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from support_ai.assistant.schema import (
    AssistResult,
    Draft,
    JudgmentReason,
    KBSource,
    StepInfo,
    Tone,
)
from support_ai.core.llm.base import Usage
from support_ai.kb.base import Article, KBHit

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = str(REPO_ROOT / "pages" / "3_Reply_Assistant.py")

ARTICLE = Article(
    id="edit-birth-data", title="Edit birth data", text="Open Profile then Birth details."
)


def test_reply_assistant_page_runs_without_exceptions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = AppTest.from_file(PAGE_PATH)

    at.run()

    assert not at.exception


def _clean_result() -> AssistResult:
    kb_source = KBSource(
        article_id=ARTICLE.id,
        title=ARTICLE.title,
        quote=ARTICLE.text,
        text=ARTICLE.text,
    )
    drafts = [
        Draft(tone=Tone.FORMAL, text="Dear [Customer name], formal draft. [Agent name]"),
        Draft(tone=Tone.EMPATHETIC, text="I understand your situation, empathetic draft."),
        Draft(tone=Tone.SHORT, text="Short draft with the answer."),
    ]
    steps = [
        StepInfo(
            step="retrieval", model_used="in_memory", latency_ms=5, cost_usd=0.0001, attempts=[]
        ),
        StepInfo(
            step="generation",
            model_used="gpt-5.4-nano",
            latency_ms=100,
            usage=Usage(input_tokens=10, output_tokens=20),
            cost_usd=0.001,
            attempts=[],
        ),
    ]
    return AssistResult(
        summary="Customer wants to edit their birth time.",
        kb_source=kb_source,
        kb_candidates=[KBHit(article=ARTICLE, score=0.9)],
        drafts=drafts,
        needs_human_judgment=False,
        steps=steps,
    )


def test_clean_result_shows_three_draft_tabs_and_no_error_box(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = AppTest.from_file(PAGE_PATH)
    at.session_state["last_assist"] = _clean_result()

    at.run()

    assert not at.exception
    assert len(at.tabs) == 3
    assert len(at.error) == 0

    draft_areas = {
        ta.key.split("_")[1]: ta.value
        for ta in at.text_area
        if ta.key and ta.key.startswith("draft_")
    }
    assert draft_areas == {
        "formal": "Dear [Customer name], formal draft. [Agent name]",
        "empathetic": "I understand your situation, empathetic draft.",
        "short": "Short draft with the answer.",
    }


def _flagged_account_specific_result() -> AssistResult:
    steps = [
        StepInfo(
            step="retrieval", model_used="in_memory", latency_ms=5, cost_usd=0.0001, attempts=[]
        ),
        StepInfo(
            step="generation",
            model_used="gpt-5.4-nano",
            latency_ms=100,
            cost_usd=0.001,
            attempts=[],
        ),
    ]
    return AssistResult(
        summary="Customer asks about their refund status.",
        kb_source=None,
        kb_candidates=[KBHit(article=ARTICLE, score=0.5)],
        drafts=[],
        needs_human_judgment=True,
        judgment_reasons=[JudgmentReason.ACCOUNT_SPECIFIC],
        account_specific_evidence="I requested a refund on September 20th",
        steps=steps,
    )


def test_flagged_account_specific_result_shows_error_and_evidence_no_drafts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = AppTest.from_file(PAGE_PATH)
    at.session_state["last_assist"] = _flagged_account_specific_result()

    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "Decide yourself" in at.error[0].value
    assert any("I requested a refund on September 20th" in caption.value for caption in at.caption)
    assert len(at.tabs) == 0
    draft_areas = [ta.key for ta in at.text_area if ta.key and ta.key.startswith("draft_")]
    assert draft_areas == []
