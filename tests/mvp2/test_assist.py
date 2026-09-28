"""`assist()` pipeline tests, using FakeProvider and a stub `KnowledgeBase`. No network."""

import itertools

import pytest
from fakes import FakeProvider, make_result, register_fake

from support_ai.assistant.assist import assist
from support_ai.assistant.schema import JudgmentReason, LLMReply, Tone
from support_ai.core import config
from support_ai.core.config import ModelConfig
from support_ai.core.llm.errors import LLMProviderError
from support_ai.core.llm.gateway import Attempt
from support_ai.kb.base import Article, KBHit, RetrievalError, SearchResult, register_knowledge_base

_name_counter = itertools.count()

ARTICLE = Article(
    id="edit-birth-data",
    title="Edit birth data",
    text="Open Profile then Birth details and tap the time field.",
)

VALID_REPLY_DATA = {
    "summary": "Customer asks how to edit their birth time.",
    "kb_article_id": "edit-birth-data",
    "kb_quote": "Open Profile then Birth details and tap the time field.",
    "account_specific_evidence": None,
    "conflicting_article_ids": [],
    "formal": "Dear [Customer name], ... [Agent name]",
    "empathetic": "I understand this matters to you ...",
    "short": "Hi [Customer name], open Profile then Birth details.",
}


def _no_sleep(_seconds: float) -> None:
    return None


def _fake_chain(fake: FakeProvider, monkeypatch, *, model_id: str = "fake-model") -> list[str]:
    """Point a one-model chain at `fake`, without touching the real 'openai' registration."""
    provider_name = f"fake-assist-{next(_name_counter)}"
    register_fake(provider_name, fake)
    model_config = ModelConfig(
        provider=provider_name,
        model_id=model_id,
        input_price_per_1m=0.2,
        output_price_per_1m=1.25,
        timeout_s=1.0,
    )
    monkeypatch.setattr("support_ai.assistant.assist.MODEL_REGISTRY", {model_id: model_config})
    return [model_id]


class StubKB:
    """A `KnowledgeBase` that replays one scripted `SearchResult`, or raises."""

    name = "stub"

    def __init__(self, result_or_exc: SearchResult | Exception):
        self._result_or_exc = result_or_exc

    def search(self, query: str, *, k: int) -> SearchResult:
        if isinstance(self._result_or_exc, Exception):
            raise self._result_or_exc
        return self._result_or_exc


def _search_result(*, has_match: bool = True, hits: list[KBHit] | None = None) -> SearchResult:
    default_hits = [KBHit(article=ARTICLE, score=0.9)]
    return SearchResult(
        hits=hits if hits is not None else default_hits,
        has_match=has_match,
        latency_ms=5,
        cost_usd=0.0001,
    )


def test_normal_full_reply_returns_three_drafts_and_kb_source_with_summed_cost(monkeypatch):
    fake = FakeProvider(
        responses=[make_result(VALID_REPLY_DATA, input_tokens=50, output_tokens=80)]
    )
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result())

    result = assist("How do I edit my birth time?", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert len(result.drafts) == 3
    assert {d.tone for d in result.drafts} == {Tone.FORMAL, Tone.EMPATHETIC, Tone.SHORT}
    assert result.kb_source is not None
    assert result.kb_source.article_id == "edit-birth-data"
    assert result.needs_human_judgment is False
    assert result.judgment_reasons == []
    assert result.total_cost_usd == pytest.approx(
        result.steps[0].cost_usd + result.steps[1].cost_usd
    )
    assert result.total_cost_usd > 0
    assert fake.calls[0]["schema"] is LLMReply


def test_gap_before_generation_runs_summary_only_mode(monkeypatch):
    fake = FakeProvider(responses=[make_result({"summary": "Customer asks about promo codes."})])
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result(has_match=False, hits=[]))

    result = assist("Do you have promo codes?", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.drafts == []
    assert result.judgment_reasons == [JudgmentReason.KB_NOT_FOUND]
    assert result.needs_human_judgment is True
    assert result.summary == "Customer asks about promo codes."
    system_used = fake.calls[0]["system"]
    assert "You summarize support tickets" in system_used  # summary_v1.md, not reply_v1.md


def test_gap_after_generation_when_model_cites_no_article(monkeypatch):
    data = {**VALID_REPLY_DATA, "kb_article_id": None, "kb_quote": None}
    fake = FakeProvider(responses=[make_result(data)])
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result(has_match=True))

    result = assist("How do I edit my birth time?", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.KB_NOT_FOUND]
    assert result.drafts == []
    # full mode DID run (retrieval had a match); the gap was only caught after generation
    system_used = fake.calls[0]["system"]
    assert "You draft reply suggestions" in system_used


def test_quote_unverified_hides_drafts(monkeypatch):
    data = {**VALID_REPLY_DATA, "kb_quote": "This sentence is nowhere in the article."}
    fake = FakeProvider(responses=[make_result(data)])
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result())

    result = assist("How do I edit my birth time?", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.KB_QUOTE_UNVERIFIED]
    assert result.drafts == []


def test_account_specific_evidence_flags_and_hides_drafts(monkeypatch):
    ticket = "I was charged twice this month, please check my account."
    data = {**VALID_REPLY_DATA, "account_specific_evidence": "I was charged twice this month"}
    fake = FakeProvider(responses=[make_result(data)])
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result())

    result = assist(ticket, kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.ACCOUNT_SPECIFIC]
    assert result.account_specific_evidence == "I was charged twice this month"
    assert result.drafts == []


def test_conflicting_kb_flags_and_hides_drafts(monkeypatch):
    article_a = Article(id="a1", title="A", text="The cancellation window is 24 hours.")
    article_b = Article(id="a2", title="B", text="The cancellation window is 48 hours.")
    hits = [KBHit(article=article_a, score=0.9), KBHit(article=article_b, score=0.8)]
    data = {
        **VALID_REPLY_DATA,
        "kb_article_id": "a1",
        "kb_quote": "The cancellation window is 24 hours.",
        "conflicting_article_ids": ["a1", "a2"],
    }
    fake = FakeProvider(responses=[make_result(data)])
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result(hits=hits))

    result = assist("How late can I cancel?", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.CONFLICTING_KB]
    assert set(result.conflicting_article_ids) == {"a1", "a2"}
    assert result.drafts == []


def test_retrieval_failed_via_retrieval_error_runs_summary_only(monkeypatch):
    attempts = [Attempt(model="text-embedding-3-small", outcome="timeout")]
    exc = RetrievalError("embedding call failed", attempts=attempts)
    kb = StubKB(exc)
    fake = FakeProvider(responses=[make_result({"summary": "Customer asks something."})])
    chain = _fake_chain(fake, monkeypatch)

    result = assist("Some question", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.RETRIEVAL_FAILED]
    assert result.kb_candidates == []
    assert result.kb_source is None
    retrieval_step = result.steps[0]
    assert retrieval_step.model_used == "stub"
    assert len(retrieval_step.attempts) == 1
    assert retrieval_step.attempts[0].outcome == "timeout"


def test_unexpected_exception_in_kb_search_is_treated_as_retrieval_failed(monkeypatch):
    kb = StubKB(RuntimeError("boom"))
    fake = FakeProvider(responses=[make_result({"summary": "Customer asks something."})])
    chain = _fake_chain(fake, monkeypatch)

    result = assist("Some question", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.RETRIEVAL_FAILED]
    assert result.steps[0].model_used == "stub"
    assert result.steps[0].attempts[0].outcome == "unexpected_error"


def test_generation_failed_on_every_model(monkeypatch):
    fake = FakeProvider(
        responses=[LLMProviderError("boom"), LLMProviderError("boom"), LLMProviderError("boom")]
    )
    chain = _fake_chain(fake, monkeypatch)
    kb = StubKB(_search_result())

    result = assist("How do I edit my birth time?", kb=kb, reply_chain=chain, sleep=_no_sleep)

    assert result.judgment_reasons == [JudgmentReason.GENERATION_FAILED]
    assert result.summary == "Summary unavailable: reply generation failed."
    assert result.drafts == []
    assert result.kb_candidates != []  # retrieval succeeded; only generation failed
    assert result.steps[1].model_used == "none"


@pytest.mark.parametrize("ticket_text", ["", "   ", "\n\t"])
def test_empty_ticket_makes_no_calls_at_all(ticket_text):
    class NoCallKB:
        name = "no-call"

        def search(self, query: str, *, k: int) -> SearchResult:
            raise AssertionError("kb.search must not be called for empty input")

    result = assist(ticket_text, kb=NoCallKB())

    assert result.needs_human_judgment is False
    assert result.judgment_reasons == []
    assert result.drafts == []
    assert result.steps == []


def test_swap_a_registered_backend_is_used_via_config_kb_backend(monkeypatch):
    """Proves the retrieval layer is replaceable: register a stub under a fresh name,
    point `config.KB_BACKEND` at it, and `assist()` (with no `kb=` argument) must use it."""
    backend_name = f"stub-backend-{next(_name_counter)}"
    kb = StubKB(_search_result(has_match=False, hits=[]))
    register_knowledge_base(backend_name, lambda: kb)
    monkeypatch.setattr(config, "KB_BACKEND", backend_name)

    fake = FakeProvider(responses=[make_result({"summary": "ok"})])
    chain = _fake_chain(fake, monkeypatch)

    result = assist("Some question", reply_chain=chain, sleep=_no_sleep)  # no kb= passed

    assert result.judgment_reasons == [JudgmentReason.KB_NOT_FOUND]
    assert result.steps[0].model_used == "stub"
