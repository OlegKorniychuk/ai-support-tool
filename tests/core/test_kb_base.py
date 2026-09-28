"""KB registry + protocol tests. No network, no real backend."""

import itertools

import pytest

from support_ai.core import config
from support_ai.core.llm.gateway import Attempt
from support_ai.kb.base import (
    Article,
    KBHit,
    KnowledgeBase,
    RetrievalError,
    SearchResult,
    get_knowledge_base,
    register_knowledge_base,
)

_name_counter = itertools.count()


def _unique_name() -> str:
    return f"stub-kb-{next(_name_counter)}"


class _StubKB:
    name = "stub"

    def search(self, query: str, *, k: int) -> SearchResult:
        return SearchResult(hits=[], has_match=False, latency_ms=1)


def test_stub_satisfies_the_knowledge_base_protocol():
    assert isinstance(_StubKB(), KnowledgeBase)


def test_register_and_get_is_lazy_and_memoized():
    name = _unique_name()
    built: list[int] = []

    def factory() -> _StubKB:
        built.append(1)
        return _StubKB()

    register_knowledge_base(name, factory)
    assert built == []  # not built until first get_knowledge_base() call

    kb1 = get_knowledge_base(name)
    kb2 = get_knowledge_base(name)

    assert built == [1]  # built exactly once
    assert kb1 is kb2  # memoized


def test_get_unknown_backend_raises_keyerror_listing_registered_names():
    name = _unique_name()
    register_knowledge_base(name, _StubKB)

    with pytest.raises(KeyError) as exc_info:
        get_knowledge_base("does-not-exist")

    assert name in str(exc_info.value)


def test_get_defaults_to_config_kb_backend(monkeypatch):
    name = _unique_name()
    register_knowledge_base(name, _StubKB)
    monkeypatch.setattr(config, "KB_BACKEND", name)

    kb = get_knowledge_base()

    assert isinstance(kb, _StubKB)


def test_search_result_defaults():
    result = SearchResult(has_match=False, latency_ms=5)
    assert result.hits == []
    assert result.cost_usd == 0.0
    assert result.usage is None
    assert result.attempts == []


def test_search_result_get_hit_finds_by_article_id():
    article = Article(id="a1", title="Title", text="Body")
    hit = KBHit(article=article, score=0.9)
    result = SearchResult(hits=[hit], has_match=True, latency_ms=5)

    assert result.get_hit("a1") is hit
    assert result.get_hit("missing") is None


def test_retrieval_error_carries_attempts():
    attempts = [Attempt(model="m", outcome="timeout")]

    error = RetrievalError("boom", attempts=attempts)

    assert error.attempts == attempts
    assert str(error) == "boom"


def test_retrieval_error_defaults_attempts_to_empty_list():
    assert RetrievalError("boom").attempts == []
