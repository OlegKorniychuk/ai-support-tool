"""`InMemoryKnowledgeBase` tests, using `FakeProvider`. No network calls, no real KB dir."""

import itertools
import json

import pytest
from fakes import FakeProvider, make_embedding_result, register_fake

from support_ai.core.config import ModelConfig
from support_ai.core.llm.errors import LLMTimeout
from support_ai.kb.base import Article, RetrievalError, get_knowledge_base
from support_ai.kb.in_memory import InMemoryKnowledgeBase

_name_counter = itertools.count()


def _unique_provider_name() -> str:
    return f"fake-embed-{next(_name_counter)}"


def _register_embedding_fake(
    fake: FakeProvider, *, model_id: str = "embed-model", input_price_per_1m: float = 0.0
) -> ModelConfig:
    provider_name = _unique_provider_name()
    register_fake(provider_name, fake)
    return ModelConfig(
        provider=provider_name,
        model_id=model_id,
        kind="embedding",
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=0.0,
        timeout_s=1.0,
    )


def _article(id_: str, title: str, text: str) -> Article:
    return Article(id=id_, title=title, text=text)


def _no_sleep(_seconds: float) -> None:
    return None


# Deterministic, hand-picked vectors keyed by the *exact* text `InMemoryKnowledgeBase`
# embeds (`f"{title}\n\n{text}"` for articles, the raw string for queries), so cosine
# scores are fully controlled and ranking is meaningful rather than incidental.
_ALPHA = _article("art-alpha", "Alpha", "alpha body")
_BETA = _article("art-beta", "Beta", "beta body")
_GAMMA = _article("art-gamma", "Gamma", "gamma body")
_RANK_VECTORS = {
    "Alpha\n\nalpha body": [1.0, 0.0, 0.0],
    "Beta\n\nbeta body": [0.6, 0.8, 0.0],  # cosine 0.6 against "query-a"
    "Gamma\n\ngamma body": [0.0, 0.0, 1.0],  # orthogonal to "query-a"
    "query-a": [1.0, 0.0, 0.0],
    "query-b": [0.0, 1.0, 0.0],  # cosine 0.8 against Beta, 0 against the others
}


def _lookup_embed(texts: list[str]) -> list[list[float]]:
    return [_RANK_VECTORS[text] for text in texts]


def _rank_kb(*, min_score: float = 0.0) -> tuple[InMemoryKnowledgeBase, FakeProvider]:
    fake = FakeProvider(embed_fn=_lookup_embed)
    embedding_config = _register_embedding_fake(fake)
    kb = InMemoryKnowledgeBase(
        [_ALPHA, _BETA, _GAMMA],
        cache_path=None,
        embedding_config=embedding_config,
        min_score=min_score,
        sleep=_no_sleep,
    )
    return kb, fake


def test_search_ranks_hits_by_cosine_similarity_best_first():
    kb, _ = _rank_kb()

    result = kb.search("query-a", k=3)

    assert [hit.article.id for hit in result.hits] == ["art-alpha", "art-beta", "art-gamma"]
    assert result.hits[0].score == pytest.approx(1.0)
    assert result.hits[1].score == pytest.approx(0.6)
    assert result.hits[2].score == pytest.approx(0.0)


def test_search_returns_only_top_k():
    kb, _ = _rank_kb()

    result = kb.search("query-a", k=2)

    assert [hit.article.id for hit in result.hits] == ["art-alpha", "art-beta"]


def test_search_k_larger_than_article_count_returns_everything():
    kb, _ = _rank_kb()

    result = kb.search("query-a", k=100)

    assert len(result.hits) == 3


def test_has_match_true_when_top_score_at_or_above_threshold():
    kb, _ = _rank_kb(min_score=0.5)  # top score for "query-b" is 0.8 (Beta)

    result = kb.search("query-b", k=3)

    assert result.has_match is True


def test_has_match_false_when_top_score_below_threshold():
    kb, _ = _rank_kb(min_score=0.95)  # top score for "query-b" is 0.8 (Beta)

    result = kb.search("query-b", k=3)

    assert result.has_match is False


def test_get_knowledge_base_in_memory_returns_an_in_memory_instance():
    kb = get_knowledge_base("in_memory")
    assert isinstance(kb, InMemoryKnowledgeBase)


def test_constructor_does_no_io_and_does_not_build_the_index():
    kb = InMemoryKnowledgeBase(cache_path=None)
    assert kb._index is None  # noqa: SLF001 - white-box check that __init__ has no I/O


def test_cache_hit_on_second_instance_embeds_only_the_query(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeProvider()  # default hash-based embed_fn: fine, ranking isn't asserted here
    embedding_config = _register_embedding_fake(fake)
    articles = [_article("x", "X", "x body"), _article("y", "Y", "y body")]

    kb1 = InMemoryKnowledgeBase(
        list(articles), cache_path=cache_path, embedding_config=embedding_config, sleep=_no_sleep
    )
    kb1.search("first query", k=2)
    assert len(fake.embed_calls) == 2  # one batched build call (2 articles) + one query call

    kb2 = InMemoryKnowledgeBase(
        list(articles), cache_path=cache_path, embedding_config=embedding_config, sleep=_no_sleep
    )
    kb2.search("second query", k=2)

    assert len(fake.embed_calls) == 3  # no new build call: the cache fully hits
    assert fake.embed_calls[-1]["texts"] == ["second query"]


def test_only_the_changed_article_is_reembedded(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeProvider()
    embedding_config = _register_embedding_fake(fake)
    articles_v1 = [_article("x", "X", "x body"), _article("y", "Y", "y body")]

    kb1 = InMemoryKnowledgeBase(
        list(articles_v1),
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )
    kb1.search("q1", k=2)

    articles_v2 = [_article("x", "X", "x body"), _article("y", "Y", "y body CHANGED")]
    kb2 = InMemoryKnowledgeBase(
        list(articles_v2),
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )
    kb2.search("q2", k=2)

    build_call_2 = fake.embed_calls[2]  # 0: kb1 build, 1: kb1 query, 2: kb2 build
    assert build_call_2["texts"] == ["Y\n\ny body CHANGED"]


def test_model_change_reembeds_every_article(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeProvider()
    embedding_config_v1 = _register_embedding_fake(fake, model_id="embed-v1")
    articles = [_article("x", "X", "x body"), _article("y", "Y", "y body")]

    kb1 = InMemoryKnowledgeBase(
        list(articles),
        cache_path=cache_path,
        embedding_config=embedding_config_v1,
        sleep=_no_sleep,
    )
    kb1.search("q1", k=2)

    embedding_config_v2 = _register_embedding_fake(fake, model_id="embed-v2")
    kb2 = InMemoryKnowledgeBase(
        list(articles),
        cache_path=cache_path,
        embedding_config=embedding_config_v2,
        sleep=_no_sleep,
    )
    kb2.search("q2", k=2)

    build_call_2 = fake.embed_calls[2]
    assert set(build_call_2["texts"]) == {"X\n\nx body", "Y\n\ny body"}


def test_deleted_article_is_pruned_from_the_cache_file(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeProvider()
    embedding_config = _register_embedding_fake(fake)
    articles_v1 = [_article("x", "X", "x body"), _article("y", "Y", "y body")]

    kb1 = InMemoryKnowledgeBase(
        list(articles_v1),
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )
    kb1.search("q1", k=2)
    assert set(json.loads(cache_path.read_text())["entries"]) == {"x", "y"}

    articles_v2 = [_article("x", "X", "x body")]  # "y" removed
    kb2 = InMemoryKnowledgeBase(
        list(articles_v2),
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )
    kb2.search("q2", k=1)

    assert set(json.loads(cache_path.read_text())["entries"]) == {"x"}


def test_corrupt_cache_file_is_treated_as_empty(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not valid json {{{")
    fake = FakeProvider()
    embedding_config = _register_embedding_fake(fake)

    kb = InMemoryKnowledgeBase(
        [_article("x", "X", "x body")],
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )
    result = kb.search("q", k=1)

    assert len(result.hits) == 1
    assert len(fake.embed_calls) == 2  # build (cache ignored) + query


def test_unwritable_cache_path_does_not_raise(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a directory")
    cache_path = blocked / "cache.json"  # parent exists but isn't a directory
    fake = FakeProvider()
    embedding_config = _register_embedding_fake(fake)

    kb = InMemoryKnowledgeBase(
        [_article("x", "X", "x body")],
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )
    result = kb.search("q", k=1)  # must not raise despite the cache write failing

    assert len(result.hits) == 1
    assert not cache_path.exists()


def test_embed_failure_raises_retrieval_error_and_next_search_retries_the_build(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeProvider(embed_responses=[LLMTimeout("t1"), LLMTimeout("t2")])
    embedding_config = _register_embedding_fake(fake)
    kb = InMemoryKnowledgeBase(
        [_article("x", "X", "x body")],
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )

    with pytest.raises(RetrievalError) as exc_info:
        kb.search("q1", k=1)

    assert len(exc_info.value.attempts) == 2
    assert all(a.outcome == "timeout" for a in exc_info.value.attempts)
    assert not cache_path.exists()  # nothing memoized or persisted on failure

    # The next search retries the build from scratch: script two fresh successes (build +
    # query) and confirm it now succeeds.
    fake.embed_responses = [make_embedding_result([[1.0]]), make_embedding_result([[1.0]])]
    result = kb.search("q2", k=1)

    assert len(result.hits) == 1
    assert len(fake.embed_calls) == 4  # 2 failed attempts + build + query on the retry


def test_cost_and_usage_are_summed_over_build_and_query(tmp_path):
    cache_path = tmp_path / "cache.json"
    fake = FakeProvider(
        embed_responses=[
            make_embedding_result([[1.0, 0.0]], input_tokens=7),  # index build (1 article)
            make_embedding_result([[1.0, 0.0]], input_tokens=3),  # query
        ]
    )
    embedding_config = _register_embedding_fake(fake, input_price_per_1m=2.0)
    kb = InMemoryKnowledgeBase(
        [_article("x", "X", "x body")],
        cache_path=cache_path,
        embedding_config=embedding_config,
        sleep=_no_sleep,
    )

    result = kb.search("q", k=1)

    assert result.usage.input_tokens == 10  # 7 + 3
    assert result.cost_usd == pytest.approx(10 / 1_000_000 * 2.0)
