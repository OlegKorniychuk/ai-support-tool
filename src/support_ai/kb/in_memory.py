"""In-memory, embeddings-based KB retrieval — the default `KnowledgeBase` backend.

Loads articles from `data/kb/*.md` (`kb/loader.py`), embeds each with the configured
embedding model, and answers `search()` with pure-Python cosine similarity over an index
built lazily on first use. Vectors are cached to disk (`config.KB_CACHE_PATH`) so a
Streamlit container, or a test run, doesn't re-embed the whole KB on every start; only
articles whose text (or the embedding model) changed since the last build are re-embedded.
This module has no special status besides being the default: swapping to a real vector DB
means writing a different `kb/<backend>.py` against the same `KnowledgeBase` protocol.
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from support_ai.core import config
from support_ai.core.config import ModelConfig
from support_ai.core.cost import cost_for_usage
from support_ai.core.llm.base import Usage
from support_ai.core.llm.gateway import Attempt, EmbeddingFailed, embed_with_retry
from support_ai.kb.base import Article, KBHit, RetrievalError, SearchResult
from support_ai.kb.loader import load_articles

_Vector = list[float]


@dataclass
class _IndexEntry:
    article: Article
    vector: _Vector


def _embedded_text(article: Article) -> str:
    """Text embedded for one article: title first, so it weighs into the vector too."""
    return f"{article.title}\n\n{article.text}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine(a: _Vector, b: _Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryKnowledgeBase:
    """`KnowledgeBase` backed by an in-memory list of embedded articles.

    No I/O happens in `__init__`. The index (articles + vectors) is built lazily on the
    first `search()` call and memoized on the instance, so constructing an
    `InMemoryKnowledgeBase` is always cheap and side-effect-free — handy for
    `get_knowledge_base()`'s lazy registry, and for tests that never call `search()`.
    """

    name = "in_memory"

    def __init__(
        self,
        articles: list[Article] | None = None,
        *,
        cache_path: Path | None = config.KB_CACHE_PATH,
        embedding_config: ModelConfig = config.MODEL_REGISTRY[config.EMBEDDING_MODEL],
        min_score: float = config.KB_MIN_SCORE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._articles = articles
        self._cache_path = cache_path
        self._embedding_config = embedding_config
        self._min_score = min_score
        self._sleep = sleep
        self._index: list[_IndexEntry] | None = None

    def search(self, query: str, *, k: int) -> SearchResult:
        start = time.monotonic()
        attempts: list[Attempt] = []
        input_tokens = 0
        cost_usd = 0.0

        try:
            index, build_usage, build_cost = self._ensure_index(attempts)
            input_tokens += build_usage.input_tokens
            cost_usd += build_cost

            query_result = embed_with_retry(
                self._embedding_config, texts=[query], sleep=self._sleep
            )
        except EmbeddingFailed as exc:
            raise RetrievalError(str(exc), attempts=attempts + exc.attempts) from exc
        except Exception as exc:  # noqa: BLE001 - normalize any build/lookup/config failure
            raise RetrievalError(str(exc), attempts=attempts) from exc

        attempts.extend(query_result.attempts)
        input_tokens += query_result.result.usage.input_tokens
        cost_usd += cost_for_usage(query_result.result.usage, self._embedding_config)

        query_vector = query_result.result.vectors[0]
        scored = [
            KBHit(article=entry.article, score=_cosine(query_vector, entry.vector))
            for entry in index
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        hits = scored[:k]
        has_match = bool(hits) and hits[0].score >= self._min_score

        return SearchResult(
            hits=hits,
            has_match=has_match,
            latency_ms=int((time.monotonic() - start) * 1000),
            cost_usd=cost_usd,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
            attempts=attempts,
        )

    def _ensure_index(self, attempts: list[Attempt]) -> tuple[list[_IndexEntry], Usage, float]:
        """Build the index if it isn't already, reusing cached vectors where possible.

        Returns the index plus this call's own embedding usage/cost for articles that had
        to be (re-)embedded — zero if the index was already memoized, or if every article's
        cached vector could be reused. Appends any embedding attempts to `attempts` (only
        reached on success; a raised `EmbeddingFailed` carries its own attempt log, and
        nothing gets memoized, so the next `search()` retries the build from scratch).
        """
        if self._index is not None:
            return self._index, Usage(input_tokens=0, output_tokens=0), 0.0

        articles = self._articles if self._articles is not None else load_articles(config.KB_DIR)
        cached_entries = self._read_cache_entries()

        reused: dict[str, _Vector] = {}
        to_embed: list[Article] = []
        texts_to_embed: list[str] = []
        for article in articles:
            digest = _sha256(_embedded_text(article))
            cached = cached_entries.get(article.id)
            if cached is not None and cached.get("sha256") == digest:
                reused[article.id] = cached["vector"]
            else:
                to_embed.append(article)
                texts_to_embed.append(_embedded_text(article))

        usage = Usage(input_tokens=0, output_tokens=0)
        cost = 0.0
        vectors: dict[str, _Vector] = dict(reused)
        if to_embed:
            gateway_result = embed_with_retry(
                self._embedding_config, texts=texts_to_embed, sleep=self._sleep
            )
            attempts.extend(gateway_result.attempts)
            usage = gateway_result.result.usage
            cost = cost_for_usage(usage, self._embedding_config)
            for article, vector in zip(to_embed, gateway_result.result.vectors, strict=True):
                vectors[article.id] = vector

        index = [_IndexEntry(article=article, vector=vectors[article.id]) for article in articles]
        self._write_cache(articles=articles, vectors=vectors)
        self._index = index
        return index, usage, cost

    def _read_cache_entries(self) -> dict[str, dict]:
        """The cache file's `entries`, or `{}` if there is none, it's unreadable, or it was
        built for a different embedding model — all treated the same: re-embed everything.
        """
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict) or data.get("model") != self._embedding_config.model_id:
            return {}
        entries = data.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _write_cache(self, *, articles: list[Article], vectors: dict[str, _Vector]) -> None:
        """Persist the full current index to disk. Entries for ids no longer in `articles`
        are dropped simply by not being written. Never raises: on a read-only or otherwise
        unwritable filesystem, the in-memory index still works, just without persistence.
        """
        if self._cache_path is None:
            return
        payload = {
            "model": self._embedding_config.model_id,
            "entries": {
                article.id: {
                    "sha256": _sha256(_embedded_text(article)),
                    "vector": vectors[article.id],
                }
                for article in articles
            },
        }
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
