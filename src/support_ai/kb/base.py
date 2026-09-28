"""KB retrieval contracts: the swappable boundary between `assistant/` and any backend.

`assistant/` (added later) imports only this module, plus `get_knowledge_base`. It never
imports a concrete backend. Swapping in a real vector DB means adding a new
`kb/<backend>.py` that implements `KnowledgeBase`, registering it in `kb/__init__.py`
(mirroring `core/llm/__init__.py`'s provider registration), and changing
`config.KB_BACKEND` — nothing in `assistant/` changes.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from support_ai.core import config
from support_ai.core.llm.base import Usage
from support_ai.core.llm.gateway import Attempt


class Article(BaseModel):
    id: str
    title: str
    text: str
    tags: list[str] = []


class KBHit(BaseModel):
    article: Article
    # Backend-specific scale; higher = more relevant. Never compared across backends.
    score: float


class SearchResult(BaseModel):
    hits: list[KBHit] = []  # best first
    # The adapter's own relevance call: score scales differ per backend (cosine similarity,
    # a vector DB's distance metric, ...), so the pipeline never thresholds scores itself.
    has_match: bool
    latency_ms: int
    cost_usd: float = 0.0
    usage: Usage | None = None
    attempts: list[Attempt] = []

    # `Attempt` is a plain stdlib dataclass (shared with the LLM gateway), not a pydantic
    # model or a type pydantic validates out of the box; see `ClassifyResult` for the same
    # workaround.
    model_config = {"arbitrary_types_allowed": True}

    def get_hit(self, article_id: str) -> KBHit | None:
        """The hit for `article_id` among `hits`, or `None` if it wasn't retrieved."""
        return next((hit for hit in self.hits if hit.article.id == article_id), None)


class RetrievalError(Exception):
    """Normalized failure any adapter raises: backend down, embedding call failed, and so
    on. Carries the attempt log, like `EmbeddingFailed`, so callers can report it the same
    way.
    """

    def __init__(self, message: str, *, attempts: list[Attempt] | None = None) -> None:
        self.attempts = attempts or []
        super().__init__(message)


@runtime_checkable
class KnowledgeBase(Protocol):
    """A search-only retrieval backend.

    Ingestion — building or updating the index — is each backend's own offline concern, not
    part of this protocol: `kb/in_memory.py` builds its index from `data/kb/*.md` and caches
    it to disk on first use; a vector-DB backend would instead have its own offline
    ingestion script that populates the DB ahead of time. A `KnowledgeBase` only searches.
    """

    name: str

    def search(self, query: str, *, k: int) -> SearchResult: ...


_FACTORIES: dict[str, Callable[[], KnowledgeBase]] = {}
_INSTANCES: dict[str, KnowledgeBase] = {}


def register_knowledge_base(name: str, factory: Callable[[], KnowledgeBase]) -> None:
    """Register a KB backend factory under `name`. Adapters call this on import.

    Registration is lazy: the factory runs only the first time `get_knowledge_base(name)`
    is called, so importing a backend module never requires building its index.
    """
    _FACTORIES[name] = factory


def get_knowledge_base(name: str | None = None) -> KnowledgeBase:
    """Return the (lazily constructed, memoized) KB backend registered under `name`.

    `name` defaults to `config.KB_BACKEND`, read at call time (not import time) so tests can
    monkeypatch it.
    """
    name = name or config.KB_BACKEND
    if name not in _INSTANCES:
        try:
            factory = _FACTORIES[name]
        except KeyError:
            raise KeyError(
                f"Unknown KB backend {name!r}. Registered: {sorted(_FACTORIES)}"
            ) from None
        _INSTANCES[name] = factory()
    return _INSTANCES[name]
