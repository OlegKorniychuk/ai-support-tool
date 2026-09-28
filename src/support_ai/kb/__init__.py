"""KB retrieval: the swappable boundary between `assistant/` and any backend.

`assistant/` (added later) depends only on `kb.base` — its types (`Article`, `KBHit`,
`SearchResult`, `RetrievalError`), its `KnowledgeBase` protocol, and `get_knowledge_base` —
never on a concrete backend module. A backend is search-only; building/updating its index
is its own offline concern (see `KnowledgeBase`'s docstring).

To add a backend: write `kb/<backend>.py` implementing `KnowledgeBase`, import it below and
call `register_knowledge_base("<name>", <factory>)` (lazy, like `core/llm/__init__.py`
registering `OpenAIProvider`), then set `config.KB_BACKEND = "<name>"`. Nothing in
`assistant/` has to change.

`kb/in_memory.py`'s registration as `"in_memory"` (the default `config.KB_BACKEND`) is
added here in a later task, once that adapter exists.
"""

from support_ai.kb.base import (
    Article,
    KBHit,
    KnowledgeBase,
    RetrievalError,
    SearchResult,
    get_knowledge_base,
    register_knowledge_base,
)

__all__ = [
    "Article",
    "KBHit",
    "KnowledgeBase",
    "RetrievalError",
    "SearchResult",
    "get_knowledge_base",
    "register_knowledge_base",
]
