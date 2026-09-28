"""A scripted fake `LLMProvider` for tests. Never touches the network."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from support_ai.core.llm.base import EmbeddingResult, LLMResult, Usage, register_provider
from support_ai.core.llm.errors import LLMError


def _hash_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic small-dim vector per text, derived from a hash of its content.

    Identical text always yields an identical vector, and different text yields (with
    overwhelming probability) a different one — good enough for tests of ranking/cosine
    logic (e.g. the KB adapter) without needing a real embedding model or scripted vectors.
    """
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([byte / 255.0 for byte in digest[:8]])
    return vectors


@dataclass
class FakeProvider:
    """Replays `responses` in order for `complete_structured`, and scripted or computed
    outcomes for `embed`.

    Each `responses` entry is either an `LLMResult` to return or an `LLMError` instance to
    raise. Every call is recorded in `calls` (`complete_structured`) or `embed_calls`
    (`embed`) so tests can assert on prompts, texts and models used.

    `embed` replays `embed_responses` (a list of `EmbeddingResult | LLMError`, one entry
    per call) when given; otherwise it computes vectors from `embed_fn` (default:
    `_hash_embed`), so most tests don't have to script embeddings by hand.
    """

    responses: list[LLMResult | LLMError] = field(default_factory=list)
    embed_responses: list[EmbeddingResult | LLMError] | None = None
    embed_fn: Callable[[list[str]], list[list[float]]] = _hash_embed
    name: ClassVar[str] = "fake"
    calls: list[dict] = field(default_factory=list)
    embed_calls: list[dict] = field(default_factory=list)

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model: str,
        timeout_s: float,
    ) -> LLMResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "model": model,
                "timeout_s": timeout_s,
            }
        )
        if not self.responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        outcome = self.responses.pop(0)
        if isinstance(outcome, LLMError):
            raise outcome
        return outcome

    def embed(
        self,
        *,
        texts: list[str],
        model: str,
        timeout_s: float,
    ) -> EmbeddingResult:
        self.embed_calls.append({"texts": texts, "model": model, "timeout_s": timeout_s})
        if self.embed_responses is not None:
            if not self.embed_responses:
                raise AssertionError("FakeProvider ran out of scripted embed_responses")
            outcome = self.embed_responses.pop(0)
            if isinstance(outcome, LLMError):
                raise outcome
            return outcome
        return EmbeddingResult(
            vectors=self.embed_fn(texts),
            usage=Usage(input_tokens=len(texts), output_tokens=0),
            model=model,
            latency_ms=0,
        )


def register_fake(name: str, provider: FakeProvider) -> None:
    """Register `provider` in the real provider registry under `name`, for gateway tests."""
    register_provider(name, lambda: provider)


def make_result(
    data: dict,
    *,
    model: str = "fake-model",
    input_tokens: int = 10,
    output_tokens: int = 10,
    latency_ms: int = 5,
) -> LLMResult:
    """Convenience builder for a scripted successful `LLMResult`."""
    return LLMResult(
        data=data,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        latency_ms=latency_ms,
    )


def make_embedding_result(
    vectors: list[list[float]],
    *,
    model: str = "fake-embedding-model",
    input_tokens: int = 10,
    latency_ms: int = 5,
) -> EmbeddingResult:
    """Convenience builder for a scripted successful `EmbeddingResult`."""
    return EmbeddingResult(
        vectors=vectors,
        usage=Usage(input_tokens=input_tokens, output_tokens=0),
        model=model,
        latency_ms=latency_ms,
    )
