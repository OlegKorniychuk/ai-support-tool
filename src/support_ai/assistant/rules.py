"""Deterministic post-LLM rules: the human-in-the-loop decision for the Reply Assistant.

The LLM only reports facts — a cited article and quote, evidence that the answer depends
on the customer's own account, ids of conflicting articles. This module verifies every one
of those facts against `search` and `ticket` and is the only place that sets
`needs_human_judgment` and its reasons (SPEC_MVP2.md, "Human-judgment rules"). A fact that
doesn't verify is dropped rather than trusted, and noted in `dropped_evidence` so the eval
can count how often the model reported something it shouldn't have.
"""

from support_ai.assistant.schema import (
    Draft,
    Judgment,
    JudgmentReason,
    KBSource,
    LLMReply,
    LLMSummary,
    Tone,
)
from support_ai.core.text import quote_in_text
from support_ai.kb.base import SearchResult


def pre_generation_reasons(search: SearchResult | None) -> list[JudgmentReason]:
    """Reasons decidable before generation, from retrieval alone.

    `search` is `None` when retrieval itself raised (`RetrievalError`): the caller has no
    `SearchResult` to pass, since embedding the ticket never produced one.
    """
    if search is None:
        return [JudgmentReason.RETRIEVAL_FAILED]
    if not search.has_match:
        return [JudgmentReason.KB_NOT_FOUND]
    return []


def judge(
    ticket: str,
    search: SearchResult | None,
    reply: LLMReply | LLMSummary | None,
    pre_reasons: list[JudgmentReason],
) -> Judgment:
    """Decide what to show for one ticket, from the LLM's reported facts plus verification.

    Starts from `pre_reasons` (retrieval-only reasons, decided before generation ran) and
    adds any reason found in `reply`. `reply is None` means the generation call failed on
    every model in the chain. An `LLMSummary` (summary-only mode, already flagged before
    generation) carries no KB or evidence claims to check. Drafts are built only when the
    final reason list is empty; a flagged ticket never shows drafts, per SPEC_MVP2.md.
    """
    reasons = list(pre_reasons)
    kb_source: KBSource | None = None
    account_specific_evidence: str | None = None
    conflicting_article_ids: list[str] = []
    dropped_evidence: list[str] = []

    if reply is None:
        reasons.append(JudgmentReason.GENERATION_FAILED)
    elif isinstance(reply, LLMReply):
        kb_source, kb_reason = _verify_kb_source(reply, search)
        if kb_reason is not None:
            reasons.append(kb_reason)

        account_specific_evidence, evidence_reason, evidence_dropped = _verify_evidence(
            reply.account_specific_evidence, ticket
        )
        if evidence_reason is not None:
            reasons.append(evidence_reason)
        dropped_evidence.extend(evidence_dropped)

        conflicting_article_ids, conflict_reason, conflict_dropped = _verify_conflicts(
            reply.conflicting_article_ids, search
        )
        if conflict_reason is not None:
            reasons.append(conflict_reason)
        dropped_evidence.extend(conflict_dropped)
    # else: LLMSummary — summary-only mode, no drafts and nothing further to verify.

    reasons = _dedupe_ordered(reasons)
    drafts = _build_drafts(reply) if not reasons else []

    return Judgment(
        kb_source=kb_source,
        drafts=drafts,
        reasons=reasons,
        account_specific_evidence=account_specific_evidence,
        conflicting_article_ids=conflicting_article_ids,
        dropped_evidence=dropped_evidence,
    )


def _verify_kb_source(
    reply: LLMReply, search: SearchResult | None
) -> tuple[KBSource | None, JudgmentReason | None]:
    """Verify the cited article and its quote (the `kb_not_found` / `kb_quote_unverified`
    rows): the model cited no article, the cited article isn't among the retrieved hits, or
    the quote doesn't verify against that article's text.
    """
    if reply.kb_article_id is None:
        return None, JudgmentReason.KB_NOT_FOUND

    hit = search.get_hit(reply.kb_article_id) if search is not None else None
    quote_ok = (
        hit is not None
        and reply.kb_quote is not None
        and quote_in_text(reply.kb_quote, hit.article.text)
    )
    if not quote_ok:
        return None, JudgmentReason.KB_QUOTE_UNVERIFIED

    kb_source = KBSource(
        article_id=hit.article.id,
        title=hit.article.title,
        quote=reply.kb_quote,
        text=hit.article.text,
    )
    return kb_source, None


def _verify_evidence(
    evidence: str | None, ticket: str
) -> tuple[str | None, JudgmentReason | None, list[str]]:
    """Verify `account_specific_evidence` is really a quote from `ticket` (the
    `account_specific` row). Unverified (but non-null) evidence is dropped, not trusted: a
    made-up quote is no proof the answer depends on the customer's own account.
    """
    if evidence is None:
        return None, None, []
    if quote_in_text(evidence, ticket):
        return evidence, JudgmentReason.ACCOUNT_SPECIFIC, []
    return None, None, [f"account_specific_evidence not found in ticket: {evidence!r}"]


def _verify_conflicts(
    conflicting_ids: list[str], search: SearchResult | None
) -> tuple[list[str], JudgmentReason | None, list[str]]:
    """Verify `conflicting_article_ids` (the `conflicting_kb` row): at least 2 distinct ids,
    all among the retrieved hits. Anything short of that is dropped, not trusted.
    """
    distinct = list(dict.fromkeys(conflicting_ids))  # de-dupe, preserve order
    retrieved_ids = {hit.article.id for hit in search.hits} if search is not None else set()

    if len(distinct) >= 2 and all(article_id in retrieved_ids for article_id in distinct):
        return distinct, JudgmentReason.CONFLICTING_KB, []
    if distinct:
        return [], None, [f"conflicting_article_ids not usable: {distinct!r}"]
    return [], None, []


def _build_drafts(reply: LLMReply | LLMSummary | None) -> list[Draft]:
    """The 3 drafts, in `Tone` declaration order (formal, empathetic, short)."""
    if not isinstance(reply, LLMReply):
        return []
    return [
        Draft(tone=Tone.FORMAL, text=reply.formal),
        Draft(tone=Tone.EMPATHETIC, text=reply.empathetic),
        Draft(tone=Tone.SHORT, text=reply.short),
    ]


def _dedupe_ordered(reasons: list[JudgmentReason]) -> list[JudgmentReason]:
    """De-dupe `reasons` and order them by `JudgmentReason`'s declaration order."""
    present = set(reasons)
    return [reason for reason in JudgmentReason if reason in present]
