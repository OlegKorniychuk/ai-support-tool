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
from support_ai.core.text import quote_in_text, strip_markers
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
        kb_source, kb_reason, kb_dropped = _verify_kb_source(reply, search)
        if kb_reason is not None:
            reasons.append(kb_reason)
        dropped_evidence.extend(kb_dropped)

        account_specific_evidence, evidence_reason, evidence_dropped = _verify_evidence(
            reply.account_specific_evidence, ticket
        )
        if evidence_reason is not None:
            reasons.append(evidence_reason)
        dropped_evidence.extend(evidence_dropped)

        conflicting_article_ids, conflict_reason, conflict_dropped = _verify_conflicts(
            reply.conflicting_article_ids, reply.kb_article_id, search
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
) -> tuple[KBSource | None, JudgmentReason | None, list[str]]:
    """Verify the cited article and its quote (the `kb_not_found` / `kb_quote_unverified`
    rows): the model cited no article, the cited article isn't among the retrieved hits, or
    the quote doesn't verify against that article's text. Markers are stripped from
    `kb_quote` before verifying (and from the stored quote) in case the model echoed a
    whole wrapped block instead of copying the article text.

    A cited-but-rejected quote is logged in the returned dropped-evidence list (cited id +
    raw quote) so the eval can see what was rejected, not just that something was.
    """
    if reply.kb_article_id is None:
        return None, JudgmentReason.KB_NOT_FOUND, []

    hit = search.get_hit(reply.kb_article_id) if search is not None else None
    quote = strip_markers(reply.kb_quote) if reply.kb_quote is not None else None
    if hit is not None and quote is not None and quote_in_text(quote, hit.article.text):
        kb_source = KBSource(
            article_id=hit.article.id, title=hit.article.title, quote=quote, text=hit.article.text
        )
        return kb_source, None, []

    dropped = [f"kb_quote unverified for {reply.kb_article_id!r}: {reply.kb_quote!r}"]
    return None, JudgmentReason.KB_QUOTE_UNVERIFIED, dropped


def _verify_evidence(
    evidence: str | None, ticket: str
) -> tuple[str | None, JudgmentReason | None, list[str]]:
    """Verify `account_specific_evidence` is really a quote from `ticket` (the
    `account_specific` row). Markers are stripped before verifying (and from the stored
    evidence) in case the model echoed the whole wrapped ticket block instead of a short
    quote from it. Unverified (but non-null) evidence is dropped, not trusted: a made-up
    quote is no proof the answer depends on the customer's own account.
    """
    if evidence is None:
        return None, None, []
    stripped = strip_markers(evidence)
    if quote_in_text(stripped, ticket):
        return stripped, JudgmentReason.ACCOUNT_SPECIFIC, []
    return None, None, [f"account_specific_evidence not found in ticket: {evidence!r}"]


def _verify_conflicts(
    conflicting_ids: list[str], cited_id: str | None, search: SearchResult | None
) -> tuple[list[str], JudgmentReason | None, list[str]]:
    """Verify the conflict set (the `conflicting_kb` row): `conflicting_article_ids` plus
    the cited article itself, deduped — the model doesn't always repeat `kb_article_id`
    inside `conflicting_article_ids` even when it's one side of the conflict it's
    reporting, so the set it names is completed with the id it cited. At least 2 distinct
    ids, all among the retrieved hits. Anything short of that is dropped, not trusted; a
    reply that names no conflict at all (`conflicting_ids` empty) is never flagged just
    because it happens to cite an article.
    """
    if not conflicting_ids:
        return [], None, []

    ids = [*conflicting_ids, *([cited_id] if cited_id is not None else [])]
    distinct = list(dict.fromkeys(ids))  # de-dupe, preserve order
    retrieved_ids = {hit.article.id for hit in search.hits} if search is not None else set()

    if len(distinct) >= 2 and all(article_id in retrieved_ids for article_id in distinct):
        return distinct, JudgmentReason.CONFLICTING_KB, []
    return [], None, [f"conflicting_article_ids not usable: {distinct!r}"]


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
