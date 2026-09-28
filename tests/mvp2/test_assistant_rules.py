"""Every human-judgment reason, and the verification/dropping rules behind it
(SPEC_MVP2.md, "Human-judgment rules"). Fixtures are built by hand from `kb.base`, not
loaded from `data/kb/`, so these tests don't drift with the real KB content.
"""

from support_ai.assistant.rules import judge, pre_generation_reasons
from support_ai.assistant.schema import JudgmentReason, LLMReply, LLMSummary, Tone
from support_ai.core.text import wrap_ticket
from support_ai.kb.base import Article, KBHit, SearchResult

ART_A = Article(
    id="art-a", title="Article A", text="Open Profile → Settings and tap Save to change name."
)
ART_B = Article(
    id="art-b", title="Article B", text="Cancel at least 24 hours before renewal to avoid a charge."
)
ART_C = Article(
    id="art-c", title="Article C", text="Cancel at least 48 hours before renewal to avoid a charge."
)

SEARCH = SearchResult(
    hits=[
        KBHit(article=ART_A, score=0.9),
        KBHit(article=ART_B, score=0.8),
        KBHit(article=ART_C, score=0.7),
    ],
    has_match=True,
    latency_ms=5,
)
SEARCH_NO_MATCH = SearchResult(hits=[], has_match=False, latency_ms=5)


def _reply(**overrides) -> LLMReply:
    fields = dict(
        summary="Customer asks how to change their name.",
        kb_article_id="art-a",
        kb_quote="Open Profile → Settings and tap Save",
        account_specific_evidence=None,
        conflicting_article_ids=[],
        formal="Dear customer, open Profile → Settings and tap Save.",
        empathetic="I understand you'd like to update your name — open Profile → Settings.",
        short="Open Profile → Settings and tap Save.",
    )
    fields.update(overrides)
    return LLMReply.model_validate(fields)


# --- pre_generation_reasons ------------------------------------------------------------


def test_pre_generation_reasons_retrieval_failed_when_search_is_none():
    assert pre_generation_reasons(None) == [JudgmentReason.RETRIEVAL_FAILED]


def test_pre_generation_reasons_kb_not_found_when_no_match():
    assert pre_generation_reasons(SEARCH_NO_MATCH) == [JudgmentReason.KB_NOT_FOUND]


def test_pre_generation_reasons_clean_when_has_match():
    assert pre_generation_reasons(SEARCH) == []


# --- judge: the clean path --------------------------------------------------------------


def test_judge_clean_reply_shows_three_drafts_in_tone_order_and_kb_source():
    result = judge(ticket="How do I change my name?", search=SEARCH, reply=_reply(), pre_reasons=[])

    assert result.reasons == []
    assert result.needs_human_judgment is False
    assert [draft.tone for draft in result.drafts] == [Tone.FORMAL, Tone.EMPATHETIC, Tone.SHORT]
    assert result.kb_source is not None
    assert result.kb_source.article_id == "art-a"
    assert result.kb_source.quote == "Open Profile → Settings and tap Save"


# --- judge: kb_not_found -----------------------------------------------------------------


def test_judge_kb_not_found_when_model_cites_no_article():
    reply = _reply(kb_article_id=None, kb_quote=None)
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.KB_NOT_FOUND]
    assert result.needs_human_judgment is True
    assert result.drafts == []
    assert result.kb_source is None


def test_judge_kb_not_found_carried_over_from_pre_generation_reasons():
    reply = LLMSummary(summary="Customer asks about something not in the KB.")
    result = judge(
        ticket="anything",
        search=SEARCH_NO_MATCH,
        reply=reply,
        pre_reasons=[JudgmentReason.KB_NOT_FOUND],
    )

    assert result.reasons == [JudgmentReason.KB_NOT_FOUND]
    assert result.drafts == []
    assert result.kb_source is None


# --- judge: kb_quote_unverified -----------------------------------------------------------


def test_judge_kb_quote_unverified_when_cited_id_not_retrieved():
    reply = _reply(kb_article_id="art-not-retrieved")
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.KB_QUOTE_UNVERIFIED]
    assert result.drafts == []
    assert result.kb_source is None
    # logged so the eval can see *why* it was rejected, not just that it was (see r014).
    assert len(result.dropped_evidence) == 1
    assert "art-not-retrieved" in result.dropped_evidence[0]


def test_judge_kb_quote_unverified_when_quote_is_a_paraphrase():
    reply = _reply(kb_quote="Go to your settings and hit save to update your name")
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.KB_QUOTE_UNVERIFIED]
    assert result.drafts == []
    assert result.kb_source is None
    assert len(result.dropped_evidence) == 1
    assert "art-a" in result.dropped_evidence[0]
    assert "Go to your settings and hit save to update your name" in result.dropped_evidence[0]


def test_judge_kb_quote_with_prompt_markers_still_verifies_and_is_stored_stripped():
    """The model can echo a whole `<<<KB ...>>>` block instead of copying just the quoted
    sentence — markers must be stripped before verifying, and the stored quote is the
    stripped text."""
    quote_with_markers = (
        "<<<KB id=art-a title=Article A>>>\nOpen Profile → Settings and tap Save\n<<<END KB>>>"
    )
    reply = _reply(kb_quote=quote_with_markers)
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == []
    assert result.kb_source is not None
    assert result.kb_source.quote == "Open Profile → Settings and tap Save"


# --- judge: account_specific ---------------------------------------------------------------


def test_judge_account_specific_kept_when_evidence_verifies_and_kb_source_still_shown():
    ticket = "I was charged twice on March 3rd for my subscription."
    reply = _reply(account_specific_evidence="charged twice on March 3rd")
    result = judge(ticket=ticket, search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.ACCOUNT_SPECIFIC]
    assert result.account_specific_evidence == "charged twice on March 3rd"
    assert result.drafts == []
    # The verified KB source is kept even though the ticket is flagged for another reason,
    # so the agent can still see it.
    assert result.kb_source is not None
    assert result.kb_source.article_id == "art-a"


def test_judge_account_specific_evidence_with_prompt_markers_still_verifies():
    """The model can accidentally report the whole wrapped ticket (markers included) as
    evidence instead of a short quote from it — markers must be stripped before verifying,
    and the stored evidence is the stripped text (see r026)."""
    ticket = "My Premium access just stopped working yesterday out of nowhere."
    reply = _reply(account_specific_evidence=wrap_ticket(ticket))
    result = judge(ticket=ticket, search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.ACCOUNT_SPECIFIC]
    assert result.account_specific_evidence == ticket
    assert "<<<TICKET>>>" not in result.account_specific_evidence


def test_judge_account_specific_evidence_dropped_when_not_found_in_ticket():
    ticket = "How do I change my name?"
    reply = _reply(account_specific_evidence="a quote that is not in the ticket")
    result = judge(ticket=ticket, search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == []
    assert result.needs_human_judgment is False
    assert result.account_specific_evidence is None
    assert len(result.dropped_evidence) == 1
    assert "a quote that is not in the ticket" in result.dropped_evidence[0]
    # Not flagged, so drafts are still shown.
    assert len(result.drafts) == 3


# --- judge: conflicting_kb -------------------------------------------------------------------


def test_judge_conflicting_kb_kept_when_two_distinct_retrieved_ids():
    # default _reply() cites "art-a"; the conflict set includes it too (see below).
    reply = _reply(conflicting_article_ids=["art-b", "art-c"])
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.CONFLICTING_KB]
    assert set(result.conflicting_article_ids) == {"art-a", "art-b", "art-c"}
    assert result.drafts == []


def test_judge_conflicting_kb_includes_the_cited_article_even_when_not_repeated():
    """The model cites "art-a" but names only the *other* side of the conflict
    ("art-b") in `conflicting_article_ids` — it doesn't always repeat its own citation
    there. That's still a reported 2-way conflict (art-a vs art-b), so it must be kept,
    with the full set (cited id included) returned — this is the r028/r029 fix."""
    reply = _reply(conflicting_article_ids=["art-b"])
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [JudgmentReason.CONFLICTING_KB]
    assert set(result.conflicting_article_ids) == {"art-a", "art-b"}
    assert result.drafts == []


def test_judge_conflicting_kb_dropped_when_still_only_one_distinct_id():
    # The model names only the article it already cited as "conflicting" — still just 1
    # distinct id once deduped, so there's no second side to the conflict.
    reply = _reply(conflicting_article_ids=["art-a"])
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == []
    assert result.conflicting_article_ids == []
    assert len(result.dropped_evidence) == 1
    assert len(result.drafts) == 3


def test_judge_conflicting_kb_dropped_when_an_id_was_not_retrieved():
    reply = _reply(conflicting_article_ids=["art-b", "art-not-retrieved"])
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == []
    assert result.conflicting_article_ids == []
    assert len(result.dropped_evidence) == 1
    assert len(result.drafts) == 3


def test_judge_conflicting_kb_not_flagged_when_model_reports_no_conflict():
    # Citing an article alone must never manufacture a conflict out of thin air.
    reply = _reply(conflicting_article_ids=[])
    result = judge(ticket="anything", search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == []
    assert result.conflicting_article_ids == []
    assert result.dropped_evidence == []
    assert len(result.drafts) == 3


# --- judge: retrieval_failed / generation_failed --------------------------------------------


def test_judge_retrieval_failed_summary_only():
    reply = LLMSummary(summary="Customer asks how to change their name.")
    result = judge(
        ticket="anything", search=None, reply=reply, pre_reasons=[JudgmentReason.RETRIEVAL_FAILED]
    )

    assert result.reasons == [JudgmentReason.RETRIEVAL_FAILED]
    assert result.drafts == []
    assert result.kb_source is None
    assert result.account_specific_evidence is None
    assert result.conflicting_article_ids == []


def test_judge_generation_failed_when_reply_is_none():
    result = judge(ticket="anything", search=SEARCH, reply=None, pre_reasons=[])

    assert result.reasons == [JudgmentReason.GENERATION_FAILED]
    assert result.drafts == []
    assert result.kb_source is None


# --- judge: ordering and dedupe ---------------------------------------------------------------


def test_judge_dedupes_and_orders_reasons_by_declaration():
    # pre_reasons handed out of declaration order, and GENERATION_FAILED duplicated by the
    # reply=None branch: the result must still be deduped and ordered by JudgmentReason.
    result = judge(
        ticket="anything",
        search=SEARCH,
        reply=None,
        pre_reasons=[JudgmentReason.GENERATION_FAILED, JudgmentReason.KB_NOT_FOUND],
    )

    assert result.reasons == [JudgmentReason.KB_NOT_FOUND, JudgmentReason.GENERATION_FAILED]
    assert result.drafts == []


def test_judge_multiple_reply_reasons_are_ordered_by_declaration():
    ticket = "I was charged twice on March 3rd for my subscription."
    reply = _reply(
        kb_article_id=None,
        kb_quote=None,
        account_specific_evidence="charged twice on March 3rd",
        conflicting_article_ids=["art-c", "art-b"],
    )
    result = judge(ticket=ticket, search=SEARCH, reply=reply, pre_reasons=[])

    assert result.reasons == [
        JudgmentReason.KB_NOT_FOUND,
        JudgmentReason.ACCOUNT_SPECIFIC,
        JudgmentReason.CONFLICTING_KB,
    ]
    assert result.drafts == []
    assert result.account_specific_evidence == "charged twice on March 3rd"
    assert set(result.conflicting_article_ids) == {"art-b", "art-c"}
