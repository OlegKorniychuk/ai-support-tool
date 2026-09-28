"""Tests for scripts/run_reply_eval.py: the expected-vs-actual mapping (`to_eval_record`)
and an end-to-end run against a stub KB + FakeProvider, writing into a tmp results dir.

Loaded by file path (not `import scripts.run_reply_eval`), like `tests/mvp1/test_run_eval.py`
does, since `scripts/` isn't a package on `support-ai`'s import path. No network calls.
"""

import csv
import importlib.util
import itertools
import json
from pathlib import Path

from fakes import FakeProvider, make_result, register_fake

from support_ai.assistant.dataset import ReplyCase
from support_ai.assistant.schema import (
    AssistResult,
    Draft,
    JudgmentReason,
    KBSource,
    StepInfo,
    Tone,
)
from support_ai.core import config
from support_ai.core.config import ModelConfig
from support_ai.core.llm.base import Usage
from support_ai.kb.base import Article, KBHit, SearchResult, register_knowledge_base

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "run_reply_eval", REPO_ROOT / "scripts" / "run_reply_eval.py"
)
run_reply_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_reply_eval)

_name_counter = itertools.count()

ARTICLE = Article(
    id="edit-birth-data",
    title="Edit birth data",
    text="Open Profile then Birth details and tap the time field.",
)

VALID_DATA = {
    "summary": "Customer asks how to edit their birth time.",
    "kb_article_id": "edit-birth-data",
    "kb_quote": "Open Profile then Birth details and tap the time field.",
    "account_specific_evidence": None,
    "conflicting_article_ids": [],
    "formal": "Dear [Customer name], open Profile then Birth details to change it. [Agent name]",
    "empathetic": "I understand this matters to you — open Profile then Birth details.",
    "short": "Open Profile then Birth details and tap the time field.",
}


def _clean_assist_result() -> AssistResult:
    kb_source = KBSource(
        article_id=ARTICLE.id, title=ARTICLE.title, quote=ARTICLE.text, text=ARTICLE.text
    )
    drafts = [
        Draft(tone=Tone.FORMAL, text=VALID_DATA["formal"]),
        Draft(tone=Tone.EMPATHETIC, text=VALID_DATA["empathetic"]),
        Draft(tone=Tone.SHORT, text=VALID_DATA["short"]),
    ]
    steps = [
        StepInfo(
            step="retrieval", model_used="in_memory", latency_ms=5, cost_usd=0.0001, attempts=[]
        ),
        StepInfo(
            step="generation",
            model_used="gpt-5.4-nano",
            latency_ms=100,
            usage=Usage(input_tokens=10, output_tokens=20, cached_input_tokens=2),
            cost_usd=0.001,
            attempts=[],
        ),
    ]
    return AssistResult(
        summary=VALID_DATA["summary"],
        kb_source=kb_source,
        kb_candidates=[KBHit(article=ARTICLE, score=0.9)],
        drafts=drafts,
        needs_human_judgment=False,
        steps=steps,
    )


def test_to_eval_record_clean_result():
    case = ReplyCase(
        id="t1",
        text="How do I edit my birth time?",
        expected_kb_ids=["edit-birth-data"],
        expected_needs_judgment=False,
        tags=["answerable"],
    )

    record = run_reply_eval.to_eval_record(case, _clean_assist_result())

    assert record.ticket_id == "t1"
    assert record.ticket_text == "How do I edit my birth time?"
    assert record.retrieved_ids == ["edit-birth-data"]
    assert record.cited_article_id == "edit-birth-data"
    assert record.kb_quote == ARTICLE.text
    assert record.drafts == {
        "formal": VALID_DATA["formal"],
        "empathetic": VALID_DATA["empathetic"],
        "short": VALID_DATA["short"],
    }
    assert record.tone_check is not None
    assert record.actual_needs_judgment is False
    assert record.actual_reasons == []
    assert record.error is None
    assert record.input_tokens == 10
    assert record.output_tokens == 20
    assert record.cached_input_tokens == 2
    assert record.latency_ms == 105
    assert record.cost_usd > 0


def test_to_eval_record_flagged_result_has_no_drafts_or_tone_check():
    case = ReplyCase(
        id="t2",
        text="I was charged twice, please check my account.",
        expected_kb_ids=["refund-policy"],
        expected_needs_judgment=True,
        expected_reasons=[JudgmentReason.ACCOUNT_SPECIFIC],
        tags=["account_specific"],
    )
    steps = [
        StepInfo(
            step="retrieval", model_used="in_memory", latency_ms=5, cost_usd=0.0001, attempts=[]
        ),
        StepInfo(
            step="generation",
            model_used="gpt-5.4-nano",
            latency_ms=100,
            cost_usd=0.001,
            attempts=[],
        ),
    ]
    result = AssistResult(
        summary="Customer asks about a duplicate charge.",
        kb_source=None,
        kb_candidates=[KBHit(article=ARTICLE, score=0.5)],
        drafts=[],
        needs_human_judgment=True,
        judgment_reasons=[JudgmentReason.ACCOUNT_SPECIFIC],
        account_specific_evidence="I was charged twice",
        steps=steps,
    )

    record = run_reply_eval.to_eval_record(case, result)

    assert record.actual_needs_judgment is True
    assert record.actual_reasons == ["account_specific"]
    assert record.drafts == {}
    assert record.tone_check is None
    assert record.cited_article_id is None
    assert record.error is None  # account_specific is an expected outcome, not a failure


def test_to_eval_record_generation_failed_sets_error_but_keeps_candidates():
    case = ReplyCase(
        id="t3", text="Some question", expected_kb_ids=[], expected_needs_judgment=False
    )
    steps = [
        StepInfo(
            step="retrieval", model_used="in_memory", latency_ms=5, cost_usd=0.0001, attempts=[]
        ),
        StepInfo(step="generation", model_used="none", latency_ms=0, cost_usd=0.0, attempts=[]),
    ]
    result = AssistResult(
        summary="Summary unavailable: reply generation failed.",
        kb_source=None,
        kb_candidates=[KBHit(article=ARTICLE, score=0.9)],
        drafts=[],
        needs_human_judgment=True,
        judgment_reasons=[JudgmentReason.GENERATION_FAILED],
        steps=steps,
    )

    record = run_reply_eval.to_eval_record(case, result)

    assert record.error == "generation_failed"
    assert record.drafts == {}
    assert record.tone_check is None
    assert record.retrieved_ids == ["edit-birth-data"]  # retrieval succeeded independently


class _StubKB:
    name = "stub"

    def search(self, query: str, *, k: int) -> SearchResult:
        return SearchResult(
            hits=[KBHit(article=ARTICLE, score=0.9)], has_match=True, latency_ms=5, cost_usd=0.0001
        )


def _fake_chain(fake: FakeProvider, monkeypatch, *, model_id: str = "fake-model") -> str:
    """Point a one-model chain at `fake`, mirroring test_assist.py's `_fake_chain`."""
    provider_name = f"fake-reply-eval-{next(_name_counter)}"
    register_fake(provider_name, fake)
    model_config = ModelConfig(
        provider=provider_name,
        model_id=model_id,
        input_price_per_1m=0.2,
        output_price_per_1m=1.25,
        timeout_s=1.0,
    )
    monkeypatch.setattr("support_ai.assistant.assist.MODEL_REGISTRY", {model_id: model_config})
    return model_id


def test_run_reply_eval_end_to_end_writes_json_and_one_summary_row(tmp_path, monkeypatch):
    cases = [
        ReplyCase(
            id="t1",
            text="How do I edit my birth time?",
            expected_kb_ids=["edit-birth-data"],
            expected_needs_judgment=False,
            tags=["answerable"],
        ),
        ReplyCase(
            id="t2",
            text="How do I change my birth place?",
            expected_kb_ids=["edit-birth-data"],
            expected_needs_judgment=False,
            tags=["answerable"],
        ),
    ]
    fake = FakeProvider(responses=[make_result(VALID_DATA), make_result(VALID_DATA)])
    model_id = _fake_chain(fake, monkeypatch)

    backend_name = f"stub-backend-{next(_name_counter)}"
    register_knowledge_base(backend_name, _StubKB)
    monkeypatch.setattr(config, "KB_BACKEND", backend_name)
    monkeypatch.setattr(run_reply_eval, "RESULTS_DIR", tmp_path)

    records = run_reply_eval.run_reply_eval(model_id, "v1", cases)
    assert len(records) == 2

    run_name = f"{model_id}_v1_20260101T000000Z"
    json_path = tmp_path / f"{run_name}.json"
    run_reply_eval._write_json(records, json_path)
    run_reply_eval._append_summary_row(run_name, model_id, "v1", "20260101T000000Z", records)

    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["ticket_id"] for row in data] == ["t1", "t2"]
    assert data[0]["cited_article_id"] == "edit-birth-data"
    assert data[0]["drafts"]["formal"] == VALID_DATA["formal"]

    summary_path = tmp_path / "summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == run_reply_eval.REPLY_SUMMARY_COLUMNS
    assert len(rows) == 2  # header + exactly one data row
    row = dict(zip(rows[0], rows[1], strict=True))
    assert row["run"] == run_name
    assert row["model"] == model_id
    assert row["prompt_version"] == "v1"
    assert row["n_tickets"] == "2"
    assert row["retrieval_hit_rate"] == "1.0"
    assert row["citation_accuracy"] == "1.0"
    assert row["error_rate"] == "0.0"
