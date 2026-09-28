#!/usr/bin/env python
"""Run one model, with no fallback, over the reply-assistant test set and write
results/reply/.

Using a single-model chain (no fallback to another model) keeps the model comparison
clean: every record for a run reflects exactly one model's behavior. Retrieval always
uses the default KB backend (`config.KB_BACKEND`) — this script compares reply models,
not retrieval backends.

Usage: uv run python scripts/run_reply_eval.py --model gpt-5.4-nano [--prompt v1] [--limit 3]
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from support_ai.assistant.assist import assist
from support_ai.assistant.dataset import ReplyCase, load_reply_cases
from support_ai.assistant.schema import AssistResult, JudgmentReason
from support_ai.assistant.tone_checks import check_drafts
from support_ai.core.config import DEFAULT_REPLY_PROMPT_VERSION, MODEL_REGISTRY, SHORT_MAX_WORDS
from support_ai.core.cost import forecast
from support_ai.eval.reply_metrics import (
    ReplyEvalRecord,
    cache_hit_rate,
    citation_accuracy,
    cost_per_ticket,
    distinct_ok_rate,
    drafts_shown_rate,
    dropped_evidence_rate,
    error_rate,
    formal_ok_rate,
    judgment_precision,
    judgment_recall,
    latency_p50,
    latency_p95,
    reason_match_rate,
    retrieval_hit_rate,
    short_ok_rate,
)
from support_ai.eval.summary_csv import append_row

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "reply"
# `run` (the JSON file's stem) lets `record_judgement.py` join a judge_summary.csv row back
# to the summary.csv row it judged, without ever rewriting either file — see summary_csv.py.
REPLY_SUMMARY_COLUMNS = [
    "timestamp",
    "run",
    "model",
    "prompt_version",
    "n_tickets",
    "retrieval_hit_rate",
    "citation_accuracy",
    "judgment_recall",
    "judgment_precision",
    "reason_match_rate",
    "drafts_shown_rate",
    "short_ok_rate",
    "formal_ok_rate",
    "distinct_ok_rate",
    "dropped_evidence_rate",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_ticket_usd",
    "cost_per_10k_tickets_usd",
    "cache_hit_rate",
]


def _error_from_reasons(reasons: list[JudgmentReason]) -> str | None:
    """The one hard-failure reason a record reports as `error`, if any.

    `kb_not_found`, `kb_quote_unverified`, `account_specific` and `conflicting_kb` are
    expected, meaningful outcomes for some tickets in the test set; `retrieval_failed` and
    `generation_failed` mean the pipeline itself broke down, not a judgment call.
    """
    if JudgmentReason.RETRIEVAL_FAILED in reasons:
        return "retrieval_failed"
    if JudgmentReason.GENERATION_FAILED in reasons:
        return "generation_failed"
    return None


def to_eval_record(case: ReplyCase, result: AssistResult) -> ReplyEvalRecord:
    """Map one ticket's expected labels and `assist()`'s actual output to a record."""
    tone_check = (
        check_drafts(result.drafts, short_max_words=SHORT_MAX_WORDS) if result.drafts else None
    )
    input_tokens = sum(step.usage.input_tokens if step.usage else 0 for step in result.steps)
    output_tokens = sum(step.usage.output_tokens if step.usage else 0 for step in result.steps)
    cached_input_tokens = sum(
        step.usage.cached_input_tokens if step.usage else 0 for step in result.steps
    )

    return ReplyEvalRecord(
        ticket_id=case.id,
        ticket_text=case.text,
        tags=case.tags,
        expected_kb_ids=case.expected_kb_ids,
        retrieved_ids=[hit.article.id for hit in result.kb_candidates],
        cited_article_id=result.kb_source.article_id if result.kb_source else None,
        expected_needs_judgment=case.expected_needs_judgment,
        actual_needs_judgment=result.needs_human_judgment,
        expected_reasons=[reason.value for reason in case.expected_reasons],
        actual_reasons=[reason.value for reason in result.judgment_reasons],
        dropped_evidence=result.dropped_evidence,
        summary=result.summary,
        kb_quote=result.kb_source.quote if result.kb_source else None,
        drafts={draft.tone.value: draft.text for draft in result.drafts},
        tone_check=tone_check,
        latency_ms=result.total_latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cost_usd=result.total_cost_usd,
        error=_error_from_reasons(result.judgment_reasons),
    )


def run_reply_eval(
    model: str, prompt_version: str, cases: list[ReplyCase]
) -> list[ReplyEvalRecord]:
    """Answer every ticket with `model` only (no fallback) and record expected vs actual."""
    records = []
    for case in cases:
        result = assist(case.text, reply_chain=[model], prompt_version=prompt_version)
        records.append(to_eval_record(case, result))
    return records


def _write_json(records: list[ReplyEvalRecord], path: Path) -> None:
    payload = [record.model_dump() for record in records]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _append_summary_row(
    run_name: str,
    model: str,
    prompt_version: str,
    timestamp: str,
    records: list[ReplyEvalRecord],
) -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    per_ticket_cost = cost_per_ticket(records)
    row = {
        "timestamp": timestamp,
        "run": run_name,
        "model": model,
        "prompt_version": prompt_version,
        "n_tickets": len(records),
        "retrieval_hit_rate": round(retrieval_hit_rate(records), 4),
        "citation_accuracy": round(citation_accuracy(records), 4),
        "judgment_recall": round(judgment_recall(records), 4),
        "judgment_precision": round(judgment_precision(records), 4),
        "reason_match_rate": round(reason_match_rate(records), 4),
        "drafts_shown_rate": round(drafts_shown_rate(records), 4),
        "short_ok_rate": round(short_ok_rate(records), 4),
        "formal_ok_rate": round(formal_ok_rate(records), 4),
        "distinct_ok_rate": round(distinct_ok_rate(records), 4),
        "dropped_evidence_rate": round(dropped_evidence_rate(records), 4),
        "error_rate": round(error_rate(records), 4),
        "latency_p50_ms": round(latency_p50(records), 1),
        "latency_p95_ms": round(latency_p95(records), 1),
        "cost_per_ticket_usd": round(per_ticket_cost, 6),
        "cost_per_10k_tickets_usd": round(forecast(per_ticket_cost, 10_000), 2),
        "cache_hit_rate": round(cache_hit_rate(records), 4),
    }
    append_row(summary_path, REPLY_SUMMARY_COLUMNS, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    chat_models = sorted(name for name, cfg in MODEL_REGISTRY.items() if cfg.kind == "chat")
    parser.add_argument("--model", required=True, choices=chat_models)
    parser.add_argument("--prompt", default=DEFAULT_REPLY_PROMPT_VERSION, dest="prompt_version")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_reply_cases()
    if args.limit is not None:
        cases = cases[: args.limit]

    records = run_reply_eval(args.model, args.prompt_version, cases)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{args.model}_{args.prompt_version}_{timestamp}"
    json_path = RESULTS_DIR / f"{run_name}.json"
    _write_json(records, json_path)
    _append_summary_row(run_name, args.model, args.prompt_version, timestamp, records)

    per_ticket_cost = cost_per_ticket(records)
    print(f"Wrote {len(records)} record(s) to {json_path} and {RESULTS_DIR / 'summary.csv'}")
    print(
        f"retrieval_hit_rate={retrieval_hit_rate(records):.4f}  "
        f"citation_accuracy={citation_accuracy(records):.4f}  "
        f"judgment_recall={judgment_recall(records):.4f}  "
        f"judgment_precision={judgment_precision(records):.4f}  "
        f"drafts_shown_rate={drafts_shown_rate(records):.4f}  "
        f"error_rate={error_rate(records):.4f}"
    )
    print(
        f"cost_per_ticket_usd={per_ticket_cost:.6f}  "
        f"cost_per_10k_tickets_usd={forecast(per_ticket_cost, 10_000):.2f}"
    )
    print(f"Next: /judge-replies {json_path}")


if __name__ == "__main__":
    main()
