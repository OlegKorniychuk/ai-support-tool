#!/usr/bin/env python
"""Run one model, with no fallback, over the synthetic test set and write results/.

Using a single-model chain (no fallback to another model) keeps the model comparison
clean: every record for a run reflects exactly one model's behavior.

Usage: uv run python scripts/run_eval.py --model gpt-5.4-nano --prompt v1 [--limit 3]
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from support_ai.classifier.classify import classify
from support_ai.classifier.dataset import TicketCase, load_tickets
from support_ai.core.config import MODEL_REGISTRY
from support_ai.core.cost import forecast
from support_ai.eval.metrics import (
    EvalRecord,
    category_accuracy,
    cost_per_ticket,
    human_review_recall,
    latency_p50,
    latency_p95,
    priority_accuracy,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUMMARY_COLUMNS = [
    "timestamp",
    "model",
    "prompt_version",
    "n_tickets",
    "category_accuracy",
    "priority_accuracy",
    "human_review_recall",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_ticket_usd",
    "cost_per_10k_tickets_usd",
]


def _to_eval_record(ticket: TicketCase, result) -> EvalRecord:
    classification = result.classification
    return EvalRecord(
        ticket_id=ticket.id,
        expected_category=ticket.expected_category.value,
        actual_category=classification.category.value,
        expected_priority=ticket.expected_priority.value,
        actual_priority=classification.priority.value,
        expected_needs_review=ticket.expected_needs_review,
        actual_needs_review=classification.needs_human_review,
        latency_ms=result.latency_ms,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd=result.cost_usd,
        error="classification_failed" if result.model_used == "none" else None,
    )


def run_eval(model: str, prompt_version: str, tickets: list[TicketCase]) -> list[EvalRecord]:
    """Classify every ticket with `model` only (no fallback) and record expected vs actual."""
    records = []
    for ticket in tickets:
        result = classify(ticket.text, model_chain=[model], prompt_version=prompt_version)
        records.append(_to_eval_record(ticket, result))
    return records


def _write_json(records: list[EvalRecord], path: Path) -> None:
    payload = [record.model_dump() for record in records]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _append_summary_row(
    model: str, prompt_version: str, timestamp: str, records: list[EvalRecord]
) -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    is_new_file = not summary_path.exists()
    per_ticket_cost = cost_per_ticket(records)
    row = {
        "timestamp": timestamp,
        "model": model,
        "prompt_version": prompt_version,
        "n_tickets": len(records),
        "category_accuracy": round(category_accuracy(records), 4),
        "priority_accuracy": round(priority_accuracy(records), 4),
        "human_review_recall": round(human_review_recall(records), 4),
        "latency_p50_ms": round(latency_p50(records), 1),
        "latency_p95_ms": round(latency_p95(records), 1),
        "cost_per_ticket_usd": round(per_ticket_cost, 6),
        "cost_per_10k_tickets_usd": round(forecast(per_ticket_cost, 10_000), 2),
    }
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--prompt", default="v1", dest="prompt_version")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tickets = load_tickets()
    if args.limit is not None:
        tickets = tickets[: args.limit]

    records = run_eval(args.model, args.prompt_version, tickets)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{args.model}_{args.prompt_version}_{timestamp}"
    _write_json(records, RESULTS_DIR / f"{run_name}.json")
    _append_summary_row(args.model, args.prompt_version, timestamp, records)

    print(f"Wrote {len(records)} record(s) to results/{run_name}.json and results/summary.csv")


if __name__ == "__main__":
    main()
