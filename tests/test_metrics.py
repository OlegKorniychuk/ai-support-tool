from support_ai.eval.metrics import (
    EvalRecord,
    category_accuracy,
    cost_per_ticket,
    human_review_recall,
    latency_p50,
    latency_p95,
    priority_accuracy,
)


def _record(**overrides) -> EvalRecord:
    defaults = {
        "ticket_id": "t1",
        "expected_category": "refund_request",
        "actual_category": "refund_request",
        "expected_priority": "P2",
        "actual_priority": "P2",
        "expected_needs_review": True,
        "actual_needs_review": True,
        "latency_ms": 100,
        "input_tokens": 50,
        "output_tokens": 20,
        "cost_usd": 0.001,
    }
    return EvalRecord.model_validate({**defaults, **overrides})


def test_category_accuracy_all_correct():
    records = [_record(), _record(ticket_id="t2")]
    assert category_accuracy(records) == 1.0


def test_category_accuracy_partial():
    records = [_record(), _record(ticket_id="t2", actual_category="other")]
    assert category_accuracy(records) == 0.5


def test_category_accuracy_empty_records_is_zero():
    assert category_accuracy([]) == 0.0


def test_priority_accuracy_partial():
    records = [_record(), _record(ticket_id="t2", actual_priority="P4")]
    assert priority_accuracy(records) == 0.5


def test_human_review_recall_counts_only_expected_review_tickets():
    records = [
        _record(ticket_id="t1", expected_needs_review=True, actual_needs_review=True),
        _record(ticket_id="t2", expected_needs_review=True, actual_needs_review=False),
        _record(ticket_id="t3", expected_needs_review=False, actual_needs_review=False),
    ]
    # 1 of 2 expected-review tickets was actually flagged; t3 doesn't count
    assert human_review_recall(records) == 0.5


def test_human_review_recall_with_no_expected_review_tickets_is_perfect():
    records = [_record(expected_needs_review=False, actual_needs_review=False)]
    assert human_review_recall(records) == 1.0


def test_human_review_recall_full_marks():
    records = [
        _record(ticket_id="t1", expected_needs_review=True, actual_needs_review=True),
        _record(ticket_id="t2", expected_needs_review=True, actual_needs_review=True),
    ]
    assert human_review_recall(records) == 1.0


def test_latency_percentiles_on_handmade_records():
    latencies = [100, 200, 300, 400]
    records = [_record(ticket_id=str(i), latency_ms=ms) for i, ms in enumerate(latencies)]
    assert latency_p50(records) == 200
    assert latency_p95(records) == 400


def test_latency_percentiles_single_record():
    records = [_record(latency_ms=250)]
    assert latency_p50(records) == 250
    assert latency_p95(records) == 250


def test_cost_per_ticket_averages():
    records = [_record(cost_usd=0.002), _record(ticket_id="t2", cost_usd=0.004)]
    assert cost_per_ticket(records) == 0.003


def test_cost_per_ticket_empty_records_is_zero():
    assert cost_per_ticket([]) == 0.0
