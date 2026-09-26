from support_ai.classifier.dataset import load_tickets
from support_ai.classifier.schema import Category

MIN_TICKETS = 15
MAX_TICKETS = 30


def test_loader_parses_all_tickets():
    cases = load_tickets()
    assert MIN_TICKETS <= len(cases) <= MAX_TICKETS
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids)), "ticket ids must be unique"


def test_every_category_is_covered():
    cases = load_tickets()
    covered = {case.expected_category for case in cases}
    assert covered == set(Category)


def test_at_least_two_aggressive_tickets():
    cases = load_tickets()
    aggressive = [c for c in cases if "aggressive" in c.tags]
    assert len(aggressive) >= 2


def test_at_least_two_mixed_topic_tickets():
    cases = load_tickets()
    mixed = [c for c in cases if "mixed_topics" in c.tags]
    assert len(mixed) >= 2


def test_at_least_two_non_english_tickets():
    cases = load_tickets()
    non_english = [c for c in cases if "non_english" in c.tags]
    assert len(non_english) >= 2
    languages = {tag for c in non_english for tag in c.tags if tag in {"uk", "es", "ru"}}
    assert len(languages) >= 2, "non-English tickets should cover more than one language"


def test_has_an_empty_or_too_short_ticket():
    cases = load_tickets()
    assert any("empty" in c.tags or "too_short" in c.tags for c in cases)


def test_has_a_prompt_injection_ticket():
    cases = load_tickets()
    assert any("injection" in c.tags for c in cases)


def test_has_a_legal_or_chargeback_threat_ticket():
    cases = load_tickets()
    assert any("legal_threat" in c.tags for c in cases)


def test_has_usage_help_tickets_in_more_than_one_language():
    cases = load_tickets()
    usage_help = [c for c in cases if "usage_help" in c.tags]
    assert len(usage_help) >= 2
    non_english = [c for c in usage_help if "non_english" in c.tags]
    assert non_english, "at least one usage_help ticket should be non-English"


def test_has_a_human_request_ticket():
    cases = load_tickets()
    assert any("human_request" in c.tags for c in cases)


def test_has_an_expert_request_that_is_not_a_human_request():
    cases = load_tickets()
    assert any("expert_request_not_human" in c.tags for c in cases)


def test_has_an_unrelated_ticket():
    cases = load_tickets()
    assert any("unrelated" in c.tags for c in cases)


def test_has_a_pure_injection_ticket_that_must_not_escalate():
    cases = load_tickets()
    pure = [c for c in cases if "injection" in c.tags and c.expected_category == "unclear"]
    assert pure
    assert all(not c.expected_needs_review for c in pure)


def test_has_a_vague_app_quality_complaint():
    cases = load_tickets()
    assert any("vague_complaint" in c.tags for c in cases)


def test_has_general_feedback_tickets_positive_and_non_english():
    cases = load_tickets()
    feedback = [c for c in cases if c.expected_category == "general_feedback"]
    assert len(feedback) >= 3
    assert any("positive" in c.tags for c in feedback)
    assert any("non_english" in c.tags for c in feedback)
    assert all(not c.expected_needs_review for c in feedback)
