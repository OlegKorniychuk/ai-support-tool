from support_ai.classifier.dataset import load_tickets
from support_ai.classifier.schema import Category

MIN_TICKETS = 15
MAX_TICKETS = 20


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
