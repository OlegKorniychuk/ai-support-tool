"""Checks on the synthetic knowledge base in `data/kb/`.

Besides "every article loads", this pins the two properties the MVP 2 test set depends on:
the planted contradictions (for `conflicting_kb`) and the deliberate topic gaps (for
`kb_not_found`). Editing an article so either one disappears should fail here, not
silently change eval results.
"""

import re
from pathlib import Path

import pytest

from support_ai.kb.loader import load_articles

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"
ARTICLES = {article.id: article for article in load_articles(KB_DIR)}


def test_kb_has_about_twenty_articles():
    assert 18 <= len(ARTICLES) <= 22


@pytest.mark.parametrize(
    "article_id,fact",
    [
        ("subscription-plans", "at least 24 hours before your renewal date"),
        ("cancel-subscription", "at least 48 hours before your renewal date"),
        ("booking-expert-session", "returned within 3 business days"),
        ("missed-expert-session", "returned automatically within 24 hours"),
    ],
)
def test_planted_contradictions_are_present(article_id, fact):
    assert fact in ARTICLES[article_id].text


@pytest.mark.parametrize("pattern", [r"promo", r"coupon", r"discount", r"gift", r"famil"])
def test_deliberate_gaps_stay_uncovered(pattern):
    hits = [a.id for a in ARTICLES.values() if re.search(pattern, a.text + a.title, re.I)]
    assert hits == []
