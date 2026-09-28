#!/usr/bin/env python
"""Print each reply-eval ticket's KB retrieval scores, to calibrate `config.KB_MIN_SCORE`.

For every ticket in `data/reply_tickets.jsonl`, searches the default KB backend and prints
its top-3 hits and whether an expected article was retrieved. Then reports whether "gap"
tickets (`expected_kb_ids` empty — the tool should say "I don't know") and "non-gap"
tickets (`expected_kb_ids` non-empty — an article should answer them) separate cleanly by
best score, and suggests a threshold at their midpoint.

Usage: uv run python scripts/kb_scores.py
Cost: ~30 ticket embeddings on the default embedding model (a few $0.0001).
"""

from dataclasses import dataclass

from support_ai.assistant.dataset import load_reply_cases
from support_ai.core import config
from support_ai.kb import get_knowledge_base


@dataclass
class ScoreRow:
    """One ticket's retrieval outcome: input to `suggest_threshold` and the printed table."""

    ticket_id: str
    tags: list[str]
    expected_kb_ids: list[str]
    top3: list[tuple[str, float]]  # (article id, score), best first
    is_gap: bool  # `expected_kb_ids` empty: this ticket has no article that should answer it

    @property
    def best_score(self) -> float:
        return self.top3[0][1] if self.top3 else 0.0

    @property
    def hit(self) -> bool:
        """An expected id is anywhere in the top-3. Always `False` for gap tickets — they
        have no expected ids, so this isn't a meaningful signal for them."""
        top_ids = {article_id for article_id, _ in self.top3}
        return any(expected_id in top_ids for expected_id in self.expected_kb_ids)


def suggest_threshold(rows: list[ScoreRow]) -> tuple[float | None, str]:
    """Suggest a `KB_MIN_SCORE` separating gap tickets' best scores from non-gap tickets'.

    Returns `(threshold, details)` when every gap ticket's best score is strictly below
    every non-gap ticket's best score — `threshold` is their midpoint, rounded to 2 dp (a
    tie, margin exactly 0, does *not* count as separable: a `>=` threshold at that value
    would also match the tied gap ticket). Returns `(None, details)` when the two groups
    overlap (names the overlapping tickets) or when either group is empty.
    """
    gap_scores = [row.best_score for row in rows if row.is_gap]
    non_gap_scores = [row.best_score for row in rows if not row.is_gap]

    if not gap_scores or not non_gap_scores:
        return None, "Need at least one gap ticket and one non-gap ticket to calibrate."

    gap_max = max(gap_scores)
    non_gap_min = min(non_gap_scores)
    margin = non_gap_min - gap_max

    if margin > 0:
        threshold = round((gap_max + non_gap_min) / 2, 2)
        details = (
            f"Separable: gap max={gap_max:.4f}, non-gap min={non_gap_min:.4f}, "
            f"margin={margin:.4f}. Suggested KB_MIN_SCORE={threshold}."
        )
        return threshold, details

    overlapping = sorted(
        row.ticket_id
        for row in rows
        if (row.is_gap and row.best_score >= non_gap_min)
        or (not row.is_gap and row.best_score <= gap_max)
    )
    details = (
        f"Overlap: gap max={gap_max:.4f} >= non-gap min={non_gap_min:.4f} "
        f"(margin={margin:.4f}). Overlapping ticket(s): {overlapping}."
    )
    return None, details


def _score_rows() -> list[ScoreRow]:
    kb = get_knowledge_base()
    rows = []
    for case in load_reply_cases():
        search = kb.search(case.text, k=config.KB_TOP_K)
        top3 = [(hit.article.id, hit.score) for hit in search.hits]
        rows.append(
            ScoreRow(
                ticket_id=case.id,
                tags=case.tags,
                expected_kb_ids=case.expected_kb_ids,
                top3=top3,
                is_gap=not case.expected_kb_ids,
            )
        )
    return rows


def _print_table(rows: list[ScoreRow]) -> None:
    ordered = sorted(rows, key=lambda row: row.best_score)
    header = f"{'best':>6}  {'group':<8}{'hit':<6}{'id':<8}{'tags':<30}{'expected':<24}top-3"
    print(header)
    print("-" * len(header))
    for row in ordered:
        group = "gap" if row.is_gap else "non-gap"
        top3_str = ", ".join(f"{article_id}:{score:.4f}" for article_id, score in row.top3)
        print(
            f"{row.best_score:6.4f}  {group:<8}{str(row.hit):<6}{row.ticket_id:<8}"
            f"{','.join(row.tags):<30}{','.join(row.expected_kb_ids):<24}{top3_str}"
        )


def main() -> None:
    rows = _score_rows()
    _print_table(rows)

    threshold, details = suggest_threshold(rows)
    print()
    print(details)


if __name__ == "__main__":
    main()
