"""Loads and validates the synthetic test set (`data/tickets.jsonl`)."""

import json
from pathlib import Path

from pydantic import BaseModel

from support_ai.classifier.schema import Category, NextStep, Priority

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "tickets.jsonl"


class TicketCase(BaseModel):
    """One hand-labeled ticket in the synthetic test set."""

    id: str
    text: str
    expected_category: Category
    expected_next_step: NextStep
    expected_priority: Priority
    tags: list[str] = []

    @property
    def expected_needs_review(self) -> bool:
        """Review is rule-based (rules.py): only `escalate_human` responses are flagged."""
        return self.expected_next_step is NextStep.ESCALATE_HUMAN


def load_tickets(path: Path = DEFAULT_DATASET_PATH) -> list[TicketCase]:
    """Read and validate every non-blank line of the JSONL test set."""
    cases: list[TicketCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            cases.append(TicketCase.model_validate(data))
    return cases
