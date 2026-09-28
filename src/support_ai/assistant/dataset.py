"""Loads and validates the reply-assistant test set (`data/reply_tickets.jsonl`)."""

import json
from pathlib import Path

from pydantic import BaseModel, model_validator

from support_ai.assistant.schema import JudgmentReason

DEFAULT_REPLY_DATASET_PATH = Path(__file__).resolve().parents[3] / "data" / "reply_tickets.jsonl"


class ReplyCase(BaseModel):
    """One hand-labeled ticket in the reply-assistant test set.

    `expected_kb_ids` is empty for a KB gap; `expected_reasons` is the set of
    `JudgmentReason` values a passing run must report (SPEC_MVP2.md, "Human-judgment
    rules"). The two must agree on whether the ticket is flagged at all — see the
    validator below.
    """

    id: str
    text: str
    expected_kb_ids: list[str] = []
    expected_needs_judgment: bool
    expected_reasons: list[JudgmentReason] = []
    tags: list[str] = []

    @model_validator(mode="after")
    def _needs_judgment_matches_reasons(self) -> "ReplyCase":
        if self.expected_needs_judgment != bool(self.expected_reasons):
            raise ValueError(
                f"{self.id}: expected_needs_judgment={self.expected_needs_judgment} "
                f"disagrees with expected_reasons={self.expected_reasons}"
            )
        return self


def load_reply_cases(path: Path = DEFAULT_REPLY_DATASET_PATH) -> list[ReplyCase]:
    """Read and validate every non-blank line of the JSONL test set."""
    cases: list[ReplyCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            cases.append(ReplyCase.model_validate(data))
    return cases
