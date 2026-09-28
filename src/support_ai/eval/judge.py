"""Pydantic schema for judge verdicts (Sonnet, via the `.claude/skills/judge-replies`
skill) and the validation/aggregation logic that ties them to a reply-eval run.

The judge never calls `assist()` or an OpenAI model directly: it only reads a run's JSON
output (`results/reply/<run>.json`, a list of `ReplyEvalRecord`) and writes verdicts back
in this shape (`results/reply/<run>.judge.json`). `validate_against_run` is the boundary
between "the judge's output is shaped right" (pydantic) and "the judge's verdicts actually
match this run" (every ticket accounted for, drafts present iff verdicts expect them) —
`scripts/record_judgement.py` runs both before ever writing a summary row.
"""

from pydantic import BaseModel, ConfigDict, Field

from support_ai.assistant.schema import Tone
from support_ai.eval.reply_metrics import ReplyEvalRecord

# `scripts/record_judgement.py`'s summary row, in the order written to
# `results/reply/judge_summary.csv`. Joined to that run's `summary.csv` row by `run`.
JUDGE_SUMMARY_COLUMNS = [
    "run",
    "judged_at",
    "judge_model",
    "rubric_version",
    "n_tickets",
    "n_drafted",
    "tone_score_formal",
    "tone_score_empathetic",
    "tone_score_short",
    "faithful_rate",
    "addresses_request_rate",
    "summary_accurate_rate",
    "distinct_rate",
]


class DraftVerdict(BaseModel):
    """The judge's score for one drafted tone of one ticket."""

    model_config = ConfigDict(extra="forbid")

    tone: Tone
    tone_score: int = Field(ge=1, le=5)
    # Every factual claim traceable to the ticket or the cited article; no promise, amount,
    # date or exception the article doesn't make. See rubric.md for the full definition.
    faithful: bool
    addresses_request: bool
    note: str = ""


class TicketVerdict(BaseModel):
    """The judge's verdict for one ticket: its summary, plus one `DraftVerdict` per tone
    when drafts were shown (none when the ticket was flagged for human judgment)."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    summary_accurate: bool
    # Whether the 3 tones read as genuinely different, not just reworded. `None` when the
    # ticket has no drafts to compare — there's nothing to judge as distinct.
    distinct: bool | None
    drafts: list[DraftVerdict] = []
    note: str = ""


class JudgeRun(BaseModel):
    """One judge pass over one reply-eval run."""

    model_config = ConfigDict(extra="forbid")

    run: str  # the run file's stem, e.g. "gpt-5.4-nano_v1_20260928T120000Z"
    judge_model: str
    rubric_version: str
    judged_at: str
    verdicts: list[TicketVerdict] = []


def validate_against_run(judge_run: JudgeRun, records: list[ReplyEvalRecord]) -> None:
    """Check that `judge_run`'s verdicts actually match `records` (the run it claims to
    judge), beyond what pydantic alone can check.

    Raises `ValueError` listing every problem found, not just the first, so a judge run
    with several mistakes can be fixed in one pass:
    - a verdict naming a ticket id that isn't in `records`
    - a record with no verdict at all
    - more than one verdict for the same ticket id
    - a ticket that got drafts (`record.drafts` non-empty) whose verdict doesn't have
      exactly one `DraftVerdict` per `Tone`, or whose `distinct` is `None`
    - a ticket with no drafts whose verdict has any `DraftVerdict`, or whose `distinct`
      isn't `None`
    """
    record_ids = {record.ticket_id for record in records}
    records_by_id = {record.ticket_id: record for record in records}
    verdict_ids = [verdict.ticket_id for verdict in judge_run.verdicts]

    problems: list[str] = []

    seen: set[str] = set()
    duplicates: set[str] = set()
    for ticket_id in verdict_ids:
        if ticket_id in seen:
            duplicates.add(ticket_id)
        seen.add(ticket_id)
    if duplicates:
        problems.append(f"duplicate verdicts for ticket id(s): {sorted(duplicates)}")

    unknown = sorted(set(verdict_ids) - record_ids)
    if unknown:
        problems.append(f"verdicts for unknown ticket id(s): {unknown}")

    missing = sorted(record_ids - set(verdict_ids))
    if missing:
        problems.append(f"missing verdict(s) for ticket id(s): {missing}")

    expected_tones = set(Tone)
    for verdict in judge_run.verdicts:
        record = records_by_id.get(verdict.ticket_id)
        if record is None:
            continue  # already reported as "unknown" above

        has_drafts = bool(record.drafts)
        draft_tones = [draft.tone for draft in verdict.drafts]
        if has_drafts:
            if set(draft_tones) != expected_tones or len(draft_tones) != len(expected_tones):
                problems.append(
                    f"{verdict.ticket_id}: expected exactly one DraftVerdict per tone "
                    f"{sorted(t.value for t in Tone)}, got {[t.value for t in draft_tones]}"
                )
            if verdict.distinct is None:
                problems.append(f"{verdict.ticket_id}: has drafts but distinct is null")
        else:
            if verdict.drafts:
                problems.append(f"{verdict.ticket_id}: has no drafts but verdict has draft(s)")
            if verdict.distinct is not None:
                problems.append(f"{verdict.ticket_id}: has no drafts but distinct isn't null")

    if problems:
        raise ValueError(
            f"Judge run {judge_run.run!r} doesn't match its records:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )


def mean_tone_score(verdicts: list[TicketVerdict], tone: Tone) -> float:
    """Mean `tone_score` across every draft verdict for `tone`. 0.0 when there are none."""
    scores = [
        draft.tone_score for verdict in verdicts for draft in verdict.drafts if draft.tone == tone
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def faithful_rate(verdicts: list[TicketVerdict]) -> float:
    """Fraction of all draft verdicts marked faithful. 0.0 when there are no drafts."""
    drafts = [draft for verdict in verdicts for draft in verdict.drafts]
    if not drafts:
        return 0.0
    return sum(draft.faithful for draft in drafts) / len(drafts)


def addresses_request_rate(verdicts: list[TicketVerdict]) -> float:
    """Fraction of all draft verdicts marked as addressing the customer's request. 0.0
    when there are no drafts."""
    drafts = [draft for verdict in verdicts for draft in verdict.drafts]
    if not drafts:
        return 0.0
    return sum(draft.addresses_request for draft in drafts) / len(drafts)


def summary_accurate_rate(verdicts: list[TicketVerdict]) -> float:
    """Fraction of all ticket verdicts (drafted or flagged) with an accurate summary. 0.0
    on an empty judge run."""
    if not verdicts:
        return 0.0
    return sum(verdict.summary_accurate for verdict in verdicts) / len(verdicts)


def distinct_rate(verdicts: list[TicketVerdict]) -> float:
    """Fraction of drafted tickets (`distinct is not None`) judged distinct. 0.0 when
    there are no drafted tickets."""
    drafted = [verdict for verdict in verdicts if verdict.distinct is not None]
    if not drafted:
        return 0.0
    return sum(verdict.distinct for verdict in drafted) / len(drafted)
