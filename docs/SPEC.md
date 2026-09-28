# Spec: MVP 1: AI Ticket Classifier

## Assumptions

1. This is a demo prototype, not production. It has no auth, no real ticket ingestion, and one user at a time.
2. All LLM calls go directly to the OpenAI API (no OpenRouter). The key comes from `OPENAI_API_KEY` (read from `.env` locally, `st.secrets` when deployed). The budget is about $10 of OpenAI credit.
3. It is deployed on Streamlit Community Cloud from GitHub `main`.
4. Model IDs and prices are checked against OpenAI's current model list and pricing during implementation.
5. The test set is synthetic and written by the author. No real customer data is used.
6. MVP 2 will later be added to the same app and reuse `core/`.

## Objective

Nebula support gets thousands of tickets a month, and all of them are routed by hand. The goal is a prototype AI classifier. It takes a ticket's text and returns a **category**, a **priority**, a **recommended next step** and a **needs-human-review flag**. It must also produce evidence of its quality, cost and robustness through an evaluation harness and a model comparison.

**Users:** support leads and the reviewer of the assignment. They paste a ticket into the UI and inspect the result and the evaluation results.

### User stories

- As a support lead, I paste a ticket and within seconds I see its category, priority, next step, confidence, and whether a human must review it and why.
- As a reviewer, I open the Eval page and see the test set table (input → expected → actual → pass/fail), accuracy, the model comparison (quality, latency, cost) and a cost forecast.
- As a support lead, I get a sensible result when a ticket is aggressive, mixes several topics or is not in English.

### Taxonomy

Final prompt: `v6` (`src/support_ai/classifier/prompts/v6.md`). v5 replaced the v1–v4 taxonomy from scratch; v6 keeps that taxonomy and moves priority into code. Older prompts stay in `prompts/` as change history but target earlier schemas and can't be run against the current one.

**Category and response.** Each ticket gets exactly one category, then exactly one response (`next_step`) from that category's own list. Within a category, responses are ordered from least to most serious; when several fit, the most serious wins.

| Category | Responses (least → most serious) |
|---|---|
| `general_question` — a question about Nebula or how the app works | `send_user_guide` (can't say what they want, or how the whole app works), `send_kb_answer` (any question about one specific action) |
| `quality_complaint` — a complaint about the app or the service | `generic_reply` (vague / general quality), `record_feature_request`, `create_bug_ticket` (specific bug), `escalate_human` (support ignored a specific earlier incident) |
| `expert_complaint` — a complaint about a Nebula expert | `generic_reply` (no specific expert), `record_expert_complaint` (service quality), `escalate_human` (unacceptable behavior incl. asking to pay outside Nebula, or a paid session that did not happen) |
| `payment_issue` — charges, subscriptions, refunds | `generic_reply` (vague, e.g. "too expensive"), `send_refund_policy` (refund demand), `escalate_human` (paid but not received, a cancellation threat, or charged because of an app bug) |
| `threat` | `generic_reply` (vague threat, e.g. "you'll regret this" — not violence), `escalate_human` (violence, self-harm, legal action, chargeback or bank dispute) |
| `other` | `no_reply` (unrelated to Nebula, or too short / too vague to infer meaning), `escalate_human` (Nebula-related but fits no category) |

Anything that can't be definitely placed in one of the five named categories goes to `other`.

**Mixed tickets:** pick the category and response with the highest priority; ties go to the category that comes first in `threat`, `payment_issue`, `expert_complaint`, `quality_complaint`, `general_question`, `other`. The category definitions themselves send a charge caused by an app bug to `payment_issue` and a paid expert session that did not happen to `expert_complaint`.

**Priority** is how fast a person must act, not how upset the customer is. The LLM does **not** set it: each (category, response) pair has a **base** priority, and code raises it **one level** only when the LLM quotes the ticket's statement of a listed fact. Tone, caps, "urgent" and future threats ("I'll sue") never raise it; actions already taken or scheduled do.

| Pair | Base | Raise to | Raise if the ticket states |
|---|---|---|---|
| `quality_complaint` / `create_bug_ticket` | P3 | P2 | a core function (only: log in, open the app, book/start an expert session, access paid content) is fully unusable, or data was lost |
| `quality_complaint` / `escalate_human` | P2 | — | — |
| `expert_complaint` / `record_expert_complaint` | P3 | — | — |
| `expert_complaint` / `escalate_human` | P2 | P1 | harassment, sexual content, threats or discrimination by the expert |
| `payment_issue` / `send_refund_policy` | P3 | P2 | duplicate, after-cancellation or unauthorized charge |
| `payment_issue` / `escalate_human` | P2 | P1 | paid and has no access at all to anything they paid for (one missing item doesn't count) |
| `threat` / `escalate_human` | P2 | P1 | violence or self-harm (always), or a legal/financial step already taken or scheduled (lawyer involved, filed or dated complaint, filed chargeback) |
| `other` / `escalate_human` | P3 | — | — |
| every other pair | P4 | — | — |

The same table lives in `rules.PRIORITY_TABLE`; `tests/test_prompts.py` parses the prompt's table and fails if the two drift apart. The LLM returns `priority_raise_evidence`: the shortest verbatim quote from the ticket (in its original language) that states the chosen pair's raise fact, or `null`. `rules.py` sets the raised priority only if the pair can be raised **and** the quote really appears in the ticket (case, whitespace and surrounding quote marks ignored); otherwise it uses the base and drops the quote. So the output carries evidence exactly when priority was raised. An invalid pair gets P3 (and is flagged).

Each response also comes with a one-line free-text `next_step_note`.

### Output schema

Validated with pydantic (`LLMClassification` uses `extra="forbid"`, so an unexpected extra key is a validation error, not a silently-dropped one). The shape is enforced through structured output (`responses.parse(text_format=LLMClassification)`), so the prompt carries no JSON block. Field order is deliberate: the model commits to `category` and `next_step` before looking for evidence against that pair's raise condition.

```json
{
  "category": "payment_issue",
  "next_step": "send_refund_policy",
  "priority_raise_evidence": "мене двічі списали кошти",
  "next_step_note": "Send the refund policy; the customer was charged twice.",
  "language": "uk",
  "tone": "aggressive",
  "confidence": 0.82,
  "rationale": "short explanation",
  "priority": "P2",
  "needs_human_review": false,
  "review_reasons": []
}
```

The LLM fills every field except `priority`, `needs_human_review` and `review_reasons`. Deterministic code in `rules.py` sets those, so neither the priority nor the escalation decision is delegated to the model (this is the D1.7 decision).

### Human-in-the-loop rules (flag only)

A ticket reaches a human only through the rules — never because the customer asked for a person. `needs_human_review = true` if any of these holds:

- the response is `escalate_human` — reason `escalate_human` (every P1 pair is an `escalate_human` pair)
- the (category, response) pair is not in `PRIORITY_TABLE` — reason `invalid_next_step`
- classification failed and the fallback result was used — reason `classification_failed`

`confidence` is informational only. "Let me talk to a real person" on its own is classified by its underlying issue. Empty or whitespace-only input is not a classification failure: it returns a deterministic `other` / `no_reply` / `P4` result with no LLM call and no flag.

The UI shows a red "Needs human review" badge with the reasons. Nothing is auto-routed.

## Tech Stack

- Python 3.12, managed with uv
- Streamlit (multipage app)
- `openai` Python SDK (OpenAI API directly)
- pydantic v2 for schemas and validation
- pandas for eval tables
- python-dotenv
- Dev tools: ruff, pytest, pytest-mock

**Models to compare**, in order from cheapest:

1. `gpt-5.4-nano`
2. `gpt-5.4-mini`
3. `gpt-5`

Development and prompt iteration start on `gpt-5.4-nano`. The default model in production config is the cheapest one that meets the accuracy targets.

## Commands

```
Install:     uv sync
Dev app:     uv run streamlit run streamlit_app.py
Tests:       uv run pytest -q
Lint:        uv run ruff check . --fix
Format:      uv run ruff format .
Eval (1):    uv run python scripts/run_eval.py --model gpt-5.4-nano --prompt v6
Export deps: uv export --no-hashes > requirements.txt   # for Streamlit Cloud, if needed
```

## Project Structure

```
streamlit_app.py              → entry point / home page
pages/
  1_Classifier.py             → paste ticket → result + review badge
  2_Eval.py                   → reads results/, shows tables, accuracy, model comparison, cost
src/support_ai/
  core/                       → shared with MVP 2
    config.py                 → settings, model registry (price per 1M tokens, timeout), default chain + prompt
    cost.py                   → token usage → $ per call, forecasts
    llm/
      base.py                 → LLMProvider protocol, LLMResult, Usage
      errors.py               → normalized error hierarchy
      gateway.py              → retry, repair-retry and model fallback (provider-agnostic)
      openai_provider.py      → the only module that imports `openai`
  classifier/
    schema.py                 → pydantic models + enums
    prompts/v1.md … v6.md     → versioned prompts; v6 is current, older ones are change history
    classify.py               → build prompt → call LLM → validate → apply rules
    rules.py                  → priority table + evidence check, deterministic HITL rules
  eval/
    metrics.py                → pure accuracy/latency/cost metric functions
scripts/
  classify_one.py             → classify one ticket from the command line
  run_eval.py                 → runs the test set with one model, writes results/
data/tickets.jsonl            → 45 synthetic tickets with expected category, next step, priority
results/                      → eval runs (<model>_<prompt>_<date>.json + summary.csv), committed
tests/                        → unit tests (LLM mocked) + Streamlit page smoke tests
docs/
  REQUIREMENTS_MVP1.md, REQUIREMENTS_MVP2.md, SPEC.md
README.md                     → the write-up: D1.1–D1.7 and X1–X5
```

## Code Style

Code is typed, uses small pure functions, keeps I/O at the edges, uses pydantic at the boundaries, and puts no LLM calls in UI files.

```python
class Classification(LLMClassification):
    """The full classification, with priority and human review applied by `rules.py`."""

    priority: Priority
    needs_human_review: bool = False
    review_reasons: list[str] = []


def classify(ticket: str, *, model_chain: list[str] | None = None, ...) -> ClassifyResult:
    """Classify one ticket. Never raises; on failure returns a fallback flagged for review."""
    ...
    gateway_result = complete_with_fallback(chain, system=system, user=user, schema=LLMClassification)
    llm_classification = LLMClassification.model_validate(gateway_result.result.data)
    classification = apply_rules(llm_classification, ticket=ticket)
```

- ruff defaults plus isort. Line length 100.
- snake_case for modules and functions, PascalCase for models, UPPER_CASE for constants.
- Prompts live in `.md` files, never inline strings.
- Comments only where the reason is not obvious from the code.

## Failure Handling (X3)

- **Invalid JSON or schema mismatch:** structured output (`responses.parse` with the pydantic schema) makes this rare. If validation still fails, make one repair retry that sends the validation error back to the model. If that fails too, move to the next model.
- **Timeout:** per model in `config.py` (nano 20 s, mini 25 s, gpt-5 45 s with `reasoning_effort=low`). One retry, then the next model.
- **Rate limit or 5xx:** exponential backoff (1 s, 2 s), at most 2 retries, then the next model in the fallback chain. SDK retries are off (`max_retries=0`); the gateway owns every retry.
- **All models fail, or any unexpected error (e.g. no API key):** return a fallback result (`category=other`, `next_step=escalate_human`, `priority=P3`, `confidence=0`, `needs_human_review=true`, reasons `escalate_human` + `classification_failed`). `classify()` never raises and the UI never crashes. Every result carries an `attempts` log (model, outcome, detail).
- **Empty or whitespace-only input:** not a failure. Return a deterministic `category=other`, `next_step=no_reply`, `priority=P4` result with no LLM call and `needs_human_review=false`.

## Caching (X5)

No app-level cache is built. Ticket texts are almost always unique free text, so an exact-match
cache would rarely hit. A semantic (similarity-based) cache could catch near-duplicates, but risks
serving a wrong label for a ticket that only looks similar to a cached one, for a small saving
given real-world ticket volume and variety. That risk-to-benefit ratio is not worth the added
code. The static system prompt is still sent first in each request so that OpenAI's provider-side
prompt caching can apply automatically; this needs no application code.

Provider-side caching is **measured**, not just assumed. The provider adapter records
`cached_input_tokens` (a subset of `input_tokens`). `cost.py` bills these at the model's
`cached_input_price_per_1m` (about 10% of the full input price for the OpenAI models), so the cost
per ticket and the 10k forecast reflect the discount. The eval records a `cache_hit_rate` (cached
input tokens ÷ all input tokens) per run, and both pages show cached tokens.

## Testing Strategy

- **Unit tests (pytest, no network):**
  - schema validation
  - `rules.py`: priority for every table pair, evidence verification, and each HITL rule
  - prompt ↔ code contract: the current prompt's priority table equals `rules.PRIORITY_TABLE`
  - `cost.py` math, including cached-token pricing
  - `core/llm/gateway.py` retry, repair-retry and fallback logic, using a fake provider that simulates invalid output, timeouts and 429s; `openai_provider.py` error mapping with a mocked SDK client
  - `classify()` normal, fallback and empty-ticket paths
  - eval metrics and the `summary.csv` writer
  - Streamlit pages render without exceptions (`AppTest`)
- **Eval (LLM, run manually, not in pytest):**
  - `scripts/run_eval.py` over `data/tickets.jsonl`
  - per-ticket pass/fail on category, response (next step) and priority, plus the priority evidence
  - category / response / priority accuracy, human-review recall and precision, latency p50/p95, cost per ticket and per 10k, cache hit rate
- **Test set:**
  - one ticket per (category, response) pair at base priority, and a raised ticket for every raisable pair
  - one ticket per escalation trigger (violence, self-harm, legal action, chargeback filed or threatened, …)
  - edge cases: non-English (uk, es, fr), mixed categories, prompt injection, too short, aggressive tone, "I want a real person"
  - every label is checked in pytest against `PRIORITY_TABLE`
- **Manual test:** paste the edge-case tickets into the Classifier page, locally and on the deployed app.

## Boundaries

- **Always:**
  - validate LLM output with pydantic
  - version prompts as new files; never edit old ones
  - log tokens and cost for every call
  - run `ruff` and `pytest` before commits
  - keep secrets in `.env` or `st.secrets`
- **Ask first:**
  - adding dependencies beyond the Tech Stack list
  - changing the taxonomy or HITL rules
  - adding paid models beyond the 3 listed
  - any eval run expected to cost more than $1
- **Never:**
  - commit API keys
  - let the LLM decide `needs_human_review` or `priority`
  - use real customer data
  - call the network in unit tests
  - delete old prompt versions, or eval results of the current taxonomy (pre-v5 runs were cleared deliberately when the taxonomy was replaced, since they are not comparable)

## Success Criteria

- [ ] The app is deployed on Streamlit Community Cloud. Pasting a ticket returns a valid result in under 10 s with the cheapest passing model.
- [x] Every edge-case ticket returns a valid schema. None crash. (Final v6 runs: 0 errors across 45 tickets × 3 models.)
- [x] The eval shows **category accuracy ≥ 85%** and **priority accuracy ≥ 75%** for the chosen default model. (Final v6, `gpt-5.4-nano`: 95.6% / 75.6%.)
- [ ] 100% of tickets expected to need review are flagged (HITL recall = 1.0 on the test set), and human-review precision is tracked alongside it so escalations stay both complete and correct. (Final v6: **not met by the default `gpt-5.4-nano`** — recall 14/17 (82%), precision 14/15; met by `gpt-5.4-mini` (17/17, precision 17/17) and `gpt-5` (17/17, precision 17/18).)
- [x] The model comparison table covers all 3 models: accuracy, p50 latency, cost per ticket, cost per 10k tickets. (`results/summary.csv`, final v6 rows.)
- [x] The failure paths (invalid JSON, timeout, 429, all models down) are covered by unit tests and all pass.
- [ ] At least 3 prompt versions are committed (done: v1–v6), and README explains the changes.
- [ ] README covers D1.1–D1.7 and X1–X5.

## Decisions

- The accuracy targets are confirmed: category ≥ 85%, priority ≥ 75%.
- The UI has no model or prompt picker. After the eval, one model and one prompt version are fixed in `config.py`: prompt `v6`, default chain `gpt-5.4-nano` → `gpt-5.4-mini` → `gpt-5`. Nano is the cheapest model and meets the accuracy targets, but not HITL recall (82%); mini meets every target (priority 93.3%, recall 100%) at about 3.7× the cost ($11.71 vs $3.20 per 10k), and `gpt-5` scores highest (priority 95.6%) at about 14× the cost and 3.6× the p50 latency. Model comparison happens only through `scripts/run_eval.py` and the Eval page.
- Priority is computed by code from a quoted piece of evidence (v6), not chosen by the LLM. A variant where the LLM also names the raise fact from a fixed list (an unreleased v7) scored worse and was dropped.
- Some v6 prompt examples closely resemble test tickets (t002, t005, t006, t035), so v6 scores on those tickets are likely somewhat optimistic.
- The project uses OpenAI only; OpenRouter was dropped. Prompt iteration starts on `gpt-5.4-nano`. The fallback chain is nano → mini → gpt-5, all from the same provider, so an OpenAI-wide outage leads to the fallback result.

## Open Questions

None.
