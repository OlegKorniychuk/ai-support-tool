# Spec: MVP 2: Agent Reply Assistant

## Assumptions

1. Demo prototype, same as MVP 1: no auth, no real ticket ingestion, one user at a time.
2. Same app, stack and deployment as MVP 1: Streamlit multipage app on Streamlit Community
   Cloud from GitHub `main`; OpenAI only for the app; key from `OPENAI_API_KEY`.
3. **Every ticket has already been through MVP 1 and is a `general_question`.** MVP 2 does
   not classify or prioritize; FR-5 is covered upstream by MVP 1. MVP 1 code, prompts and
   data are not changed (except moving two shared helpers into `core/`, see structure).
4. The knowledge base is synthetic, written for this project, English only. It describes an
   invented Nebula-like astrology app with expert consultations; it is not real Nebula policy.
5. Model ids and prices (incl. `text-embedding-3-small`) are re-checked against OpenAI's
   pricing pages during implementation.
6. OpenAI budget for MVP 2 is $5–10. The LLM judge runs in Claude Code (Sonnet), so it
   spends no OpenAI credit.

## Objective

Support agents spend much of their time searching the KB and writing replies to general
questions. MVP 2 is an internal assistant: the agent pastes a ticket and gets a short
summary, three ready-to-edit reply drafts in distinct tones, the KB article + verified quote
behind them, and a clear mark when the agent must decide alone. The assistant suggests; the
agent decides.

**Users:** internal support agents; the assignment reviewer (Reply Eval page).

### User stories

- As an agent, I paste a general question and within seconds see a 1–2 sentence summary,
  the KB article with the quote that backs the reply, and three drafts (formal, empathetic,
  short) I can edit.
- As an agent, when the KB has no answer, the answer depends on the customer's own account,
  or KB articles contradict each other, I see a red "Decide yourself" banner with reasons
  and no AI drafts.
- As a reviewer, I open Reply Eval and see per-ticket results (retrieval, grounding, flags,
  tone checks, judge scores), the 3-model comparison and cost.

### Answers to the requirements' open questions

| Question | Answer |
|---|---|
| Category and priority values | Out of scope: set upstream by MVP 1; all MVP 2 tickets are `general_question` |
| KB content and size | ~20 synthetic English articles in `data/kb/` (list below) |
| When 2 variants instead of 3 | Never. Drafts are shown as exactly 3 or not at all (0 when flagged) |
| Test set size | 30 new general-question tickets in `data/reply_tickets.jsonl` |

The requirements' Evaluation row asks for expected category and priority; for MVP 2 the
expected labels are instead the KB article(s) and the human-judgment flag + reasons.

### Pipeline

```
ticket → retrieve: KnowledgeBase.search(ticket, k=KB_TOP_K) → top-3 hits + has_match
       → pre-generation rules (code): retrieval failed or has_match is false?
       → generate: "full" (summary, KB citation, judgment evidence, 3 drafts)
                   or "summary_only" (summary) when already flagged
       → post-generation rules (code): verify quote, evidence, conflict ids; hide drafts if flagged
       → AssistResult
```

`assist()` never raises. Empty/whitespace input → deterministic result, no LLM call, no
flag, no drafts.

### Retrieval (D2.1)

- Retrieval lives behind a swappable interface, `src/support_ai/kb/`, not inside
  `assistant/`. `kb/base.py` defines the data types — `Article(id, title, text, tags)`,
  `KBHit(article, score)`, `SearchResult(hits, has_match, latency_ms, cost_usd,
  usage | None, attempts)`, `RetrievalError` — and a `KnowledgeBase` protocol with one
  method, `search(query, *, k) -> SearchResult`. The interface is search-only; each backend
  owns its own ingestion. A registry (`register_knowledge_base(name, factory)` /
  `get_knowledge_base()`, keyed by `KB_BACKEND` in `config.py`, lazy and memoized — the same
  pattern as `core/llm/base.py`'s provider registry) picks the active backend. `assistant/`
  only imports `kb/base.py` and calls `get_knowledge_base().search(...)`.
- `kb/loader.py`: `data/kb/<id>.md`, a `key: value` header (`id`, `title`, `tags`) above
  `---`, then the body. Parsed by a small own function (no YAML dependency).
- `kb/in_memory.py` is the default adapter (`KB_BACKEND=in_memory`): builds an index from
  `kb/loader.py` articles (title + body) embedded with `text-embedding-3-small`, cached on
  disk in `.cache/kb_embeddings.json` (key = sha256(article text) + embedding model id).
  Ticket embedded per request. Cosine similarity in pure Python (20 vectors; no numpy
  dependency). Returns the top-`KB_TOP_K` hits with scores. Swapping to a real vector DB is a
  new `kb/<backend>.py` implementing the same protocol, its own offline ingestion script, and
  a `KB_BACKEND` change — nothing in `assistant/` changes.
- Why embeddings: non-English tickets match English articles; retrieval is a separate,
  measurable step (hit@3); scales past what fits in a prompt. Rejected: whole KB in prompt
  (doesn't scale, nothing to evaluate), keyword/BM25 (fails across languages), fixed
  topic → article map (brittle for varied how-to questions).
- `has_match` is decided by the adapter, not the pipeline, because score scales differ per
  backend. For `in_memory`, `KB_MIN_SCORE` is the adapter's own setting: best score below it
  → `has_match = false`. Calibrated on the test set with `scripts/kb_scores.py` and recorded
  in `config.py`.

### Knowledge base (~20 articles)

Getting started (user guide) · Edit birth date/time/place · Login and password reset ·
App won't open / crashes · Reporting a bug · Feature requests and feedback · Notifications ·
Subscription plans · Cancel a subscription · Refund policy · Duplicate or unauthorized
charge · Payment methods · Booking an expert session · Missed or cancelled expert session ·
Expert code of conduct · Rating an expert · How readings are personalized · Privacy and data
use · Delete account and data · Support hours and response times.

- Near-duplicates on purpose (cancel vs refund vs duplicate charge; login vs app won't open).
- Two planted contradictions, as in a drifting real KB (e.g. cancellation deadline stated
  differently in "Subscription plans" and "Cancel a subscription") for `conflicting_kb`.
- Deliberate gaps (promo codes, gift subscriptions, family sharing) for `kb_not_found`.

### Reply generation (D2.2)

One structured-output call. Static system prompt (versioned `.md`, cacheable); the user
message carries the wrapped ticket (`<<<TICKET>>>` markers, as MVP 1) and the top-3 articles
in `<<<KB id=…>>>` markers.

```json
{
  "summary": "Customer asks how to change the birth time in their profile.",
  "kb_article_id": "edit-birth-data",
  "kb_quote": "Open Profile → Birth details and tap the time field",
  "account_specific_evidence": null,
  "conflicting_article_ids": [],
  "formal": "Dear [Customer name], ...",
  "empathetic": "I understand how important accurate birth details are ...",
  "short": "Hi [Customer name], open Profile → Birth details ..."
}
```

- `LLMReply` (`extra="forbid"`). Field order is deliberate: the model commits to the source,
  quote and judgment evidence before writing drafts.
  - `kb_article_id` / `kb_quote`: null when no retrieved article answers the question.
  - `account_specific_evidence`: shortest verbatim quote from the ticket showing the answer
    depends on this customer's own data (their charge, renewal date, session history), or
    null.
  - `conflicting_article_ids`: ids of retrieved articles that give contradictory answers to
    this question, or empty.
  - Drafts are always written in full mode; code decides whether they are shown.
- `LLMSummary` = `summary` only (summary-only mode).
- Tones (all English, same facts, all facts from the cited article, no promises the KB
  doesn't make; `[Customer name]` / `[Agent name]` placeholders):
  - **formal** — greeting and sign-off, complete sentences, no contractions, slang or emoji.
  - **empathetic** — opens by naming the customer's specific situation and feeling; warm.
  - **short** — at most `SHORT_MAX_WORDS` (50) words: the answer plus one next step.
- Model chain: starts `gpt-5.4-nano` → `gpt-5.4-mini` → `gpt-5`; the default becomes the
  cheapest model that meets the targets after the eval (as in MVP 1).

### Human-judgment rules (D2.5, D2.6)

Set by code in `assistant/rules.py`. The LLM only reports facts (article, quote, evidence,
article ids); code verifies them and decides. `needs_human_judgment = true` if any reason
applies; then **no drafts are shown**.

| Reason | When (checked by code) | Why the agent must decide |
|---|---|---|
| `kb_not_found` | adapter reports no match (`has_match = false`), or model cited no article | a reply would have to invent policy |
| `kb_quote_unverified` | quote not found (normalized) in the cited article, or cited article not among the retrieved | the draft may rest on a made-up fact |
| `account_specific` | `account_specific_evidence` is found (normalized) in the ticket | the tool can't see the customer's account; only the agent can check it |
| `conflicting_kb` | ≥ 2 distinct `conflicting_article_ids`, all among the retrieved | the KB contradicts itself; the agent picks the right answer and reports the conflict |
| `retrieval_failed` | embedding call failed after retries | no KB grounding |
| `generation_failed` | reply call failed on every model | no drafts to offer |

Evidence not found in the ticket, or conflict ids not among the retrieved, are dropped and
logged in the result (as MVP 1 drops unverified priority evidence); the eval counts them.

**UI signal:** red `st.error` banner "Decide yourself — no AI drafts" with the reasons (and
the verified evidence quote / conflicting article titles); the drafts area is replaced by
that banner; summary and KB candidates are still shown. The result carries
`needs_human_judgment` + `judgment_reasons` so a future queue could route on them. Nothing
is auto-sent.

### Output (`AssistResult`)

`summary`, `kb_source` (id, title, verified quote, full text) or null, `kb_candidates`
(top-3 with scores), `drafts` (3 × `{tone, text}` or empty), `needs_human_judgment`,
`judgment_reasons`, `account_specific_evidence`, `conflicting_article_ids`,
`dropped_evidence`, and per-step metadata (`step`, `model_used`, `latency_ms`, `usage`,
`cost_usd`, `attempts`) plus totals.

## Tech Stack

Same as MVP 1: Python 3.12 / uv, Streamlit, `openai` SDK, pydantic v2, pandas,
python-dotenv; ruff, pytest, pytest-mock. **No new dependencies** (pure-Python cosine, own
header parser, `hashlib`).

Models: `gpt-5.4-nano` / `gpt-5.4-mini` / `gpt-5` (reply, compared); `text-embedding-3-small`
(retrieval); Claude Sonnet via Claude Code skill (judge, eval only, outside the app).

## Commands

```
Install:      uv sync
Dev app:      uv run streamlit run streamlit_app.py
Tests (core): uv run pytest tests/core tests/mvp2 -q
Tests (mvp1): uv run pytest tests/mvp1 -q
Tests (all):  uv run pytest -q
Lint:         uv run ruff check . --fix
Format:       uv run ruff format .
Assist one:   uv run python scripts/assist_one.py "How do I change my birth time?"
Reply eval:   uv run python scripts/run_reply_eval.py --model gpt-5.4-nano [--limit 5]
Judge a run:  /judge-replies results/reply/<run>.json          (in Claude Code)
Record judge: uv run python scripts/record_judgement.py results/reply/<run>.judge.json
KB threshold: uv run python scripts/kb_scores.py
Judge agree:  uv run python scripts/judge_agreement.py
```

Tests are split into three suites so MVP 1's own tests don't have to run for every MVP 2
change: `tests/core/` (shared `core/` modules plus the shared home page), `tests/mvp1/`
(classifier + its eval/scripts/pages), `tests/mvp2/` (this MVP's own tests). Every MVP 2
task runs `Tests (core)`. `Tests (mvp1)` is run once for a task only when that task touches
an MVP 1-owned path (`src/support_ai/classifier/`, `src/support_ai/eval/metrics.py`,
`scripts/run_eval.py`, `scripts/classify_one.py`, `pages/1_*`, `pages/2_*`,
`data/tickets.jsonl`) or changes a shared `core/` module's contract that MVP 1 depends on,
plus at checkpoints. `Tests (all)` (`uv run pytest -q`) always runs the full suite and is
what CI-equivalent verification before a commit ultimately reports.

## Project Structure

New and changed paths (everything else as in `docs/SPEC.md`):

```
streamlit_app.py              → home page lists both tools (changed)
pages/
  3_Reply_Assistant.py        → paste ticket → summary, KB source, 3 editable drafts or banner
  4_Reply_Eval.py             → reads results/reply/: per-ticket table, judge columns,
                                 model comparison, cost
src/support_ai/
  core/
    config.py                 → + EMBEDDING_MODEL, DEFAULT_REPLY_CHAIN,
                                 DEFAULT_REPLY_PROMPT_VERSION, KB_BACKEND, KB_TOP_K,
                                 KB_MIN_SCORE (in-memory adapter setting), SHORT_MAX_WORDS
                                 (changed)
    text.py                   → wrap_ticket, normalize, quote_in_text — moved from
                                 classifier (classifier imports them; behavior unchanged)
    llm/base.py               → + `embed()` on LLMProvider, EmbeddingResult (changed)
    llm/gateway.py            → + embed_with_retry (same retry budgets, no model fallback) (changed)
    llm/openai_provider.py    → + embeddings call + error mapping (changed)
  kb/                         → swappable retrieval package (new); assistant/ imports only
                                 kb/base.py
    base.py                   → Article, KBHit, SearchResult, RetrievalError, KnowledgeBase
                                 protocol; register_knowledge_base/get_knowledge_base
                                 registry keyed by KB_BACKEND
    loader.py                 → parses data/kb/*.md headers + body
    in_memory.py              → default adapter: embeds via the gateway, KB_MIN_SCORE
                                 threshold, .cache/kb_embeddings.json cache, pure-Python
                                 cosine
  assistant/
    schema.py                 → LLMReply, LLMSummary, Draft, KBSource, AssistResult
    rules.py                  → human-judgment rules, evidence/quote/conflict verification
    assist.py                 → the pipeline; never raises
    tone_checks.py            → deterministic draft checks (word count, contractions, overlap)
    dataset.py                → ReplyCase loader for data/reply_tickets.jsonl
    prompts/reply_v1.md, summary_v1.md
  eval/
    metrics.py                → `_percentile` becomes public `percentile` (changed)
    reply_metrics.py          → pure metric functions incl. judge aggregates
    summary_csv.py            → shared summary.csv read/append helpers, used by both eval
                                 scripts
    judge.py                  → JudgeVerdict schema (judge output)
scripts/
  assist_one.py, run_reply_eval.py, record_judgement.py, kb_scores.py, judge_agreement.py
.claude/skills/judge-replies/
  SKILL.md                    → committed repo skill (project-scoped, versioned in git,
                                 `model: sonnet`): read run → write <run>.judge.json → run
                                 recorder
  rubric.md                   → judge rubric (tone definitions shared with reply_v1.md)
data/
  kb/*.md                     → ~20 synthetic articles
  reply_tickets.jsonl         → 30 general-question tickets
.cache/                       → KB embedding cache (gitignored)
results/reply/                → runs, *.judge.json, summary.csv, judge_summary.csv
                                 (append-only, joined to summary.csv by run name),
                                 hand_scores.csv (committed)
tests/
  fakes.py                    → shared scripted FakeProvider, stays at tests/ root
                                 (importable from every suite via pytest `pythonpath`)
  core/                        → shared core/ modules (incl. embeddings) + the shared
                                 home page's smoke test (test_home_page.py)
  mvp1/                        → classifier + its eval/scripts/pages (moved from tests/,
                                 unchanged behavior; see docs/SPEC.md)
  mvp2/                        → this MVP's own tests (new)
docs/SPEC_MVP2.md             → this spec
README.md                     → + MVP 2 part (Ukrainian): D2.1–D2.6, X1–X5 for MVP 2
```

## Code Style

As MVP 1: typed, small pure functions, I/O at the edges, pydantic at boundaries, prompts in
`.md` files, no LLM calls in pages, ruff (line length 100).

```python
def assist(ticket: str, *, reply_chain: list[str] | None = None, ...) -> AssistResult:
    """Summarize, ground and draft replies for one general question. Never raises."""
    if not ticket.strip():
        return _empty_result()
    retrieval = get_knowledge_base().search(ticket, k=KB_TOP_K)
    reasons = pre_generation_reasons(retrieval)
    mode = "summary_only" if reasons else "full"
    reply = _generate(ticket, retrieval, mode=mode, chain=reply_chain)
    return finalize(ticket, retrieval, reply, reasons)  # rules.py hides drafts if flagged
```

## Failure Handling (X3)

- **Invalid JSON / schema mismatch:** structured output; one repair retry; then next model;
  then `generation_failed`.
- **Timeout, 429, 5xx:** MVP 1 gateway budgets (1 timeout retry; backoff 1 s, 2 s, max 2),
  then next model. Embeddings: same budgets, no fallback model → `retrieval_failed`.
- **Retrieval fails:** summary-only call without KB; flagged; no KB source.
- **Generation fails on every model:** KB candidates still shown; summary "unavailable";
  flagged.
- **KB index can't be built** (e.g. no key): treated as `retrieval_failed`; retried on the
  next request.
- **Empty input:** deterministic, no calls, not flagged.
- Every step records its `attempts`; the page shows a notice when a fallback model was used.

## Caching (X5)

- **KB embeddings: cached.** Key = sha256(article text) + embedding model id. Stored in
  `.cache/kb_embeddings.json` and memoized in process, so a Streamlit container embeds the
  KB at most once. Invalidation: an article whose text changed, or a different embedding
  model, is re-embedded; entries for deleted articles are dropped on rebuild.
- **Provider prompt caching:** static system prompts first, variable content in the user
  message; `cached_input_tokens` and `cache_hit_rate` measured as in MVP 1.
- **Not cached:** ticket embeddings and LLM responses — tickets are unique free text, and a
  cached draft could repeat outdated policy after a KB edit.

## Cost (X4)

Cost per ticket = ticket embedding + reply (or summary-only) call. Reply Eval shows cost
per step, per ticket and per 10k tickets. Levers documented: cheapest passing reply model;
summary-only mode when retrieval already fails; static cacheable prompts; KB embedded once;
judging on Claude instead of OpenAI.

## Testing Strategy

Every new MVP 2 test file lives under `tests/mvp2/`, except tests of a shared `core/`
module (e.g. embeddings on the provider/gateway), which live under `tests/core/` alongside
the classifier's `core/` tests — see Commands for which suite each task must run.

- **Unit (pytest, no network; `FakeProvider` gains a scripted `embed()`):**
  - KB loader and header parser; cosine / top-k; embedding cache hit, invalidation on text
    or model change, pruning
  - every human-judgment reason; quote, evidence and conflict-id verification; dropped
    evidence; drafts hidden whenever flagged
  - `assist()`: normal, kb gap (pre and post generation), quote unverified, account
    specific, conflicting KB, retrieval failed, generation failed, empty input
  - tone checks: word count, contractions, pairwise overlap
  - prompt ↔ code contract: `SHORT_MAX_WORDS` and tone names match in `reply_v1.md`,
    skill `rubric.md` and config/schema
  - dataset labels: every `expected_kb_ids` exists in `data/kb/`; reasons are valid names
  - reply metrics, `eval/summary_csv.py` writer/reader shared by both eval scripts,
    `record_judgement.py` validation (unknown ticket ids, missing tickets, bad scores
    rejected) and its `judge_summary.csv` append
  - embeddings error mapping in `openai_provider.py` (mocked SDK) — `tests/core/`
  - pages 3 and 4 render (`AppTest`)
- **Reply eval (OpenAI, manual):** `scripts/run_reply_eval.py --model X` over 30 tickets,
  one model, no fallback. Per ticket: retrieved ids vs `expected_kb_ids`, cited id, quote
  verified, expected vs actual flag and reasons, drafts shown, deterministic tone checks,
  latency, tokens, cost. Writes `results/reply/<model>_<prompt>_<ts>.json` + a
  `summary.csv` row.
- **Judge (Claude Code skill `/judge-replies`, Sonnet, no OpenAI tokens):**
  - The skill is a committed repo file, `.claude/skills/judge-replies/{SKILL.md,rubric.md}`
    (project-scoped, versioned in git like prompts, `model: sonnet` in frontmatter) — not a
    user-level skill. Anyone who opens the repo in Claude Code gets `/judge-replies`.
  - Runs on Sonnet (skill frontmatter `model`; if unsupported, the skill delegates to a
    Sonnet subagent — confirmed at implementation).
  - Reads the run file and cited KB articles; per drafted ticket scores each draft:
    `tone_score` 1–5, `faithful` (no claim outside ticket + cited article),
    `addresses_request`; per ticket `distinct` (tones differ, not just wording); for every
    ticket `summary_accurate`.
  - Writes `results/reply/<run>.judge.json` (`JudgeVerdict` schema, defined in
    `src/support_ai/eval/judge.py`, with judge model and rubric version), then runs
    `record_judgement.py`, which validates it with pydantic and **appends** a row to
    `results/reply/judge_summary.csv` (joined to that run's `summary.csv` row by run name) —
    judge results are append-only, like the rest of the eval results; `summary.csv` rows are
    never rewritten.
  - Calibrated against the author's hand scores on 10 tickets (30 drafts), scored in chat
    with Claude and recorded to `results/reply/hand_scores.csv`; `scripts/judge_agreement.py`
    computes agreement and the number goes in README. A different vendor judging OpenAI
    drafts avoids self-preference bias.
- **Test set (`data/reply_tickets.jsonl`, 30):** ~20 answerable how-to questions (incl.
  near-duplicate topics and 3 non-English), 3 KB gaps, 3 account-specific, 3 conflicting-KB,
  1 prompt injection. Each has `expected_kb_ids` (empty = gap), `expected_needs_judgment`,
  `expected_reasons`, `tags`.
- **Manual:** paste edge cases (gap, account-specific, conflict, non-English, injection)
  locally and on the deployed app.

## Boundaries

- **Always:** validate LLM output (and judge output) with pydantic; verify quotes, evidence
  and ids in code; version prompts as new files; log tokens and cost per step; run `ruff`
  and `pytest` before commits; secrets in `.env` / `st.secrets`.
- **Ask first:** new dependencies; any change to MVP 1 behavior, prompts or data; changing
  the human-judgment reasons; adding OpenAI models beyond the listed ones; any eval run
  expected to cost > $1 of OpenAI credit.
- **Never:** show drafts when `needs_human_judgment`; let the LLM set the flag directly;
  use OpenAI models for judging; present KB content as real Nebula policy; use real
  customer data; call the network in unit tests; delete old prompt versions or eval results.

## Success Criteria

- [ ] Reply Assistant page deployed; a pasted ticket returns in < 15 s; eval p50 ≤ 6 s with
      the default model.
- [ ] 0 crashes / schema failures across the 30-ticket set for every reply model.
- [ ] Retrieval hit@3 ≥ 90% and cited article ∈ `expected_kb_ids` ≥ 85% (answerable tickets).
- [ ] Human-judgment recall = 100%, precision ≥ 85%; reason match reported.
- [ ] Whenever drafts are shown: exactly 3, quote verified; short ≤ 50 words and formal
      contraction-free in ≥ 95% of tickets.
- [ ] Judge (Sonnet): mean `tone_score` ≥ 4.0 per tone; `faithful` ≥ 95% of drafts;
      `addresses_request` ≥ 90%; `summary_accurate` ≥ 95%.
- [ ] Reply model comparison (nano / mini / gpt-5): quality, p50/p95 latency, cost per
      ticket and per 10k; default chosen and justified.
- [ ] Judge agreement with hand scores reported.
- [ ] README MVP 2 part covers D2.1–D2.6 and X1–X5 for MVP 2 (incl. one bad-output example
      and what failed first on tones).

## Open Questions

None.
