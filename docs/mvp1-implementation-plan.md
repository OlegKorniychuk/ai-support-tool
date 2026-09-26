# Implementation Plan: MVP 1 (Ticket Classifier Pipeline)

## Context

`SPEC.md` defines MVP 1 (Python + Streamlit, OpenAI only). The goal of this plan is a **working pipeline**: paste a ticket and get a valid, schema-checked classification, at whatever accuracy. The AI provider must be swappable without touching app logic.

The repo currently holds only `REQUIREMENTS.md`, `SPEC.md` and `.gitignore`, so every file is new.

**In scope:**

- a layer that isolates the AI provider (OpenAI adapter)
- retries and fallback to another model
- the fixed human-review rules
- prompt v1
- the Classifier page
- the full synthetic test set (15+ tickets)
- the eval runner and Eval page, checked on 3–5 tickets only
- cost tracking

**Out of scope (later):**

- prompt refinement (v2+)
- the full eval run and model comparison
- deploy
- the docs

**Dropped:** result caching. Ticket texts are almost always unique, so the cache would rarely hit. This is the X5 answer: caching is argued against, not built.

## Architecture Decisions

- **Provider isolation.** Only `core/llm/openai_provider.py` imports `openai`. Everything else depends on an `LLMProvider` protocol and on the error types in `core/llm/errors.py`. Switching provider means adding one adapter file, registering it, and changing config. `classifier/` and `pages/` stay untouched.

  ```python
  class LLMProvider(Protocol):
      name: str
      def complete_structured(self, *, system: str, user: str, schema: type[BaseModel],
                              model: str, timeout_s: float) -> LLMResult: ...
  # LLMResult(data: dict, usage: Usage(input_tokens, output_tokens), model: str, latency_ms: int)
  # Errors: LLMTimeout, LLMRateLimited, LLMInvalidOutput, LLMProviderError (all subclass LLMError)
  ```

- **Retry and fallback live in `core/llm/gateway.py`, independent of the provider.** They work only with the normalized errors and a configured model chain, so they behave the same for any provider.
- **Model registry in `core/config.py`.** Each entry holds the provider name, model id, price in and out per 1M tokens, and a timeout per model (gpt-5 is slower). One module-level constant sets the default chain: nano → mini → gpt-5.
- **Fake provider in tests (`tests/fakes.py`).** It is scripted with responses or errors, so all pipeline tests run without network. The real OpenAI adapter is checked by one manual smoke call.
- **The human-review flag is deterministic code** (`classifier/rules.py`). The LLM output schema excludes `needs_human_review` and `review_reasons`.

## Dependency Graph

```
scaffold ─┬─ schema ────────────────┬─ rules ─┐
          └─ llm types/errors/registry/cost ─┬─ openai adapter
                                             └─ gateway ──────┴─ classify (+ prompt v1) ─┬─ Classifier page
                                                                                         └─ run_eval ─ Eval page
test set (tickets.jsonl) ────────────────────────────────────────────────────────────────┘
```

## Task List

### Phase 1: Foundation

#### Task 0: Update SPEC for no caching

**Description:** In `SPEC.md`:

- Replace the Caching (X5) section with a short rationale for not caching: tickets are unique free text, so exact-match hits are near zero, and semantic caching risks wrong labels for a small saving. Provider-side prompt caching of the static system prompt still applies automatically; no app code is needed for it.
- Remove `cache.py` from the project structure.
- Remove "cache key and invalidation" from the testing strategy.

**Acceptance:**

- [ ] SPEC.md has no references to building a cache.

**Verify:** `grep -n -i cache SPEC.md` shows only the rationale.
**Files:** `SPEC.md`
**Scope:** XS

#### Task 1: Project scaffold

**Description:** Set up the uv project with a src layout, ruff and pytest config, and a hello-world Streamlit entry point.
**Acceptance:**

- [ ] `uv sync` installs the packages: streamlit, openai, pydantic, pandas, python-dotenv; dev tools ruff, pytest, pytest-mock.
- [ ] The `support_ai` package can be imported.
- [ ] `.env.example` lists `OPENAI_API_KEY`, and `.env` is gitignored.

**Verify:** `uv run pytest -q` passes (0 tests is OK), `uv run ruff check .` is clean, and `uv run streamlit run streamlit_app.py` renders the home page.
**Files:** `pyproject.toml`, `.env.example`, `.gitignore`, `streamlit_app.py`, `src/support_ai/__init__.py`
**Scope:** S

#### Task 2: Classifier schema

**Description:** Pydantic models and enums (Category, Priority, NextStep) as in the SPEC taxonomy. There are two models: `LLMClassification`, which the LLM returns, and `Classification`, which adds `needs_human_review` and `review_reasons`.
**Acceptance:**

- [ ] Invalid enum values and confidence outside 0–1 are rejected.
- [ ] `LLMClassification.model_json_schema()` has no review fields.

**Verify:** `uv run pytest tests/test_schema.py -q`
**Files:** `src/support_ai/classifier/schema.py`, `tests/test_schema.py`
**Scope:** S

#### Task 3: LLM abstraction contracts, model registry and cost

**Description:** Add `LLMProvider` protocol, `LLMResult`/`Usage`, error hierarchy, provider registry (`get_provider(name)`), model registry with prices/timeouts, `cost.py` (usage → $ per call, forecast per N tickets), and `FakeProvider` for tests.
**Acceptance:**

- [ ] Nothing in `core/llm/` except the future adapter imports `openai`.
- [ ] Cost math is correct for known token counts.
- [ ] `forecast(cost_per_ticket, 10_000)` works.

**Verify:** `uv run pytest tests/test_cost.py -q`, plus `grep -r "import openai" src/` returning nothing yet.
**Files:** `src/support_ai/core/llm/base.py`, `src/support_ai/core/llm/errors.py`, `src/support_ai/core/config.py`, `src/support_ai/core/cost.py`, `tests/fakes.py` (+ `tests/test_cost.py`)
**Scope:** M

### Checkpoint: Foundation

- [ ] Tests pass and ruff is clean. Review the provider interface before building on it.

### Phase 2: Core Pipeline

#### Task 4: OpenAI adapter

**Description:** Implement `OpenAIProvider` using the SDK's structured outputs, parsed against the pydantic schema. Map SDK exceptions onto our errors: timeout → `LLMTimeout`, 429 → `LLMRateLimited`, parse or refusal → `LLMInvalidOutput`, other → `LLMProviderError`. Fill `Usage` and latency. Register the adapter as `"openai"`. Check the model ids and prices against current OpenAI docs.
**Acceptance:**

- [ ] A unit test with a mocked SDK client shows each exception mapped to the right error type.
- [ ] One real call on gpt-5.4-nano returns a parsed dict and a non-zero usage.

**Verify:** `uv run pytest tests/test_openai_provider.py -q`, plus a manual one-off call via `uv run python -c ...` or the smoke CLI from Task 7.
**Files:** `src/support_ai/core/llm/openai_provider.py`, `src/support_ai/core/llm/__init__.py` (registry wiring), `tests/test_openai_provider.py`
**Scope:** S

#### Task 5: Gateway with retries and fallback

**Description:** `complete_with_fallback(chain, system, user, schema)`:

- rate limit or provider error: exponential backoff, at most 2 retries
- timeout: 1 retry
- invalid output: 1 repair retry that appends the validation error to the prompt
- after that: move to the next model
- all models fail: raise `AllModelsFailed`, recording the attempts made

It returns the result plus the model actually used and the attempt log. Backoff sleeps can be injected so tests run fast.
**Acceptance:**

- [ ] Tests with `FakeProvider` cover: success on first try, recovery after invalid JSON, recovery after a 429, a timeout leading to fallback to the next model, and every model failing.

**Verify:** `uv run pytest tests/test_gateway.py -q`
**Files:** `src/support_ai/core/llm/gateway.py`, `tests/test_gateway.py`
**Scope:** S

#### Task 6: Human-review rules

**Description:** `apply_rules(LLMClassification, *, failed=False) -> Classification`. The rules are those in SPEC: confidence below the threshold (0.7, configurable), refund or expert complaint, P1, mixed topics, and the failure fallback.
**Acceptance:**

- [ ] A table-driven test covers each rule on its own, rules combined, and no rules firing (not flagged).

**Verify:** `uv run pytest tests/test_rules.py -q`
**Files:** `src/support_ai/classifier/rules.py`, `tests/test_rules.py`
**Scope:** S

#### Task 7: Prompt v1 and the classify pipeline

**Description:**

- `prompts/v1.md`: a first working prompt, with the taxonomy, priority rules and a note on multilingual and aggressive input. The ticket goes in a delimited block, as a guard against prompt injection.
- `classify(ticket) -> ClassifyResult`: render the prompt, call the gateway, validate, apply the rules and compute the cost.
  - It never raises. On `AllModelsFailed` or empty input it returns the fallback result, flagged for review.
- `scripts/classify_one.py "text"`: prints the result as JSON.

**Acceptance:**

- [ ] Tests with `FakeProvider` cover the normal path, the fallback path and the empty-ticket path.
- [ ] `classifier/` does not import `openai`.
- [ ] One real ticket gives valid JSON on the CLI.

**Verify:** `uv run pytest tests/test_classify.py -q`, `uv run python scripts/classify_one.py "I was charged twice, refund now!"`, `grep -r "openai" src/support_ai/classifier` returning nothing.
**Files:** `src/support_ai/classifier/prompts/v1.md`, `src/support_ai/classifier/classify.py`, `scripts/classify_one.py`, `tests/test_classify.py`
**Scope:** M

### Checkpoint: Core Pipeline

- [ ] All tests pass.
- [ ] The CLI classifies a real English ticket and a real non-English ticket end to end, with cost printed.
- [ ] Human check: to test the decoupling, trace how a second provider would be added. It should take one new file plus a config change.

### Phase 3: UI + Eval Harness

#### Task 8: Classifier page

**Description:**

- A text area and a Classify button, with a spinner while it runs.
- Results: category, secondary categories, priority, next step and note, language, tone, confidence, rationale.
- A red "Needs human review" badge listing the reasons.
- The model used, latency and cost.
- A friendly message on fallback.
- The page calls `classify()` only, with no LLM details, and reads the API key from `.env` or `st.secrets`.

**Acceptance:**

- [ ] Pasting a ticket shows the result.
- [ ] A missing API key shows a clear error instead of a stack trace.

**Verify:** A manual run of `uv run streamlit run streamlit_app.py` with 3 tickets: normal, aggressive and Ukrainian.
**Files:** `pages/1_Classifier.py`, `src/support_ai/core/config.py` (key loading)
**Scope:** S

#### Task 9: Synthetic test set

**Description:** `data/tickets.jsonl` with 15–20 tickets. Each has `id`, `text`, `expected_category`, `expected_priority`, `expected_needs_review` and `tags`. It covers every category plus:

- at least 2 aggressive, at least 2 mixed-topic and at least 2 non-English tickets (uk, es, …)
- one each of: empty or too short, prompt injection, legal or chargeback threat

Add a loader that validates it against pydantic.

**Acceptance:**

- [ ] The loader test passes.
- [ ] The test asserts the coverage of tags and categories.

**Verify:** `uv run pytest tests/test_dataset.py -q`
**Files:** `data/tickets.jsonl`, `src/support_ai/classifier/dataset.py`, `tests/test_dataset.py`
**Scope:** S

#### Task 10: Eval runner

**Description:** `scripts/run_eval.py --model <id> --prompt v1 [--limit N]` runs a single model with no fallback, so the comparison stays clean. It writes:

- `results/<model>_<prompt>_<ts>.json`, per ticket: expected, actual, pass/fail, latency, tokens, cost
- a row appended to `results/summary.csv`: category accuracy, priority accuracy, human-review recall, latency p50 and p95, cost per ticket, cost per 10k tickets

Metrics are pure functions in `eval/metrics.py`.

**Acceptance:**

- [ ] Metrics are unit-tested on handmade records.
- [ ] `--limit 3` against nano writes both files.

**Verify:** `uv run pytest tests/test_metrics.py -q`, then `uv run python scripts/run_eval.py --model gpt-5.4-nano --prompt v1 --limit 3`
**Files:** `src/support_ai/eval/metrics.py`, `scripts/run_eval.py`, `tests/test_metrics.py`
**Scope:** M

#### Task 11: Eval page

**Description:** Reads `results/`. Shows the summary table (model comparison) and a run picker. For the chosen run it shows a per-ticket table (input → expected → actual → pass/fail, with failures highlighted), accuracy metrics, and the cost forecast. If there are no results yet, it shows an empty state.
**Acceptance:**

- [ ] The page renders the run from Task 10.
- [ ] With an empty `results/`, the page shows a hint instead of crashing.

**Verify:** Manual check in Streamlit.
**Files:** `pages/2_Eval.py`
**Scope:** S

### Checkpoint: Complete

- [ ] `uv run pytest -q` passes and `uv run ruff check .` is clean.
- [ ] Classifier page: an edge-case ticket (injection, mixed topics, non-English) returns a valid result and the badge appears when expected.
- [ ] Eval page shows a 3-ticket run with cost.
- [ ] `grep -rn "import openai\|from openai" src/ pages/ scripts/` matches only `openai_provider.py`.
- [ ] Review with the user before prompt refinement and the full eval.

## Risks and Mitigations

| Risk                                                                                            | Impact | Mitigation                                                                                           |
| ----------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| The model IDs (gpt-5.4-nano/mini) or their structured-output support differ from what's assumed | High   | Check in Task 4 before anything else. The registry is config, so fixing it is a one-line change.     |
| gpt-5 (a reasoning model) is slow and hits the 20 s timeout                                     | Med    | Timeout per model in the registry. Set low reasoning effort as a provider option if it is supported. |
| The SDK's structured-output API shape changes                                                   | Med    | It is confined to the adapter and covered by a mocked test.                                          |
| Real calls in the eval burn credit                                                              | Low    | Use `--limit` and run nano only in this plan. The full run is deferred.                              |
| The prompt-injection ticket overrides the rules                                                 | Med    | The ticket goes in a delimited block, and the human-review flag is computed in code, not by the LLM. |

## Parallelization

- Tasks 2, 3 and 9 can run in parallel after Task 1.
- Tasks 4, 5 and 6 can run in parallel after Task 3 (Task 6 needs only Task 2).
- Task 7 has to wait for Tasks 4, 5 and 6.

## Verification (end to end)

1. `uv sync && uv run pytest -q && uv run ruff check .`
2. `uv run python scripts/classify_one.py "Верніть гроші, експерт грубив мені!"` gives valid JSON with `needs_human_review: true`.
3. `uv run python scripts/run_eval.py --model gpt-5.4-nano --prompt v1 --limit 3`, then check `results/`.
4. `uv run streamlit run streamlit_app.py`: classify on the Classifier page and view the run on the Eval page.
