# Requirements: AI Support Tooling (2 MVPs)

## Background

Nebula receives thousands of support tickets a month: subscription questions, bugs, complaints about experts, and refund requests. Today all routing is done by hand, and agents spend a lot of time searching the internal knowledge base (KB) and writing replies.

---

## MVP 1: Ticket Classifier

### Functional

- **F1.1** Takes raw ticket text as input.
- **F1.2** Returns:
  - category
  - priority
  - recommended next step
- **F1.3** AI is the core component that makes the classification.
- **F1.4** Handles these edge cases:
  - aggressive or abusive tone
  - mixed topics in one ticket
  - non-English text
- **F1.5** Flags ticket types that must not be processed automatically and sends them to a human-in-the-loop mechanism (at least one).

### Testing

- **T1.1** Uses a self-generated synthetic set of at least 10 tickets, including the edge cases in F1.4.
- **T1.2** Each test case has an expected category and priority.

### Documentation

- **D1.1** Model or tool chosen, with reasons.
- **D1.2** Prompt structure, and how the prompt evolved from the first version to the final one.
- **D1.3** Category list and the logic behind it.
- **D1.4** Known failure modes and how each is mitigated.
- **D1.5** Automation limits: which ticket types are excluded from automatic handling and why, plus the proposed human-in-the-loop mechanism.
- **D1.6** 2–3 edge cases where the AI gave an unexpected result, and what was done about each.
- **D1.7** One decision made by the author rather than delegated to AI, with the reason.

---

## MVP 2: Agent Reply Assistant (internal)

### Functional

- **F2.1** Internal tool for support agents. Any format is allowed (web UI, chat bot, script, etc.).
- **F2.2** The agent pastes ticket text as input.
- **F2.3** Returns:
  - a short summary of the ticket
  - 2–3 reply drafts in different tones: **formal**, **empathetic**, **short**
  - a link to or quote from the knowledge base
- **F2.4** The knowledge base can be simulated as a set of texts.
- **F2.5** The UI marks scenarios where the agent must decide personally and not rely on AI drafts.

### Documentation

- **D2.1** Retrieval approach and why it was chosen (if retrieval is used).
- **D2.2** Prompt structure for getting distinct tones, and what did not work on the first try.
- **D2.3** Trade-offs made on purpose because of time or tool limits.
- **D2.4** One concrete example of bad AI output and how it was fixed.
- **D2.5** List of scenarios where human judgment is required, with the reasoning for each.
- **D2.6** How the UI technically signals that human judgment is required.

---

## Cross-cutting (evaluation criteria)

| #   | Area             | Requirement                                                                                                                           |
| --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| X1  | Evaluation       | Fixed test set with expected category and priority. Table of input → expected → actual → pass/fail. Overall accuracy. Error analysis. |
| X2  | Model comparison | Several models compared on quality, speed, and cost, with a justified final choice.                                                   |
| X3  | Failure handling | Defined behavior for invalid JSON, timeouts, and rate limits, plus fallback scenarios.                                                |
| X4  | Cost             | Cost per ticket, a forecast for 10k tickets/month, and ideas for cutting cost without losing quality.                                 |
| X5  | Caching          | Decide whether caching is needed. If yes: what to cache and when to invalidate it. If no: the reasoning.                              |

## Out of scope

- Production routing integration and live ticket ingestion (the task only asks for prototypes).
