# AI Support Assistant — Requirements

2026-09-27

## Goal

Build an internal AI assistant that cuts the time support agents spend searching the knowledge base and writing replies. The agent pastes a ticket and gets a summary, ready-to-edit reply drafts and the knowledge-base source behind them. Users are internal support agents only; the assistant suggests, the agent decides.

## Functional requirements

One pasted ticket in, one response out with summary, reply variants, KB source, category and priority.

| ID   | Requirement                                                                          | Notes                                                |
| ---- | ------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| FR-1 | Agent pastes ticket text into the tool                                               | Format is open: web UI, Slack bot, script or other   |
| FR-2 | Tool returns a short summary of the request                                          |                                                      |
| FR-3 | Tool returns 2–3 reply variants in distinct tones: formal, empathetic, short         | Tones must be clearly different, not rewordings      |
| FR-4 | Tool returns a link to or quote from the knowledge base backing the reply            | KB may be simulated as a set of texts                |
| FR-5 | Tool assigns a category and priority to the ticket                                   | Implied by the evaluation criteria; taxonomy not given |
| FR-6 | Interface marks tickets where the agent must decide without relying on AI variants   | See Automation boundaries                            |

## Automation boundaries

The tool must know where AI suggestions stop and the agent's own judgment is mandatory.

- Define a list of concrete scenarios where the agent must decide alone, not from AI variants.
- Explain the logic behind each scenario.
- Propose how the interface technically marks these tickets for the agent.

## Quality requirements

These are the evaluation criteria; each needs evidence, not just a claim.

| Area             | Requirement                                                                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Evaluation       | Fixed test set with expected category and priority; table of input → expected → actual → pass/fail; overall accuracy; analysis of errors |
| Model comparison | Several models compared on quality, speed and cost; final choice justified                                                                    |
| Failure handling | Defined behavior on invalid JSON, timeout and rate limit; fallback scenarios for each                                                         |
| Cost             | Cost of one ticket; forecast for 10k tickets/month; ways to cut cost without losing quality                                                   |
| Caching          | Decide whether caching is needed; if yes, what is cached and when it is invalidated; if no, why it makes no sense here                        |

## Documentation deliverables

The working MVP ships with written reasoning covering these points.

- [ ] Retrieval approach and why it was chosen (if retrieval is used)
- [ ] Prompt structure used to get variants in different tones
- [ ] Trade-offs made on purpose because of time or tool limits
- [ ] Reply-generation prompts: how each tone was achieved and what did not work the first time
- [ ] One concrete example of bad AI output and how it was fixed
- [ ] List of human-judgment scenarios with the logic of each
- [ ] Evaluation, model comparison, failure handling, cost and caching write-ups

## Constraints and open questions

The task fixes outputs, not implementation; architecture and stack are left to the solution.

- Delivery format is free: web UI, Slack bot, script or anything else.
- Knowledge base may be simulated as a set of texts.

Open questions the task leaves unanswered:

- [ ] What are the category and priority values?
- [ ] What content and size should the simulated knowledge base have?
- [ ] When is 2 variants acceptable instead of 3?
- [ ] How large should the fixed test set be?
