rubric_version: v1

# Reply-assistant judging rubric

Shared with `src/support_ai/assistant/prompts/reply_v1.md` (the drafting prompt): the tone
rules below are restated exactly as they appear there, so the model that writes the drafts
and the judge that scores them agree on what each tone means. Judge only the drafts (and,
for every ticket, the summary) — not whether retrieval found the right article or whether
the human-judgment flag was correct. Those are checked separately, by code.

## formal

A greeting and a sign-off (`[Agent name]`). Complete sentences. No contractions, no slang,
no emoji.

## empathetic

Opens by naming the customer's specific situation and how they likely feel about it
(frustrated, worried, confused, relieved, ...), in a warm tone — then gives the same facts
as the other two drafts.

## short

At most 50 words: the answer, plus one concrete next step. No greeting needed.

## `tone_score` (1–5)

Score each present tone against its rule above and against general reply quality:

- **5** — Follows the tone rule exactly; clear, correctly grounded, reads like something a
  good agent would send as-is.
- **4** — Follows the tone rule; correct and usable, but a little clunky, generic or
  slightly long/short of the mark.
- **3** — Mostly follows the tone rule but has a real flaw (e.g. formal draft with one
  contraction, empathetic draft that doesn't actually name the situation, short draft a few
  words over 50) that an agent would want to fix before sending.
- **2** — Breaks the tone rule in an obvious way, or is confusing, generic filler that
  barely engages with the question.
- **1** — Wrong tone entirely, unusable, or so far off the rule it isn't recognizable as
  that draft's tone.

## `faithful`

`true` only if every factual claim in the draft is traceable to `ticket_text` or the cited
article's text (read the full article, not just `kb_quote`). `false` if the draft states or
implies any promise, amount, date, exception or policy detail the article doesn't make —
including a confident-sounding invented detail, or a fact from a *different* KB topic. A
draft with no cited article (`kb_article_id` was null) is faithful only if it doesn't invent
any policy at all — acknowledging the question and saying the agent will follow up is fine.
Placeholders (`[Customer name]`, `[Agent name]`) are always fine and never count against
faithfulness.

## `addresses_request`

`true` if the draft actually answers what the customer in `ticket_text` asked (or, for a
no-article case, politely acknowledges it without answering) — not a generic non-answer, not
an answer to a different question than the one asked, not missing a part of a multi-part
question that the cited article could have answered.

## `summary_accurate`

`true` if `summary` correctly states what the customer wants, in a sentence or two, without
adding anything `ticket_text` doesn't say or missing the customer's actual ask. Judge every
ticket's summary, drafted or flagged.

## `distinct` (drafted tickets only; `null` for flagged tickets)

`true` if the 3 tones genuinely differ from each other — in opening, structure and phrasing,
not just a word swapped here and there — while still stating the same facts. `false` if two
or more drafts read as the same reply lightly reworded (this is what
`tone_checks.DISTINCT_MAX_OVERLAP` catches deterministically for exact word overlap; you're
judging the same idea by reading, which also catches paraphrased near-duplicates the word
check misses).
