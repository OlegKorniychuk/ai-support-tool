---
name: judge-replies
description: Judge MVP 2 reply-eval drafts with Sonnet and record the verdicts. Use when asked to judge a reply-eval run, e.g. "/judge-replies results/reply/gpt-5.4-nano_v1_<ts>.json" — scores tone, faithfulness and distinctness per rubric.md and writes results/reply/<run>.judge.json, then records it.
model: sonnet
argument-hint: <results/reply/run.json>
allowed-tools:
  - Read
  - Write
  - Bash(uv run python scripts/record_judgement.py:*)
---

# /judge-replies

Judges one reply-assistant eval run: the run file given as `$ARGUMENTS` (a JSON array of
`ReplyEvalRecord`, e.g. `results/reply/gpt-5.4-nano_v1_20260928T120000Z.json`). You are the
judge — score every drafted ticket against `rubric.md`, write the verdicts, then record
them. A different vendor (you, Sonnet) judges OpenAI's drafts to avoid self-preference bias
— see SPEC_MVP2.md.

## Steps

1. **Read the run file** given as `$ARGUMENTS`. Note its filename without `.json` — that's
   the `run` value you'll write back (e.g. `gpt-5.4-nano_v1_20260928T120000Z`).
2. **Read `rubric.md`** (next to this file) in full before scoring anything.
3. **For every record in the run:**
   - Read `ticket_text`, `summary`, `drafts` (a `{tone: text}` map, empty when the ticket
     was flagged for human judgment), and `kb_quote`.
   - If `cited_article_id` is set, read `data/kb/<cited_article_id>.md` for the full
     article text — a draft may state facts from anywhere in the article, not only the
     quoted sentence.
   - Judge the drafts and the summary, not whether retrieval or the human-judgment flag was
     correct — that's a different, code-checked metric. A flagged ticket (`drafts` empty)
     still gets a `summary_accurate` judgment, but no `DraftVerdict`s and `distinct: null`.
   - Score each tone present in `drafts` per `rubric.md`: `tone_score` (1–5),
     `faithful`, `addresses_request`, and an optional one-line `note` on anything notable.
   - Set `distinct: true/false` for a drafted ticket (do the 3 tones actually read
     differently, not just reworded?); `null` for a flagged ticket.
4. **Write `results/reply/<run>.judge.json`**, exactly matching the `JudgeRun` shape
   (`src/support_ai/eval/judge.py`) — one `TicketVerdict` per record, in any order:

   ```json
   {
     "run": "gpt-5.4-nano_v1_20260928T120000Z",
     "judge_model": "claude-sonnet-5",
     "rubric_version": "v1",
     "judged_at": "2026-09-28T14:05:00Z",
     "verdicts": [
       {
         "ticket_id": "r002",
         "summary_accurate": true,
         "distinct": true,
         "drafts": [
           {"tone": "formal", "tone_score": 5, "faithful": true, "addresses_request": true, "note": ""},
           {"tone": "empathetic", "tone_score": 4, "faithful": true, "addresses_request": true, "note": ""},
           {"tone": "short", "tone_score": 5, "faithful": true, "addresses_request": true, "note": ""}
         ],
         "note": ""
       },
       {"ticket_id": "r021", "summary_accurate": true, "distinct": null, "drafts": [], "note": "kb gap, correctly flagged"}
     ]
   }
   ```

5. **Run the recorder**: `uv run python scripts/record_judgement.py results/reply/<run>.judge.json`.
6. **If it reports errors** (unknown/missing/duplicate ticket ids, a drafted ticket without
   all 3 tones, a mismatched `distinct`), fix `results/reply/<run>.judge.json` and rerun
   step 5. Don't guess past the tool's own error messages — they name every problem.
7. **Finish with a one-paragraph summary** (in your reply, not a file) of weak spots: which
   tone scored lowest and why, any unfaithful or off-request drafts, any drafts that read as
   near-duplicates, and any inaccurate summaries. Point at specific ticket ids.
