"""Reply Assistant: grounds a general-question ticket in the KB and drafts replies.

`schema.py` defines the LLM output and result types; `rules.py` turns the LLM's reported
facts into the human-judgment decision, verified against the ticket and the KB
(SPEC_MVP2.md, "Human-judgment rules"). `assist.py`, the pipeline that ties retrieval,
generation and these rules together, is added in a later task.
"""
