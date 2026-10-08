# Changelog

Each entry matches a project snapshot (arag_project.zip). Newest first.

## 2026-10-08 — Answer quality fixes
- Fix: refusals that cite a source now show that source; no dangling [n] markers.
- Conflict rule clarified: a general rule vs a contradicting specific rule is a conflict
  unless a passage says which takes precedence.
- LLM now sees top 5 passages (RETRIEVAL_TOP_K); retrieval metrics still measured at 3.
- Fixed seed for more repeatable answers (best effort).
- New: `python -m eval.run_answers --repeat N` consistency check.

## 2026-10-07 — LLM answer generator
- Azure OpenAI answers with citations; statuses answered / refused / conflict / unverified / fallback.
- No LLM for real-time or clarify routes (templates keep numbers exact).
- `python -m eval.run_answers`: answer correctness, refusal correctness, citation coverage, latency, cost.

## 2026-10-07 — Embeddings, vector default, .env
- Azure embeddings with change detection (data/embeddings.json); vector / hybrid / bm25 retrievers.
- Measured: vector 100% vs hybrid 90% vs BM25 90% chunk hit@3; RETRIEVER=vector by default.
- .env support (python-dotenv); Codespaces secrets take precedence.

## Earlier — Local phase (no LLM)
- Mock API, chunking with change detection, BM25, rule router with held-out set,
  pipeline, Streamlit UI, logging, feedback, clarification follow-ups.
