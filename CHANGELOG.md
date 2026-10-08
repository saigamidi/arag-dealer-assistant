# Changelog

Each entry matches a project snapshot (arag_project.zip). Newest first.

## 2026-10-08 — LLM router + stale-index warning
- LLMRouter: regex extracts IDs (exact, free); the LLM decides intent only (live lookup? policy info?
  part or order?); code maps that to static / realtime / hybrid / clarify. Rules decide if the LLM fails.
- TieredRouter (default, ROUTER=tiered): rules fast path when an ID and lookup words are present,
  LLM for everything else. Saves LLM calls on the clearest questions.
- Router few-shot examples are invented and checked by a test not to overlap the evaluation sets.
- `python -m eval.run_eval --only router` compares rules / llm / tiered on main and held-out sets,
  with LLM calls, tokens and fast-path share.
- Clarification uses the LLM's hint: "ETA on 10482" now suggests ORD-10482.
- K2 mitigation: app sidebar warns when documents changed since the last ingest or embeddings
  are out of date (`check_freshness` in src/ingest.py).
- Answer log and UI show which router decided each route.

## 2026-10-08 — Rule 9 refined; judge fix (H03, S03)
- H03 regression (from rule 9): a correct answer with a caveat was labeled "refused".
  Rule 9 now distinguishes the MAIN question (refuse) from a missing detail (answer + mention gap).
- Answers apply the rule to the specific case asked.
- S03 was a judge error (correct answer marked wrong). Judge prompt now accepts a general rule
  that, applied to the asked case, gives the expected result. Measurement change, noted as such.

## 2026-10-08 — Status consistency (O03, round 2)
- After the scope fix, O03's answer text was a correct refusal but labeled "answered".
  Prompt rule 9: the status must match what the answer says.
- Eval label "Conflict flagged (S10)" -> "Conflicts flagged" (now covers S10 and S16).
- Finding (S16): the knowledge base had been edited (30 -> 45 days) during a change-detection
  test; the model quoted it correctly and the stale expected answer flagged it. Ground truth must
  be versioned with the knowledge base.

## 2026-10-08 — Scope-inference fix (O03)
- Finding: the model extended "Premium Care and Comprehensive cover X" to Essential Plus,
  a cited but ungrounded answer (refusal correctness 96%).
- Prompt rule 5 generalized: a statement applies only to the items it names (plans, models, parts).
- Test set 40 -> 42 questions: S16 (second conflict: special-order part returns) and
  S17 (scope: FLT-2002 is listed for SUV Sport, not SUV Touring) to validate beyond O03.
- Note: retrieval and routing baselines now use 22 / 42 labeled queries; re-run for comparability.

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
