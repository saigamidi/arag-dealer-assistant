## ARAG v1.0 — Dealer Support Assistant

First complete release: an agentic RAG assistant that answers dealer support questions from policy documents and live inventory/order systems, with cited answers, explained routing and graceful fallbacks.

### Highlights
- **Tiered router:** regex for IDs, rules fast path, LLM for intent, rules fallback. Held-out routing accuracy **80% → 100% (15/15)**, with 25% fewer LLM calls than LLM-only routing.
- **Vector retrieval** (Azure OpenAI embeddings): **100%** hit@3 on 22 labelled queries; beat hybrid search (91%) in a measured comparison.
- **Cited LLM answers** with an explicit status (answered / refused / conflict) and code-level guardrails; real-time answers use templates for exact numbers.
- **Graceful degradation** for every external dependency; stale-index warning at startup.
- **Evaluation harness** for retrieval, routing and answers, plus 167 automated tests that never call the cloud.

### Results
Answer correctness 96% · citation coverage 100% · conflicts flagged 2/2 · latency p90 ≈ 3 s · ≈ $0.0004 per answered question.
Below target: refusal correctness 93–97%, consistency 90%. See known issues K1–K5.

### Status
**Conditional go for shadow mode** (1 of 4 gates met). Not for live use.

### Documents
- `docs/ARCHITECTURE.md` — solution architecture and decision records
- `docs/ARAG_Model_PRD.docx` — PRD v2.3
- `docs/ARAG_v1.0_Release_Review.docx` — release review and retrospective
- `docs/ARAG_RAID_Log.xlsx` — risks, assumptions, issues, dependencies, open decisions

All data is synthetic.
