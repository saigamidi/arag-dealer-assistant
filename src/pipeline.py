"""
ARAG pipeline: one call from question to answer.

    question -> router -> (retrieval) + (live API calls) -> generator -> answer
                                                                       -> log

Every interaction is logged with its route, retrieved chunks, API calls,
latency and (later) the agent's feedback (PRD > Non-functional requirements:
Observability).

Try it (mock API must be running for live data):
    python -m src.pipeline "Is ELC-3030 in stock, and can the dealer return it once opened?"
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.api_client import ApiClient, ApiResult
from src.conversation import PendingClarification, resolve_follow_up
from src.generator import Citation, Generator, StubGenerator
from src.retrieve import make_retriever
from src.router import RuleRouter

LOG_DIR = Path("data/logs")
TOP_K = 3


@dataclass
class Answer:
    answer_id: str
    query: str
    route: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    api_results: list[ApiResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    latency_ms: int = 0
    generator: str = "stub"
    interpreted_as: str | None = None          # set when a follow-up was merged with its question
    follow_up_of: str | None = None            # answer_id of the clarification it answered
    pending: PendingClarification | None = None   # set when this answer asks a clarifying question


class Assistant:
    def __init__(self, retriever=None, router=None, generator: Generator | None = None,
                 api: ApiClient | None = None, log_dir: Path = LOG_DIR):
        self.retriever = retriever or make_retriever()
        # Clarification candidates ("BRK-1020 or BRK-1021?") use keyword search:
        # exact part names matter more than meaning there.
        self.router = router or RuleRouter(getattr(self.retriever, "keyword_retriever", self.retriever))
        self.generator = generator or StubGenerator()
        self.api = api or ApiClient()
        self.log_dir = log_dir

    def ask(self, query: str, simulate: str | None = None,
            pending: PendingClarification | None = None) -> Answer:
        """
        `pending` is the clarification from the previous answer, if any. When the
        new message answers it (e.g. just "BRK-1020"), the two are combined.
        """
        start = time.perf_counter()
        resolved = resolve_follow_up(query, pending)
        effective = resolved or query
        decision = self.router.route(effective)

        results = self.retriever.search(effective, k=TOP_K) if decision.needs_retrieval else []
        api_results = [self.api.call(c["endpoint"], c["id"], c.get("warehouse"), simulate)
                       for c in decision.api_calls]
        generated = self.generator.generate(effective, decision, results, api_results)

        answer = Answer(
            answer_id=uuid.uuid4().hex[:12], query=query, route=decision.route,
            text=generated.text, citations=generated.citations, api_results=api_results,
            reasons=decision.reasons, latency_ms=int((time.perf_counter() - start) * 1000),
            generator=getattr(self.generator, "name", "unknown"),
            interpreted_as=resolved,
            follow_up_of=pending.answer_id if (resolved and pending) else None,
        )
        if decision.route == "clarify":
            answer.pending = PendingClarification(
                original_query=effective, missing=decision.missing or "part",
                answer_id=answer.answer_id, candidates=decision.candidates,
                suggested_id=decision.suggested_id)
        self._log_interaction(answer, decision, results)
        return answer

    # ------------------------------------------------------------ logging
    def _append(self, filename: str, record: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_interaction(self, answer: Answer, decision, results) -> None:
        self._append("interactions.jsonl", {
            "answer_id": answer.answer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query": answer.query,
            "interpreted_as": answer.interpreted_as,
            "follow_up_of": answer.follow_up_of,
            "route": answer.route,
            "reasons": answer.reasons,
            "entities": {"parts": decision.part_ids, "orders": decision.order_ids,
                         "warehouse": decision.warehouse},
            "retrieved": [{"chunk_id": r.chunk_id, "score": round(r.score, 3)} for r in results],
            "api_calls": [{"endpoint": r.endpoint, "id": r.id, "ok": r.ok, "error_code": r.error_code}
                          for r in answer.api_results],
            "latency_ms": answer.latency_ms,
            "generator": answer.generator,
            "retriever": type(self.retriever).__name__,
            "retrieval_fallback": getattr(self.retriever, "last_fallback", None),
        })

    def record_feedback(self, answer_id: str, rating: str, comment: str = "") -> None:
        if rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        self._append("feedback.jsonl", {
            "answer_id": answer_id, "rating": rating, "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Is ELC-3030 in stock, and can the dealer return it once opened?"
    a = Assistant().ask(question)
    print(f"\nQuery: {a.query}\nRoute: {a.route}   ({a.latency_ms} ms)\n")
    print(a.text.replace("**", ""))
    print()
