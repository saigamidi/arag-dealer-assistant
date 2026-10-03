"""
ARAG router: decide which path a question takes.

Routes
  static    -> knowledge base only (RAG)                e.g. "How long is the standard warranty?"
  realtime  -> live API only                            e.g. "How many BRK-1020 are in stock?"
  hybrid    -> knowledge base + live API                e.g. "Is ELC-3030 in stock, and can it be returned?"
  clarify   -> a live lookup is needed but the ID is missing
                                                        e.g. "Where is my order?"

v1 (now, free):   RuleRouter - regex for IDs + keyword lists for intent.
v2 (cloud, later): LLMRouter  - same interface, an LLM classifies intent.
Both return a RouteDecision, so the rest of the pipeline doesn't change.

Every decision carries human-readable reasons, so a wrong route can be
debugged by reading why it was chosen.

Try it (from the project root):
    python -m src.router "Is ELC-3030 in stock, and can the dealer return it once opened?"
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Literal, Protocol

Route = Literal["static", "realtime", "hybrid", "clarify"]

# ---------------------------------------------------------------- entities
PART_ID = re.compile(r"\b([A-Za-z]{3})-(\d{4})\b")
ORDER_ID = re.compile(r"\bORD-(\d{5})\b", re.I)
BARE_ORDER_NUMBER = re.compile(r"\border\s*(?:number|no\.?|#)?\s*(\d{5})\b", re.I)
WAREHOUSE = re.compile(r"\b(?:WH-)?(north|south|east|west)\b(?:\s+warehouse)?", re.I)

# ---------------------------------------------------------------- intent keywords
# Live-data intent: the answer changes during the day.
REALTIME_PATTERNS = [
    r"\bin stock\b", r"\bstock\b", r"\bavailab", r"\binventory\b", r"\bhow many\b",
    r"\bquantit", r"\bwarehouses?\b", r"\bstatus\b", r"\btrack", r"\bdelivered\b",
    r"\barriv", r"\bshipped\b", r"\bwhere is\b", r"\bwhere's\b", r"\bwhen will\b",
    r"\beta\b", r"\bright now\b", r"\bcheck\b",
]
# Policy / reference intent: the answer is in the knowledge base.
POLICY_PATTERNS = [
    r"\bpolic", r"\bwarrant", r"\bcover", r"\breturn", r"\brefund", r"\bcancel",
    r"\bdiscount", r"\bfees?\b", r"\bcharge\b", r"\beligib", r"\ballowed\b", r"\brules?\b",
    r"\bhow long\b", r"\bhow often\b", r"\btypical", r"\busually\b", r"\bprocess\b",
    r"\bwhat happens\b", r"\bprice\b", r"\bcost\b", r"\bcompatib", r"\bfits?\b",
]


def _matches(patterns: list[str], text: str) -> list[str]:
    hits = []
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            hits.append(m.group(0).lower())
    return hits


# ---------------------------------------------------------------- decision
@dataclass
class RouteDecision:
    route: Route
    part_ids: list[str] = field(default_factory=list)
    order_ids: list[str] = field(default_factory=list)
    warehouse: str | None = None
    reasons: list[str] = field(default_factory=list)
    clarification: str | None = None          # question to ask the agent, for "clarify"
    candidates: list[dict] = field(default_factory=list)   # suggested parts, for "clarify"
    missing: str | None = None                # "part" or "order", for "clarify"
    suggested_id: str | None = None           # e.g. ORD-10482 guessed from "order 10482"

    @property
    def needs_retrieval(self) -> bool:
        return self.route in ("static", "hybrid")

    @property
    def api_calls(self) -> list[dict]:
        """The live lookups the pipeline should make, e.g. for the mock API."""
        if self.route not in ("realtime", "hybrid"):
            return []
        calls = [{"endpoint": "inventory", "id": p, "warehouse": self.warehouse} for p in self.part_ids]
        calls += [{"endpoint": "orders", "id": o} for o in self.order_ids]
        return calls


class Router(Protocol):
    def route(self, query: str) -> RouteDecision: ...


# ---------------------------------------------------------------- rule-based router
class RuleRouter:
    """
    Decision rules (in order):
      1. An order ID always needs a live lookup -> realtime, or hybrid if there's also policy intent.
      2. A part ID with live intent -> realtime, or hybrid if there's also policy intent.
         A part ID without live intent (e.g. "price of BRK-1020") -> static (catalog).
      3. Live intent but no ID -> clarify, unless it's a general policy question
         (e.g. "How often is order status updated?") -> static.
      4. Everything else -> static. Out-of-scope questions also go here; deciding
         "I don't know" is the generator's job, not the router's.
    """

    def __init__(self, retriever=None):
        # Optional: used only to suggest candidate parts when asking for clarification.
        self.retriever = retriever

    def route(self, query: str) -> RouteDecision:
        part_ids = list(dict.fromkeys(f"{a.upper()}-{b}" for a, b in PART_ID.findall(query)))
        part_ids = [p for p in part_ids if not p.startswith("ORD")]
        order_ids = list(dict.fromkeys(f"ORD-{n}" for n in ORDER_ID.findall(query)))
        wh = WAREHOUSE.search(query)
        warehouse = f"WH-{wh.group(1).upper()}" if wh else None

        live = _matches(REALTIME_PATTERNS, query)
        policy = _matches(POLICY_PATTERNS, query)
        d = RouteDecision(route="static", part_ids=part_ids, order_ids=order_ids, warehouse=warehouse)

        if order_ids:
            d.reasons.append(f"order ID found: {', '.join(order_ids)} (always needs a live lookup)")
            if policy:
                d.route = "hybrid"
                d.reasons.append(f"policy intent: {', '.join(policy)}")
            else:
                d.route = "realtime"
            if part_ids and not live:
                d.part_ids = []               # parts mentioned inside an order question
            return d

        if part_ids:
            d.reasons.append(f"part ID found: {', '.join(part_ids)}")
            if live and policy:
                d.route = "hybrid"
                d.reasons.append(f"live intent: {', '.join(live)}; policy intent: {', '.join(policy)}")
            elif live:
                d.route = "realtime"
                d.reasons.append(f"live intent: {', '.join(live)}")
            else:
                d.reasons.append("no live intent, so the catalog answers it")
            return d

        if live and not policy:
            d.route = "clarify"
            d.reasons.append(f"live intent ({', '.join(live)}) but no part or order ID")
            d.clarification, d.candidates, d.missing, d.suggested_id = self._clarify(query)
            return d

        if live and policy:
            d.reasons.append(f"general question about live systems (policy intent: {', '.join(policy)})")
        else:
            d.reasons.append("no IDs and no live intent")
        return d

    def _clarify(self, query: str) -> tuple[str, list[dict], str, str | None]:
        """Return (question to ask, candidate parts, what's missing, suggested ID)."""
        bare = BARE_ORDER_NUMBER.search(query)
        if bare:
            suggestion = f"ORD-{bare.group(1)}"
            return f"Did you mean order {suggestion}? Order IDs look like ORD-12345.", [], "order", suggestion
        if re.search(r"\border", query, re.I):
            return "Which order? Please share the order ID (format ORD-12345).", [], "order", None

        candidates = []
        if self.retriever is not None:
            results = [r for r in self.retriever.search(query, k=8) if r.chunk.get("part_id")]
            if results:
                top = results[0].score
                candidates = [{"part_id": r.chunk["part_id"],
                               "description": r.chunk["text"].split("Description: ")[-1].splitlines()[0]}
                              for r in results if r.score >= 0.8 * top][:3]
        if candidates:
            options = " or ".join(f"{c['part_id']} ({c['description']})" for c in candidates)
            return f"Which part do you mean: {options}?", candidates, "part", None
        return "Which part? Please share the part ID (format ABC-1234, e.g. BRK-1020).", [], "part", None


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Is ELC-3030 in stock, and can the dealer return it once opened?"
    try:
        from src.retrieve import BM25Retriever
        router = RuleRouter(BM25Retriever())
    except FileNotFoundError:
        router = RuleRouter()
    d = router.route(question)
    print(f"\nQuery:     {question}\nRoute:     {d.route}")
    if d.part_ids or d.order_ids:
        print(f"Entities:  parts={d.part_ids} orders={d.order_ids} warehouse={d.warehouse}")
    if d.api_calls:
        print(f"API calls: {d.api_calls}")
    print(f"Retrieval: {'yes' if d.needs_retrieval else 'no'}")
    for r in d.reasons:
        print(f"Why:       {r}")
    if d.clarification:
        print(f"Ask agent: {d.clarification}")
    print()
