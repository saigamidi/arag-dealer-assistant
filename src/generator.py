"""
Answer generation.

v1 (now, free):   StubGenerator - no LLM.
    * Live data (inventory, orders) is formatted with templates. This part is
      production-ready: numbers should be stated exactly, not paraphrased.
    * Knowledge base questions show the most relevant passages with citations
      ("preview mode"). Writing a natural answer from them, and deciding
      "I don't know", is the LLM's job in the cloud phase.
v2 (cloud, later): LLMGenerator - same interface, writes the answer from the
    passages and live data, cites sources, and refuses when they don't answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.api_client import ApiResult
from src.retrieve import SearchResult
from src.router import RouteDecision

STATUS_LABEL = {"in_stock": "In stock", "low_stock": "Low stock", "backordered": "Backordered"}


@dataclass
class Citation:
    number: int
    chunk_id: str
    source_file: str
    title: str
    section: str
    text: str
    score: float | None = None


@dataclass
class GeneratedAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    mode: str = "stub"
    # "answered", "refused" (not in the sources), "conflict" (sources disagree),
    # "unverified" (answer without a citation), "live_only", "clarify", "fallback"
    status: str = "answered"
    tokens_in: int = 0
    tokens_out: int = 0
    note: str | None = None


class Generator(Protocol):
    def generate(self, query: str, decision: RouteDecision, results: list[SearchResult],
                 api_results: list[ApiResult], context: list[dict] | None = None) -> GeneratedAnswer: ...


# --------------------------------------------------------------------------
# Live data formatting (shared by stub and, later, the LLM generator)
# --------------------------------------------------------------------------
def format_inventory(d: dict) -> str:
    label = STATUS_LABEL.get(d["status"], d["status"])
    breakdown = ", ".join(f"{w}: {q}" for w, q in d["quantities_by_warehouse"].items() if q > 0)
    single_wh = len(d["quantities_by_warehouse"]) == 1
    where = f" at {next(iter(d['quantities_by_warehouse']))}" if single_wh else " in total"
    line = f"**{d['part_id']}** ({d['description']}): **{label}**, {d['total_quantity']} units{where}"
    if breakdown and not single_wh:
        line += f" ({breakdown})"
    line += "."
    if d.get("estimated_restock_date"):
        line += f" Estimated restock: {d['estimated_restock_date']}."
    if d.get("special_order"):
        line += " This is a special-order part."
    return line


def format_order(d: dict) -> str:
    line = f"**{d['order_id']}** is **{d['status']}** (placed {d['placed_date']}"
    if d["status"] == "Delivered" and d.get("delivered_date"):
        line += f", delivered {d['delivered_date']})."
    else:
        line += f", expected delivery {d['expected_delivery']})."
    if d.get("is_delayed"):
        line += f" **Delayed**: revised delivery {d['revised_delivery']}"
        line += f" ({d['delay_reason']})." if d.get("delay_reason") else "."
    if d.get("tracking_number"):
        line += f" Tracking: {d['tracking_number']}."
    line += " Free cancellation: " + ("still possible." if d["can_cancel_free_of_charge"]
                                      else "no longer available (order is past Placed).")
    return line


def format_api_result(r: ApiResult) -> str:
    if not r.ok:
        return f"⚠️ {r.id}: {r.message}"
    return format_inventory(r.data) if r.endpoint == "inventory" else format_order(r.data)


def data_freshness(api_results: list[ApiResult]) -> str | None:
    stamps = sorted({r.data.get("data_as_of") for r in api_results if r.ok and r.data.get("data_as_of")})
    return f"_Live data as of {stamps[-1].replace('T', ' ').replace('Z', ' UTC')}._" if stamps else None


# --------------------------------------------------------------------------
# Stub generator
# --------------------------------------------------------------------------
def _excerpt(chunk_text: str, limit: int = 320) -> str:
    body = chunk_text.split("\n\n", 1)[-1].replace("\n", " ").strip()
    return body if len(body) <= limit else body[:limit].rsplit(" ", 1)[0] + "…"


class StubGenerator:
    name = "stub"

    def generate(self, query, decision, results, api_results, context=None) -> GeneratedAnswer:
        if decision.route == "clarify":
            return GeneratedAnswer(text=decision.clarification or "Could you share more details?",
                                   status="clarify")

        parts: list[str] = []
        citations: list[Citation] = []

        if api_results:
            parts.append("**Live data**")
            parts += [f"- {format_api_result(r)}" for r in api_results]
            fresh = data_freshness(api_results)
            if fresh:
                parts.append(fresh)

        if decision.needs_retrieval:
            for i, r in enumerate(results, start=1):
                c = r.chunk
                citations.append(Citation(i, c["chunk_id"], c["source_file"], c["title"],
                                          c["section"], c["text"], round(r.score, 2)))
            if citations:
                if parts:
                    parts.append("")
                parts.append("**From the knowledge base** _(preview mode: most relevant passages; "
                             "written answers arrive with the LLM in the cloud phase)_")
                parts += [f"- [{c.number}] *{c.title} › {c.section}*: {_excerpt(c.text)}" for c in citations]
            else:
                parts.append("I couldn't find anything in the knowledge base that matches this question.")

        status = "live_only" if (api_results and not decision.needs_retrieval) else "answered"
        return GeneratedAnswer(text="\n".join(parts), citations=citations, status=status)
