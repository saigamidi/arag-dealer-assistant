"""
ARAG evaluation harness (no LLM needed yet).

Part 1 - Retrieval: how well the retriever finds the right chunks for the
labeled static and hybrid questions in eval/test_set.csv.
Part 2 - Routing: whether each of the 40 questions takes the right path
(static / realtime / hybrid / clarify) and the right IDs are extracted.

Metrics
  * Doc hit@3   - an expected DOCUMENT appears in the top 3   (PRD target: >= 80%)
  * Chunk hit@3 - an expected CHUNK appears in the top 3      (stricter)
  * Recall@3    - share of expected chunks found in the top 3 (matters when an answer needs 2 chunks)
  * MRR         - 1 / rank of the first correct chunk, averaged (rewards ranking it first)

It also compares top scores for answerable vs out-of-scope questions, which
informs the "I don't know" threshold used later by the generator.

Usage (from the project root):
    python -m eval.run_eval                     # retrieval + routing
    python -m eval.run_eval --only router       # one part only (retrieval | router)
    python -m eval.run_eval --show-all          # print every query, not just misses

Routing is also scored on eval/router_holdout.csv: 15 new questions written
after the rules were finished and never used to tune them. The gap between
the two scores shows how well the rules generalize.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.retrieve import BM25Retriever
from src.router import RuleRouter

TEST_SET = Path("eval/test_set.csv")
ROUTER_HOLDOUT = Path("eval/router_holdout.csv")   # written AFTER the rules, never used to tune them
RESULTS_DIR = Path("eval/results")
K = 3
PRD_TARGET_DOC_HIT = 0.80
PRD_TARGET_ROUTING = 0.90
ROUTES = ["static", "realtime", "hybrid", "clarify"]


def load_test_set(path: Path = TEST_SET) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_retrieval(retriever, name: str, show_all: bool = False) -> dict:
    rows = load_test_set()
    per_query = []

    for row in rows:
        expected = [c for c in row["expected_chunks"].split("|") if c]
        results = retriever.search(row["query"], k=10)
        top_ids = [r.chunk_id for r in results]
        top_k = top_ids[:K]
        top_score = results[0].score if results else 0.0

        record = {"query_id": row["query_id"], "category": row["category"], "query": row["query"],
                  "top_score": round(top_score, 3), "retrieved_top3": "|".join(top_k)}

        if expected:
            expected_docs = {c.split("::")[0] for c in expected}
            first_rank = next((i + 1 for i, cid in enumerate(top_ids) if cid in expected), None)
            record.update({
                "expected": "|".join(expected),
                "doc_hit": int(any(cid.split("::")[0] in expected_docs for cid in top_k)),
                "chunk_hit": int(any(cid in expected for cid in top_k)),
                "recall": round(sum(c in top_k for c in expected) / len(expected), 3),
                "rr": round(1 / first_rank, 3) if first_rank else 0.0,
            })
        per_query.append(record)

    scored = [r for r in per_query if "chunk_hit" in r]

    def mean(key, subset):
        return round(sum(r[key] for r in subset) / len(subset), 3) if subset else None

    summary = {"retriever": name, "k": K, "queries_scored": len(scored),
               "doc_hit@3": mean("doc_hit", scored), "chunk_hit@3": mean("chunk_hit", scored),
               "recall@3": mean("recall", scored), "mrr": mean("rr", scored),
               "by_category": {}}
    for cat in sorted({r["category"] for r in scored}):
        subset = [r for r in scored if r["category"] == cat]
        summary["by_category"][cat] = {"n": len(subset), "doc_hit@3": mean("doc_hit", subset),
                                       "chunk_hit@3": mean("chunk_hit", subset)}

    # Score separation: can a threshold tell answerable from out-of-scope?
    answerable = [r["top_score"] for r in scored]
    oos = [r["top_score"] for r in per_query if r["category"] == "out_of_scope"]
    summary["score_separation"] = {
        "answerable_min": min(answerable), "answerable_median": round(statistics.median(answerable), 3),
        "out_of_scope_max": max(oos), "out_of_scope_median": round(statistics.median(oos), 3),
        "overlap": max(oos) >= min(answerable),
    }

    print_report(summary, scored, per_query, show_all)
    save(summary, per_query, name)
    return summary


def print_report(summary, scored, per_query, show_all):
    pct = lambda v: f"{v * 100:.0f}%"
    status = "MEETS" if summary["doc_hit@3"] >= PRD_TARGET_DOC_HIT else "BELOW"
    print(f"\nRetrieval evaluation: {summary['retriever']}  ({summary['queries_scored']} labeled queries)")
    print("-" * 64)
    print(f"Doc hit@3    {pct(summary['doc_hit@3']):>5}   ({status} PRD target of {pct(PRD_TARGET_DOC_HIT)})")
    print(f"Chunk hit@3  {pct(summary['chunk_hit@3']):>5}")
    print(f"Recall@3     {pct(summary['recall@3']):>5}")
    print(f"MRR          {summary['mrr']:.2f}")
    for cat, m in summary["by_category"].items():
        print(f"  {cat:<8} n={m['n']:<3} doc hit {pct(m['doc_hit@3']):>5}   chunk hit {pct(m['chunk_hit@3']):>5}")

    s = summary["score_separation"]
    print(f"\nTop-score separation (for the 'I don't know' threshold)")
    print(f"  Answerable:   min {s['answerable_min']:.2f}, median {s['answerable_median']:.2f}")
    print(f"  Out-of-scope: max {s['out_of_scope_max']:.2f}, median {s['out_of_scope_median']:.2f}")
    print("  -> Scores overlap: a BM25 score threshold alone can't separate them."
          if s["overlap"] else "  -> No overlap: a threshold between these values would work.")

    misses = [r for r in scored if not r["chunk_hit"]]
    to_show = scored if show_all else misses
    if to_show:
        print(f"\n{'All queries' if show_all else 'Misses (expected chunk not in top 3)'}:")
        for r in to_show:
            mark = "OK  " if r["chunk_hit"] else "MISS"
            print(f"  [{mark}] {r['query_id']}  {r['query']}")
            print(f"         expected:  {r['expected']}")
            print(f"         retrieved: {r['retrieved_top3'] or '(nothing)'}")
    print()


def save(summary, per_query, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"retrieval_{name}_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["query_id", "category", "query", "expected", "retrieved_top3",
                  "doc_hit", "chunk_hit", "recall", "rr", "top_score"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_query)

    # One line per run in history.csv, so improvements are easy to compare over time.
    history = RESULTS_DIR / "history.csv"
    new = not history.exists()
    with open(history, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_at", "retriever", "doc_hit@3", "chunk_hit@3", "recall@3", "mrr"])
        w.writerow([stamp, name, summary["doc_hit@3"], summary["chunk_hit@3"],
                    summary["recall@3"], summary["mrr"]])
    (RESULTS_DIR / f"retrieval_{name}_latest.json").write_text(json.dumps(summary, indent=2))




# ==========================================================================
# Routing evaluation
# ==========================================================================
def evaluate_router(router, name: str, show_all: bool = False, test_set: Path = TEST_SET) -> dict:
    rows = load_test_set(test_set)
    per_query = []
    for row in rows:
        d = router.route(row["query"])
        extracted = set(d.part_ids + d.order_ids)
        expected_entity = row["expected_entity"].strip()
        per_query.append({
            "query_id": row["query_id"], "category": row["category"], "query": row["query"],
            "expected_route": row["expected_route"], "predicted_route": d.route,
            "route_ok": int(d.route == row["expected_route"]),
            "expected_entity": expected_entity, "extracted": "|".join(sorted(extracted)),
            "entity_ok": int(extracted == {expected_entity}) if expected_entity else None,
            "reasons": " / ".join(d.reasons),
        })

    acc = lambda subset: round(sum(r["route_ok"] for r in subset) / len(subset), 3)
    with_entity = [r for r in per_query if r["entity_ok"] is not None]
    summary = {
        "router": name, "test_set": test_set.name, "queries": len(per_query), "routing_accuracy": acc(per_query),
        "entity_accuracy": round(sum(r["entity_ok"] for r in with_entity) / len(with_entity), 3),
        "by_category": {c: {"n": len(sub), "accuracy": acc(sub)}
                        for c in dict.fromkeys(r["category"] for r in per_query)
                        for sub in [[r for r in per_query if r["category"] == c]]},
        "confusion": {e: {p: sum(1 for r in per_query if r["expected_route"] == e and r["predicted_route"] == p)
                          for p in ROUTES} for e in ROUTES},
    }

    pct = lambda v: f"{v * 100:.0f}%"
    status = "MEETS" if summary["routing_accuracy"] >= PRD_TARGET_ROUTING else "BELOW"
    print(f"\nRouting evaluation: {name} on {test_set.name}  ({len(per_query)} queries)")
    print("-" * 64)
    print(f"Routing accuracy   {pct(summary['routing_accuracy']):>5}   ({status} PRD target of {pct(PRD_TARGET_ROUTING)})")
    print(f"Entity extraction  {pct(summary['entity_accuracy']):>5}   ({len(with_entity)} queries with an ID)")
    for cat, m in summary["by_category"].items():
        print(f"  {cat:<13} n={m['n']:<3} {pct(m['accuracy']):>5}")
    print("\nConfusion matrix (rows = expected, columns = predicted)")
    print("  " + " " * 10 + "".join(f"{p:>10}" for p in ROUTES))
    for e in ROUTES:
        print(f"  {e:<10}" + "".join(f"{summary['confusion'][e][p]:>10}" for p in ROUTES))

    wrong = [r for r in per_query if not r["route_ok"] or r["entity_ok"] == 0]
    to_show = per_query if show_all else wrong
    if to_show:
        print(f"\n{'All queries' if show_all else 'Errors'}:")
        for r in to_show:
            mark = "OK  " if r["route_ok"] and r["entity_ok"] != 0 else "MISS"
            print(f"  [{mark}] {r['query_id']}  {r['query']}")
            print(f"         expected: {r['expected_route']:<9} got: {r['predicted_route']:<9} "
                  f"entities: {r['extracted'] or '-'}")
            print(f"         why: {r['reasons']}")
    else:
        print("\nNo routing or entity errors.")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with open(RESULTS_DIR / f"router_{name}_{test_set.stem}_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)
    history = RESULTS_DIR / "router_history.csv"
    new = not history.exists()
    with open(history, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_at", "router", "test_set", "routing_accuracy", "entity_accuracy"])
        w.writerow([stamp, name, test_set.name, summary["routing_accuracy"], summary["entity_accuracy"]])
    (RESULTS_DIR / f"router_{name}_{test_set.stem}_latest.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate ARAG retrieval and routing.")
    p.add_argument("--only", choices=["retrieval", "router"], help="Run just one part")
    p.add_argument("--show-all", action="store_true", help="Show every query, not just misses")
    args = p.parse_args()
    retriever = BM25Retriever()
    if args.only in (None, "retrieval"):
        evaluate_retrieval(retriever, "bm25", args.show_all)
    if args.only in (None, "router"):
        router = RuleRouter(retriever)
        evaluate_router(router, "rules", args.show_all, TEST_SET)
        if ROUTER_HOLDOUT.exists():
            evaluate_router(router, "rules", args.show_all, ROUTER_HOLDOUT)
