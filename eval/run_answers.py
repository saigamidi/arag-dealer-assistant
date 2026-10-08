"""
ARAG answer evaluation (uses the LLM, costs a few cents per run).

Runs every question in eval/test_set.csv through the full assistant (router ->
retrieval -> mock API -> LLM generator) and scores the answers.

Metrics (PRD targets in brackets)
  * Answer correctness   [>= 85%]  an LLM judge compares each answer with the expected answer
  * Refusal correctness  [100%]    out-of-scope -> refused; answerable -> not refused
  * Citation coverage    [100%]    answered knowledge-base questions cite at least one source
  * Cites expected source          a cited chunk is one of the labeled expected chunks
  * Conflicts flagged              questions labeled flag_conflict (S10, S16) get status "conflict"
  * Latency p90          [< 5s]
  * Tokens and estimated cost per query

The judge uses the same model as the generator, so it can be lenient with its
own mistakes. Every answer is saved to eval/results/ for a manual review pass.

Usage (from the project root; the mock API is called in-process, no server needed):
    python -m eval.run_answers
    python -m eval.run_answers --only S10,O01      # selected questions
    python -m eval.run_answers --only S10 --repeat 5   # consistency: ask each question 5 times
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from mock_api.main import app
from src.api_client import ApiClient
from src.llm import AzureChatClient
from src.llm_generator import LLMGenerator
from src.pipeline import Assistant

TEST_SET = Path("eval/test_set.csv")
RESULTS_DIR = Path("eval/results")
SCORED = {"static", "hybrid", "out_of_scope"}           # categories the LLM answers
# Approximate prices in USD per 1M tokens; set these to your deployment's price.
PRICE_IN = float(os.getenv("CHAT_PRICE_IN_PER_M", "0.40"))
PRICE_OUT = float(os.getenv("CHAT_PRICE_OUT_PER_M", "1.60"))

JUDGE_PROMPT = """You grade answers from a dealer support assistant.
Given a QUESTION, the EXPECTED answer (ground truth) and the ACTUAL answer, decide whether the ACTUAL answer
is correct: it must state the key facts of the EXPECTED answer (numbers, IDs, yes/no, conditions) and must not
contradict them. Extra correct detail is fine. Wording does not matter.
If the ACTUAL answer states a general rule that, applied to the case in the QUESTION, gives the EXPECTED result,
it is correct (e.g. "refunds are prorated after 30 days" correctly answers a question about day 45).
Ignore the "Live data" block formatting; judge the facts.
Return only JSON: {"correct": true | false, "reason": "<one short sentence>"}"""


def judge(client, question: str, expected: str, actual: str) -> tuple[bool, str, int, int]:
    res = client.complete_json(JUDGE_PROMPT, f"QUESTION: {question}\nEXPECTED: {expected}\nACTUAL: {actual}",
                               max_tokens=120)
    return bool(res.data.get("correct")), str(res.data.get("reason", "")), res.tokens_in, res.tokens_out


def pct(v):
    return "n/a" if v is None else f"{v * 100:.0f}%"


def run(only: set[str] | None = None) -> dict:
    rows = list(csv.DictReader(open(TEST_SET, encoding="utf-8")))
    if only:
        rows = [r for r in rows if r["query_id"] in only]

    chat = AzureChatClient()
    assistant = Assistant(generator=LLMGenerator(chat),
                          api=ApiClient("http://testserver", http=TestClient(app)),
                          log_dir=RESULTS_DIR / "answer_logs")
    records = []
    judge_in = judge_out = 0

    for row in rows:
        cat = row["category"]
        ans = assistant.ask(row["query"])
        expected_chunks = {c for c in row["expected_chunks"].split("|") if c}
        cited = [c.chunk_id for c in ans.citations]
        rec = {"query_id": row["query_id"], "category": cat, "query": row["query"],
               "route": ans.route, "status": ans.status, "answer": ans.text.replace("\n", " "),
               "cited": "|".join(cited), "expected_answer": row["expected_answer"],
               "latency_ms": ans.latency_ms, "tokens_in": ans.tokens_in, "tokens_out": ans.tokens_out}

        if cat in SCORED:
            should_refuse = cat == "out_of_scope"
            rec["refusal_ok"] = int((ans.status == "refused") == should_refuse)
            if not should_refuse:
                rec["has_citation"] = int(bool(cited))
                rec["cites_expected"] = int(bool(expected_chunks & set(cited))) if expected_chunks else None
                ok, reason, ti, to = judge(chat, row["query"], row["expected_answer"], ans.text)
                judge_in, judge_out = judge_in + ti, judge_out + to
                rec["correct"], rec["judge_reason"] = int(ok), reason
            else:
                rec["correct"] = rec["refusal_ok"]
                rec["judge_reason"] = "refused as expected" if rec["refusal_ok"] else "should have refused"
            if row["expected_behavior"] == "flag_conflict":
                rec["conflict_ok"] = int(ans.status == "conflict")
        records.append(rec)
        mark = "OK  " if rec.get("correct", 1) and rec.get("refusal_ok", 1) else "MISS"
        print(f"[{mark}] {row['query_id']:<4} {ans.status:<10} {row['query'][:60]}")

    def mean(key, subset=None):
        vals = [r[key] for r in (subset or records) if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    scored = [r for r in records if r["category"] in SCORED]
    answerable = [r for r in scored if r["category"] != "out_of_scope"]
    llm_calls = [r for r in records if r["tokens_in"]]
    latencies = sorted(r["latency_ms"] for r in llm_calls) or [0]
    p90 = latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))]
    gen_in, gen_out = sum(r["tokens_in"] for r in records), sum(r["tokens_out"] for r in records)
    gen_cost = gen_in / 1e6 * PRICE_IN + gen_out / 1e6 * PRICE_OUT
    judge_cost = judge_in / 1e6 * PRICE_IN + judge_out / 1e6 * PRICE_OUT

    summary = {
        "model": chat.name, "retriever": type(assistant.retriever).__name__, "questions": len(records),
        "answer_correctness": mean("correct", answerable),
        "refusal_correctness": mean("refusal_ok", scored),
        "citation_coverage": mean("has_citation", answerable),
        "cites_expected_source": mean("cites_expected", answerable),
        "conflict_flagged": mean("conflict_ok"),
        "latency_p90_ms": p90,
        "avg_cost_per_llm_query_usd": round(gen_cost / max(len(llm_calls), 1), 6),
        "run_cost_usd": round(gen_cost + judge_cost, 4),
    }

    print(f"\nAnswer evaluation: {summary['model']} with {summary['retriever']}  ({len(records)} questions)")
    print("-" * 64)
    print(f"Answer correctness     {pct(summary['answer_correctness']):>5}   (PRD target >= 85%, LLM-judged)")
    print(f"Refusal correctness    {pct(summary['refusal_correctness']):>5}   (PRD target 100%)")
    print(f"Citation coverage      {pct(summary['citation_coverage']):>5}   (PRD target 100%)")
    print(f"Cites expected source  {pct(summary['cites_expected_source']):>5}")
    n_conf = sum(1 for r in records if r.get("conflict_ok") is not None)
    print(f"Conflicts flagged      {pct(summary['conflict_flagged']):>5}   ({n_conf} conflict questions)")
    print(f"Latency p90            {p90 / 1000:.1f}s    (PRD target < 5s)")
    print(f"Cost per LLM query     ${summary['avg_cost_per_llm_query_usd']:.5f}   "
          f"(this run incl. judge: ${summary['run_cost_usd']:.4f}; prices are estimates)")

    misses = [r for r in scored if not r.get("correct") or not r.get("refusal_ok")
              or r.get("conflict_ok") == 0]
    if misses:
        print("\nTo review:")
        for r in misses:
            print(f"  [{r['query_id']}] {r['query']}")
            print(f"     status: {r['status']} | expected: {r['expected_answer']}")
            print(f"     answer: {r['answer'][:220]}")
            print(f"     judge:  {r.get('judge_reason', '')}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fields = list(dict.fromkeys(k for r in records for k in r))
    with open(RESULTS_DIR / f"answers_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    history = RESULTS_DIR / "answer_history.csv"
    new = not history.exists()
    with open(history, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_at"] + list(summary))
        w.writerow([stamp] + list(summary.values()))
    (RESULTS_DIR / "answers_latest.json").write_text(json.dumps(summary, indent=2))
    return summary


def consistency(only: set[str] | None, repeat: int) -> None:
    """Ask each question `repeat` times; report how often the status and cited sources match the first run."""
    rows = list(csv.DictReader(open(TEST_SET, encoding="utf-8")))
    rows = [r for r in rows if r["category"] in SCORED and (not only or r["query_id"] in only)]
    assistant = Assistant(generator=LLMGenerator(AzureChatClient()),
                          api=ApiClient("http://testserver", http=TestClient(app)),
                          log_dir=RESULTS_DIR / "answer_logs")
    stable = 0
    print(f"\nConsistency check: {len(rows)} question(s) x {repeat} runs")
    print("-" * 64)
    for row in rows:
        runs = [assistant.ask(row["query"]) for _ in range(repeat)]
        sigs = [(a.status, tuple(sorted(c.chunk_id for c in a.citations))) for a in runs]
        same = sum(sig == sigs[0] for sig in sigs)
        stable += same == repeat
        statuses = ", ".join(a.status for a in runs)
        print(f"[{'STABLE' if same == repeat else 'VARIES'}] {row['query_id']:<4} {same}/{repeat} identical  ({statuses})")
    print(f"\nFully consistent: {stable}/{len(rows)} questions ({stable / max(len(rows), 1) * 100:.0f}%)\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate ARAG answers with the LLM.")
    p.add_argument("--only", help="Comma-separated query IDs, e.g. S10,O01")
    p.add_argument("--repeat", type=int, default=0, help="Ask each question N times and report consistency")
    a = p.parse_args()
    only = set(a.only.split(",")) if a.only else None
    if a.repeat > 1:
        consistency(only, a.repeat)
    else:
        run(only)
