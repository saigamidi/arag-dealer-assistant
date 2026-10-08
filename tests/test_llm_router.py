"""
Tests for the LLM and tiered routers, using a scripted fake chat client
(no cloud calls, no cost).
Run from the project root with:  pytest -v
"""

import pytest

from src.llm import FakeChatClient
from src.router import ROUTER_PROMPT, LLMRouter, RuleRouter, TieredRouter, make_router


def llm_router(*responses, fail=False):
    return LLMRouter(FakeChatClient(responses=list(responses), fail=fail), RuleRouter())


LIVE_PART = {"needs_live_lookup": True, "lookup_type": "part", "needs_policy_info": False, "reason": "stock"}
LIVE_ORDER = {"needs_live_lookup": True, "lookup_type": "order", "needs_policy_info": False, "reason": "status"}
POLICY = {"needs_live_lookup": False, "lookup_type": None, "needs_policy_info": True, "reason": "policy"}
BOTH_ORDER = {"needs_live_lookup": True, "lookup_type": "order", "needs_policy_info": True, "reason": "both"}


# ---------------------------------------------------------------- mapping LLM intent to routes
def test_live_lookup_with_id_is_realtime():
    d = llm_router(LIVE_PART).route("BRK-1001 qty at WH-EAST")        # held-out X07: rules said static
    assert d.route == "realtime" and d.part_ids == ["BRK-1001"] and d.warehouse == "WH-EAST"
    assert d.decided_by == "llm" and d.api_calls[0]["warehouse"] == "WH-EAST"


def test_live_and_policy_with_id_is_hybrid():
    d = llm_router(BOTH_ORDER).route("Can I still cancel ORD-10503?")
    assert d.route == "hybrid" and d.order_ids == ["ORD-10503"]


def test_live_without_id_asks_for_clarification():
    d = llm_router(LIVE_PART).route("Do you have the left LED headlight for the SUV Sport?")
    assert d.route == "clarify" and d.missing == "part"


def test_bare_number_with_order_hint_suggests_order_id():
    d = llm_router(LIVE_ORDER).route("What's the ETA on 10482?")        # held-out X14
    assert d.route == "clarify" and d.suggested_id == "ORD-10482"


def test_policy_only_is_static_even_with_part_id():
    d = llm_router(POLICY).route("Is ELC-3010 covered by the standard warranty?")
    assert d.route == "static" and d.api_calls == [] and d.needs_retrieval


def test_ids_always_come_from_regex_not_the_llm():
    router = llm_router(LIVE_PART)
    router.route("Is brk-1020 in stock?")
    _, user = router.client.prompts[0]
    assert "BRK-1020" in user                      # normalised by regex, passed to the LLM as context


def test_malformed_llm_output_is_safe():
    d = llm_router({"needs_live_lookup": "maybe", "lookup_type": "galaxy"}).route("What is the warranty?")
    assert d.route in ("static", "clarify")


# ---------------------------------------------------------------- resilience and cost tracking
def test_llm_failure_falls_back_to_rules():
    d = llm_router(fail=True).route("Track ORD-10215")
    assert d.route == "realtime" and d.decided_by == "rules (LLM unavailable)"


def test_router_counts_calls_and_tokens():
    router = llm_router(POLICY, POLICY)
    router.route("What is the return window?")
    router.route("How long is the warranty?")
    assert router.calls == 2 and router.tokens_in == 200 and router.tokens_out == 40


# ---------------------------------------------------------------- tiered router
def test_tiered_fast_path_skips_llm_for_clear_lookups():
    llm = llm_router(POLICY)
    tiered = TieredRouter(llm)
    d = tiered.route("How many BRK-1020 calipers are in stock?")
    assert d.route == "realtime" and d.decided_by == "rules (fast path)"
    assert llm.client.prompts == [] and tiered.fast_path == 1


def test_tiered_sends_unclear_questions_to_llm():
    llm = llm_router(LIVE_PART)
    tiered = TieredRouter(llm)
    d = tiered.route("Need 3 x FLT-2010 today, can you confirm we have them?")   # held-out X10
    assert d.route == "realtime" and d.decided_by == "llm" and len(llm.client.prompts) == 1


def test_tiered_sends_questions_without_ids_to_llm():
    llm = llm_router(POLICY)
    TieredRouter(llm).route("How long is the standard warranty?")
    assert len(llm.client.prompts) == 1


# ---------------------------------------------------------------- config and prompt hygiene
def test_make_router_modes(monkeypatch):
    assert isinstance(make_router("rules"), RuleRouter)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert isinstance(make_router("tiered"), RuleRouter)            # no secrets -> safe fallback
    with pytest.raises(ValueError):
        make_router("magic")


def test_router_prompt_examples_do_not_leak_test_questions():
    """Few-shot examples must not copy evaluation questions, or the held-out score is meaningless."""
    import csv
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    questions = [r["query"].lower() for f in ("eval/test_set.csv", "eval/router_holdout.csv")
                 for r in csv.DictReader(open(root / f, encoding="utf-8"))]
    examples = [line.split('"')[1].lower() for line in ROUTER_PROMPT.splitlines() if line.startswith('- "')]
    assert examples and not any(e in questions for e in examples)
    ids_in_examples = {"acc-4010", "ord-20001", "acc-4020"}
    assert not any(i in q for q in questions for i in ids_in_examples)
