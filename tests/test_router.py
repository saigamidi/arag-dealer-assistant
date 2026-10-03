"""
Tests for the rule-based router.
Run from the project root with:  pytest -v
"""

import json
from pathlib import Path

import pytest

from src import ingest as ing
from src.retrieve import BM25Retriever
from src.router import RuleRouter

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
router = RuleRouter()


# ---------------------------------------------------------------- routes
@pytest.mark.parametrize("query, route", [
    ("How long is the standard warranty on a new vehicle?", "static"),
    ("What is the list price of BRK-1020?", "static"),                 # part ID, catalog question
    ("How often is order status updated?", "static"),                  # live words, general question
    ("What is the towing capacity of the SUV Sport?", "static"),       # out of scope -> generator decides
    ("How many BRK-1020 calipers are in stock?", "realtime"),
    ("Track ORD-10215", "realtime"),
    ("Is ELC-3030 in stock, and can the dealer return it once opened?", "hybrid"),
    ("Can order ORD-10495 still be cancelled free of charge?", "hybrid"),
    ("Where is my order?", "clarify"),
    ("Is the part in stock?", "clarify"),
])
def test_routes(query, route):
    assert router.route(query).route == route


# ---------------------------------------------------------------- entities
def test_ids_are_normalised_and_deduplicated():
    d = router.route("check stock for acc-4040 and ACC-4040")
    assert d.part_ids == ["ACC-4040"]


def test_order_id_is_not_mistaken_for_a_part_id():
    d = router.route("status of ord-10482")
    assert d.order_ids == ["ORD-10482"] and d.part_ids == []


def test_multiple_parts_produce_multiple_api_calls():
    d = router.route("Are BRK-1020 and BRK-1021 in stock?")
    assert [c["id"] for c in d.api_calls] == ["BRK-1020", "BRK-1021"]


@pytest.mark.parametrize("text", ["at the north warehouse", "in WH-NORTH", "North warehouse"])
def test_warehouse_extraction(text):
    d = router.route(f"Is FLT-2001 available {text}?")
    assert d.warehouse == "WH-NORTH"
    assert d.api_calls[0]["warehouse"] == "WH-NORTH"


def test_static_route_makes_no_api_calls_but_needs_retrieval():
    d = router.route("What is the restocking fee for vehicle returns?")
    assert d.api_calls == [] and d.needs_retrieval


def test_realtime_route_skips_retrieval():
    assert not router.route("Track ORD-10215").needs_retrieval


def test_every_decision_explains_itself():
    assert router.route("Track ORD-10215").reasons


# ---------------------------------------------------------------- clarification
def test_clarify_asks_for_order_id():
    d = router.route("Where is my order?")
    assert "ORD-12345" in d.clarification


def test_clarify_suggests_order_id_from_bare_number():
    d = router.route("Where is order 10482?")
    assert d.route == "clarify" and "ORD-10482" in d.clarification


def test_clarify_suggests_candidate_parts(tmp_path):
    if not KB_DIR.exists():
        pytest.skip("knowledge_base folder not found")
    ing.ingest(KB_DIR, tmp_path)
    chunks = [json.loads(l) for l in (tmp_path / "chunks.jsonl").read_text().splitlines() if l.strip()]
    d = RuleRouter(BM25Retriever(chunks)).route("Can you check stock for the front brake caliper?")
    assert d.route == "clarify"
    assert {c["part_id"] for c in d.candidates} == {"BRK-1020", "BRK-1021"}
    assert "BRK-1020" in d.clarification and "BRK-1021" in d.clarification
