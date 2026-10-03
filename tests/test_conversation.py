"""
Tests for clarification follow-ups (multi-turn continuity).
Run from the project root with:  pytest -v
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_api.main import app
from src import ingest as ing
from src.api_client import ApiClient
from src.conversation import PendingClarification, resolve_follow_up
from src.pipeline import Assistant
from src.retrieve import BM25Retriever

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

CALIPERS = PendingClarification(
    original_query="Can you check stock for the front brake caliper?", missing="part", answer_id="a1",
    candidates=[{"part_id": "BRK-1020", "description": "Brake caliper assembly (front, left)"},
                {"part_id": "BRK-1021", "description": "Brake caliper assembly (front, right)"}])
ORDER = PendingClarification(original_query="Where is my order?", missing="order", answer_id="a2")
SUGGESTED = PendingClarification(original_query="Where is order 10482?", missing="order",
                                 answer_id="a3", suggested_id="ORD-10482")


# ---------------------------------------------------------------- resolving replies
@pytest.mark.parametrize("reply, expected_id", [
    ("BRK-1020", "BRK-1020"),
    ("brk-1021", "BRK-1021"),
    ("the left one", "BRK-1020"),
    ("right", "BRK-1021"),
    ("the first one", "BRK-1020"),
    ("2", "BRK-1021"),
])
def test_part_replies_are_resolved(reply, expected_id):
    assert resolve_follow_up(reply, CALIPERS) == f"{CALIPERS.original_query} ({expected_id})"


@pytest.mark.parametrize("reply", ["ORD-10482", "10482", "#10482"])
def test_order_replies_are_resolved(reply):
    assert resolve_follow_up(reply, ORDER) == "Where is my order? (ORD-10482)"


def test_yes_accepts_suggested_id():
    assert resolve_follow_up("yes", SUGGESTED) == "Where is order 10482? (ORD-10482)"


@pytest.mark.parametrize("reply", [
    "How long is the standard warranty on a new vehicle?",   # new question
    "What's the price of BRK-1020?",                         # has its own intent
    "front",                                                 # matches both candidates
    "yes",                                                   # two candidates, no suggestion
    "hmm not sure",                                          # no ID, no choice
])
def test_new_or_unclear_messages_are_not_merged(reply):
    assert resolve_follow_up(reply, CALIPERS) is None


def test_no_pending_means_no_merge():
    assert resolve_follow_up("BRK-1020", None) is None


# ---------------------------------------------------------------- end to end
@pytest.fixture
def assistant(tmp_path):
    if not KB_DIR.exists():
        pytest.skip("knowledge_base folder not found")
    ing.ingest(KB_DIR, tmp_path / "data")
    chunks = [json.loads(l) for l in (tmp_path / "data" / "chunks.jsonl").read_text().splitlines() if l.strip()]
    return Assistant(retriever=BM25Retriever(chunks),
                     api=ApiClient("http://testserver", http=TestClient(app)), log_dir=tmp_path / "logs")


def test_clarify_then_part_id_returns_live_stock(assistant):
    first = assistant.ask("Can you check stock for the front brake caliper?")
    assert first.route == "clarify" and first.pending is not None
    assert {c["part_id"] for c in first.pending.candidates} == {"BRK-1020", "BRK-1021"}

    second = assistant.ask("BRK-1020", pending=first.pending)
    assert second.route == "realtime"
    assert second.interpreted_as.endswith("(BRK-1020)")
    assert second.follow_up_of == first.answer_id
    assert "9 units" in second.text
    assert second.pending is None


def test_clarify_then_order_number(assistant):
    first = assistant.ask("Where is my order?")
    second = assistant.ask("10388", pending=first.pending)
    assert second.route == "realtime" and "Delayed" in second.text


def test_new_question_after_clarify_is_answered_on_its_own(assistant):
    first = assistant.ask("Where is my order?")
    second = assistant.ask("How long is the standard warranty on a new vehicle?", pending=first.pending)
    assert second.route == "static" and second.interpreted_as is None


def test_follow_up_is_logged(assistant):
    first = assistant.ask("Where is my order?")
    assistant.ask("ORD-10482", pending=first.pending)
    entry = json.loads((assistant.log_dir / "interactions.jsonl").read_text().splitlines()[-1])
    assert entry["follow_up_of"] == first.answer_id
    assert entry["interpreted_as"] == "Where is my order? (ORD-10482)"


# ---------------------------------------------------------------- UI
def test_ui_carries_clarification_to_next_message(monkeypatch):
    from streamlit.testing.v1 import AppTest
    if not Path("data/chunks.jsonl").exists():
        pytest.skip("run `python -m src.ingest` first")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30).run()
    at.chat_input[0].set_value("Can you check stock for the front brake caliper?").run()
    assert at.session_state.clarification is not None
    at.chat_input[0].set_value("BRK-1020").run()
    last = at.session_state.messages[-1]["answer"]
    assert last.route == "realtime" and last.interpreted_as.endswith("(BRK-1020)")
    assert at.session_state.clarification is None
