"""
End-to-end tests: router -> retrieval -> mock API -> stub generator -> logs.
Uses FastAPI's TestClient, so the mock API server doesn't need to be running.
Run from the project root with:  pytest -v
"""

import json
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from mock_api.main import app
from src import ingest as ing
from src.api_client import ApiClient
from src.pipeline import Assistant
from src.retrieve import BM25Retriever

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    if not KB_DIR.exists():
        pytest.skip("knowledge_base folder not found")
    out = tmp_path_factory.mktemp("data")
    ing.ingest(KB_DIR, out)
    chunks = [json.loads(l) for l in (out / "chunks.jsonl").read_text().splitlines() if l.strip()]
    return BM25Retriever(chunks)


@pytest.fixture
def assistant(retriever, tmp_path):
    api = ApiClient("http://testserver", http=TestClient(app))
    return Assistant(retriever=retriever, api=api, log_dir=tmp_path / "logs")


# ---------------------------------------------------------------- routes end to end
def test_static_question_returns_cited_passages(assistant):
    a = assistant.ask("How long is the standard warranty on a new vehicle?")
    assert a.route == "static"
    assert a.citations and a.citations[0].chunk_id == "warranty_policy::overview"
    assert "[1]" in a.text and a.api_results == []


def test_realtime_question_formats_live_stock(assistant):
    a = assistant.ask("How many BRK-1020 calipers are in stock?")
    assert a.route == "realtime" and a.citations == []
    assert "Low stock" in a.text and "9 units" in a.text and "WH-NORTH: 4" in a.text
    assert "Live data as of" in a.text


def test_hybrid_question_has_live_data_and_sources(assistant):
    a = assistant.ask("Is ELC-3030 in stock, and can the dealer return it once opened?")
    assert a.route == "hybrid"
    assert "Backordered" in a.text and "2026-10-21" in a.text
    assert a.citations


def test_delayed_order_shows_revised_date(assistant):
    a = assistant.ask("When will order ORD-10388 arrive?")
    assert "Delayed" in a.text and "2026-10-21" in a.text


def test_warehouse_specific_stock(assistant):
    a = assistant.ask("Is FLT-2001 available at the north warehouse?")
    assert "210 units at WH-NORTH" in a.text


def test_clarify_asks_instead_of_calling_api(assistant):
    a = assistant.ask("Where is my order?")
    assert a.route == "clarify" and a.api_results == [] and "ORD-12345" in a.text


# ---------------------------------------------------------------- failure handling
def test_unknown_part_explained_not_crashed(assistant):
    a = assistant.ask("How many BRK-9999 are in stock?")
    assert a.api_results[0].error_code == "PART_NOT_FOUND"
    assert "No part with ID BRK-9999" in a.text


def test_api_outage_is_handled_gracefully(assistant):
    a = assistant.ask("Track ORD-10215", simulate="error")
    assert a.api_results[0].error_code == "SERVICE_UNAVAILABLE"
    assert "temporarily unavailable" in a.text


def test_timeout_is_handled_gracefully(retriever, tmp_path):
    class SlowHttp:
        def get(self, *args, **kwargs):
            raise requests.Timeout()
    assistant = Assistant(retriever=retriever, api=ApiClient("http://x", http=SlowHttp()),
                          log_dir=tmp_path)
    a = assistant.ask("Track ORD-10215")
    assert a.api_results[0].error_code == "TIMEOUT"
    assert "did not respond in time" in a.text


def test_api_not_running_is_handled(retriever, tmp_path):
    class DownHttp:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError()
    a = Assistant(retriever=retriever, api=ApiClient("http://x", http=DownHttp()),
                  log_dir=tmp_path).ask("Track ORD-10215")
    assert a.api_results[0].error_code == "CONNECTION_ERROR"
    assert "could not be reached" in a.text


# ---------------------------------------------------------------- observability
def test_interaction_is_logged(assistant):
    a = assistant.ask("Is BRK-1020 in stock and is it covered by a parts warranty?")
    log = [json.loads(l) for l in (assistant.log_dir / "interactions.jsonl").read_text().splitlines()]
    entry = log[-1]
    assert entry["answer_id"] == a.answer_id and entry["route"] == "hybrid"
    assert entry["entities"]["parts"] == ["BRK-1020"]
    assert entry["retrieved"] and entry["api_calls"][0]["ok"] is True
    assert entry["latency_ms"] >= 0


def test_feedback_is_recorded(assistant):
    a = assistant.ask("Track ORD-10215")
    assistant.record_feedback(a.answer_id, "down", "wrong order")
    fb = json.loads((assistant.log_dir / "feedback.jsonl").read_text().splitlines()[-1])
    assert fb == {**fb, "answer_id": a.answer_id, "rating": "down", "comment": "wrong order"}
    with pytest.raises(ValueError):
        assistant.record_feedback(a.answer_id, "maybe")


# ---------------------------------------------------------------- UI smoke test
def test_streamlit_app_loads():
    from streamlit.testing.v1 import AppTest
    if not Path("data/chunks.jsonl").exists():
        pytest.skip("run `python -m src.ingest` first")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Dealer Support Assistant"
