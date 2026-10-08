"""
Tests for the LLM answer generator, using a scripted fake chat client
(no cloud calls, no cost).
Run from the project root with:  pytest -v
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_api.main import app
from src import ingest as ing
from src.api_client import ApiClient
from src.generator import StubGenerator
from src.llm import FakeChatClient
from src.llm_generator import SYSTEM_PROMPT, LLMGenerator, make_generator
from src.pipeline import Assistant
from src.retrieve import BM25Retriever

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    if not KB_DIR.exists():
        pytest.skip("knowledge_base folder not found")
    out = tmp_path_factory.mktemp("data")
    ing.ingest(KB_DIR, out)
    return BM25Retriever([json.loads(l) for l in (out / "chunks.jsonl").read_text().splitlines() if l.strip()])


def make_assistant(retriever, tmp_path, responses, fail=False):
    client = FakeChatClient(responses=responses, fail=fail)
    a = Assistant(retriever=retriever, generator=LLMGenerator(client),
                  api=ApiClient("http://testserver", http=TestClient(app)), log_dir=tmp_path / "logs")
    return a, client


# ---------------------------------------------------------------- answering
def test_static_answer_with_valid_citation(retriever, tmp_path):
    a, client = make_assistant(retriever, tmp_path, [
        {"status": "answered", "answer": "3 years or 60,000 miles, whichever comes first [1].", "citations": [1]}])
    ans = a.ask("How long is the standard warranty on a new vehicle?")
    assert ans.status == "answered" and ans.generator == "llm"
    assert "60,000 miles" in ans.text and len(ans.citations) == 1
    assert ans.tokens_in == 100 and ans.tokens_out == 20
    system, user = client.prompts[0]
    assert system == SYSTEM_PROMPT and "PASSAGES:" in user and "[1]" in user


def test_invalid_citation_numbers_are_dropped(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [
        {"status": "answered", "answer": "Within 30 days [1][9].", "citations": [1, 9, "x"]}])
    ans = a.ask("How many days does a dealer have to submit a warranty claim after a repair?")
    assert [c.number for c in ans.citations] == [1]


def test_answer_without_citation_is_marked_unverified(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [{"status": "answered", "answer": "Probably 30 days.", "citations": []}])
    ans = a.ask("How many days does a dealer have to submit a warranty claim after a repair?")
    assert ans.status == "unverified"
    assert "Verify it before replying" in ans.text


def test_refusal_keeps_sources_it_cites(retriever, tmp_path):
    """Regression: a refusal that cites [1] must show source [1] (it used to be dropped)."""
    a, _ = make_assistant(retriever, tmp_path, [
        {"status": "refused", "answer": "The policy covers mats [1] but not this case.", "citations": [1]}])
    ans = a.ask("Can a dealer return an unopened set of floor mats?")
    assert ans.status == "refused" and [c.number for c in ans.citations] == [1]
    assert "[1]" in ans.text


def test_refusal_without_sources_has_no_dangling_markers(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [
        {"status": "refused", "answer": "The sources don't cover towing capacity [7].", "citations": [7]}])
    ans = a.ask("What is the towing capacity of the SUV Sport?")
    assert ans.status == "refused" and ans.citations == [] and "[7]" not in ans.text


def test_every_marker_in_text_has_a_source(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [
        {"status": "answered", "answer": "Within 30 days [1][2].", "citations": []}])
    ans = a.ask("How many days does a dealer have to submit a warranty claim after a repair?")
    import re
    markers = {int(n) for n in re.findall(r"\[(\d+)\]", ans.text)}
    assert markers <= {c.number for c in ans.citations}


def test_prompt_defines_general_vs_specific_rule_as_conflict():
    assert "general" in SYSTEM_PROMPT and "precedence" in SYSTEM_PROMPT


def test_llm_sees_more_passages_than_the_metric_k(retriever, tmp_path):
    a, client = make_assistant(retriever, tmp_path, [{"status": "answered", "answer": "x [1]", "citations": [1]}])
    a.ask("Can a dealer return an unopened set of floor mats?")
    _, user = client.prompts[0]
    assert "Return Eligibility" in user and "Non-Returnable Items" in user     # both sides of the conflict


def test_conflict_is_flagged(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [
        {"status": "conflict", "answer": "The catalog allows it [1] but the return policy does not [2].",
         "citations": [1, 2]}])
    ans = a.ask("Can a dealer return an unopened set of floor mats?")
    assert ans.status == "conflict" and "Sources disagree" in ans.text


def test_garbage_status_defaults_safely(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [{"status": "maybe", "answer": "Yes [1].", "citations": [1]}])
    assert a.ask("Are brake pads covered under the standard warranty?").status == "answered"


# ---------------------------------------------------------------- routes that skip the LLM
def test_realtime_question_does_not_call_llm(retriever, tmp_path):
    a, client = make_assistant(retriever, tmp_path, [{}])
    ans = a.ask("How many BRK-1020 calipers are in stock?")
    assert client.prompts == [] and "9 units" in ans.text and ans.status == "live_only"


def test_clarify_does_not_call_llm(retriever, tmp_path):
    a, client = make_assistant(retriever, tmp_path, [{}])
    ans = a.ask("Where is my order?")
    assert client.prompts == [] and ans.status == "clarify"


def test_hybrid_gives_llm_the_live_data_and_keeps_exact_numbers(retriever, tmp_path):
    a, client = make_assistant(retriever, tmp_path, [
        {"status": "answered", "answer": "It is backordered; once opened it can't be returned [1].", "citations": [1]}])
    ans = a.ask("Is ELC-3030 in stock, and can the dealer return it once opened?")
    _, user = client.prompts[0]
    assert "LIVE DATA" in user and "ELC-3030" in user and "Backordered" in user
    assert "Estimated restock: 2026-10-21" in ans.text      # template block, exact


def test_catalog_row_context_includes_linked_notes(retriever, tmp_path):
    a, client = make_assistant(retriever, tmp_path, [{"status": "answered", "answer": "x [1]", "citations": [1]}])
    a.ask("Is ELC-3030 in stock, and can the dealer return it once opened?")
    _, user = client.prompts[0]
    assert "Special-Order Notice" in user


# ---------------------------------------------------------------- resilience and config
def test_llm_failure_falls_back_to_passages(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [{}], fail=True)
    ans = a.ask("How long is the standard warranty on a new vehicle?")
    assert ans.status == "fallback" and ans.citations
    assert "AI answer unavailable" in ans.note


def test_status_and_tokens_are_logged(retriever, tmp_path):
    a, _ = make_assistant(retriever, tmp_path, [{"status": "answered", "answer": "Yes [1].", "citations": [1]}])
    a.ask("Are brake pads covered under the standard warranty?")
    entry = json.loads((tmp_path / "logs" / "interactions.jsonl").read_text().splitlines()[-1])
    assert entry["status"] == "answered" and entry["tokens"] == {"in": 100, "out": 20} and entry["cited"]


def test_make_generator_respects_config(monkeypatch):
    assert isinstance(make_generator("stub"), StubGenerator)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert isinstance(make_generator("llm"), StubGenerator)        # no secrets -> safe fallback
    with pytest.raises(ValueError):
        make_generator("magic")


def test_prompt_limits_rules_to_the_items_they_name():
    """Regression for O03: the model extended a two-plan rule to a third plan."""
    assert "applies ONLY to the items it names" in SYSTEM_PROMPT


def test_test_set_covers_two_conflicts_and_scope_inference():
    import csv
    rows = {r["query_id"]: r for r in csv.DictReader(open(Path(__file__).resolve().parent.parent / "eval/test_set.csv"))}
    assert sum(r["expected_behavior"] == "flag_conflict" for r in rows.values()) == 2
    assert {"S16", "S17", "O03"} <= rows.keys()


def test_prompt_requires_status_to_match_answer():
    """Regression for O03: a correct refusal was labeled 'answered'."""
    assert "The status must match your answer" in SYSTEM_PROMPT
    # Regression for H03: a correct answer with a caveat must stay "answered".
    assert "MAIN" in SYSTEM_PROMPT and "only a detail is" in SYSTEM_PROMPT


def test_judge_accepts_rules_applied_to_the_case():
    """Regression for S03: the judge rejected a correct general-rule answer."""
    from eval.run_answers import JUDGE_PROMPT
    assert "applied to the case" in JUDGE_PROMPT
