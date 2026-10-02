"""
Tests for BM25 retrieval and rank fusion.
Run from the project root with:  pytest -v
"""

import json
from pathlib import Path

import pytest

from src import ingest as ing
from src.retrieve import (BM25Retriever, SearchResult, expand_query,
                          reciprocal_rank_fusion, tokenize)

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    if not KB_DIR.exists():
        pytest.skip("knowledge_base folder not found")
    out = tmp_path_factory.mktemp("data")
    ing.ingest(KB_DIR, out)
    chunks = [json.loads(l) for l in (out / "chunks.jsonl").read_text().splitlines() if l.strip()]
    return BM25Retriever(chunks)


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_keeps_ids_whole():
    tokens = tokenize("Is BRK-1020 in stock? Order ORD-10482 too.")
    assert "brk-1020" in tokens and "ord-10482" in tokens
    assert "1020" not in tokens


def test_tokenizer_drops_stopwords_and_stems_plurals():
    assert tokenize("What are the filters for the SUV") == ["filter", "suv"]


def test_glossary_expands_system_names():
    assert "warranty claims system" in expand_query("Where is the claims portal?")


# ---------------------------------------------------------------- search
def test_exact_part_id_ranks_first(retriever):
    results = retriever.search("price of BRK-1020", k=3)
    assert results[0].chunk_id == "parts_catalog_brakes::BRK-1020"


def test_left_and_right_calipers_are_not_confused(retriever):
    top = retriever.search("BRK-1021", k=1)[0]
    assert top.chunk_id == "parts_catalog_brakes::BRK-1021"


def test_policy_question_finds_policy_section(retriever):
    ids = [r.chunk_id for r in retriever.search("deadline to submit a warranty claim", k=3)]
    assert "warranty_policy::claim-process" in ids


def test_results_are_ranked_and_limited(retriever):
    results = retriever.search("brake warranty", k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score


def test_empty_or_stopword_only_query_returns_nothing(retriever):
    assert retriever.search("", k=3) == []
    assert retriever.search("what is the", k=3) == []


def test_expand_related_adds_notes_for_catalog_rows(retriever):
    results = retriever.search("ELC-3030", k=1)
    context_ids = [c["chunk_id"] for c in retriever.expand_related(results)]
    assert context_ids[0] == "parts_catalog_electrical::ELC-3030"
    assert "parts_catalog_electrical::special-order-notice" in context_ids


# ---------------------------------------------------------------- rank fusion
def _results(ids):
    return [SearchResult({"chunk_id": cid}, 1.0, i + 1) for i, cid in enumerate(ids)]


def test_rrf_rewards_agreement_between_retrievers():
    keyword = _results(["a", "b", "c"])
    vector = _results(["d", "b", "e"])
    fused = [r.chunk_id for r in reciprocal_rank_fusion([keyword, vector], k=5)]
    assert fused[0] == "b"                     # 2nd in both lists beats 1st in only one
    assert set(fused) == {"a", "b", "c", "d", "e"}


def test_rrf_respects_k():
    fused = reciprocal_rank_fusion([_results(["a", "b", "c", "d"])], k=2)
    assert [r.chunk_id for r in fused] == ["a", "b"]
