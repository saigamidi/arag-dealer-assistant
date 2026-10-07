"""
Tests for embeddings, vector search and hybrid search.
Uses an offline stand-in embedder and a fake Azure client: no cloud calls, no cost.
Run from the project root with:  pytest -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import ingest as ing
from src.embeddings import AzureEmbedder, EmbeddingStore, HashEmbedder, cosine, normalise
from src.retrieve import (BM25Retriever, HybridRetriever, ResilientVectorRetriever,
                          VectorRetriever, make_retriever)

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    if not KB_DIR.exists():
        pytest.skip("knowledge_base folder not found")
    out = tmp_path_factory.mktemp("data")
    ing.ingest(KB_DIR, out)
    return [json.loads(l) for l in (out / "chunks.jsonl").read_text().splitlines() if l.strip()]


@pytest.fixture
def store(tmp_path, chunks):
    s = EmbeddingStore(tmp_path / "embeddings.json")
    s.sync(chunks, HashEmbedder())
    return s


# ---------------------------------------------------------------- vector maths
def test_normalised_vectors_have_cosine_one_with_themselves():
    v = normalise([3.0, 4.0])
    assert abs(cosine(v, v) - 1.0) < 1e-9


# ---------------------------------------------------------------- store and change detection
def test_first_sync_embeds_every_chunk(tmp_path, chunks):
    report = EmbeddingStore(tmp_path / "e.json").sync(chunks, HashEmbedder())
    assert report["embedded"] == len(chunks) == report["total"]


def test_second_sync_embeds_nothing(store, chunks):
    report = store.sync(chunks, HashEmbedder())
    assert report["embedded"] == 0 and report["unchanged"] == len(chunks)


def test_store_persists_and_reloads(store, chunks):
    reloaded = EmbeddingStore(store.path)
    assert reloaded.model == "hash:256"
    assert reloaded.vector(chunks[0]["chunk_id"]) is not None


def test_changed_chunk_is_re_embedded_and_deleted_chunk_dropped(store, chunks):
    edited = [dict(c) for c in chunks[1:]]          # first chunk deleted
    edited[0]["content_hash"] = "changed"            # second chunk edited
    report = store.sync(edited, HashEmbedder())
    assert report["embedded"] == 1 and report["removed"] == 1
    assert store.vector(chunks[0]["chunk_id"]) is None


def test_changing_model_rebuilds_everything(store, chunks):
    report = store.sync(chunks, HashEmbedder(dims=128))
    assert report["rebuild"] is True and report["embedded"] == len(chunks)


# ---------------------------------------------------------------- Azure embedder (fake client)
class FakeAzureClient:
    def __init__(self):
        self.calls = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.calls.append(len(input))
        data = [SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=10 * len(input)))


def test_azure_embedder_batches_and_counts_tokens(tmp_path):
    client = FakeAzureClient()
    emb = AzureEmbedder("arag-embed", client=client, cache_path=tmp_path / "cache.json")
    vecs = emb.embed([f"text {i}" for i in range(130)])
    assert len(vecs) == 130
    assert client.calls == [64, 64, 2]               # batched, not one call per chunk
    assert emb.tokens_used == 1300


def test_query_cache_avoids_paying_twice(tmp_path):
    client = FakeAzureClient()
    emb = AzureEmbedder("arag-embed", client=client, cache_path=tmp_path / "cache.json")
    emb.embed_query("restocking fee")
    emb.embed_query("restocking fee")
    assert client.calls == [1]
    # A new embedder instance reads the cache from disk.
    emb2 = AzureEmbedder("arag-embed", client=client, cache_path=tmp_path / "cache.json")
    emb2.embed_query("restocking fee")
    assert client.calls == [1]


def test_missing_secrets_raise_clear_error(monkeypatch):
    for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_EMBED_DEPLOYMENT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="secrets"):
        AzureEmbedder()


# ---------------------------------------------------------------- retrievers
def test_vector_retriever_finds_matching_chunk(store, chunks):
    vr = VectorRetriever(chunks, store, HashEmbedder())
    top = vr.search("restocking fee new vehicle returns", k=3)
    assert "return_policy::new-vehicle-returns" in [r.chunk_id for r in top]
    assert top[0].score >= top[-1].score


def test_vector_retriever_rejects_mismatched_model(store, chunks):
    with pytest.raises(ValueError, match="rebuild"):
        VectorRetriever(chunks, store, HashEmbedder(dims=64))


def test_vector_retriever_requires_all_embeddings(tmp_path, chunks):
    partial = EmbeddingStore(tmp_path / "p.json")
    partial.sync(chunks[:10], HashEmbedder())
    with pytest.raises(ValueError, match="no embedding"):
        VectorRetriever(chunks, partial, HashEmbedder())


def test_hybrid_keeps_exact_part_id_on_top(store, chunks):
    hybrid = HybridRetriever(BM25Retriever(chunks), VectorRetriever(chunks, store, HashEmbedder()))
    assert hybrid.search("BRK-1021", k=1)[0].chunk_id == "parts_catalog_brakes::BRK-1021"


def test_hybrid_falls_back_to_keyword_when_embeddings_fail(store, chunks):
    class Broken(HashEmbedder):
        def embed_query(self, text):
            raise ConnectionError("Azure down")
    hybrid = HybridRetriever(BM25Retriever(chunks), VectorRetriever(chunks, store, Broken()))
    results = hybrid.search("warranty claim deadline", k=3)
    assert results and "keyword search only" in hybrid.last_fallback
    assert [r.chunk_id for r in results] == [r.chunk_id for r in hybrid.bm25.search("warranty claim deadline", k=3)]


def test_make_retriever_falls_back_to_bm25_without_secrets(chunks, monkeypatch, capsys):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    r = make_retriever("hybrid", chunks=chunks)
    assert isinstance(r, BM25Retriever)
    assert "using BM25 only" in capsys.readouterr().out


def test_make_retriever_builds_hybrid_when_ready(store, chunks):
    r = make_retriever("hybrid", chunks=chunks, embedder=HashEmbedder(), store=store)
    assert isinstance(r, HybridRetriever)


def test_make_retriever_defaults_to_vector(store, chunks, monkeypatch):
    monkeypatch.delenv("RETRIEVER", raising=False)
    r = make_retriever(chunks=chunks, embedder=HashEmbedder(), store=store)
    assert isinstance(r, ResilientVectorRetriever)


def test_retriever_env_var_selects_mode(store, chunks, monkeypatch):
    monkeypatch.setenv("RETRIEVER", "hybrid")
    assert isinstance(make_retriever(chunks=chunks, embedder=HashEmbedder(), store=store), HybridRetriever)
    monkeypatch.setenv("RETRIEVER", "bm25")
    assert isinstance(make_retriever(chunks=chunks), BM25Retriever)


def test_unknown_retriever_is_rejected(chunks):
    with pytest.raises(ValueError, match="Unknown RETRIEVER"):
        make_retriever("magic", chunks=chunks)


def test_vector_mode_falls_back_to_keyword_on_failure(store, chunks):
    class Broken(HashEmbedder):
        def embed_query(self, text):
            raise ConnectionError("Azure down")
    r = ResilientVectorRetriever(VectorRetriever(chunks, store, Broken()), BM25Retriever(chunks))
    assert r.search("warranty claim deadline", k=3)
    assert "keyword search only" in r.last_fallback


def test_dotenv_file_is_loaded_without_overriding_env(tmp_path, monkeypatch):
    from dotenv import load_dotenv
    env = tmp_path / ".env"
    env.write_text("RETRIEVER=bm25\nARAG_TEST_ONLY=from_file\n")
    monkeypatch.setenv("RETRIEVER", "hybrid")          # e.g. a Codespaces secret
    monkeypatch.delenv("ARAG_TEST_ONLY", raising=False)
    load_dotenv(env, override=False)
    import os
    assert os.environ["RETRIEVER"] == "hybrid"        # real env var wins
    assert os.environ["ARAG_TEST_ONLY"] == "from_file"
    monkeypatch.delenv("ARAG_TEST_ONLY")
