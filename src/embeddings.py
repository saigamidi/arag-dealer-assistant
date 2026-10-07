"""
ARAG embeddings: turn each chunk into a vector (a list of numbers that
captures its meaning), and keep those vectors in sync with the chunks.

Embedders (same interface: embed(list_of_texts) -> list_of_vectors)
  * AzureEmbedder - Azure OpenAI deployment (text-embedding-3-small, 1536 numbers)
  * HashEmbedder  - free, offline stand-in used by the tests. It only matches
                    shared words, so it is NOT a real meaning-based model.

EmbeddingStore (data/embeddings.json)
  * One vector per chunk, saved with the chunk's content hash.
  * On each run, only new or changed chunks are embedded; vectors for deleted
    chunks are dropped. Re-running with nothing changed costs nothing.
  * Changing the embedding model rebuilds everything (vectors from different
    models can't be compared).
  * Commit the file: a fresh codespace then never needs to pay to re-embed.

Usage (from the project root, after `python -m src.ingest`):
    python -m src.embeddings              # embed new/changed chunks
    python -m src.embeddings --rebuild    # re-embed everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Protocol

EMBEDDINGS_PATH = Path("data/embeddings.json")
QUERY_CACHE_PATH = Path("data/cache/query_embeddings.json")
BATCH_SIZE = 64
# Approximate list price for text-embedding-3-small, USD per 1M tokens.
# Only used to print an estimate; check Azure pricing for your region.
PRICE_PER_MILLION_TOKENS = float(os.getenv("EMBED_PRICE_PER_M", "0.02"))


class Embedder(Protocol):
    name: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------
# Azure OpenAI embedder
# --------------------------------------------------------------------------
class AzureEmbedder:
    def __init__(self, deployment: str | None = None, client=None, cache_path: Path | None = QUERY_CACHE_PATH):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")
        if client is None:
            from openai import AzureOpenAI
            endpoint, key = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")
            if not (endpoint and key and self.deployment):
                raise RuntimeError("Azure OpenAI secrets are missing (see scripts/check_azure.py).")
            client = AzureOpenAI(azure_endpoint=endpoint, api_key=key,
                                 api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                                 timeout=10, max_retries=2)
        self.client = client
        self.name = f"azure:{self.deployment}"
        self.tokens_used = 0
        # Small disk cache for query embeddings, so re-running the evaluation
        # doesn't pay for the same questions twice.
        self.cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.name}|{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [self._cache.get(self._key(t)) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            resp = self.client.embeddings.create(model=self.deployment, input=[texts[i] for i in batch])
            self.tokens_used += getattr(resp.usage, "total_tokens", 0) or 0
            for i, item in zip(batch, resp.data):
                out[i] = item.embedding
        return [v for v in out]  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if key not in self._cache:
            self._cache[key] = self.embed([text])[0]
            self._save_cache()
        return self._cache[key]

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")


# --------------------------------------------------------------------------
# Free offline stand-in (tests only)
# --------------------------------------------------------------------------
class HashEmbedder:
    """Hashes words into a fixed-size vector. Deterministic, free, offline.
    Matches shared words only, so it's a test double, not a semantic model."""

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.name = f"hash:{dims}"
        self.tokens_used = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for w in re.findall(r"[a-z0-9-]+", text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dims] += 1.0 if (h >> 8) % 2 else -1.0
        return v

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# --------------------------------------------------------------------------
# Vector maths
# --------------------------------------------------------------------------
def normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two already-normalised vectors (1 = same meaning)."""
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Store with change detection
# --------------------------------------------------------------------------
class EmbeddingStore:
    def __init__(self, path: Path = EMBEDDINGS_PATH):
        self.path = path
        self.model: str | None = None
        self.vectors: dict[str, dict] = {}       # chunk_id -> {"hash", "vector"}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.model = data.get("model")
            self.vectors = data.get("vectors", {})

    def vector(self, chunk_id: str) -> list[float] | None:
        rec = self.vectors.get(chunk_id)
        return rec["vector"] if rec else None

    def sync(self, chunks: list[dict], embedder: Embedder, rebuild: bool = False) -> dict:
        """Embed new/changed chunks, drop deleted ones, save. Returns a report."""
        if rebuild or self.model != embedder.name:
            rebuild = True
            self.vectors = {}
        current = {c["chunk_id"]: c for c in chunks}
        todo = [c for cid, c in current.items()
                if self.vectors.get(cid, {}).get("hash") != c["content_hash"]]
        removed = sorted(set(self.vectors) - set(current))
        for cid in removed:
            del self.vectors[cid]

        before = getattr(embedder, "tokens_used", 0)
        if todo:
            vecs = embedder.embed([c["text"] for c in todo])
            for c, v in zip(todo, vecs):
                self.vectors[c["chunk_id"]] = {"hash": c["content_hash"],
                                               "vector": [round(x, 6) for x in normalise(v)]}
        self.model = embedder.name
        self.save()
        tokens = getattr(embedder, "tokens_used", 0) - before
        return {"rebuild": rebuild, "model": self.model, "embedded": len(todo),
                "removed": len(removed), "unchanged": len(current) - len(todo),
                "total": len(self.vectors), "tokens": tokens,
                "estimated_cost_usd": round(tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS, 6)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"model": self.model, "vectors": self.vectors}), encoding="utf-8")


def default_embedder() -> Embedder:
    return AzureEmbedder()


if __name__ == "__main__":
    from src.retrieve import load_chunks
    p = argparse.ArgumentParser(description="Embed the ARAG chunks.")
    p.add_argument("--rebuild", action="store_true", help="Re-embed every chunk")
    a = p.parse_args()
    r = EmbeddingStore().sync(load_chunks(), default_embedder(), rebuild=a.rebuild)
    print(f"\n{'Full rebuild' if r['rebuild'] else 'Incremental update'} with {r['model']}")
    print(f"  Embedded: {r['embedded']}, unchanged: {r['unchanged']}, removed: {r['removed']}, total: {r['total']}")
    print(f"  Tokens: {r['tokens']:,}  (~${r['estimated_cost_usd']:.6f} at ${PRICE_PER_MILLION_TOKENS}/1M tokens)\n")
