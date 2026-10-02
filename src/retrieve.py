"""
ARAG retrieval: find the most relevant chunks for a question.

v1 (now, free):   BM25Retriever   - keyword search, strong on exact part IDs
v2 (cloud, later): VectorRetriever - semantic search using cloud embeddings
                   HybridRetriever - combines both with Reciprocal Rank Fusion

Every retriever has the same interface:  search(query, k) -> list[SearchResult]
so the rest of the system doesn't care which one is plugged in.

Try it (from the project root):
    python -m src.retrieve "Which front brake pads fit the Sedan EX?"
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/chunks.jsonl")

# Part and order IDs are kept as single tokens ("brk-1020"), so BM25 matches
# them exactly instead of splitting them into "brk" and "1020".
ID_PATTERN = re.compile(r"\b[a-z]{3}-\d{4,5}\b")
WORD_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "my", "of", "on", "or",
    "our", "the", "their", "this", "to", "we", "what", "when", "where", "which", "who",
    "will", "with", "you", "your", "any", "there", "that", "should", "would", "me", "they",
}

# Glossary from the PRD content audit: one system appears under several names,
# plus UK/US spelling. Kept deliberately small and NOT tuned to the test set,
# so evaluation shows honestly where keyword search falls short.
GLOSSARY = {
    "claims portal": "warranty claims system service portal",
    "warranty claims system": "claims portal service portal",
    "service portal": "claims portal warranty claims system",
    "tyre": "tire",
    "tyres": "tires",
}


def light_stem(word: str) -> str:
    """Very light stemming: plural -> singular (filters -> filter)."""
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    text = text.lower()
    ids = ID_PATTERN.findall(text)
    rest = ID_PATTERN.sub(" ", text)
    words = [light_stem(w) for w in WORD_PATTERN.findall(rest) if w not in STOPWORDS]
    return ids + words


def expand_query(query: str) -> str:
    q = query.lower()
    extra = [synonyms for term, synonyms in GLOSSARY.items() if term in q]
    return query + (" " + " ".join(extra) if extra else "")


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.ingest` first.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Common interface
# --------------------------------------------------------------------------
@dataclass
class SearchResult:
    chunk: dict
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk["chunk_id"]


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[SearchResult]: ...


# --------------------------------------------------------------------------
# BM25 keyword retriever
# --------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, chunks: list[dict] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        tokens = tokenize(expand_query(query))
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(self.chunks[i], float(scores[i]), r + 1)
                for r, i in enumerate(ranked[:k]) if scores[i] > 0]

    def expand_related(self, results: list[SearchResult]) -> list[dict]:
        """
        Small-to-big: add the notes sections linked to any catalog row in the
        results (e.g. ELC-3030 -> Special-Order Notice), without duplicates.
        Used when building the LLM's context, not for ranking.
        """
        seen, context = set(), []
        for r in results:
            for cid in [r.chunk_id] + r.chunk.get("related_chunk_ids", []):
                if cid not in seen and cid in self.by_id:
                    seen.add(cid)
                    context.append(self.by_id[cid])
        return context


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion (used by the hybrid retriever in the cloud phase)
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], k: int = 3,
                           rrf_k: int = 60) -> list[SearchResult]:
    """
    Combine rankings from several retrievers. Each chunk scores
    sum(1 / (rrf_k + rank)) across lists, so chunks ranked well by more than
    one retriever rise to the top. Uses ranks, not raw scores, because BM25
    and vector scores are on different scales.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for results in result_lists:
        for r in results:
            fused[r.chunk_id] = fused.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + r.rank)
            chunks[r.chunk_id] = r.chunk
    ordered = sorted(fused, key=fused.get, reverse=True)[:k]
    return [SearchResult(chunks[cid], fused[cid], i + 1) for i, cid in enumerate(ordered)]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which front brake pads fit the Sedan EX?"
    retriever = BM25Retriever()
    print(f"\nQuery: {question}\nTokens: {tokenize(expand_query(question))}\n")
    for r in retriever.search(question, k=3):
        first_line = r.chunk["text"].split("\n\n", 1)[-1].splitlines()[0][:80]
        print(f"#{r.rank}  score {r.score:6.2f}  {r.chunk_id}\n     {first_line}")
    print()
