"""
Inspect chunks by eye after ingestion.

Usage (from the project root):
    python -m src.inspect_chunks                      # summary by document
    python -m src.inspect_chunks --doc return_policy  # every chunk of one document
    python -m src.inspect_chunks --id parts_catalog_brakes::BRK-1020
    python -m src.inspect_chunks --type catalog_row   # all catalog rows
    python -m src.inspect_chunks --search floor       # chunks containing a word
"""

import argparse
import json
from collections import Counter
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise SystemExit("No chunks found. Run `python -m src.ingest` first.")
    return [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def show(chunk: dict) -> None:
    print("=" * 72)
    print(f"ID:      {chunk['chunk_id']}")
    print(f"Type:    {chunk['chunk_type']}   Domain: {chunk['domain']}   "
          f"Words: {len(chunk['text'].split())}")
    if chunk["related_chunk_ids"]:
        print(f"Related: {', '.join(chunk['related_chunk_ids'])}")
    print("-" * 72)
    print(chunk["text"])
    print()


def summary(chunks: list[dict]) -> None:
    by_doc = Counter(c["doc_id"] for c in chunks)
    by_type = Counter(c["chunk_type"] for c in chunks)
    words = [len(c["text"].split()) for c in chunks]
    print(f"\n{len(chunks)} chunks from {len(by_doc)} documents "
          f"({by_type['section']} sections, {by_type['catalog_row']} catalog rows)")
    print(f"Chunk size (words): min {min(words)}, avg {sum(words) // len(words)}, max {max(words)}\n")
    print(f"{'Document':<32}{'Domain':<22}Chunks")
    for doc, n in sorted(by_doc.items()):
        domain = next(c["domain"] for c in chunks if c["doc_id"] == doc)
        print(f"{doc:<32}{domain:<22}{n}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect ARAG chunks.")
    p.add_argument("--doc", help="Show all chunks from one document (file name without .md)")
    p.add_argument("--id", help="Show one chunk by ID")
    p.add_argument("--type", choices=["section", "catalog_row"], help="Show chunks of one type")
    p.add_argument("--search", help="Show chunks whose text contains this word (case-insensitive)")
    a = p.parse_args()

    chunks = load_chunks()
    selected = chunks
    if a.doc:
        selected = [c for c in selected if c["doc_id"] == a.doc]
    if a.id:
        selected = [c for c in selected if c["chunk_id"] == a.id]
    if a.type:
        selected = [c for c in selected if c["chunk_type"] == a.type]
    if a.search:
        selected = [c for c in selected if a.search.lower() in c["text"].lower()]

    if not any([a.doc, a.id, a.type, a.search]):
        summary(chunks)
    elif not selected:
        print("No matching chunks.")
    else:
        for c in selected:
            show(c)
        print(f"{len(selected)} chunk(s) shown.")
