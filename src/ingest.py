"""
ARAG ingestion: load the knowledge base, chunk it, and keep chunks in sync.

Chunking strategy (see PRD > Key Decisions):
  * Policy / process documents -> one chunk per "## " section.
    Text before the first section becomes an "Overview" chunk.
  * Catalog tables (first column "Part ID") -> one chunk per row, with the
    column headers repeated so each part keeps its compatibility and price.
  * Every chunk starts with a short context header (document + section),
    which helps both keyword search and, later, embeddings.

Change detection (see PRD > Risks: stale knowledge base):
  * A manifest stores a fingerprint (SHA-256 hash) of every document.
  * On each run, only new or changed documents are re-chunked; chunks of
    deleted documents are removed; unchanged documents are skipped.
  * If CHUNKER_VERSION changes (i.e. the chunking logic changed), every
    document is rebuilt automatically.
  * The run's changes are written to last_run.json so the embedding step
    (cloud, later) only re-embeds chunks that actually changed.

Usage (from the project root):
    python -m src.ingest              # incremental update
    python -m src.ingest --rebuild    # force a full rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Bump this whenever the chunking logic changes, to force a full rebuild.
CHUNKER_VERSION = "1.0"

KB_DIR = Path("knowledge_base")
OUT_DIR = Path("data")

# Domains from the PRD's Knowledge Base & Data section.
DOMAINS = {
    "warranty_policy": "warranty_coverage",
    "extended_warranty_options": "warranty_coverage",
    "recall_procedures": "warranty_coverage",
    "return_policy": "commercial_policies",
    "trade_in_policy": "commercial_policies",
    "financing_incentives": "commercial_policies",
    "parts_catalog_brakes": "parts_catalog",
    "parts_catalog_electrical": "parts_catalog",
    "parts_catalog_filters_fluids": "parts_catalog",
    "accessories_catalog": "parts_catalog",
    "service_scheduling": "service_operations",
    "customer_complaints_process": "service_operations",
    "dealer_onboarding": "dealer_operations",
    "dealer_portal_access": "dealer_operations",
    "inventory_management": "dealer_operations",
    "inventory_lookup_faq": "realtime_boundary",
    "order_status_faq": "realtime_boundary",
}


@dataclass
class Chunk:
    chunk_id: str               # stable ID, e.g. "return_policy::parts-returns" or "parts_catalog_brakes::BRK-1020"
    doc_id: str                 # file name without extension
    source_file: str
    title: str                  # document title (the "# " heading)
    section: str                # section heading, or the part ID for catalog rows
    domain: str
    chunk_type: str             # "section" or "catalog_row"
    text: str                   # what gets searched (and later embedded)
    part_id: str | None = None
    related_chunk_ids: list[str] = field(default_factory=list)  # for "small-to-big" retrieval
    content_hash: str = ""


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(section_heading, section_body), ...])."""
    title_match = re.search(r"^# (.+)$", markdown, re.M)
    title = title_match.group(1).strip() if title_match else "Untitled"
    body = markdown[title_match.end():] if title_match else markdown

    parts = re.split(r"^## (.+)$", body, flags=re.M)
    sections = []
    preamble = parts[0].strip()
    if preamble:
        sections.append(("Overview", preamble))
    for heading, content in zip(parts[1::2], parts[2::2]):
        sections.append((heading.strip(), content.strip()))
    return title, sections


def parse_table(block: str) -> tuple[list[str], list[list[str]], str]:
    """
    Find the first Markdown table in a block.
    Return (headers, rows, remaining_text_without_the_table).
    """
    lines = block.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), None)
    if start is None:
        return [], [], block
    end = start
    while end < len(lines) and lines[end].strip().startswith("|"):
        end += 1
    table_lines = [l.strip() for l in lines[start:end]]

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    headers = cells(table_lines[0])
    rows = [cells(l) for l in table_lines[2:]]          # skip the |---| divider
    remaining = "\n".join(lines[:start] + lines[end:]).strip()
    return headers, rows, remaining


def is_catalog_table(headers: list[str]) -> bool:
    return bool(headers) and headers[0].lower() == "part id"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_document(path: Path) -> list[Chunk]:
    doc_id = path.stem
    # Strip Markdown bold markers so they don't clutter search or answers.
    markdown = path.read_text(encoding="utf-8").replace("**", "")
    title, sections = split_sections(markdown)
    domain = DOMAINS.get(doc_id, "general")

    section_chunks: list[Chunk] = []
    row_chunks: list[Chunk] = []

    for heading, content in sections:
        headers, rows, remaining = parse_table(content)

        if is_catalog_table(headers):
            # One chunk per part, headers repeated as "Header: value" lines.
            for row in rows:
                part_id = row[0]
                fields = "\n".join(f"{h}: {v}" for h, v in zip(headers, row))
                row_chunks.append(Chunk(
                    chunk_id=f"{doc_id}::{part_id}",
                    doc_id=doc_id, source_file=path.name, title=title,
                    section=part_id, domain=domain, chunk_type="catalog_row",
                    part_id=part_id,
                    text=f"Document: {title}\nPart: {part_id}\n\n{fields}",
                ))
            content = remaining          # any prose around the table stays a section
            if not content:
                continue

        section_chunks.append(Chunk(
            chunk_id=f"{doc_id}::{slugify(heading)}",
            doc_id=doc_id, source_file=path.name, title=title,
            section=heading, domain=domain, chunk_type="section",
            text=f"Document: {title}\nSection: {heading}\n\n{content}",
        ))

    # Small-to-big links: a catalog row points to the notes sections of its
    # document (e.g. ELC-3030 -> "Special-Order Notice"), so context lost by
    # row-level chunking can be pulled back in at answer time.
    section_ids = [c.chunk_id for c in section_chunks]
    for row in row_chunks:
        row.related_chunk_ids = section_ids

    chunks = section_chunks + row_chunks
    for c in chunks:
        c.content_hash = sha256(c.text)
    return chunks


# --------------------------------------------------------------------------
# Incremental ingestion
# --------------------------------------------------------------------------
def load_previous(out_dir: Path) -> tuple[dict, dict[str, dict]]:
    manifest_path, chunks_path = out_dir / "manifest.json", out_dir / "chunks.jsonl"
    if not (manifest_path.exists() and chunks_path.exists()):
        return {}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return manifest, chunks


def ingest(kb_dir: Path = KB_DIR, out_dir: Path = OUT_DIR, rebuild: bool = False) -> dict:
    """Chunk the knowledge base and write chunks.jsonl, manifest.json, last_run.json."""
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_dir.resolve()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, old_chunks = load_previous(out_dir)
    if manifest.get("chunker_version") != CHUNKER_VERSION:
        rebuild = True
    old_docs: dict = {} if rebuild else manifest.get("documents", {})
    if rebuild:
        old_chunks = {}

    current_files = {p.stem: p for p in sorted(kb_dir.glob("*.md"))}
    new_docs: dict[str, dict] = {}
    all_chunks: dict[str, dict] = {}
    report = {"unchanged": [], "added": [], "updated": [], "removed": [],
              "chunks_added": [], "chunks_updated": [], "chunks_removed": []}

    for doc_id, path in current_files.items():
        doc_hash = sha256(path.read_text(encoding="utf-8"))
        previous = old_docs.get(doc_id)

        if previous and previous["hash"] == doc_hash:
            # Unchanged: carry the existing chunks forward untouched.
            report["unchanged"].append(doc_id)
            for cid in previous["chunk_ids"]:
                all_chunks[cid] = old_chunks[cid]
            new_docs[doc_id] = previous
            continue

        chunks = chunk_document(path)
        new_ids = {c.chunk_id for c in chunks}
        old_ids = set(previous["chunk_ids"]) if previous else set()

        for c in chunks:
            all_chunks[c.chunk_id] = asdict(c)
            if c.chunk_id not in old_ids:
                report["chunks_added"].append(c.chunk_id)
            elif old_chunks.get(c.chunk_id, {}).get("content_hash") != c.content_hash:
                report["chunks_updated"].append(c.chunk_id)
        report["chunks_removed"] += sorted(old_ids - new_ids)
        report["updated" if previous else "added"].append(doc_id)
        new_docs[doc_id] = {"hash": doc_hash, "chunk_ids": [c.chunk_id for c in chunks]}

    # Documents that disappeared: drop all their chunks.
    for doc_id in sorted(set(old_docs) - set(current_files)):
        report["removed"].append(doc_id)
        report["chunks_removed"] += old_docs[doc_id]["chunk_ids"]

    unknown = [d for d in current_files if d not in DOMAINS]

    # Write outputs.
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"chunker_version": CHUNKER_VERSION, "updated_at": timestamp, "documents": new_docs},
        indent=2), encoding="utf-8")
    report.update({"rebuild": rebuild, "total_documents": len(new_docs),
                   "total_chunks": len(all_chunks), "unknown_domain_docs": unknown,
                   "run_at": timestamp})
    (out_dir / "last_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def check_freshness(kb_dir: Path = KB_DIR, out_dir: Path = OUT_DIR) -> dict:
    """
    Compare the knowledge base on disk with the last ingest, without changing anything.
    Used at app startup so a stale index is visible instead of silent (PRD risk K2).
    Also checks embeddings: chunks whose vectors are missing or out of date.
    Returns {"stale": bool, "changed": [...], "added": [...], "removed": [...], "embeddings_stale": int}.
    """
    manifest_path = out_dir / "manifest.json"
    result = {"stale": False, "changed": [], "added": [], "removed": [], "embeddings_stale": 0,
              "never_ingested": not manifest_path.exists()}
    if result["never_ingested"]:
        result["stale"] = True
        return result
    docs = json.loads(manifest_path.read_text(encoding="utf-8")).get("documents", {})
    current = {p.stem: p for p in kb_dir.glob("*.md")} if kb_dir.exists() else {}
    for doc_id, path in current.items():
        if doc_id not in docs:
            result["added"].append(doc_id)
        elif docs[doc_id]["hash"] != sha256(path.read_text(encoding="utf-8")):
            result["changed"].append(doc_id)
    result["removed"] = sorted(set(docs) - set(current))

    emb_path, chunks_path = out_dir / "embeddings.json", out_dir / "chunks.jsonl"
    if emb_path.exists() and chunks_path.exists():
        vectors = json.loads(emb_path.read_text(encoding="utf-8")).get("vectors", {})
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                if vectors.get(c["chunk_id"], {}).get("hash") != c["content_hash"]:
                    result["embeddings_stale"] += 1
    result["stale"] = bool(result["changed"] or result["added"] or result["removed"]
                           or result["embeddings_stale"])
    return result


def print_summary(report: dict) -> None:
    mode = "Full rebuild" if report["rebuild"] else "Incremental update"
    print(f"\n{mode} complete: {report['total_documents']} documents, {report['total_chunks']} chunks")
    print(f"  Documents  -> unchanged: {len(report['unchanged'])}, added: {len(report['added'])}, "
          f"updated: {len(report['updated'])}, removed: {len(report['removed'])}")
    print(f"  Chunks     -> added: {len(report['chunks_added'])}, updated: {len(report['chunks_updated'])}, "
          f"removed: {len(report['chunks_removed'])}")
    for label in ("added", "updated", "removed"):
        if report[label]:
            print(f"  {label.capitalize()} documents: {', '.join(report[label])}")
    if report["unknown_domain_docs"]:
        print(f"  WARNING: no domain mapping for {', '.join(report['unknown_domain_docs'])} "
              f"(add them to DOMAINS in src/ingest.py)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk the ARAG knowledge base.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore the manifest and rebuild everything")
    args = parser.parse_args()
    print_summary(ingest(rebuild=args.rebuild))
