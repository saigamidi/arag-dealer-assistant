"""
Tests for ingestion, chunking and change detection.
Run from the project root with:  pytest -v
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from src import ingest as ing

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
pytestmark = pytest.mark.skipif(not KB_DIR.exists(), reason="knowledge_base folder not found")


@pytest.fixture
def workspace(tmp_path):
    """A private copy of the knowledge base, so tests never touch the real data/ folder."""
    kb = tmp_path / "knowledge_base"
    shutil.copytree(KB_DIR, kb)
    return kb, tmp_path / "data"


def read_chunks(out_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (out_dir / "chunks.jsonl").read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- chunking
def test_full_ingest_counts(workspace):
    kb, out = workspace
    report = ing.ingest(kb, out)
    chunks = read_chunks(out)
    assert report["total_documents"] == 17
    assert sum(c["chunk_type"] == "catalog_row" for c in chunks) == 29
    assert report["unknown_domain_docs"] == []


def test_every_section_heading_becomes_a_chunk(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    sections = {(c["doc_id"], c["section"]) for c in read_chunks(out) if c["chunk_type"] == "section"}
    for path in kb.glob("*.md"):
        for heading in re.findall(r"^## (.+)$", path.read_text(), re.M):
            assert (path.stem, heading.strip()) in sections, f"missing {path.stem} > {heading}"


def test_catalog_row_keeps_headers_compatibility_and_price(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    row = next(c for c in read_chunks(out) if c["chunk_id"] == "parts_catalog_brakes::BRK-1020")
    assert row["part_id"] == "BRK-1020"
    assert "Compatible Models: SUV Sport, SUV Touring" in row["text"]
    assert "List Price (USD): $142.00" in row["text"]
    assert "BRK-1021" not in row["text"]          # neighbouring row must not leak in


def test_catalog_rows_link_to_notes_sections(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    row = next(c for c in read_chunks(out) if c["chunk_id"] == "parts_catalog_electrical::ELC-3030")
    assert "parts_catalog_electrical::special-order-notice" in row["related_chunk_ids"]


def test_policy_tables_stay_inside_their_section(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    plans = next(c for c in read_chunks(out) if c["chunk_id"] == "extended_warranty_options::available-plans")
    assert "Premium Care" in plans["text"] and "$2,299" in plans["text"]


def test_preamble_becomes_overview_chunk(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    ids = {c["chunk_id"] for c in read_chunks(out)}
    assert "warranty_policy::overview" in ids


def test_chunks_are_unique_non_empty_and_have_context_header(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    chunks = read_chunks(out)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c["text"].startswith("Document: ")
        assert len(c["text"].split()) > 10
        assert "**" not in c["text"]


# ---------------------------------------------------------------- change detection
def test_second_run_changes_nothing(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    report = ing.ingest(kb, out)
    assert len(report["unchanged"]) == 17
    assert not (report["chunks_added"] or report["chunks_updated"] or report["chunks_removed"])


def test_edited_document_only_updates_its_changed_chunks(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    path = kb / "return_policy.md"
    path.write_text(path.read_text().replace("within **30 days**", "within **45 days**"))
    report = ing.ingest(kb, out)
    assert report["updated"] == ["return_policy"]
    assert report["chunks_updated"] == ["return_policy::parts-returns"]
    text = next(c for c in read_chunks(out) if c["chunk_id"] == "return_policy::parts-returns")["text"]
    assert "45 days" in text


def test_removed_section_is_deleted(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    path = kb / "trade_in_policy.md"
    path.write_text(path.read_text().split("## Dealer Discretion")[0])
    report = ing.ingest(kb, out)
    assert report["chunks_removed"] == ["trade_in_policy::dealer-discretion"]


def test_deleted_document_removes_all_its_chunks(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    (kb / "dealer_onboarding.md").unlink()
    report = ing.ingest(kb, out)
    assert report["removed"] == ["dealer_onboarding"]
    assert not any(c["doc_id"] == "dealer_onboarding" for c in read_chunks(out))


def test_new_document_is_added_and_flagged_without_domain(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    (kb / "tyre_policy.md").write_text("# Tyre Policy\n\n## Coverage\nTyres are covered by the tyre maker for 5 years.\n")
    report = ing.ingest(kb, out)
    assert report["added"] == ["tyre_policy"]
    assert report["unknown_domain_docs"] == ["tyre_policy"]


def test_chunker_version_change_forces_rebuild(workspace, monkeypatch):
    kb, out = workspace
    ing.ingest(kb, out)
    monkeypatch.setattr(ing, "CHUNKER_VERSION", "999")
    assert ing.ingest(kb, out)["rebuild"] is True



# ---------------------------------------------------------------- freshness check (startup warning)
def test_freshness_clean_after_ingest(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    assert ing.check_freshness(kb, out)["stale"] is False


def test_freshness_detects_edited_added_and_removed_docs(workspace):
    kb, out = workspace
    ing.ingest(kb, out)
    (kb / "return_policy.md").write_text((kb / "return_policy.md").read_text().replace("30 days", "45 days"))
    (kb / "new_policy.md").write_text("# New\n\n## Rule\nSomething new.\n")
    (kb / "dealer_onboarding.md").unlink()
    f = ing.check_freshness(kb, out)
    assert f["stale"] and f["changed"] == ["return_policy"] and f["added"] == ["new_policy"]
    assert f["removed"] == ["dealer_onboarding"]


def test_freshness_detects_stale_embeddings(workspace):
    """The real K2 case: chunks re-ingested but embeddings not refreshed."""
    from src.embeddings import EmbeddingStore, HashEmbedder
    kb, out = workspace
    ing.ingest(kb, out)
    chunks = read_chunks(out)
    EmbeddingStore(out / "embeddings.json").sync(chunks, HashEmbedder())
    assert ing.check_freshness(kb, out)["stale"] is False
    (kb / "return_policy.md").write_text((kb / "return_policy.md").read_text().replace("30 days", "45 days"))
    ing.ingest(kb, out)                                   # documents re-ingested...
    f = ing.check_freshness(kb, out)                       # ...but embeddings not refreshed
    assert f["stale"] and f["embeddings_stale"] == 1 and not f["changed"]


def test_freshness_reports_never_ingested(tmp_path):
    assert ing.check_freshness(KB_DIR, tmp_path / "nothing")["never_ingested"] is True
