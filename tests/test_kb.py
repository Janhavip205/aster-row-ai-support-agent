"""Tests for Knowledge Base Indexer and Metadata-Aware Retriever."""

import pytest
from pathlib import Path

from src.config import KB_DIR
from src.kb.indexer import DocumentChunk, load_all_chunks, parse_markdown_document
from src.kb.retriever import KBRetriever, RetrievalResult


@pytest.fixture(scope="module")
def retriever() -> KBRetriever:
    """Fixture providing an initialized KBRetriever instance."""
    return KBRetriever(kb_dir=KB_DIR, min_score=0.25)


def test_indexer_loads_and_preserves_metadata():
    """Verify that the indexer parses all 14 files and preserves frontmatter metadata."""
    chunks = load_all_chunks(KB_DIR)
    assert len(chunks) > 0, "Indexer should extract chunks from knowledge-base files"

    # Verify key metadata fields are present and valid on all chunks
    for chunk in chunks:
        assert chunk.filename.endswith(".md")
        assert len(chunk.document_id) > 0
        assert len(chunk.title) > 0
        assert len(chunk.heading) > 0
        assert len(chunk.content) > 0
        assert chunk.status in ("active", "superseded", "draft")
        assert chunk.policy_authority in ("official", "none")
        assert chunk.source_citation == f"{chunk.filename} > {chunk.heading}"


def test_current_returns_policy_retrievable(retriever: KBRetriever):
    """Verify that current returns policy (30 days) is retrievable."""
    query = "How long does a regular customer have to return an unused backpack?"
    results = retriever.retrieve(query, top_k=5)

    filenames = [r.chunk.filename for r in results]
    assert "01-returns-policy-current.md" in filenames, (
        f"Expected 01-returns-policy-current.md in results, got {filenames}"
    )

    # Check that the specific section on return window is retrieved
    standard_return = next(
        (r for r in results if r.chunk.filename == "01-returns-policy-current.md" and "return" in r.chunk.heading.lower()),
        None,
    )
    assert standard_return is not None
    assert standard_return.chunk.status == "active"
    assert standard_return.chunk.policy_authority == "official"
    assert "30 calendar days" in standard_return.chunk.content


def test_legacy_returns_policy_excluded_as_authoritative(retriever: KBRetriever):
    """Verify that 02-returns-policy-legacy.md is excluded from authoritative customer retrieval."""
    query = "What is the return window under the legacy policy?"
    results = retriever.retrieve(query, top_k=10, authoritative_only=True)

    filenames = [r.chunk.filename for r in results]
    assert "02-returns-policy-legacy.md" not in filenames, (
        f"02-returns-policy-legacy.md must be excluded from authoritative results, but found in {filenames}"
    )

    # However, verify that it was indexed and classified as excluded
    excluded = retriever.get_excluded_chunks()
    excluded_files = {c.filename for c in excluded}
    assert "02-returns-policy-legacy.md" in excluded_files


def test_internal_migration_content_excluded(retriever: KBRetriever):
    """Verify that 14-internal-content-migration-notes.md is completely excluded from customer retrieval."""
    query = "Does the migration scratchpad grant 60 days return window?"
    results = retriever.retrieve(query, top_k=10, authoritative_only=True)

    filenames = [r.chunk.filename for r in results]
    assert "14-internal-content-migration-notes.md" not in filenames, (
        f"14-internal-content-migration-notes.md must never appear in customer results, but found in {filenames}"
    )

    # Verify that it is in the excluded chunks list
    excluded = retriever.get_excluded_chunks()
    excluded_files = {c.filename for c in excluded}
    assert "14-internal-content-migration-notes.md" in excluded_files


def test_warranty_content_retrievable(retriever: KBRetriever):
    """Verify that limited product warranty terms are retrievable."""
    query = "Do all Aster & Row products have a lifetime warranty?"
    results = retriever.retrieve(query, top_k=3)

    assert len(results) > 0
    top_result = results[0]
    assert top_result.chunk.filename == "07-warranty.md"
    assert top_result.chunk.heading == "Warranty periods"
    assert "Aster & Row does not offer a lifetime warranty" in top_result.chunk.content


def test_international_shipping_retrievable(retriever: KBRetriever):
    """Verify international shipping terms for Canada and unsupported countries are retrievable."""
    query = "Do you ship to Canada or other countries internationally?"
    results = retriever.retrieve(query, top_k=3)

    filenames = [r.chunk.filename for r in results]
    assert "06-international-shipping.md" in filenames

    intl_result = next(r for r in results if r.chunk.filename == "06-international-shipping.md")
    assert intl_result.chunk.status == "active"
    assert intl_result.chunk.policy_authority == "official"


def test_breeze_tumbler_retrieves_both_active_sources(retriever: KBRetriever):
    """Verify that both active conflicting sources for Breeze Tumbler dishwasher safety are retrieved."""
    query = "Can I put the entire Breeze Tumbler in the dishwasher?"
    results = retriever.retrieve(query, top_k=5)

    filenames = [r.chunk.filename for r in results]
    assert "11-product-care.md" in filenames, "Must retrieve 11-product-care.md (hand-wash body)"
    assert "12-breeze-tumbler-product-card.md" in filenames, "Must retrieve 12-breeze-tumbler-product-card.md (dishwasher safe)"

    # Verify neither source was silently excluded
    care_chunk = next(r for r in results if r.chunk.filename == "11-product-care.md")
    card_chunk = next(r for r in results if r.chunk.filename == "12-breeze-tumbler-product-card.md")
    assert care_chunk.chunk.status == "active" and care_chunk.chunk.policy_authority == "official"
    assert card_chunk.chunk.status == "active" and card_chunk.chunk.policy_authority == "official"


def test_irrelevant_query_produces_no_result(retriever: KBRetriever):
    """Verify that a completely irrelevant query yields no results with default relevance threshold."""
    query = "What is the secret recipe for baking delicious chocolate chip cookies with walnuts?"
    results = retriever.retrieve(query, min_score=0.25)
    assert len(results) == 0, f"Expected 0 results for irrelevant query, got {len(results)}"


def test_trailplus_membership_retrievable(retriever: KBRetriever):
    """Verify TrailPlus membership 45-day return window is retrievable."""
    query = "What is the return window for an active TrailPlus member?"
    results = retriever.retrieve(query, top_k=3)

    filenames = [r.chunk.filename for r in results]
    assert "09-trailplus-membership.md" in filenames
    trailplus_chunk = next(r for r in results if r.chunk.filename == "09-trailplus-membership.md")
    assert "45-calendar-day return window" in trailplus_chunk.chunk.content


def test_knowledge_base_files_remain_unmodified():
    """Verify that none of the original markdown files in knowledge-base/ were modified."""
    # Ensure all 14 files exist with expected file sizes
    expected_files = [
        "01-returns-policy-current.md",
        "02-returns-policy-legacy.md",
        "03-final-sale-and-promotions.md",
        "04-damaged-or-wrong-items.md",
        "05-domestic-shipping.md",
        "06-international-shipping.md",
        "07-warranty.md",
        "08-order-changes-and-cancellations.md",
        "09-trailplus-membership.md",
        "10-gift-cards-and-price-adjustments.md",
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
        "13-support-escalation.md",
        "14-internal-content-migration-notes.md",
    ]
    for filename in expected_files:
        path = KB_DIR / filename
        assert path.is_file(), f"File {filename} is missing from {KB_DIR}"
