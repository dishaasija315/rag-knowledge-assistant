"""Unit tests for Phase 3: Document Chunking."""
import pytest
from app.ingestion.chunker import chunk_document, get_text_splitter
from app.models.schemas import ExtractedDocument, ExtractedPage, DocumentChunk


@pytest.fixture
def sample_extracted_doc():
    pages = [
        ExtractedPage(
            page_number=1,
            text="Employee Leave Policy 2025. All full-time employees are entitled to 15 days of annual paid leave.",
            char_count=98,
        ),
        ExtractedPage(
            page_number=2,
            text="Remote Work Policy. Employees may work from home up to 2 days per week with manager approval.",
            char_count=93,
        ),
    ]
    return ExtractedDocument(
        doc_id="abc123sha256hash",
        filename="company_handbook.pdf",
        total_pages=2,
        pages=pages,
    )


def test_chunk_document_metadata_preservation(sample_extracted_doc):
    chunks = chunk_document(sample_extracted_doc, chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 2

    # Check first chunk (Page 1)
    chunk1 = chunks[0]
    assert isinstance(chunk1, DocumentChunk)
    assert chunk1.doc_id == "abc123sha256hash"
    assert chunk1.filename == "company_handbook.pdf"
    assert chunk1.page_number == 1
    assert chunk1.chunk_index == 0
    assert chunk1.chunk_id == "abc123sha256hash_p1_c0"
    assert "15 days of annual paid leave" in chunk1.text
    assert chunk1.metadata["doc_id"] == "abc123sha256hash"
    assert chunk1.metadata["page_number"] == 1

    # Check second chunk (Page 2)
    chunk2 = chunks[1]
    assert chunk2.doc_id == "abc123sha256hash"
    assert chunk2.filename == "company_handbook.pdf"
    assert chunk2.page_number == 2
    assert chunk2.chunk_index == 1
    assert chunk2.chunk_id == "abc123sha256hash_p2_c1"
    assert "2 days per week" in chunk2.text
    assert chunk2.metadata["page_number"] == 2


def test_chunking_long_text_across_boundaries():
    long_text = "Section A: Detailed Terms. " * 30  # ~810 characters
    doc = ExtractedDocument(
        doc_id="doc_long_test",
        filename="long_doc.pdf",
        total_pages=1,
        pages=[ExtractedPage(page_number=1, text=long_text, char_count=len(long_text))],
    )

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1

    # Ensure all chunks have valid IDs and page numbers
    for idx, c in enumerate(chunks):
        assert c.page_number == 1
        assert c.chunk_index == idx
        assert c.chunk_id == f"doc_long_test_p1_c{idx}"
        assert len(c.text) <= 250  # Roughly within chunk size


def test_empty_pages_are_skipped():
    doc = ExtractedDocument(
        doc_id="doc_empty_pages",
        filename="sparse.pdf",
        total_pages=3,
        pages=[
            ExtractedPage(page_number=1, text="", char_count=0),
            ExtractedPage(page_number=2, text="Valid content on page 2", char_count=23),
            ExtractedPage(page_number=3, text="   ", char_count=3),
        ],
    )

    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].text == "Valid content on page 2"
