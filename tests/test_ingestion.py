"""Unit tests for Phase 2: PDF Ingestion & Text Extraction."""
import hashlib
from pathlib import Path
import fitz
import pytest

from app.ingestion.pdf_loader import (
    calculate_document_id,
    load_pdf_from_bytes,
    load_pdf_from_path,
    save_pdf,
)
from app.models.schemas import (
    ExtractedDocument,
    InvalidPDFError,
    EmptyPDFError,
)


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper to create an in-memory PDF with specified text on each page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_calculate_document_id():
    sample_bytes = b"Hello, this is a test PDF content."
    expected_hash = hashlib.sha256(sample_bytes).hexdigest()
    assert calculate_document_id(sample_bytes) == expected_hash


def test_load_valid_multipage_pdf(tmp_path):
    pages_content = [
        "Company Leave Policy 2025: Employees get 15 days of paid annual leave.",
        "Remote Work Guidelines: Eligible employees may work remotely 2 days per week.",
        "Notice Period: Standard employee notice period is 30 days upon resignation."
    ]
    pdf_bytes = create_sample_pdf_bytes(pages_content)
    filename = "test_policy.pdf"

    extracted = load_pdf_from_bytes(pdf_bytes, filename=filename, save=True, target_dir=tmp_path)

    assert isinstance(extracted, ExtractedDocument)
    assert extracted.filename == filename
    assert extracted.total_pages == 3
    assert extracted.doc_id == calculate_document_id(pdf_bytes)
    assert len(extracted.pages) == 3

    # Check 1-indexed page numbering and content
    assert extracted.pages[0].page_number == 1
    assert "15 days of paid annual leave" in extracted.pages[0].text
    assert extracted.pages[0].char_count > 0

    assert extracted.pages[1].page_number == 2
    assert "2 days per week" in extracted.pages[1].text

    assert extracted.pages[2].page_number == 3
    assert "30 days upon resignation" in extracted.pages[2].text

    # Check file was saved properly
    saved_file = Path(extracted.file_path)
    assert saved_file.exists()
    assert saved_file.read_bytes() == pdf_bytes


def test_load_from_path(tmp_path):
    pages_content = ["Single page document for testing load_pdf_from_path."]
    pdf_bytes = create_sample_pdf_bytes(pages_content)
    file_path = tmp_path / "sample_doc.pdf"
    file_path.write_bytes(pdf_bytes)

    extracted = load_pdf_from_path(file_path, save=False)

    assert extracted.filename == "sample_doc.pdf"
    assert extracted.total_pages == 1
    assert "Single page document" in extracted.pages[0].text
    assert extracted.pages[0].page_number == 1


def test_empty_bytes_raises_invalid_pdf():
    with pytest.raises(InvalidPDFError, match="empty"):
        load_pdf_from_bytes(b"", filename="empty.pdf", save=False)


def test_corrupted_pdf_bytes_raises_invalid_pdf():
    corrupted_bytes = b"NOT_A_VALID_PDF_HEADER_OR_BODY"
    with pytest.raises(InvalidPDFError, match="Failed to parse"):
        load_pdf_from_bytes(corrupted_bytes, filename="corrupt.pdf", save=False)


def test_blank_pdf_raises_empty_pdf():
    # PDF with 2 blank pages containing no text
    blank_pdf_bytes = create_sample_pdf_bytes(["", ""])
    with pytest.raises(EmptyPDFError, match="no extractable text"):
        load_pdf_from_bytes(blank_pdf_bytes, filename="blank.pdf", save=False)


def test_nonexistent_path_raises_invalid_pdf():
    with pytest.raises(InvalidPDFError, match="File not found"):
        load_pdf_from_path("non_existent_file_12345.pdf")
