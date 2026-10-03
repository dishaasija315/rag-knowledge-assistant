"""PDF loading and text extraction module using PyMuPDF (fitz)."""
import hashlib
import logging
from pathlib import Path
from typing import Union, Optional

import pymupdf
fitz = pymupdf  # PyMuPDF alias

from app.config import get_settings
from app.models.schemas import (
    ExtractedDocument,
    ExtractedPage,
    InvalidPDFError,
    EmptyPDFError,
)

logger = logging.getLogger(__name__)


def calculate_document_id(file_bytes: bytes) -> str:
    """Generate a deterministic SHA256 document identifier from file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def save_pdf(file_bytes: bytes, filename: str, target_dir: Optional[Path] = None) -> Path:
    """
    Save raw PDF bytes to the documents storage directory.
    
    Args:
        file_bytes: Raw bytes of the PDF file.
        filename: Destination filename.
        target_dir: Optional override for documents directory.
        
    Returns:
        Path to the saved PDF file.
    """
    if target_dir is None:
        target_dir = get_settings().documents_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    destination_path = target_dir / filename
    
    with open(destination_path, "wb") as f:
        f.write(file_bytes)
        
    logger.info(f"Saved PDF to: {destination_path}")
    return destination_path


def load_pdf_from_bytes(
    file_bytes: bytes,
    filename: str,
    save: bool = True,
    target_dir: Optional[Path] = None,
) -> ExtractedDocument:
    """
    Load a PDF from raw bytes, validate it, extract page-by-page text, and preserve metadata.
    
    Args:
        file_bytes: Raw binary content of the PDF.
        filename: Original filename of the PDF.
        save: Whether to persist the PDF file to disk.
        target_dir: Optional target directory if saving.
        
    Returns:
        ExtractedDocument schema populated with metadata and pages.
        
    Raises:
        InvalidPDFError: If the bytes do not represent a valid PDF.
        EmptyPDFError: If the PDF contains 0 pages or no extractable text.
    """
    if not file_bytes:
        raise InvalidPDFError(f"Cannot ingest '{filename}': file content is empty (0 bytes).")

    doc_id = calculate_document_id(file_bytes)
    
    # Validate and open PDF with PyMuPDF
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise InvalidPDFError(f"Failed to parse '{filename}' as a valid PDF: {str(e)}") from e

    total_pages = pdf_doc.page_count
    if total_pages == 0:
        pdf_doc.close()
        raise EmptyPDFError(f"The PDF '{filename}' contains 0 pages.")

    pages = []
    total_characters = 0

    try:
        for page_idx in range(total_pages):
            page = pdf_doc.load_page(page_idx)
            page_text = page.get_text("text").strip()
            char_count = len(page_text)
            total_characters += char_count

            # We record every page (1-indexed) along with its extracted text
            pages.append(
                ExtractedPage(
                    page_number=page_idx + 1,
                    text=page_text,
                    char_count=char_count,
                )
            )

        # PyMuPDF embedded metadata dictionary
        raw_meta = pdf_doc.metadata or {}
        cleaned_meta = {k: v for k, v in raw_meta.items() if v}
    finally:
        pdf_doc.close()

    if total_characters == 0:
        raise EmptyPDFError(
            f"The PDF '{filename}' has {total_pages} page(s) but contains no extractable text."
        )

    saved_path_str = None
    if save:
        saved_path = save_pdf(file_bytes, filename, target_dir=target_dir)
        saved_path_str = str(saved_path)

    logger.info(
        f"Ingested '{filename}' (ID: {doc_id[:8]}...): {total_pages} pages, {total_characters} characters."
    )

    return ExtractedDocument(
        doc_id=doc_id,
        filename=filename,
        total_pages=total_pages,
        file_path=saved_path_str,
        pages=pages,
        metadata=cleaned_meta,
    )


def load_pdf_from_path(file_path: Union[str, Path], save: bool = False) -> ExtractedDocument:
    """
    Load a PDF from an existing local filesystem path.
    
    Args:
        file_path: Path to the PDF file.
        save: Whether to copy/re-save the file to documents directory.
        
    Returns:
        ExtractedDocument.
    """
    path = Path(file_path)
    if not path.is_file():
        raise InvalidPDFError(f"File not found at path: {file_path}")

    with open(path, "rb") as f:
        file_bytes = f.read()

    return load_pdf_from_bytes(file_bytes, filename=path.name, save=save)
