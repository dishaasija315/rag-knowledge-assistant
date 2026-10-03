"""Document chunking module using LangChain RecursiveCharacterTextSplitter."""
import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.models.schemas import ExtractedDocument, DocumentChunk

logger = logging.getLogger(__name__)


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """Create a configured instance of RecursiveCharacterTextSplitter."""
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(
    doc: ExtractedDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[DocumentChunk]:
    """
    Split an ExtractedDocument into DocumentChunks, preserving page-level provenance.
    
    Each page is chunked while maintaining its exact page_number, filename, and doc_id.
    
    Args:
        doc: The ExtractedDocument containing pages and metadata.
        chunk_size: Optional override for chunk size.
        chunk_overlap: Optional override for chunk overlap.
        
    Returns:
        List of DocumentChunk instances with deterministic chunk IDs and metadata.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[DocumentChunk] = []
    global_chunk_idx = 0

    for page in doc.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        raw_chunks = splitter.split_text(page_text)
        for chunk_text in raw_chunks:
            cleaned_text = chunk_text.strip()
            if not cleaned_text:
                continue

            chunk_id = f"{doc.doc_id}_p{page.page_number}_c{global_chunk_idx}"
            chunk_metadata = {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "page_number": page.page_number,
                "chunk_id": chunk_id,
                "chunk_index": global_chunk_idx,
                "total_pages": doc.total_pages,
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=cleaned_text,
                    char_count=len(cleaned_text),
                    metadata=chunk_metadata,
                )
            )
            global_chunk_idx += 1

    logger.info(
        f"Chunked '{doc.filename}' into {len(chunks)} chunks across {doc.total_pages} pages."
    )
    return chunks
