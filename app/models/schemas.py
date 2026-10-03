"""Data models and schemas for RAG Knowledge Assistant."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# Ingestion Models
# ------------------------------------------------------------------------------

class ExtractedPage(BaseModel):
    """Represents text extracted from a single page of a PDF."""
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Extracted text content of the page")
    char_count: int = Field(..., description="Number of characters on this page")


class ExtractedDocument(BaseModel):
    """Represents a fully ingested and parsed PDF document."""
    doc_id: str = Field(..., description="Deterministic SHA256 hash of file content")
    filename: str = Field(..., description="Original filename of the PDF")
    total_pages: int = Field(..., description="Total number of pages in the PDF")
    file_path: Optional[str] = Field(None, description="Path where PDF is stored on disk")
    pages: List[ExtractedPage] = Field(default_factory=list, description="List of extracted pages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of ingestion")


# ------------------------------------------------------------------------------
# Chunking Models
# ------------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """Represents a split chunk of text with exact provenance metadata."""
    chunk_id: str = Field(..., description="Unique chunk ID e.g. {doc_id}_p{page}_c{chunk_index}")
    doc_id: str = Field(..., description="Deterministic SHA256 ID of parent document")
    filename: str = Field(..., description="Original source filename")
    page_number: int = Field(..., description="1-indexed page number where chunk originated")
    chunk_index: int = Field(..., description="Sequential index of the chunk within document")
    text: str = Field(..., description="Text content of the chunk")
    char_count: int = Field(..., description="Number of characters in the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ------------------------------------------------------------------------------
# Retrieval Models
# ------------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from vector store matching a query."""
    chunk_id: str = Field(..., description="Unique chunk ID")
    doc_id: str = Field(..., description="SHA256 ID of source document")
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="1-indexed source page number")
    chunk_index: int = Field(..., description="Sequential index of chunk in document")
    text: str = Field(..., description="Text snippet of the chunk")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full chunk metadata payload")


# ------------------------------------------------------------------------------
# Citation & API Schemas
# ------------------------------------------------------------------------------

class Citation(BaseModel):
    """Structured citation reference pointing to the exact source PDF and page."""
    filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="1-indexed page number in the source PDF")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    snippet: str = Field(..., description="Text snippet from the chunk")
    score: float = Field(default=0.0, description="Similarity score of chunk")


class QARequest(BaseModel):
    """Request model for document Q&A."""
    query: str = Field(..., min_length=1, description="Question asked by the user")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Top-k chunks to retrieve")
    doc_id: Optional[str] = Field(default=None, description="Optional single doc_id filter")
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional list of doc_ids to search within")


class QAResponse(BaseModel):
    """Response model for grounded document Q&A."""
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Grounded answer or refusal message")
    citations: List[Citation] = Field(default_factory=list, description="Exact source citations used")
    is_grounded: bool = Field(default=True, description="Whether answer passed grounding check")
    retrieved_count: int = Field(default=0, description="Number of context chunks retrieved")


class CompareRequest(BaseModel):
    """Request model for comparing two documents."""
    doc_id_a: str = Field(..., description="Document ID of first document (Doc A)")
    doc_id_b: str = Field(..., description="Document ID of second document (Doc B)")
    topic: Optional[str] = Field(default="Key differences, policies, and changes", description="Comparison topic")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Top-k chunks per document")


class DifferenceItem(BaseModel):
    """A specific difference identified between two documents."""
    topic: str = Field(..., description="Subtopic or policy area")
    document_a_detail: str = Field(..., description="Statement/rule in Document A")
    document_b_detail: str = Field(..., description="Statement/rule in Document B")
    change_type: str = Field(default="modified", description="added, removed, modified, or unchanged")


class CompareResponse(BaseModel):
    """Response model for document comparison."""
    doc_id_a: str
    filename_a: str
    doc_id_b: str
    filename_b: str
    topic: str
    comparison_summary: str = Field(..., description="Comprehensive comparison narrative")
    citations_doc_a: List[Citation] = Field(default_factory=list, description="Citations from Document A")
    citations_doc_b: List[Citation] = Field(default_factory=list, description="Citations from Document B")


class ContradictionItem(BaseModel):
    """Represents a specific detected factual contradiction."""
    claim_a: str = Field(..., description="Statement from first source")
    source_a: str = Field(..., description="Filename of first source")
    page_a: int = Field(..., description="Page number of first source")
    claim_b: str = Field(..., description="Conflicting statement from second source")
    source_b: str = Field(..., description="Filename of second source")
    page_b: int = Field(..., description="Page number of second source")
    explanation: str = Field(..., description="Explanation of why statements conflict")


class ContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    topic: str = Field(..., min_length=1, description="Topic/policy area to check for contradictions")
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional doc IDs to scope search")
    top_k: Optional[int] = Field(default=6, ge=1, le=20, description="Top-k chunks to inspect")


class ContradictionResponse(BaseModel):
    """Response model for contradiction detection."""
    topic: str
    has_contradictions: bool
    explanation: str
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    inspected_citations: List[Citation] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    """Request model for targeted summarization."""
    topic: str = Field(..., min_length=1, description="Topic to summarize (e.g. pricing, leave)")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID to scope summary")
    top_k: Optional[int] = Field(default=6, ge=1, le=20, description="Top-k chunks to summarize")


class SummarizeResponse(BaseModel):
    """Response model for targeted summarization."""
    topic: str
    summary: str
    citations: List[Citation] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Summary information for an indexed document."""
    doc_id: str
    filename: str
    total_pages: int
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response containing list of all indexed documents."""
    total_documents: int
    documents: List[DocumentInfo]


class UploadResponse(BaseModel):
    """Response returned after uploading and indexing a PDF."""
    message: str
    doc_id: str
    filename: str
    total_pages: int
    chunks_indexed: int


class DeleteResponse(BaseModel):
    """Response returned after deleting a document."""
    message: str
    doc_id: str


class HealthResponse(BaseModel):
    """API health status response."""
    status: str
    qdrant_status: str
    embedding_model: str
    gemini_configured: bool


# ------------------------------------------------------------------------------
# Ingestion & Retrieval Exceptions
# ------------------------------------------------------------------------------

class IngestionError(Exception):
    """Base exception for PDF ingestion errors."""
    pass


class InvalidPDFError(IngestionError):
    """Raised when an uploaded file is not a valid PDF or is corrupted."""
    pass


class EmptyPDFError(IngestionError):
    """Raised when an uploaded PDF has no pages or contains no extractable text."""
    pass


class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass


class EmptyQueryError(RetrievalError):
    """Raised when a query string is empty or contains only whitespace."""
    pass


class InvalidTopKError(RetrievalError):
    """Raised when top_k requested is less than 1."""
    pass
