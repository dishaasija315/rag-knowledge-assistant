"""FastAPI routes for RAG Knowledge Assistant."""
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import get_settings
from app.models.schemas import (
    UploadResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentInfo,
    QARequest,
    QAResponse,
    CompareRequest,
    CompareResponse,
    ContradictionRequest,
    ContradictionResponse,
    SummarizeRequest,
    SummarizeResponse,
    HealthResponse,
    InvalidPDFError,
    EmptyPDFError,
    RetrievalError,
)
from app.ingestion.pdf_loader import load_pdf_from_bytes
from app.core.vector_store import QdrantVectorStore
from app.core.embeddings import EmbeddingService
from app.workflows.rag_graph import get_rag_workflow
from app.comparison.comparator import get_comparator
from app.evaluation.evaluator import RAGEvaluator

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------------------
# Document Management Endpoints
# ------------------------------------------------------------------------------

@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload and index a PDF document",
)
@router.post(
    "/api/upload",
    response_model=UploadResponse,
    include_in_schema=False,
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, extract page text, chunk, embed, and store vectors in Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Uploaded: '{file.filename}'",
        )

    try:
        file_bytes = await file.read()
        extracted_doc = load_pdf_from_bytes(file_bytes=file_bytes, filename=file.filename, save=True)

        vector_store = QdrantVectorStore()
        embedding_service = EmbeddingService()
        chunks_count = vector_store.index_document(extracted_doc, embedding_service=embedding_service)

        return UploadResponse(
            message="Document successfully processed and indexed into vector store.",
            doc_id=extracted_doc.doc_id,
            filename=extracted_doc.filename,
            total_pages=extracted_doc.total_pages,
            chunks_indexed=chunks_count,
        )
    except (InvalidPDFError, EmptyPDFError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {str(e)}")


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
@router.get(
    "/api/documents",
    response_model=DocumentListResponse,
    include_in_schema=False,
)
async def list_documents():
    """Retrieve list of unique documents and statistics currently stored in Qdrant."""
    try:
        vector_store = QdrantVectorStore()
        raw_docs = vector_store.get_all_documents()
        doc_infos = [
            DocumentInfo(
                doc_id=d["doc_id"],
                filename=d["filename"],
                total_pages=d.get("total_pages", 0),
                chunk_count=d["chunk_count"],
            )
            for d in raw_docs
        ]
        return DocumentListResponse(
            total_documents=len(doc_infos),
            documents=doc_infos,
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document by document_id",
)
@router.delete(
    "/api/documents/{document_id}",
    response_model=DeleteResponse,
    include_in_schema=False,
)
async def delete_document(document_id: str):
    """Delete all vector points and metadata for a document from Qdrant."""
    try:
        vector_store = QdrantVectorStore()
        vector_store.delete_document(doc_id=document_id)
        return DeleteResponse(
            message=f"Document '{document_id}' successfully removed from vector store.",
            doc_id=document_id,
        )
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ------------------------------------------------------------------------------
# RAG Operations Endpoints
# ------------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QAResponse,
    summary="Grounded document Q&A via LangGraph",
)
@router.post(
    "/api/query",
    response_model=QAResponse,
    include_in_schema=False,
)
async def query_documents(request: QARequest):
    """Execute grounded question-answering workflow with source citations."""
    try:
        workflow = get_rag_workflow()
        response = workflow.run(
            query=request.query,
            top_k=request.top_k,
            doc_id=request.doc_id,
            doc_ids=request.doc_ids,
        )
        return response
    except RetrievalError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing Q&A query: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two documents on a topic",
)
@router.post(
    "/api/compare",
    response_model=CompareResponse,
    include_in_schema=False,
)
async def compare_documents(request: CompareRequest):
    """Perform targeted semantic comparison between two indexed documents."""
    try:
        comparator = get_comparator()
        response = comparator.compare_documents(
            doc_id_a=request.doc_id_a,
            doc_id_b=request.doc_id_b,
            topic=request.topic,
            top_k=request.top_k or 5,
        )
        return response
    except Exception as e:
        logger.error(f"Error comparing documents: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/contradictions",
    response_model=ContradictionResponse,
    summary="Detect factual contradictions across documents",
)
@router.post(
    "/api/contradictions",
    response_model=ContradictionResponse,
    include_in_schema=False,
)
async def detect_contradictions(request: ContradictionRequest):
    """Analyze retrieved chunks across documents for factual contradictions."""
    try:
        comparator = get_comparator()
        response = comparator.detect_contradictions(
            topic=request.topic,
            doc_ids=request.doc_ids,
            top_k=request.top_k or 6,
        )
        return response
    except Exception as e:
        logger.error(f"Error detecting contradictions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Targeted summarization of sections on a topic",
)
@router.post(
    "/api/summarize",
    response_model=SummarizeResponse,
    include_in_schema=False,
)
async def summarize_topic(request: SummarizeRequest):
    """Retrieve and summarize sections of documents on a specific topic."""
    try:
        comparator = get_comparator()
        response = comparator.summarize_topic(
            topic=request.topic,
            doc_id=request.doc_id,
            top_k=request.top_k or 6,
        )
        return response
    except Exception as e:
        logger.error(f"Error summarizing topic: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/evaluation",
    summary="Run automated RAG evaluation benchmark",
)
@router.get(
    "/api/evaluation",
    include_in_schema=False,
)
async def run_evaluation(top_k: int = 4):
    """Execute evaluation benchmark and return accuracy, grounding, and citation metrics."""
    try:
        evaluator = RAGEvaluator()
        return evaluator.run_evaluation(top_k=top_k)
    except Exception as e:
        logger.error(f"Error running evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ------------------------------------------------------------------------------
# System Health Endpoint
# ------------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
@router.get(
    "/api/health",
    response_model=HealthResponse,
    include_in_schema=False,
)
async def health_check():
    """Verify health status of FastAPI, Qdrant database, and embedding services."""
    settings = get_settings()
    qdrant_status = "unavailable"

    try:
        store = QdrantVectorStore()
        store.ensure_collection_exists()
        info = store.get_collection_info()
        qdrant_status = info.get("status", "connected")
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    gemini_ready = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here")

    return HealthResponse(
        status="healthy",
        qdrant_status=qdrant_status,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        gemini_configured=gemini_ready,
    )
