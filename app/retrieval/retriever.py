"""Basic retrieval module querying Qdrant using dense vector embeddings."""
import logging
from typing import List, Optional

from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

from app.config import get_settings
from app.core.embeddings import EmbeddingService
from app.core.vector_store import QdrantVectorStore
from app.models.schemas import (
    RetrievedChunk,
    RetrievalError,
    EmptyQueryError,
    InvalidTopKError,
)

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """Retrieves relevant document chunks from Qdrant vector database."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        default_top_k: Optional[int] = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.default_top_k = default_top_k or settings.DEFAULT_TOP_K
        self.score_threshold = settings.SIMILARITY_SCORE_THRESHOLD

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Perform semantic vector search in Qdrant and return top-k matching chunks.
        
        Args:
            query: User search query string.
            top_k: Number of chunks to retrieve (defaults to config DEFAULT_TOP_K).
            doc_id: Optional filter for a single document ID.
            doc_ids: Optional filter for a list of document IDs.
            score_threshold: Optional minimum cosine similarity cutoff score.
            collection_name: Optional override for Qdrant collection.
            
        Returns:
            List of RetrievedChunk models ordered by relevance score descending.
            
        Raises:
            EmptyQueryError: If query is empty or whitespace.
            InvalidTopKError: If top_k < 1.
            RetrievalError: If Qdrant search encounters a connection or query error.
        """
        # Validate query string
        if not query or not query.strip():
            raise EmptyQueryError("Search query cannot be empty or whitespace-only.")

        cleaned_query = query.strip()

        # Validate top_k
        k = top_k if top_k is not None else self.default_top_k
        if k < 1:
            raise InvalidTopKError(f"top_k must be greater than or equal to 1, received: {k}")

        col_name = collection_name or self.vector_store.collection_name
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # Generate 384-dimensional query embedding
        try:
            query_vector = self.embedding_service.embed_text(cleaned_query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise RetrievalError(f"Embedding generation failed: {e}") from e

        # Construct metadata filters if document scoping is requested
        query_filter = None
        if doc_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            )
        elif doc_ids and len(doc_ids) > 0:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=doc_ids),
                    )
                ]
            )

        # Execute cosine similarity search in Qdrant
        try:
            hits = self.vector_store.search_vectors(
                query_vector=query_vector,
                limit=k,
                query_filter=query_filter,
                score_threshold=threshold,
                collection_name=col_name,
            )
        except Exception as e:
            logger.error(f"Qdrant search failed for collection '{col_name}': {e}")
            raise RetrievalError(f"Vector search failed in Qdrant: {e}") from e

        # Format hits into strongly typed RetrievedChunk models
        results: List[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    doc_id=str(payload.get("doc_id", "")),
                    filename=str(payload.get("filename", "unknown")),
                    page_number=int(payload.get("page_number", 1)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                    metadata=payload,
                )
            )

        top_score = results[0].score if results else 0.0
        logger.info(
            f"Query '{cleaned_query[:40]}...' returned {len(results)} chunks (top score: {top_score:.4f})"
        )
        return results


# Convenience module-level retriever instance
_DEFAULT_RETRIEVER: Optional[QdrantRetriever] = None


def get_retriever() -> QdrantRetriever:
    """Get or instantiate default singleton QdrantRetriever."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = QdrantRetriever()
    return _DEFAULT_RETRIEVER


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    doc_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """Top-level convenience function for document retrieval."""
    return get_retriever().retrieve(query=query, top_k=top_k, doc_id=doc_id, doc_ids=doc_ids)
