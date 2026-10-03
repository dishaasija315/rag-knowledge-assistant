"""Qdrant vector store management and persistence module."""
import logging
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.config import get_settings
from app.models.schemas import ExtractedDocument, DocumentChunk
from app.ingestion.chunker import chunk_document
from app.core.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


_SHARED_LOCAL_CLIENT: Optional[QdrantClient] = None


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk identifier."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantVectorStore:
    """Manager for Qdrant vector database collection, indexing, and point operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Optional[QdrantClient] = None,
    ):
        settings = get_settings()
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION

        self._client: Optional[QdrantClient] = client

    @property
    def client(self) -> QdrantClient:
        """Lazy-loaded QdrantClient connection."""
        if self._client is None:
            # Try connecting to Qdrant server first
            try:
                server_client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=2,
                    check_compatibility=False,
                )
                server_client.get_collections()
                self._client = server_client
                logger.info(f"Connected to Qdrant server at {self.host}:{self.port}")
            except Exception as e:
                logger.info(
                    f"Qdrant server at {self.host}:{self.port} not reachable ({e}). "
                    "Using local in-memory Qdrant client."
                )
                global _SHARED_LOCAL_CLIENT
                if _SHARED_LOCAL_CLIENT is None:
                    _SHARED_LOCAL_CLIENT = QdrantClient(location=":memory:")
                self._client = _SHARED_LOCAL_CLIENT
        return self._client

    def ensure_collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Verify collection exists in Qdrant with 384-dim COSINE configuration, creating it if needed.
        
        Args:
            collection_name: Optional collection override.
            
        Returns:
            True if collection exists or was created successfully.
        """
        col_name = collection_name or self.collection_name

        try:
            if not self.client.collection_exists(col_name):
                logger.info(
                    f"Collection '{col_name}' does not exist. Creating with size={self.dimension}, distance=COSINE..."
                )
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{col_name}' created successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection '{col_name}': {e}")
            raise

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Upsert document chunks and their embeddings into Qdrant using deterministic UUIDs.
        
        Args:
            chunks: List of DocumentChunk instances.
            embeddings: Corresponding 384-dimensional embedding vectors.
            collection_name: Optional collection name override.
            
        Returns:
            Number of points upserted.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        col_name = collection_name or self.collection_name
        self.ensure_collection_exists(col_name)

        points: List[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = chunk_id_to_uuid(chunk.chunk_id)
            payload = {
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "total_pages": chunk.metadata.get("total_pages", 0),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upsert into Qdrant
        self.client.upsert(
            collection_name=col_name,
            points=points,
            wait=True,
        )

        logger.info(f"Successfully upserted {len(points)} points into '{col_name}'.")
        return len(points)

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 4,
        query_filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[Any]:
        """
        Execute cosine similarity search for a query vector in Qdrant.
        
        Args:
            query_vector: Dense embedding vector.
            limit: Maximum points to return.
            query_filter: Optional metadata Filter condition.
            score_threshold: Optional minimum similarity score cutoff.
            collection_name: Optional collection override.
            
        Returns:
            List of ScoredPoint results.
        """
        col_name = collection_name or self.collection_name
        self.ensure_collection_exists(col_name)

        response = self.client.query_points(
            collection_name=col_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold if score_threshold and score_threshold > 0.0 else None,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def index_document(
        self,
        doc: ExtractedDocument,
        embedding_service: Optional[EmbeddingService] = None,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Complete end-to-end indexing pipeline for an ExtractedDocument:
        ExtractedDocument -> Chunker -> EmbeddingService -> Qdrant Upsert.
        
        Args:
            doc: Parsed ExtractedDocument.
            embedding_service: Optional EmbeddingService instance.
            collection_name: Optional collection name override.
            
        Returns:
            Number of chunks indexed into Qdrant.
        """
        chunks = chunk_document(doc)
        if not chunks:
            logger.warning(f"Document '{doc.filename}' produced 0 chunks. Nothing to index.")
            return 0

        embedder = embedding_service or EmbeddingService()
        embeddings = embedder.embed_chunks(chunks)

        return self.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
        )

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve collection metadata, point counts, and vector configuration.
        
        Args:
            collection_name: Optional collection override.
            
        Returns:
            Dictionary with collection status, points_count, vector_size, and distance metric.
        """
        col_name = collection_name or self.collection_name
        self.ensure_collection_exists(col_name)

        info = self.client.get_collection(col_name)
        params = info.config.params.vectors

        # Handle single vs named vector configurations
        if isinstance(params, VectorParams):
            size = params.size
            distance = params.distance.name
        else:
            # Fallback for dictionary or custom config
            size = getattr(params, "size", self.dimension)
            distance = getattr(params, "distance", "COSINE")
            if hasattr(distance, "name"):
                distance = distance.name

        return {
            "name": col_name,
            "status": info.status.name if hasattr(info.status, "name") else str(info.status),
            "points_count": info.points_count or 0,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "vector_size": size,
            "distance": str(distance),
        }

    def get_points_by_document(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all stored points and payloads belonging to a specific document ID.
        
        Args:
            doc_id: SHA256 document identifier.
            collection_name: Optional collection override.
            limit: Maximum points to fetch.
            
        Returns:
            List of point records with id, payload, and chunk details.
        """
        col_name = collection_name or self.collection_name
        self.ensure_collection_exists(col_name)

        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

        records, _ = self.client.scroll(
            collection_name=col_name,
            scroll_filter=doc_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for r in records:
            results.append({
                "id": str(r.id),
                "payload": r.payload,
            })
        return results

    def get_all_documents(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all unique documents currently indexed in the collection.
        
        Returns:
            List of dictionaries: [{ doc_id, filename, total_pages, chunk_count }, ...]
        """
        col_name = collection_name or self.collection_name
        self.ensure_collection_exists(col_name)

        records, _ = self.client.scroll(
            collection_name=col_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )

        docs_map: Dict[str, Dict[str, Any]] = {}
        for r in records:
            p = r.payload or {}
            doc_id = p.get("doc_id")
            if not doc_id:
                continue

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": p.get("filename", "unknown"),
                    "total_pages": p.get("total_pages", 0),
                    "chunk_count": 0,
                }
            docs_map[doc_id]["chunk_count"] += 1

        return list(docs_map.values())

    def delete_document(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        """
        Delete all vector points belonging to a specific document ID from Qdrant.
        
        Args:
            doc_id: SHA256 document identifier.
            collection_name: Optional collection override.
            
        Returns:
            True if deletion succeeded.
        """
        col_name = collection_name or self.collection_name
        self.ensure_collection_exists(col_name)

        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

        self.client.delete(
            collection_name=col_name,
            points_selector=models.FilterSelector(filter=doc_filter),
            wait=True,
        )
        logger.info(f"Deleted document points with doc_id '{doc_id}' from '{col_name}'.")
        return True

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete an entire collection (primarily used for test tear-downs)."""
        col_name = collection_name or self.collection_name
        if self.client.collection_exists(col_name):
            self.client.delete_collection(col_name)
            logger.info(f"Deleted collection '{col_name}'.")
            return True
        return False
