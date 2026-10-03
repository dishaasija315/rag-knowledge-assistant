"""Embedding generation module using SentenceTransformers (all-MiniLM-L6-v2)."""
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

# Global singleton holder for embedding model
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        model_name = get_settings().EMBEDDING_MODEL_NAME
        logger.info(f"Loading SentenceTransformer model: {model_name} ...")
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded successfully.")
    return _EMBEDDING_MODEL


class EmbeddingService:
    """Service providing dense vector embeddings using local SentenceTransformer."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

    @property
    def model(self) -> SentenceTransformer:
        return get_embedding_model()

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a 384-dimensional embedding vector for a single string.
        
        Args:
            text: Input text query or sentence.
            
        Returns:
            List of floats representing the dense vector.
        """
        if not text or not text.strip():
            # Return zero vector if empty
            return [0.0] * self.dimension

        vector = self.model.encode(
            text.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vector.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings in batches.
        
        Args:
            texts: List of strings.
            batch_size: Processing batch size.
            
        Returns:
            List of embedding vectors (each vector is a list of floats).
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]
        vectors = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return [vec.tolist() for vec in vectors]

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of DocumentChunk instances.
        
        Args:
            chunks: List of DocumentChunk models.
            batch_size: Processing batch size.
            
        Returns:
            List of embedding vectors corresponding 1:1 with input chunks.
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)
