"""Unit tests for Phase 3: Local Embeddings generation with SentenceTransformers."""
import math
import pytest
from app.core.embeddings import EmbeddingService
from app.models.schemas import DocumentChunk


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def test_embedding_dimension(embedding_service):
    text = "What is the annual leave allowance?"
    vector = embedding_service.embed_text(text)

    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(val, float) for val in vector)


def test_batch_embeddings(embedding_service):
    texts = [
        "First sentence for embedding.",
        "Second sentence about remote working policy.",
        "Third sentence on probation periods.",
    ]
    vectors = embedding_service.embed_texts(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384


def test_embed_chunks(embedding_service):
    chunks = [
        DocumentChunk(
            chunk_id="doc1_p1_c0",
            doc_id="doc1",
            filename="policy.pdf",
            page_number=1,
            chunk_index=0,
            text="Employees get 15 vacation days per year.",
            char_count=40,
        ),
        DocumentChunk(
            chunk_id="doc1_p1_c1",
            doc_id="doc1",
            filename="policy.pdf",
            page_number=1,
            chunk_index=1,
            text="Health insurance covers dental and vision care.",
            char_count=47,
        ),
    ]

    vectors = embedding_service.embed_chunks(chunks)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_semantic_similarity(embedding_service):
    # Test semantic relevance
    vec_query = embedding_service.embed_text("vacation days and time off")
    vec_relevant = embedding_service.embed_text("Employees are eligible for 15 annual paid leave days.")
    vec_irrelevant = embedding_service.embed_text("The internal combustion engine operates on a four-stroke cycle.")

    sim_relevant = cosine_similarity(vec_query, vec_relevant)
    sim_irrelevant = cosine_similarity(vec_query, vec_irrelevant)

    assert sim_relevant > sim_irrelevant
    assert sim_relevant > 0.4  # Strong semantic connection
