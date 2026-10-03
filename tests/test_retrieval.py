"""Unit and integration tests for Phase 5: Basic Retrieval."""
import pytest
from app.config import get_settings
from app.core.vector_store import QdrantVectorStore
from app.core.embeddings import EmbeddingService
from app.retrieval.retriever import QdrantRetriever
from app.models.schemas import (
    ExtractedDocument,
    ExtractedPage,
    RetrievedChunk,
    EmptyQueryError,
    InvalidTopKError,
    RetrievalError,
)

TEST_RETRIEVAL_COLLECTION = "test_rag_retrieval_pytest"


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


@pytest.fixture(scope="module")
def vector_store():
    store = QdrantVectorStore(collection_name=TEST_RETRIEVAL_COLLECTION)
    store.delete_collection(TEST_RETRIEVAL_COLLECTION)
    store.ensure_collection_exists(TEST_RETRIEVAL_COLLECTION)
    yield store
    store.delete_collection(TEST_RETRIEVAL_COLLECTION)


@pytest.fixture(scope="module")
def seeded_retriever(vector_store, embedding_service):
    # Seed sample documents into test collection
    doc_leave = ExtractedDocument(
        doc_id="doc_leave_id",
        filename="leave_policy_2026.pdf",
        total_pages=2,
        pages=[
            ExtractedPage(
                page_number=1,
                text="Employees receive 24 days of paid annual leave during each calendar year.",
                char_count=77,
            ),
            ExtractedPage(
                page_number=2,
                text="Sick leave is granted at 12 days per calendar year for medical recovery.",
                char_count=74,
            ),
        ],
    )

    doc_remote = ExtractedDocument(
        doc_id="doc_remote_id",
        filename="remote_policy_2026.pdf",
        total_pages=2,
        pages=[
            ExtractedPage(
                page_number=1,
                text="Eligible employees may work remotely for up to 3 working days per week.",
                char_count=73,
            ),
            ExtractedPage(
                page_number=2,
                text="Production application deployments require review and approval by at least two engineers.",
                char_count=92,
            ),
        ],
    )

    vector_store.index_document(doc_leave, embedding_service, collection_name=TEST_RETRIEVAL_COLLECTION)
    vector_store.index_document(doc_remote, embedding_service, collection_name=TEST_RETRIEVAL_COLLECTION)

    return QdrantRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        default_top_k=3,
    )


def test_successful_semantic_retrieval(seeded_retriever):
    query = "How many annual leave days do employees get?"
    results = seeded_retriever.retrieve(query=query, top_k=2, collection_name=TEST_RETRIEVAL_COLLECTION)

    assert len(results) == 2
    top_hit = results[0]

    assert isinstance(top_hit, RetrievedChunk)
    assert top_hit.doc_id == "doc_leave_id"
    assert top_hit.filename == "leave_policy_2026.pdf"
    assert top_hit.page_number == 1
    assert "24 days of paid annual leave" in top_hit.text
    assert top_hit.score > 0.4  # Strong similarity score


def test_retrieval_metadata_preservation(seeded_retriever):
    query = "remote work allowance"
    results = seeded_retriever.retrieve(query=query, top_k=1, collection_name=TEST_RETRIEVAL_COLLECTION)

    assert len(results) == 1
    hit = results[0]
    assert hit.filename == "remote_policy_2026.pdf"
    assert hit.page_number == 1
    assert hit.chunk_id == "doc_remote_id_p1_c0"
    assert hit.chunk_index == 0
    assert "3 working days per week" in hit.text
    assert "total_pages" in hit.metadata


def test_top_k_limit_behavior(seeded_retriever):
    query = "policy guidelines"
    results_1 = seeded_retriever.retrieve(query=query, top_k=1, collection_name=TEST_RETRIEVAL_COLLECTION)
    results_3 = seeded_retriever.retrieve(query=query, top_k=3, collection_name=TEST_RETRIEVAL_COLLECTION)

    assert len(results_1) == 1
    assert len(results_3) == 3


def test_similarity_score_ordering(seeded_retriever):
    query = "medical sick leave and doctor appointments"
    results = seeded_retriever.retrieve(query=query, top_k=4, collection_name=TEST_RETRIEVAL_COLLECTION)

    assert len(results) >= 2
    # Verify scores are sorted in descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_single_document_filter(seeded_retriever):
    query = "policy rules and requirements"
    # Filter explicitly to doc_remote_id
    results = seeded_retriever.retrieve(
        query=query,
        top_k=4,
        doc_id="doc_remote_id",
        collection_name=TEST_RETRIEVAL_COLLECTION,
    )

    assert len(results) > 0
    assert all(r.doc_id == "doc_remote_id" for r in results)
    assert all(r.filename == "remote_policy_2026.pdf" for r in results)


def test_multi_document_filter(seeded_retriever):
    query = "work rules"
    results = seeded_retriever.retrieve(
        query=query,
        top_k=4,
        doc_ids=["doc_leave_id"],
        collection_name=TEST_RETRIEVAL_COLLECTION,
    )

    assert len(results) > 0
    assert all(r.doc_id == "doc_leave_id" for r in results)


def test_empty_query_raises_empty_query_error(seeded_retriever):
    with pytest.raises(EmptyQueryError, match="cannot be empty"):
        seeded_retriever.retrieve("", collection_name=TEST_RETRIEVAL_COLLECTION)

    with pytest.raises(EmptyQueryError, match="cannot be empty"):
        seeded_retriever.retrieve("   \n\t  ", collection_name=TEST_RETRIEVAL_COLLECTION)


def test_invalid_top_k_raises_invalid_top_k_error(seeded_retriever):
    with pytest.raises(InvalidTopKError, match="greater than or equal to 1"):
        seeded_retriever.retrieve("test query", top_k=0, collection_name=TEST_RETRIEVAL_COLLECTION)

    with pytest.raises(InvalidTopKError, match="greater than or equal to 1"):
        seeded_retriever.retrieve("test query", top_k=-5, collection_name=TEST_RETRIEVAL_COLLECTION)
