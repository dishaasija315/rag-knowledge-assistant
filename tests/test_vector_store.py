"""Unit and integration tests for Phase 4: Qdrant Vector Store."""
import pytest
from app.config import get_settings
from app.core.vector_store import QdrantVectorStore, chunk_id_to_uuid
from app.core.embeddings import EmbeddingService
from app.models.schemas import ExtractedDocument, ExtractedPage, DocumentChunk
from app.ingestion.chunker import chunk_document


TEST_COLLECTION_NAME = "test_rag_documents_pytest"


@pytest.fixture(scope="module")
def vector_store():
    store = QdrantVectorStore(collection_name=TEST_COLLECTION_NAME)
    # Ensure fresh test collection
    store.delete_collection(TEST_COLLECTION_NAME)
    store.ensure_collection_exists(TEST_COLLECTION_NAME)
    yield store
    # Teardown: delete test collection
    store.delete_collection(TEST_COLLECTION_NAME)


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()


@pytest.fixture
def sample_document():
    pages = [
        ExtractedPage(
            page_number=1,
            text="Employee Leave Policy 2026. Employees receive 24 days of paid annual leave.",
            char_count=77,
        ),
        ExtractedPage(
            page_number=2,
            text="Remote Work Policy. Employees may work remotely up to 3 days per week.",
            char_count=71,
        ),
    ]
    return ExtractedDocument(
        doc_id="test_doc_sha256_hash_123",
        filename="test_handbook_2026.pdf",
        total_pages=2,
        pages=pages,
    )


def test_qdrant_collection_creation_and_config(vector_store):
    info = vector_store.get_collection_info(TEST_COLLECTION_NAME)

    assert info["name"] == TEST_COLLECTION_NAME
    assert info["vector_size"] == 384
    assert info["distance"] in ["Cosine", "COSINE"]
    assert info["status"] in ["green", "yellow", "ok", "GREEN", "OK"]


def test_deterministic_uuid_generation():
    chunk_id = "doc123_p1_c0"
    uuid1 = chunk_id_to_uuid(chunk_id)
    uuid2 = chunk_id_to_uuid(chunk_id)

    assert uuid1 == uuid2
    assert len(uuid1) == 36
    assert chunk_id_to_uuid("doc123_p1_c1") != uuid1


def test_index_document_and_payload_preservation(vector_store, embedding_service, sample_document):
    indexed_count = vector_store.index_document(
        doc=sample_document,
        embedding_service=embedding_service,
        collection_name=TEST_COLLECTION_NAME,
    )

    assert indexed_count == 2

    # Check collection points
    info = vector_store.get_collection_info(TEST_COLLECTION_NAME)
    assert info["points_count"] == 2

    # Retrieve points by document ID
    points = vector_store.get_points_by_document(sample_document.doc_id, collection_name=TEST_COLLECTION_NAME)
    assert len(points) == 2

    # Verify payload metadata on first point
    payloads = [p["payload"] for p in points]
    p1 = next(p for p in payloads if p["page_number"] == 1)
    assert p1["doc_id"] == sample_document.doc_id
    assert p1["filename"] == "test_handbook_2026.pdf"
    assert p1["page_number"] == 1
    assert p1["chunk_index"] == 0
    assert p1["total_pages"] == 2
    assert "24 days of paid annual leave" in p1["text"]

    p2 = next(p for p in payloads if p["page_number"] == 2)
    assert p2["page_number"] == 2
    assert "3 days per week" in p2["text"]


def test_deterministic_reindexing_prevents_duplicates(vector_store, embedding_service, sample_document):
    # Re-index the exact same document
    reindexed_count = vector_store.index_document(
        doc=sample_document,
        embedding_service=embedding_service,
        collection_name=TEST_COLLECTION_NAME,
    )

    assert reindexed_count == 2

    # Points count in Qdrant must remain 2, not 4!
    info = vector_store.get_collection_info(TEST_COLLECTION_NAME)
    assert info["points_count"] == 2


def test_get_all_documents_and_delete_document(vector_store, sample_document):
    all_docs = vector_store.get_all_documents(collection_name=TEST_COLLECTION_NAME)
    assert len(all_docs) == 1
    assert all_docs[0]["doc_id"] == sample_document.doc_id
    assert all_docs[0]["filename"] == "test_handbook_2026.pdf"
    assert all_docs[0]["chunk_count"] == 2

    # Delete document
    deleted = vector_store.delete_document(sample_document.doc_id, collection_name=TEST_COLLECTION_NAME)
    assert deleted is True

    # Check points after deletion
    info = vector_store.get_collection_info(TEST_COLLECTION_NAME)
    assert info["points_count"] == 0
