"""Unit tests for Document Comparison, Contradiction Detection, and Summarization (Phase 9, 10)."""
import pytest
from app.comparison.comparator import DocumentComparator
from app.models.schemas import RetrievedChunk, CompareResponse, ContradictionResponse, SummarizeResponse
from app.llm.gemini_client import GeminiClient


@pytest.fixture
def mock_comparator_retriever():
    class MockRetriever:
        def retrieve(self, query, top_k=5, doc_id=None, doc_ids=None):
            if doc_id == "doc_2025":
                return [
                    RetrievedChunk(
                        chunk_id="doc2025_p2_c0",
                        doc_id="doc_2025",
                        filename="policy_2025.pdf",
                        page_number=2,
                        chunk_index=0,
                        text="Employees receive 15 days of annual leave.",
                        score=0.85,
                    )
                ]
            elif doc_id == "doc_2026":
                return [
                    RetrievedChunk(
                        chunk_id="doc2026_p2_c0",
                        doc_id="doc_2026",
                        filename="policy_2026.pdf",
                        page_number=2,
                        chunk_index=0,
                        text="Employees receive 24 days of annual leave.",
                        score=0.89,
                    )
                ]
            elif "leave" in query.lower() or "conflict" in query.lower():
                return [
                    RetrievedChunk(
                        chunk_id="doc2025_p2_c0",
                        doc_id="doc_2025",
                        filename="policy_2025.pdf",
                        page_number=2,
                        chunk_index=0,
                        text="Employees receive 20 days of annual leave.",
                        score=0.85,
                    ),
                    RetrievedChunk(
                        chunk_id="doc2026_p2_c0",
                        doc_id="doc_2026",
                        filename="policy_2026.pdf",
                        page_number=2,
                        chunk_index=0,
                        text="Employees receive 24 days of annual leave.",
                        score=0.89,
                    ),
                ]
            return []

    return MockRetriever()


@pytest.fixture
def comparator(mock_comparator_retriever):
    client = GeminiClient()
    return DocumentComparator(retriever=mock_comparator_retriever, llm_client=client)


def test_compare_documents(comparator):
    res = comparator.compare_documents(
        doc_id_a="doc_2025",
        doc_id_b="doc_2026",
        topic="annual leave days",
    )
    assert isinstance(res, CompareResponse)
    assert res.filename_a == "policy_2025.pdf"
    assert res.filename_b == "policy_2026.pdf"
    assert len(res.citations_doc_a) > 0
    assert len(res.citations_doc_b) > 0


def test_detect_contradictions(comparator):
    res = comparator.detect_contradictions(topic="annual leave days conflict")
    assert isinstance(res, ContradictionResponse)
    assert res.has_contradictions is True
    assert len(res.contradictions) > 0
    assert res.contradictions[0].page_a == 1 or res.contradictions[0].page_a == 2


def test_summarize_topic(comparator):
    res = comparator.summarize_topic(topic="annual leave", doc_id="doc_2026")
    assert isinstance(res, SummarizeResponse)
    assert res.topic == "annual leave"
    assert len(res.citations) > 0
    assert res.citations[0].filename == "policy_2026.pdf"
