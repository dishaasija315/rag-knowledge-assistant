"""Unit tests for LangGraph RAG Workflow (Phase 6, 7, 8)."""
import pytest
from app.workflows.rag_graph import RAGGraphWorkflow, build_citations_from_chunks
from app.models.schemas import RetrievedChunk, QAResponse
from app.llm.gemini_client import GeminiClient
from app.llm.prompts import STRICT_REFUSAL_MESSAGE
from app.retrieval.retriever import QdrantRetriever


@pytest.fixture
def mock_retriever():
    class MockRetriever:
        def retrieve(self, query, top_k=4, doc_id=None, doc_ids=None):
            if "leave" in query.lower():
                return [
                    RetrievedChunk(
                        chunk_id="doc1_p2_c0",
                        doc_id="doc1",
                        filename="policy_2026.pdf",
                        page_number=2,
                        chunk_index=0,
                        text="Employees receive 24 days of paid annual leave per calendar year.",
                        score=0.88,
                    )
                ]
            return []

    return MockRetriever()


@pytest.fixture
def rag_workflow(mock_retriever):
    client = GeminiClient()
    return RAGGraphWorkflow(retriever=mock_retriever, llm_client=client)


def test_build_citations():
    chunks = [
        RetrievedChunk(
            chunk_id="doc1_p2_c0",
            doc_id="doc1",
            filename="policy_2026.pdf",
            page_number=2,
            chunk_index=0,
            text="Employees receive 24 days of paid annual leave.",
            score=0.85,
        ),
        # Duplicate page chunk
        RetrievedChunk(
            chunk_id="doc1_p2_c1",
            doc_id="doc1",
            filename="policy_2026.pdf",
            page_number=2,
            chunk_index=1,
            text="Carry forward is allowed up to 10 days.",
            score=0.80,
        ),
    ]

    citations = build_citations_from_chunks(chunks)
    assert len(citations) == 1  # Deduplicated by file + page
    assert citations[0].filename == "policy_2026.pdf"
    assert citations[0].page_number == 2


def test_rag_workflow_grounded_answer(rag_workflow):
    response = rag_workflow.run(query="What is the annual leave policy?")
    assert isinstance(response, QAResponse)
    assert response.is_grounded is True
    assert len(response.citations) > 0
    assert response.citations[0].filename == "policy_2026.pdf"
    assert response.citations[0].page_number == 2


def test_rag_workflow_strict_refusal_when_unsupported(rag_workflow):
    response = rag_workflow.run(query="What is the nuclear reactor core temperature?")
    assert isinstance(response, QAResponse)
    assert STRICT_REFUSAL_MESSAGE.lower() in response.answer.lower()
    assert len(response.citations) == 0
