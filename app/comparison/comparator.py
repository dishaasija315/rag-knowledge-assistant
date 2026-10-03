"""Document comparison, contradiction detection, and targeted summarization module."""
import logging
from typing import List, Optional

from app.models.schemas import (
    CompareResponse,
    Citation,
    ContradictionResponse,
    ContradictionItem,
    SummarizeResponse,
)
from app.retrieval.retriever import QdrantRetriever, get_retriever
from app.llm.gemini_client import GeminiClient
from app.workflows.rag_graph import build_citations_from_chunks

logger = logging.getLogger(__name__)


class DocumentComparator:
    """Service providing RAG-driven document comparison, contradiction detection, and summarization."""

    def __init__(
        self,
        retriever: Optional[QdrantRetriever] = None,
        llm_client: Optional[GeminiClient] = None,
    ):
        self.retriever = retriever or get_retriever()
        self.llm_client = llm_client or GeminiClient()

    def compare_documents(
        self,
        doc_id_a: str,
        doc_id_b: str,
        topic: Optional[str] = None,
        top_k: int = 5,
    ) -> CompareResponse:
        """
        Compare two documents on a specified topic using targeted semantic retrieval.
        
        Args:
            doc_id_a: SHA256 document ID of Document A.
            doc_id_b: SHA256 document ID of Document B.
            topic: Focus topic for comparison.
            top_k: Number of chunks to retrieve per document.
            
        Returns:
            CompareResponse with comparative summary and dual citations.
        """
        comparison_topic = topic.strip() if topic and topic.strip() else "policies, rules, and general terms"

        # 1. Retrieve relevant chunks for Document A
        chunks_a = self.retriever.retrieve(
            query=comparison_topic,
            top_k=top_k,
            doc_id=doc_id_a,
        )

        # 2. Retrieve relevant chunks for Document B
        chunks_b = self.retriever.retrieve(
            query=comparison_topic,
            top_k=top_k,
            doc_id=doc_id_b,
        )

        filename_a = chunks_a[0].filename if chunks_a else "Document A"
        filename_b = chunks_b[0].filename if chunks_b else "Document B"

        # 3. Generate comparative synthesis via Gemini
        summary = self.llm_client.compare_contexts(
            topic=comparison_topic,
            filename_a=filename_a,
            chunks_a=chunks_a,
            filename_b=filename_b,
            chunks_b=chunks_b,
        )

        citations_a = build_citations_from_chunks(chunks_a)
        citations_b = build_citations_from_chunks(chunks_b)

        return CompareResponse(
            doc_id_a=doc_id_a,
            filename_a=filename_a,
            doc_id_b=doc_id_b,
            filename_b=filename_b,
            topic=comparison_topic,
            comparison_summary=summary,
            citations_doc_a=citations_a,
            citations_doc_b=citations_b,
        )

    def detect_contradictions(
        self,
        topic: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 6,
    ) -> ContradictionResponse:
        """
        Identify factual contradictions across documents for a given topic.
        
        Args:
            topic: Policy/fact area to analyze for conflicts.
            doc_ids: Optional list of document IDs to scope search.
            top_k: Number of context chunks to analyze.
            
        Returns:
            ContradictionResponse containing structured contradiction items.
        """
        cleaned_topic = topic.strip()

        # Retrieve relevant chunks across documents
        chunks = self.retriever.retrieve(
            query=cleaned_topic,
            top_k=top_k,
            doc_ids=doc_ids,
        )

        if not chunks:
            return ContradictionResponse(
                topic=cleaned_topic,
                has_contradictions=False,
                explanation="No relevant context found in documents to evaluate contradictions.",
                contradictions=[],
                inspected_citations=[],
            )

        # Analyze using Gemini
        analysis = self.llm_client.detect_contradictions(
            topic=cleaned_topic,
            chunks=chunks,
        )

        contradiction_items: List[ContradictionItem] = []
        for item in analysis.get("contradictions", []):
            try:
                contradiction_items.append(
                    ContradictionItem(
                        claim_a=str(item.get("claim_a", "")),
                        source_a=str(item.get("source_a", "Document A")),
                        page_a=int(item.get("page_a", 1)),
                        claim_b=str(item.get("claim_b", "")),
                        source_b=str(item.get("source_b", "Document B")),
                        page_b=int(item.get("page_b", 1)),
                        explanation=str(item.get("explanation", "")),
                    )
                )
            except Exception as e:
                logger.warning(f"Error parsing contradiction item: {e}")

        citations = build_citations_from_chunks(chunks)

        return ContradictionResponse(
            topic=cleaned_topic,
            has_contradictions=bool(analysis.get("has_contradictions", False) or len(contradiction_items) > 0),
            explanation=str(analysis.get("explanation", "Analysis complete.")),
            contradictions=contradiction_items,
            inspected_citations=citations,
        )

    def summarize_topic(
        self,
        topic: str,
        doc_id: Optional[str] = None,
        top_k: int = 6,
    ) -> SummarizeResponse:
        """
        Generate targeted summary of sections matching a topic.
        
        Args:
            topic: Topic keyword/phrase.
            doc_id: Optional document ID to scope summary.
            top_k: Number of chunks to summarize.
            
        Returns:
            SummarizeResponse with summary text and citations.
        """
        cleaned_topic = topic.strip()
        chunks = self.retriever.retrieve(
            query=cleaned_topic,
            top_k=top_k,
            doc_id=doc_id,
        )

        summary_text = self.llm_client.summarize_context(
            topic=cleaned_topic,
            chunks=chunks,
        )

        citations = build_citations_from_chunks(chunks)

        return SummarizeResponse(
            topic=cleaned_topic,
            summary=summary_text,
            citations=citations,
        )


# Singleton comparator instance
_DEFAULT_COMPARATOR: Optional[DocumentComparator] = None


def get_comparator() -> DocumentComparator:
    """Get or instantiate singleton DocumentComparator."""
    global _DEFAULT_COMPARATOR
    if _DEFAULT_COMPARATOR is None:
        _DEFAULT_COMPARATOR = DocumentComparator()
    return _DEFAULT_COMPARATOR
