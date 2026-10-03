"""LangGraph workflow for orchestrating Query Analysis, Retrieval, Grounded Generation, and Citations."""
import logging
from typing import List, Optional, Dict, Any, TypedDict

from langgraph.graph import StateGraph, END

from app.config import get_settings
from app.models.schemas import RetrievedChunk, Citation, QAResponse
from app.retrieval.retriever import QdrantRetriever, get_retriever
from app.llm.gemini_client import GeminiClient
from app.llm.prompts import STRICT_REFUSAL_MESSAGE

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """LangGraph state representation for the RAG lifecycle."""
    query: str
    analyzed_query: str
    top_k: int
    doc_id: Optional[str]
    doc_ids: Optional[List[str]]
    retrieved_chunks: List[RetrievedChunk]
    generated_answer: str
    final_answer: str
    is_grounded: bool
    citations: List[Citation]
    retry_count: int


def build_citations_from_chunks(chunks: List[RetrievedChunk]) -> List[Citation]:
    """Convert RetrievedChunks into structured Citations, deduplicating by file and page."""
    citations: List[Citation] = []
    seen = set()

    for chunk in chunks:
        key = (chunk.filename, chunk.page_number)
        if key not in seen:
            seen.add(key)
            # Create readable snippet
            snippet = chunk.text.strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."

            citations.append(
                Citation(
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    snippet=snippet,
                    score=chunk.score,
                )
            )
    return citations


class RAGGraphWorkflow:
    """Orchestrates RAG execution via LangGraph."""

    def __init__(
        self,
        retriever: Optional[QdrantRetriever] = None,
        llm_client: Optional[GeminiClient] = None,
    ):
        self.retriever = retriever or get_retriever()
        self.llm_client = llm_client or GeminiClient()
        self.graph = self._build_graph()

    def _analyze_query_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 1: Analyze user question and extract optimized search terms."""
        query = state["query"]
        try:
            analyzed = self.llm_client.analyze_query(query)
            logger.info(f"[Workflow] Query analyzed: '{query}' -> '{analyzed}'")
            return {"analyzed_query": analyzed}
        except Exception as e:
            logger.warning(f"[Workflow] Query analysis fallback: {e}")
            return {"analyzed_query": query}

    def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 2: Retrieve relevant chunks from Qdrant using vector search."""
        search_query = state.get("analyzed_query") or state["query"]
        top_k = state.get("top_k", get_settings().DEFAULT_TOP_K)
        doc_id = state.get("doc_id")
        doc_ids = state.get("doc_ids")

        try:
            chunks = self.retriever.retrieve(
                query=search_query,
                top_k=top_k,
                doc_id=doc_id,
                doc_ids=doc_ids,
            )
            logger.info(f"[Workflow] Retrieved {len(chunks)} chunks.")
            return {"retrieved_chunks": chunks}
        except Exception as e:
            logger.error(f"[Workflow] Retrieval error: {e}")
            return {"retrieved_chunks": []}

    def _generate_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 3: Send retrieved context to Gemini and generate grounded answer."""
        chunks = state.get("retrieved_chunks", [])
        query = state["query"]

        if not chunks:
            logger.info("[Workflow] No chunks retrieved. Triggering strict refusal.")
            return {"generated_answer": STRICT_REFUSAL_MESSAGE}

        answer = self.llm_client.generate_grounded_answer(
            question=query,
            context_chunks=chunks,
        )
        return {"generated_answer": answer}

    def _validate_grounding_node(self, state: RAGState) -> Dict[str, Any]:
        """Node 4: Validate answer grounding against retrieved chunks and attach citations."""
        generated = state.get("generated_answer", "").strip()
        chunks = state.get("retrieved_chunks", [])
        query = state.get("query", "")

        # Check for refusal message
        if not generated or STRICT_REFUSAL_MESSAGE.lower() in generated.lower():
            return {
                "final_answer": STRICT_REFUSAL_MESSAGE,
                "is_grounded": True,
                "citations": [],
            }

        # Check grounding
        is_grounded = self.llm_client.validate_grounding(
            question=query,
            answer=generated,
            context_chunks=chunks,
        )

        if not is_grounded:
            logger.warning("[Workflow] Grounding validation failed. Enforcing strict refusal.")
            return {
                "final_answer": STRICT_REFUSAL_MESSAGE,
                "is_grounded": False,
                "citations": [],
            }

        # Build real citations from retrieved chunks
        citations = build_citations_from_chunks(chunks)

        return {
            "final_answer": generated,
            "is_grounded": True,
            "citations": citations,
        }

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph."""
        workflow = StateGraph(RAGState)

        # Register nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate_grounding", self._validate_grounding_node)

        # Set entry point & flow edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate_grounding")
        workflow.add_edge("validate_grounding", END)

        return workflow.compile()

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> QAResponse:
        """
        Execute the full RAG LangGraph workflow for a user query.
        
        Args:
            query: User question.
            top_k: Number of chunks to retrieve.
            doc_id: Optional single document ID filter.
            doc_ids: Optional multiple document IDs filter.
            
        Returns:
            QAResponse model with answer, citations, and grounding status.
        """
        settings = get_settings()
        k = top_k or settings.DEFAULT_TOP_K

        initial_state: RAGState = {
            "query": query,
            "analyzed_query": query,
            "top_k": k,
            "doc_id": doc_id,
            "doc_ids": doc_ids,
            "retrieved_chunks": [],
            "generated_answer": "",
            "final_answer": "",
            "is_grounded": True,
            "citations": [],
            "retry_count": 0,
        }

        # Execute compiled LangGraph
        result_state = self.graph.invoke(initial_state)

        return QAResponse(
            query=query,
            answer=result_state["final_answer"],
            citations=result_state.get("citations", []),
            is_grounded=result_state.get("is_grounded", True),
            retrieved_count=len(result_state.get("retrieved_chunks", [])),
        )


# Singleton workflow instance
_DEFAULT_RAG_WORKFLOW: Optional[RAGGraphWorkflow] = None


def get_rag_workflow() -> RAGGraphWorkflow:
    """Get or instantiate singleton RAGGraphWorkflow."""
    global _DEFAULT_RAG_WORKFLOW
    if _DEFAULT_RAG_WORKFLOW is None:
        _DEFAULT_RAG_WORKFLOW = RAGGraphWorkflow()
    return _DEFAULT_RAG_WORKFLOW
