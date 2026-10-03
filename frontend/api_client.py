"""API client for connecting Streamlit frontend to FastAPI backend."""
import logging
from typing import List, Optional, Dict, Any
import requests

from app.config import get_settings

logger = logging.getLogger(__name__)


class APIClient:
    """Client for interacting with the RAG Knowledge Assistant FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            settings = get_settings()
            self.base_url = f"http://localhost:{settings.API_PORT}"
        else:
            self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        """Check backend health status."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        return {"status": "offline", "qdrant_status": "disconnected", "gemini_configured": False}

    def upload_pdf(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Upload a PDF file to the backend for indexing."""
        files = {"file": (filename, file_bytes, "application/pdf")}
        resp = requests.post(f"{self.base_url}/documents/upload", files=files, timeout=60)
        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Upload failed: {error_detail}")
        return resp.json()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieve list of currently indexed documents."""
        try:
            resp = requests.get(f"{self.base_url}/documents", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("documents", [])
        except Exception as e:
            logger.warning(f"Failed to list documents: {e}")
        return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID."""
        resp = requests.delete(f"{self.base_url}/documents/{doc_id}", timeout=10)
        return resp.status_code == 200

    def query_rag(
        self,
        query: str,
        top_k: int = 4,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Ask a grounded question across uploaded documents."""
        payload = {
            "query": query,
            "top_k": top_k,
            "doc_ids": doc_ids,
        }
        resp = requests.post(f"{self.base_url}/query", json=payload, timeout=60)
        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Query error: {error_detail}")
        return resp.json()

    def compare_documents(
        self,
        doc_id_a: str,
        doc_id_b: str,
        topic: str = "",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Compare two documents on a specified topic."""
        payload = {
            "doc_id_a": doc_id_a,
            "doc_id_b": doc_id_b,
            "topic": topic or "General policies and differences",
            "top_k": top_k,
        }
        resp = requests.post(f"{self.base_url}/compare", json=payload, timeout=60)
        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Comparison error: {error_detail}")
        return resp.json()

    def detect_contradictions(
        self,
        topic: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """Detect conflicting statements on a topic."""
        payload = {
            "topic": topic,
            "doc_ids": doc_ids,
            "top_k": top_k,
        }
        resp = requests.post(f"{self.base_url}/contradictions", json=payload, timeout=60)
        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Contradiction check error: {error_detail}")
        return resp.json()

    def summarize_topic(
        self,
        topic: str,
        doc_id: Optional[str] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """Summarize sections of a document matching a topic."""
        payload = {
            "topic": topic,
            "doc_id": doc_id,
            "top_k": top_k,
        }
        resp = requests.post(f"{self.base_url}/summarize", json=payload, timeout=60)
        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Summarization error: {error_detail}")
        return resp.json()

    def run_evaluation(self, top_k: int = 4) -> Dict[str, Any]:
        """Run quantitative benchmark evaluation."""
        resp = requests.get(f"{self.base_url}/evaluation?top_k={top_k}", timeout=120)
        if resp.status_code != 200:
            error_detail = resp.json().get("detail", resp.text)
            raise RuntimeError(f"Evaluation error: {error_detail}")
        return resp.json()
