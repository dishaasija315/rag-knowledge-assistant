"""Integration tests for FastAPI endpoints (Phase 12)."""
import io
import pytest
from fastapi.testclient import TestClient
import pymupdf

from app.main import app
from app.core.vector_store import QdrantVectorStore


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_mock_pdf_bytes(text: str = "Test PDF Content for API") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_upload_and_list_document_flow(client):
    pdf_bytes = create_mock_pdf_bytes("Employee Attendance Guidelines 2026. Standard hours 9am to 6pm.")
    file_obj = io.BytesIO(pdf_bytes)

    # 1. Upload
    response = client.post(
        "/documents/upload",
        files={"file": ("api_test_handbook.pdf", file_obj, "application/pdf")},
    )
    assert response.status_code == 200
    upload_data = response.json()
    assert upload_data["filename"] == "api_test_handbook.pdf"
    assert upload_data["chunks_indexed"] > 0
    doc_id = upload_data["doc_id"]

    # 2. List
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Query
    query_res = client.post(
        "/query",
        json={"query": "What are standard working hours?", "top_k": 2},
    )
    assert query_res.status_code == 200
    qa_data = query_res.json()
    assert "answer" in qa_data
    assert "citations" in qa_data

    # 4. Delete
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["doc_id"] == doc_id


def test_upload_invalid_file_type(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"Not a PDF"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]
