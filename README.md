# RAG Knowledge Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for document Q&A, multi-document comparison, contradiction detection, and targeted summarization — built with FastAPI, Streamlit, LangChain, LangGraph, Qdrant, Sentence Transformers, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-Text%20Splitting-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)
![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-blue?logo=google)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [How It Works (Technical Deep-Dive)](#how-it-works-technical-deep-dive)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Interview Q&A](#interview-qa)

---

## Overview

Users upload PDF documents and ask questions about them. The system:

1. **Ingests** PDFs using PyMuPDF, preserving page-level metadata.
2. **Chunks** text using LangChain's `RecursiveCharacterTextSplitter` with full provenance tracking.
3. **Embeds** chunks locally using Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions).
4. **Stores** embeddings and metadata in a persistent Qdrant vector database.
5. **Retrieves** the most relevant chunks via cosine similarity search.
6. **Generates** strictly grounded answers using Google Gemini with real source citations.
7. **Orchestrates** the entire RAG workflow using LangGraph state machines.

The system is **strictly grounded** — if the uploaded documents don't contain the answer, the system explicitly says:

> *"I couldn't find this information in the uploaded documents."*

No hallucination. No fabricated citations. Every claim is traceable to an exact filename and page number.

---

## Key Features

| Feature | Description |
|---|---|
| **Document Q&A** | Ask questions and get answers grounded strictly in uploaded PDFs with exact source citations |
| **Multi-Document Search** | Search across all uploaded documents or filter by specific documents |
| **Document Comparison** | Compare two documents on a topic — identify additions, modifications, and removals |
| **Contradiction Detection** | Detect genuine factual conflicts across documents (ignoring version/date differences) |
| **Targeted Summarization** | Retrieve and summarize only sections relevant to a specific topic |
| **Strict Grounding** | LLM answers only from retrieved context; refuses ungrounded queries explicitly |
| **Real Citations** | Every answer includes verified filename + page number citations from actual chunks |
| **LangGraph Orchestration** | Multi-step RAG pipeline: Query Analysis → Retrieval → Generation → Grounding Validation |
| **Persistent Vector Store** | Qdrant Docker volume ensures vectors survive container restarts |
| **Evaluation Framework** | Automated benchmark measuring retrieval recall, grounding, citation accuracy, and refusal rate |

---

## Architecture

```
                                 [ User / Streamlit UI ]
                                            |
                                            v (HTTP)
                                     [ FastAPI Backend ]
                                            |
                  +-------------------------+-------------------------+
                  | (Upload Flow)                                     | (Query / Compare / Contradict / Summarize)
                  v                                                   v
         [ Save PDF to Local ]                                [ Query Embedding ]
        (data/documents/*.pdf)                              (all-MiniLM-L6-v2, 384-dim)
                  |                                                   |
                  v                                                   v
          [ PyMuPDF Parsing ]                                [ Qdrant Vector Search ]
        (Text + Page metadata)                               (Cosine Distance Top-K)
                  |                                                   |
                  v                                                   v
        [ Chunking (LangChain) ]                             [ Retrieved Chunks ]
     (doc_id, chunk_id, page, file)                       (Exact payload & metadata)
                  |                                                   |
                  v                                                   v
      [ Embeddings (MiniLM-L6) ]                            [ LangGraph Workflow ]
                  |                                         Query Analysis → Retrieve
                  v                                         → Generate → Validate
         [ Store in Qdrant ]                                          |
      (Docker Volume Persistent)                                      v
                                                            [ Grounded Answer + Citations ]
```

### LangGraph Workflow (4-Node State Machine)

```
┌──────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Analyze Query   │───>│   Retrieve   │───>│   Generate   │───>│  Validate Grounding  │───> END
│ (Optimize search │    │ (Qdrant top-k│    │ (Gemini with │    │ (Check answer against │
│  keywords)       │    │  cosine sim) │    │  context)    │    │  context, attach      │
└──────────────────┘    └──────────────┘    └──────────────┘    │  real citations)      │
                                                                └──────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Backend API** | FastAPI | REST endpoints for upload, query, compare, summarize |
| **Frontend** | Streamlit | Interactive chat UI with tabs for each feature |
| **PDF Parsing** | PyMuPDF (fitz) | Fast, accurate text extraction with page-level metadata |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Semantic text splitting with provenance metadata |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Local 384-dim dense vectors, no API cost |
| **Vector Database** | Qdrant (Docker) | Cosine similarity search with payload filtering |
| **LLM** | Google Gemini 1.5 Flash (free tier) | Grounded text generation with strict system prompts |
| **Workflow** | LangGraph | Stateful multi-step RAG orchestration |
| **Data Validation** | Pydantic v2 | Request/response schemas and settings management |
| **Testing** | Pytest | 39 unit and integration tests |

**Cost: 100% free/local** (local embeddings, local Qdrant, Gemini free tier).

---

## Project Structure

```
rag-knowledge-assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entrypoint
│   ├── config.py                   # Pydantic settings from .env
│   ├── api/
│   │   └── routes.py               # REST API endpoints (upload, query, compare, etc.)
│   ├── core/
│   │   ├── vector_store.py         # Qdrant client, collection CRUD, indexing pipeline
│   │   └── embeddings.py           # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   ├── ingestion/
│   │   ├── pdf_loader.py           # PyMuPDF text extraction with SHA256 doc hashing
│   │   └── chunker.py              # LangChain text splitter with metadata preservation
│   ├── retrieval/
│   │   └── retriever.py            # Cosine similarity retrieval with doc_id filtering
│   ├── llm/
│   │   ├── gemini_client.py        # Gemini API client with offline fallback
│   │   └── prompts.py              # Strictly grounded system prompts
│   ├── workflows/
│   │   └── rag_graph.py            # LangGraph 4-node workflow (analyze→retrieve→generate→validate)
│   ├── comparison/
│   │   └── comparator.py           # Document comparison, contradiction, and summarization
│   ├── evaluation/
│   │   └── evaluator.py            # Automated RAG benchmark framework
│   └── models/
│       └── schemas.py              # Pydantic models for all request/response payloads
│
├── frontend/
│   ├── app.py                      # Streamlit UI (5 tabs: Q&A, Compare, Contradict, Summarize, Eval)
│   └── api_client.py               # HTTP client for FastAPI backend
│
├── data/
│   ├── documents/                  # Uploaded PDF storage
│   ├── qdrant_storage/             # Persistent Qdrant volume
│   └── evaluation_dataset.json     # Benchmark test cases
│
├── tests/                          # 39 passing tests
│   ├── test_ingestion.py           # PDF loading and validation
│   ├── test_chunking.py            # Text splitting and metadata
│   ├── test_embeddings.py          # Embedding dimensions and semantics
│   ├── test_vector_store.py        # Qdrant CRUD and persistence
│   ├── test_retrieval.py           # Semantic search and filtering
│   ├── test_rag_workflow.py        # LangGraph workflow and grounding
│   ├── test_comparison.py          # Compare, contradict, summarize
│   ├── test_evaluation.py          # Evaluation framework
│   └── test_api.py                 # FastAPI endpoint tests
│
├── scripts/
│   ├── sample_docs_generator.py    # Generate test PDFs with known differences
│   └── check_health.py             # System health verification
│
├── .env.example                    # Template environment variables
├── .gitignore
├── requirements.txt
├── docker-compose.yml              # Qdrant with persistent volume
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Docker Desktop (for Qdrant)
- Google Gemini API key ([free tier](https://aistudio.google.com/apikey))

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-knowledge-assistant.git
cd rag-knowledge-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL_NAME=gemini-1.5-flash

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_documents

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_TOP_K=4
```

### 5. Start Qdrant Vector Database

```bash
docker compose up -d
```

Verify Qdrant is running:

```bash
curl http://localhost:6333/readyz
# Expected: "OK"
```

### 6. Generate Sample Test PDFs (Optional)

```bash
python scripts/sample_docs_generator.py
```

This creates two realistic HR policy documents (`policy_2025.pdf` and `policy_2026.pdf`) with known differences for testing comparison and contradiction detection.

---

## Running the Application

### Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8000
```

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

- Streamlit UI: http://localhost:8501

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload and index a PDF document |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document by ID |
| `POST` | `/query` | Grounded Q&A with source citations |
| `POST` | `/compare` | Compare two documents on a topic |
| `POST` | `/contradictions` | Detect factual contradictions across documents |
| `POST` | `/summarize` | Targeted summarization of a topic |
| `GET` | `/evaluation` | Run automated benchmark evaluation |
| `GET` | `/health` | System health status |

### Example: Document Q&A

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many days of annual leave do employees receive?",
    "top_k": 4
  }'
```

**Response:**

```json
{
  "query": "How many days of annual leave do employees receive?",
  "answer": "According to the 2026 policy, employees receive 24 days of paid annual leave per calendar year. [Source: policy_2026.pdf, Page 2]",
  "citations": [
    {
      "filename": "policy_2026.pdf",
      "page_number": 2,
      "chunk_id": "abc123_p2_c0",
      "snippet": "Employees receive 24 days of paid annual leave during each calendar year...",
      "score": 0.87
    }
  ],
  "is_grounded": true,
  "retrieved_count": 4
}
```

---

## How It Works (Technical Deep-Dive)

### 1. Document Ingestion (`app/ingestion/pdf_loader.py`)

- **PyMuPDF** (`fitz`) extracts text page-by-page from uploaded PDFs.
- A **deterministic SHA256 hash** of the file content generates a stable `doc_id` — uploading the same file twice produces the same ID, preventing duplicates.
- Empty pages are filtered. Corrupted or non-PDF files raise typed exceptions (`InvalidPDFError`, `EmptyPDFError`).
- The raw PDF is saved to `data/documents/` for persistence.

### 2. Chunking (`app/ingestion/chunker.py`)

- **LangChain `RecursiveCharacterTextSplitter`** splits page text into overlapping chunks (`chunk_size=500`, `chunk_overlap=50`).
- Each chunk carries full provenance metadata:
  - `doc_id` — SHA256 hash linking back to the parent document
  - `filename` — original PDF filename
  - `page_number` — 1-indexed page where the chunk originated
  - `chunk_id` — deterministic ID: `{doc_id}_p{page}_c{index}`
  - `chunk_index` — sequential position within the document

### 3. Embeddings (`app/core/embeddings.py`)

- **Sentence Transformers `all-MiniLM-L6-v2`** generates 384-dimensional dense vectors.
- Runs entirely locally on CPU — zero API cost.
- The model is loaded once (singleton) and reused across requests.
- Same model is used for both document chunk embeddings and query embeddings.

### 4. Vector Storage (`app/core/vector_store.py`)

- **Qdrant** stores vectors with cosine distance metric and 384-dim configuration.
- Each point includes a rich **payload**: text, doc_id, filename, page_number, chunk_id, chunk_index.
- Points use **deterministic UUID5** IDs derived from chunk_id — re-indexing the same document updates rather than duplicates.
- Docker volume `./data/qdrant_storage:/qdrant/storage` ensures persistence across container restarts.

### 5. Retrieval (`app/retrieval/retriever.py`)

- User query is embedded using the same `all-MiniLM-L6-v2` model.
- **Cosine similarity search** returns the top-K most relevant chunks.
- Supports optional **doc_id filtering** to scope search to specific documents.
- Returns `RetrievedChunk` objects with scores and full metadata.

### 6. LangGraph Workflow (`app/workflows/rag_graph.py`)

A 4-node LangGraph `StateGraph` orchestrates the RAG pipeline:

| Node | Function |
|---|---|
| **Analyze Query** | Optimizes user question into search-friendly keywords using Gemini |
| **Retrieve** | Executes Qdrant cosine similarity search with the optimized query |
| **Generate** | Sends retrieved context + question to Gemini with strictly grounded system prompt |
| **Validate Grounding** | Verifies the answer is supported by context; enforces refusal if not; attaches real citations |

### 7. Grounding & Citations

- **System prompt** explicitly forbids using outside knowledge.
- If context is insufficient, the system returns the exact refusal message.
- **Citations are extracted directly from retrieved chunks** — never generated or guessed.
- Each citation includes: filename, page_number, chunk_id, supporting text snippet, similarity score.

### 8. Advanced Features

All advanced features (comparison, contradiction detection, summarization) use the **same retrieval pipeline**:

- **Document Comparison**: Retrieves top-K chunks from Doc A and Doc B separately, then prompts Gemini for a structured diff analysis with dual-source citations.
- **Contradiction Detection**: Retrieves chunks across documents, prompts Gemini to identify genuine semantic conflicts (ignoring differences due to different dates, versions, or conditions).
- **Targeted Summarization**: Retrieves topic-relevant chunks, then generates a focused summary with citations.

---

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

**39 tests** covering:

| Test File | Coverage |
|---|---|
| `test_ingestion.py` | PDF loading, SHA256 hashing, error handling |
| `test_chunking.py` | Text splitting, metadata preservation, edge cases |
| `test_embeddings.py` | Embedding dimensions, batch processing, semantic similarity |
| `test_vector_store.py` | Qdrant CRUD, deterministic UUIDs, deduplication |
| `test_retrieval.py` | Semantic search, top-K, doc filtering, error handling |
| `test_rag_workflow.py` | LangGraph pipeline, grounding, citations |
| `test_comparison.py` | Compare, contradict, summarize |
| `test_evaluation.py` | Benchmark dataset loading, metrics structure |
| `test_api.py` | FastAPI endpoint integration |

### Run Evaluation Benchmark

```bash
python -m app.evaluation.evaluator
```

This measures:
- **Retrieval Recall@K** — Did the correct page appear in retrieved chunks?
- **Grounding Score** — Does the answer contain expected keywords from the ground truth?
- **Citation Accuracy** — Does the citation point to the correct page?
- **Refusal Accuracy** — Does the system refuse out-of-scope questions?

---

## Design Decisions

### Why PyMuPDF instead of PDFPlumber or LangChain loaders?

PyMuPDF is significantly faster and provides clean page-by-page text extraction with reliable page number tracking. LangChain's PDF loaders are wrappers that add unnecessary abstraction for our use case.

### Why `all-MiniLM-L6-v2` instead of OpenAI embeddings?

- **Free**: Runs locally on CPU with zero API cost.
- **Fast**: ~80ms per query embedding.
- **Good enough**: 384 dimensions provide strong semantic similarity for document Q&A.
- **Consistent**: Same model for document and query embeddings ensures proper vector space alignment.

### Why Qdrant instead of ChromaDB or FAISS?

- **Production-ready**: Qdrant offers payload filtering, persistence, and a REST API out of the box.
- **Docker-native**: Simple `docker compose up` with volume persistence.
- **Payload filtering**: Supports `doc_id` filtering for scoped multi-document queries.
- **Interview-relevant**: Demonstrates real vector database usage, not an in-memory prototype.

### Why LangGraph instead of a simple function chain?

LangGraph makes the RAG pipeline **inspectable and extensible**:
- Each step (query analysis → retrieval → generation → validation) is a named node.
- State is explicitly typed with `TypedDict`.
- Adding new steps (e.g., reranking, caching) requires adding a node, not refactoring the entire pipeline.
- Demonstrates understanding of **stateful agentic workflows** — a key concept in modern GenAI systems.

### Why strict grounding instead of allowing LLM general knowledge?

For a RAG system, the **entire value proposition** is that answers come from your documents. If the LLM fills gaps with general knowledge, you can't trust any answer, and citations become meaningless. Strict grounding with explicit refusal is the correct design.

---

## Interview Q&A

### Q: What is RAG and why did you use it?

**A:** RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. Instead of relying solely on the LLM's training data, we retrieve relevant document chunks and provide them as context. This ensures answers are grounded in actual uploaded documents, reduces hallucination, and enables the system to work with documents the LLM has never seen.

### Q: Walk me through what happens when a user asks a question.

**A:**
1. The user's question enters the LangGraph workflow.
2. **Query Analysis**: Gemini extracts optimized search keywords from the question.
3. **Embedding**: The optimized query is embedded into a 384-dimensional vector using the same Sentence Transformer model used for document chunks.
4. **Retrieval**: Qdrant performs cosine similarity search, returning the top-K most relevant chunks with their metadata.
5. **Generation**: The retrieved chunks are formatted into a prompt with a strict grounding system instruction, sent to Gemini.
6. **Validation**: The generated answer is checked against the retrieved context. If grounded, real citations (filename + page number extracted from chunk metadata) are attached. If not, the system returns a refusal message.

### Q: How do you prevent hallucination?

**A:** Three layers:
1. **System prompt**: Explicitly forbids using outside knowledge and requires the exact refusal message when context is insufficient.
2. **Grounding validation**: A separate LLM call verifies every claim in the answer is supported by the retrieved context.
3. **Real citations**: Citations come directly from chunk metadata (filename, page_number), never from the LLM's generation. If the LLM fabricates a citation, it won't match any retrieved chunk.

### Q: How do you generate document IDs?

**A:** SHA256 hash of the raw PDF file bytes. This is deterministic — uploading the same file twice produces the same doc_id, which prevents duplicate indexing (Qdrant point IDs are UUID5 derived from chunk_id, so re-indexing updates existing points instead of creating duplicates).

### Q: Why did you choose cosine similarity over other distance metrics?

**A:** Cosine similarity measures the angle between vectors, making it invariant to vector magnitude. This is ideal for dense embeddings from Sentence Transformers, where semantic meaning is encoded in direction rather than magnitude. Euclidean distance would be sensitive to embedding norms, which aren't semantically meaningful here.

### Q: How does your document comparison work?

**A:** It's fully RAG-driven:
1. Retrieve top-K chunks from Document A filtered by `doc_id_a`.
2. Retrieve top-K chunks from Document B filtered by `doc_id_b`.
3. Format both sets of chunks as side-by-side context.
4. Prompt Gemini to identify concrete differences with exact page citations from both documents.

This approach works because the retrieval scoping ensures we compare corresponding sections rather than random chunks.

### Q: How does contradiction detection distinguish real contradictions from version differences?

**A:** The prompt explicitly instructs Gemini: "Statements that apply to different years, different versions, different product tiers, or conditional clauses are NOT contradictions unless they claim to describe the same condition simultaneously." The system retrieves cross-document chunks on a topic, then applies this nuanced analysis. For example, "15 days leave in 2025" and "24 days leave in 2026" is a policy change, not a contradiction.

### Q: What would you improve with more time?

**A:**
1. **Hybrid search**: Combine dense vector search with BM25 sparse retrieval for better recall on keyword-heavy queries.
2. **Reranking**: Add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation to improve precision.
3. **Streaming responses**: Use Gemini's streaming API for real-time answer generation in the UI.
4. **Multi-modal support**: Extend PyMuPDF to extract tables and images from PDFs.
5. **Conversation memory**: Add session-based chat history for follow-up questions.

---

## License

This project is for educational and portfolio purposes.
