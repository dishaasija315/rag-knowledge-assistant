"""FastAPI application entrypoint for RAG Knowledge Assistant."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import router
from app.core.vector_store import QdrantVectorStore
from app.core.embeddings import get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_knowledge_assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting up RAG Knowledge Assistant backend...")
    settings = get_settings()

    # Pre-warm Qdrant connection and ensure collection
    try:
        store = QdrantVectorStore()
        store.ensure_collection_exists()
        logger.info(f"Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' ready.")
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant during startup: {e}")

    # Pre-warm Embedding model
    try:
        get_embedding_model()
        logger.info("SentenceTransformer embedding model loaded and ready.")
    except Exception as e:
        logger.warning(f"Could not pre-load embedding model: {e}")

    yield

    logger.info("Shutting down RAG Knowledge Assistant backend...")


app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="Production-ready GenAI RAG system for document Q&A, comparison, and contradiction detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


@app.get("/", summary="Root status")
async def root():
    return {
        "app": "RAG Knowledge Assistant",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
