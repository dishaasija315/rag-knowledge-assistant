"""Phase 1 Environment & Health Check Script"""
import sys
import urllib.request
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def check_python():
    print(f"[OK] Python Version: {sys.version.split()[0]}")


def check_config():
    settings = get_settings()
    print(f"[OK] Config Loaded:")
    print(f"     - Qdrant Host: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"     - Collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"     - Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"     - Chunk Size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")
    print(f"     - Document Dir: {settings.documents_dir}")


def check_qdrant(host: str = "localhost", port: int = 6333):
    url = f"http://{host}:{port}/readyz"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"[OK] Qdrant is running and healthy at http://{host}:{port}")
                return True
    except Exception as e:
        print(f"[WARN] Could not connect to Qdrant at {url}: {e}")
        print("       Run 'docker compose up -d' to start Qdrant.")
        return False


def main():
    print("=" * 60)
    print("RAG Knowledge Assistant - Phase 1 Verification")
    print("=" * 60)
    check_python()
    check_config()
    print("-" * 60)
    qdrant_ok = check_qdrant()
    print("=" * 60)
    if qdrant_ok:
        print("Phase 1 setup is complete and fully verified!")
    else:
        print("Qdrant container is not running yet. Starting Docker container is next.")


if __name__ == "__main__":
    main()
