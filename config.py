from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

DATABASE_DIR = BASE_DIR / "database"
VECTOR_DB_DIR = DATABASE_DIR / "chroma_db"

CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

ASSETS_DIR = BASE_DIR / "assets"

# AI Models
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:1.5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# RAG Configuration
TOP_K = int(os.getenv("TOP_K", 5))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.1))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Upload Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))

SUPPORTED_FILE_TYPES = [
    ".pdf"
]

# Chat history
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 6))

# --- Bulk ingestion (large-scale offline PDF loading) ---
# Number of worker processes for parallel PDF extraction/chunking (CPU-bound).
# Defaults to all logical cores.
BULK_EXTRACT_WORKERS = int(os.getenv("BULK_EXTRACT_WORKERS", os.cpu_count() or 4))

# How many chunks to accumulate before sending one batched embedding request.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 256))

# How many (text, embedding, metadata) triples to write to Chroma per call.
CHROMA_WRITE_BATCH_SIZE = int(os.getenv("CHROMA_WRITE_BATCH_SIZE", 512))

# Use a local sentence-transformers model instead of Ollama for bulk
# embedding. Much higher throughput, especially on GPU. Falls back to
# Ollama automatically if sentence-transformers/torch aren't installed.
USE_FAST_BULK_EMBEDDER = os.getenv("USE_FAST_BULK_EMBEDDER", "true").lower() == "true"
FAST_EMBED_MODEL = os.getenv("FAST_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# SQLite registry used to skip files that were already ingested (by content
# hash), so re-running bulk ingestion on the same folder is nearly instant.
REGISTRY_DB_PATH = DATABASE_DIR / "ingest_registry.sqlite3"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create Required Directories
for directory in [
    DATA_DIR,
    UPLOAD_DIR,
    DATABASE_DIR,
    VECTOR_DB_DIR,
    CACHE_DIR,
    LOG_DIR,
    ASSETS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
