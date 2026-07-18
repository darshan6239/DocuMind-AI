"""
Core: the single source of embeddings for the whole app.

CRITICAL: whatever backend is chosen here is used for BOTH indexing
(storing document chunks) and querying (embedding the user's question).
Mixing two different embedding models between ingestion and querying
would put vectors in different, non-comparable spaces and silently
break similarity search. Do not add a second embeddings path — always
go through get_embeddings() from this module.

Two backends are supported:
- Ollama (default): simple, no extra dependencies, fine for interactive
  use and small/medium corpora.
- A local sentence-transformers model (opt-in via USE_FAST_BULK_EMBEDDER):
  much higher throughput for bulk-loading large corpora (100k+ PDFs),
  and uses a GPU automatically if one is available.
"""
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL, FAST_EMBED_MODEL, OLLAMA_BASE_URL, USE_FAST_BULK_EMBEDDER
from utils.logger import get_logger

logger = get_logger(__name__)

_embeddings: Optional[Embeddings] = None
_backend_name: Optional[str] = None


class SentenceTransformersEmbeddings(Embeddings):
    """
    LangChain-compatible Embeddings wrapper around sentence-transformers,
    batching internally for throughput. Implements the same interface
    Chroma expects (embed_documents / embed_query), so it's a drop-in
    replacement for OllamaEmbeddings everywhere in the app.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # optional dep

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        device = str(getattr(self.model, "device", "cpu"))
        logger.info(f"Loaded fast embedding backend '{model_name}' on device '{device}'")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _build_embeddings() -> Embeddings:
    """Construct the embeddings backend according to configuration, with
    automatic fallback to Ollama if the fast backend can't be loaded."""
    global _backend_name

    if USE_FAST_BULK_EMBEDDER:
        try:
            embedder = SentenceTransformersEmbeddings(FAST_EMBED_MODEL)
            _backend_name = f"sentence-transformers:{FAST_EMBED_MODEL}"
            return embedder
        except Exception as e:
            logger.warning(
                f"Fast embedding backend unavailable ({e}); falling back to Ollama. "
                "Install 'sentence-transformers' and 'torch' for much faster bulk ingestion."
            )

    from langchain_ollama import OllamaEmbeddings

    _backend_name = f"ollama:{EMBEDDING_MODEL}"
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_embeddings() -> Embeddings:
    """
    Return the single, process-wide embeddings backend. Cached after
    first call so the (possibly expensive-to-load) model is only
    initialized once.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = _build_embeddings()
    return _embeddings


def get_backend_name() -> str:
    """Return a human-readable identifier of the active embedding backend
    (useful for logging / sanity-checking that ingestion and querying
    are using the same one)."""
    get_embeddings()  # ensure initialized
    return _backend_name
