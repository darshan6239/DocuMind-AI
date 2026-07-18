"""
Core: persistent Chroma vector store management.
"""
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import VECTOR_DB_DIR, TOP_K
from core.embeddings import get_embeddings, get_backend_name
from utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "documind_collection"
_BACKEND_MARKER_FILE = VECTOR_DB_DIR / ".embedding_backend"

_vector_store: Optional[Chroma] = None


def _check_backend_consistency(current_backend: str) -> None:
    """
    Guard against silently mixing embedding backends. If this database
    was previously populated with a different embedding model, similarity
    search results would be meaningless (vectors from different models
    aren't comparable). Fail loudly instead of corrupting search.
    """
    if _BACKEND_MARKER_FILE.exists():
        recorded = _BACKEND_MARKER_FILE.read_text().strip()
        if recorded and recorded != current_backend:
            raise RuntimeError(
                f"Embedding backend mismatch: this vector store was built with "
                f"'{recorded}' but the app is now configured to use "
                f"'{current_backend}'. Mixing embedding models breaks similarity "
                f"search. Either restore the original embedding config, or delete "
                f"'{VECTOR_DB_DIR}' to start a fresh index with the new backend."
            )
    else:
        _BACKEND_MARKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BACKEND_MARKER_FILE.write_text(current_backend)


def get_vector_store() -> Chroma:
    """Lazily instantiate and cache the persistent Chroma vector store."""
    global _vector_store
    if _vector_store is None:
        _check_backend_consistency(get_backend_name())
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(VECTOR_DB_DIR),
        )
    return _vector_store


def add_documents_with_embeddings(
    documents: List[Document], embeddings: List[List[float]]
) -> int:
    """
    Write chunks with already-computed embedding vectors directly into
    Chroma's underlying collection, skipping the embedding call that
    add_documents() would normally trigger. Used by the bulk ingestion
    pipeline, which embeds in large batches ahead of time for throughput.

    Args:
        documents: Chunked Document objects.
        embeddings: Parallel list of embedding vectors, one per document.

    Returns:
        Number of documents written.
    """
    if not documents:
        return 0
    if len(documents) != len(embeddings):
        raise ValueError("documents and embeddings must be the same length")

    store = get_vector_store()
    ids = [
        f"{d.metadata.get('file_hash', d.metadata.get('source', 'unknown'))}"
        f"::{d.metadata.get('chunk_id', i)}"
        for i, d in enumerate(documents)
    ]
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]

    store._collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(documents)


def add_documents(documents: List[Document]) -> int:
    """Embed and add a list of chunked documents to the vector store."""
    if not documents:
        return 0
    store = get_vector_store()
    store.add_documents(documents)
    logger.info(f"Added {len(documents)} chunk(s) to the vector store")
    return len(documents)


def similarity_search(
    query: str, k: int = TOP_K, source_filter: Optional[str] = None
) -> List[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: The user's question.
        k: Number of chunks to retrieve.
        source_filter: Optional filename to restrict the search to a
            single uploaded document.

    Returns:
        List of relevant Document chunks, ordered by relevance.
    """
    store = get_vector_store()
    filter_dict = {"source": source_filter} if source_filter else None
    return store.similarity_search(query, k=k, filter=filter_dict)


def list_sources() -> List[str]:
    """Return the distinct set of source filenames currently indexed."""
    store = get_vector_store()
    data = store.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    sources = {m["source"] for m in metadatas if m and "source" in m}
    return sorted(sources)


def count_chunks(source: Optional[str] = None) -> int:
    """Count chunks in the store, optionally filtered to one source."""
    store = get_vector_store()
    if source:
        data = store.get(where={"source": source})
        return len(data.get("ids") or [])
    return store._collection.count()


def delete_source(source: str) -> int:
    """
    Delete all chunks belonging to a given source document.

    Returns:
        Number of chunks deleted.
    """
    store = get_vector_store()
    data = store.get(where={"source": source})
    ids = data.get("ids") or []
    if ids:
        store.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunk(s) for source '{source}'")
    return len(ids)


def collection_is_empty() -> bool:
    """Return True if the vector store currently has no documents."""
    return count_chunks() == 0
