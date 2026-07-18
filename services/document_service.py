"""
Document service: manages the library of indexed documents — listing,
per-document stats, and deletion (from both the vector store and disk).
"""
from dataclasses import dataclass
from typing import List

from core.vector_store import list_sources, count_chunks, delete_source, collection_is_empty
from utils.file_utils import delete_uploaded_file, list_uploaded_files
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentInfo:
    name: str
    chunk_count: int


def get_library() -> List[DocumentInfo]:
    """Return info about every document currently indexed in the vector store."""
    return [DocumentInfo(name=src, chunk_count=count_chunks(src)) for src in list_sources()]


def is_library_empty() -> bool:
    """Return True if no documents are indexed yet."""
    return collection_is_empty()


def remove_document(filename: str) -> int:
    """
    Remove a document from both the vector store and disk.

    Args:
        filename: Name of the document to remove.

    Returns:
        Number of chunks deleted from the vector store.
    """
    deleted_chunks = delete_source(filename)
    delete_uploaded_file(filename)
    logger.info(f"Removed document '{filename}' ({deleted_chunks} chunks)")
    return deleted_chunks


def get_total_stats() -> dict:
    """Return aggregate stats across the whole library."""
    sources = list_sources()
    total_chunks = count_chunks()
    on_disk = len(list_uploaded_files())
    return {
        "num_documents": len(sources),
        "num_chunks": total_chunks,
        "num_files_on_disk": on_disk,
    }
