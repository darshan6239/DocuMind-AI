"""
Ingest service: orchestrates the pipeline from a saved PDF on disk to
searchable chunks in the vector store.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.text_processor import process_pdf
from core.vector_store import add_documents
from utils.logger import get_logger
from utils.timer import timed_block

logger = get_logger(__name__)


@dataclass
class IngestResult:
    success: bool
    filename: str
    num_chunks: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


def ingest_pdf(file_path: Path) -> IngestResult:
    """
    Extract, chunk, embed, and store a single PDF file.

    Args:
        file_path: Path to the PDF on disk (already saved).

    Returns:
        IngestResult with the outcome and chunk count.
    """
    filename = file_path.name
    try:
        with timed_block(f"ingest:{filename}") as t:
            chunks = process_pdf(file_path)
            if not chunks:
                return IngestResult(
                    success=False,
                    filename=filename,
                    error="No extractable text found in this PDF (it may be scanned/image-only).",
                )
            add_documents(chunks)

        return IngestResult(
            success=True,
            filename=filename,
            num_chunks=len(chunks),
            elapsed_seconds=t["elapsed"],
        )
    except Exception as e:
        logger.exception(f"Failed to ingest '{filename}'")
        return IngestResult(success=False, filename=filename, error=str(e))
