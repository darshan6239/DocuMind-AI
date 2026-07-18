"""
Bulk ingestion service — the fast path for loading a large corpus of
PDFs (thousands to millions of files), as opposed to the one-file-at-a-
time interactive flow used by the Streamlit upload widget.

Speed comes from four independent changes, all combined here:

1. Parallel extraction: PDF parsing/chunking is CPU-bound and
   embarrassingly parallel, so it runs across a process pool instead
   of one file at a time.
2. Batched embedding: chunks from many files are accumulated into large
   batches (hundreds at a time) before a single embedding call, instead
   of one call per file. Combined with the fast local embedding backend
   (see core/embeddings.py), this is the single biggest speedup.
3. Batched vector store writes: precomputed embeddings are written to
   Chroma in large batches via the collection API directly, instead of
   triggering one embedding + write per file.
4. Skip-cache: every file's content hash is recorded once ingested, so
   re-running over the same folder (e.g. after a crash, or because new
   files were added) only processes what's actually new.
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from config import (
    BULK_EXTRACT_WORKERS,
    CHROMA_WRITE_BATCH_SIZE,
    EMBED_BATCH_SIZE,
)
from core.embeddings import get_embeddings, get_backend_name
from core.ingest_registry import IngestRegistry, compute_file_hash
from core.text_processor import process_pdf
from core.vector_store import add_documents_with_embeddings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BulkIngestStats:
    total_discovered: int = 0
    skipped_already_ingested: int = 0
    processed: int = 0
    failed: int = 0
    total_chunks: int = 0
    elapsed_seconds: float = 0.0
    errors: List[Tuple[str, str]] = field(default_factory=list)


def discover_pdfs(root_dir: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield every .pdf file under root_dir."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    yield from root_dir.glob(pattern)


def _extract_worker(file_path_str: str) -> Tuple[str, Optional[List[Document]], Optional[str]]:
    """
    Run in a worker process: extract + chunk a single PDF. Kept as a
    module-level function (not a method/closure) so it can be pickled
    and sent to worker processes by ProcessPoolExecutor.

    Returns:
        (file_path_str, chunks_or_None, error_message_or_None)
    """
    try:
        path = Path(file_path_str)
        chunks = process_pdf(path)
        return file_path_str, chunks, None
    except Exception as e:
        return file_path_str, None, str(e)


def ingest_directory(
    root_dir: Path,
    recursive: bool = True,
    workers: int = BULK_EXTRACT_WORKERS,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    write_batch_size: int = CHROMA_WRITE_BATCH_SIZE,
    progress_callback=None,
) -> BulkIngestStats:
    """
    Ingest every PDF under a directory as fast as possible.

    Args:
        root_dir: Folder to scan for .pdf files.
        recursive: Whether to search subdirectories too.
        workers: Number of parallel extraction processes.
        embed_batch_size: Chunks accumulated before one embedding call.
        write_batch_size: Chunks accumulated before one Chroma write.
        progress_callback: Optional callable(stats: BulkIngestStats) invoked
            periodically so a CLI/UI can report progress.

    Returns:
        BulkIngestStats summarizing the run.
    """
    start = time.perf_counter()
    stats = BulkIngestStats()
    registry = IngestRegistry()
    embedder = get_embeddings()

    logger.info(f"Bulk ingestion starting: backend={get_backend_name()}, workers={workers}")

    # --- Phase 1: discover + hash-filter (fast, I/O only) ---
    candidates: List[Tuple[Path, str]] = []
    for path in discover_pdfs(root_dir, recursive=recursive):
        stats.total_discovered += 1
        file_hash = compute_file_hash(path)
        if registry.is_processed(file_hash):
            stats.skipped_already_ingested += 1
            continue
        candidates.append((path, file_hash))

    logger.info(
        f"Discovered {stats.total_discovered} PDF(s); "
        f"{stats.skipped_already_ingested} already ingested, "
        f"{len(candidates)} to process"
    )

    if not candidates:
        stats.elapsed_seconds = time.perf_counter() - start
        return stats

    hash_by_path = {str(p): h for p, h in candidates}

    # --- Phase 2: parallel extraction + chunking ---
    doc_buffer: List[Document] = []
    chunks_remaining_for_file: dict = {}  # file_hash -> chunks left to flush
    chunks_expected_for_file: dict = {}  # file_hash -> total chunk count

    def flush(buffer: List[Document]) -> None:
        """Embed and write a batch of chunks, updating registry as files complete."""
        if not buffer:
            return
        texts = [d.page_content for d in buffer]
        vectors = embedder.embed_documents(texts)

        for i in range(0, len(buffer), write_batch_size):
            add_documents_with_embeddings(
                buffer[i : i + write_batch_size], vectors[i : i + write_batch_size]
            )

        stats.total_chunks += len(buffer)

        for d in buffer:
            fh = d.metadata["file_hash"]
            chunks_remaining_for_file[fh] -= 1
            if chunks_remaining_for_file[fh] == 0:
                registry.mark_processed(fh, d.metadata["source"], chunks_expected_for_file[fh])
                stats.processed += 1

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_worker, str(path)): (path, file_hash)
            for path, file_hash in candidates
        }

        for future in as_completed(futures):
            path, file_hash = futures[future]
            file_path_str, chunks, error = future.result()

            if error is not None:
                stats.failed += 1
                stats.errors.append((path.name, error))
                logger.error(f"Failed to extract '{path.name}': {error}")
                continue

            if not chunks:
                stats.failed += 1
                stats.errors.append((path.name, "No extractable text (scanned/image-only PDF?)"))
                continue

            for chunk in chunks:
                chunk.metadata["file_hash"] = file_hash
            chunks_expected_for_file[file_hash] = len(chunks)
            chunks_remaining_for_file[file_hash] = len(chunks)

            doc_buffer.extend(chunks)

            if len(doc_buffer) >= embed_batch_size:
                flush(doc_buffer)
                doc_buffer = []

            if progress_callback:
                progress_callback(stats)

    # flush whatever's left in the buffer
    flush(doc_buffer)

    stats.elapsed_seconds = time.perf_counter() - start
    logger.info(
        f"Bulk ingestion complete: {stats.processed} processed, "
        f"{stats.skipped_already_ingested} skipped, {stats.failed} failed, "
        f"{stats.total_chunks} chunks, {stats.elapsed_seconds:.1f}s"
    )
    return stats
