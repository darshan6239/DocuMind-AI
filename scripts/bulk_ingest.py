#!/usr/bin/env python3
"""
CLI for bulk-ingesting a large folder of PDFs into DocuMind AI's
vector store as fast as possible.

Usage:
    python scripts/bulk_ingest.py /path/to/pdfs
    python scripts/bulk_ingest.py /path/to/pdfs --workers 16 --no-recursive
    python scripts/bulk_ingest.py /path/to/pdfs --embed-batch-size 512

Run this instead of uploading files one by one through the Streamlit UI
when you have a large corpus (thousands to millions of PDFs) to load.
Re-running it on the same folder is safe and fast — already-ingested
files are skipped via content-hash lookup.
"""
import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/bulk_ingest.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BULK_EXTRACT_WORKERS, CHROMA_WRITE_BATCH_SIZE, EMBED_BATCH_SIZE  # noqa: E402
from services.bulk_ingest_service import ingest_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk-ingest PDFs into DocuMind AI.")
    parser.add_argument("directory", type=str, help="Folder containing PDF files.")
    parser.add_argument(
        "--no-recursive", action="store_true", help="Don't search subdirectories."
    )
    parser.add_argument(
        "--workers", type=int, default=BULK_EXTRACT_WORKERS,
        help=f"Parallel extraction workers (default: {BULK_EXTRACT_WORKERS}, i.e. all cores).",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=EMBED_BATCH_SIZE,
        help=f"Chunks per embedding call (default: {EMBED_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--write-batch-size", type=int, default=CHROMA_WRITE_BATCH_SIZE,
        help=f"Chunks per vector-store write (default: {CHROMA_WRITE_BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.directory).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: '{root}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning '{root}' for PDFs (recursive={not args.no_recursive})...")

    last_report = {"count": 0, "time": time.perf_counter()}

    def report_progress(stats):
        # Print a lightweight running total every ~50 processed files.
        if stats.processed - last_report["count"] >= 50:
            elapsed = time.perf_counter() - last_report["time"]
            rate = (stats.processed - last_report["count"]) / elapsed if elapsed > 0 else 0
            print(
                f"  processed={stats.processed} skipped={stats.skipped_already_ingested} "
                f"failed={stats.failed} chunks={stats.total_chunks} "
                f"(~{rate:.1f} files/sec)"
            )
            last_report["count"] = stats.processed
            last_report["time"] = time.perf_counter()

    stats = ingest_directory(
        root,
        recursive=not args.no_recursive,
        workers=args.workers,
        embed_batch_size=args.embed_batch_size,
        write_batch_size=args.write_batch_size,
        progress_callback=report_progress,
    )

    print()
    print("=" * 60)
    print("Bulk ingestion complete")
    print("=" * 60)
    print(f"  Discovered:        {stats.total_discovered}")
    print(f"  Already ingested:  {stats.skipped_already_ingested}")
    print(f"  Newly processed:   {stats.processed}")
    print(f"  Failed:            {stats.failed}")
    print(f"  Total chunks:      {stats.total_chunks}")
    print(f"  Elapsed:           {stats.elapsed_seconds:.1f}s")
    if stats.processed:
        print(f"  Throughput:        {stats.processed / stats.elapsed_seconds:.2f} files/sec")

    if stats.errors:
        print()
        print(f"  {len(stats.errors)} file(s) failed:")
        for name, err in stats.errors[:20]:
            print(f"    - {name}: {err}")
        if len(stats.errors) > 20:
            print(f"    ... and {len(stats.errors) - 20} more")


if __name__ == "__main__":
    main()
