"""
Core: a lightweight SQLite-backed registry of already-ingested files,
keyed by content hash. This is what makes re-running bulk ingestion on
the same folder of a million PDFs nearly instant the second time —
every file whose hash is already recorded is skipped without touching
the vector store at all.
"""
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from config import REGISTRY_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_files (
    file_hash   TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    num_chunks  INTEGER NOT NULL,
    ingested_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_ingested_filename ON ingested_files(filename);
"""


def compute_file_hash(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute a SHA-256 hash of a file's contents, streamed in chunks so
    memory use stays flat regardless of file size.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Bytes to read per iteration.

    Returns:
        Hex digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


class IngestRegistry:
    """
    Thread-safe wrapper around a SQLite table tracking which file hashes
    have already been fully ingested.
    """

    def __init__(self, db_path: Path = REGISTRY_DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        # Ensure schema exists using a throwaway connection.
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        """Return a connection local to the current thread/process."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30)
        return self._local.conn

    def is_processed(self, file_hash: str) -> bool:
        """Return True if a file with this hash has already been ingested."""
        cur = self._conn().execute(
            "SELECT 1 FROM ingested_files WHERE file_hash = ? LIMIT 1", (file_hash,)
        )
        return cur.fetchone() is not None

    def mark_processed(self, file_hash: str, filename: str, num_chunks: int) -> None:
        """Record that a file has been fully ingested."""
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO ingested_files (file_hash, filename, num_chunks) "
            "VALUES (?, ?, ?)",
            (file_hash, filename, num_chunks),
        )
        conn.commit()

    def count(self) -> int:
        """Return the total number of files recorded as ingested."""
        cur = self._conn().execute("SELECT COUNT(*) FROM ingested_files")
        return cur.fetchone()[0]

    def forget(self, file_hash: str) -> None:
        """Remove a file's record (e.g. to force re-ingestion)."""
        conn = self._conn()
        conn.execute("DELETE FROM ingested_files WHERE file_hash = ?", (file_hash,))
        conn.commit()
