"""
Filesystem helper utilities: saving uploaded files, cleaning up,
and inspecting the upload directory.
"""
from pathlib import Path
from typing import List

from config import UPLOAD_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def save_uploaded_file(filename: str, file_bytes: bytes) -> Path:
    """
    Persist an uploaded file's bytes to the upload directory.

    Args:
        filename: Original filename of the upload.
        file_bytes: Raw file content.

    Returns:
        Path to the saved file on disk.
    """
    dest_path = UPLOAD_DIR / filename
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved upload '{filename}' ({len(file_bytes)} bytes) to {dest_path}")
    return dest_path


def delete_uploaded_file(filename: str) -> bool:
    """
    Remove a previously uploaded file from disk, if present.

    Args:
        filename: Name of the file to remove.

    Returns:
        True if a file was deleted, False if it didn't exist.
    """
    path = UPLOAD_DIR / filename
    if path.exists():
        path.unlink()
        logger.info(f"Deleted uploaded file '{filename}'")
        return True
    return False


def list_uploaded_files() -> List[Path]:
    """Return all PDF files currently saved in the upload directory."""
    return sorted(UPLOAD_DIR.glob("*.pdf"))


def file_exists(filename: str) -> bool:
    """Check whether a given filename already exists in the upload directory."""
    return (UPLOAD_DIR / filename).exists()
