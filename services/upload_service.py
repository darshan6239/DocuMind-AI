"""
Upload service: validates and persists uploaded files to disk.
Purely responsible for getting bytes safely onto disk — indexing is
handled separately by the ingest service.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.file_utils import save_uploaded_file, file_exists
from utils.validators import validate_upload
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class UploadResult:
    success: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    already_exists: bool = False


def handle_upload(filename: str, file_bytes: bytes) -> UploadResult:
    """
    Validate and save an uploaded file.

    Args:
        filename: Original filename of the upload.
        file_bytes: Raw file content.

    Returns:
        UploadResult describing the outcome.
    """
    validation = validate_upload(filename, len(file_bytes))
    if not validation.is_valid:
        logger.warning(f"Rejected upload '{filename}': {validation.error}")
        return UploadResult(success=False, error=validation.error)

    if file_exists(filename):
        return UploadResult(success=True, path=None, already_exists=True)

    path = save_uploaded_file(filename, file_bytes)
    return UploadResult(success=True, path=path)
