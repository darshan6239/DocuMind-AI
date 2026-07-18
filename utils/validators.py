"""
Validation helpers for uploaded files.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import MAX_FILE_SIZE_MB, SUPPORTED_FILE_TYPES


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None


def validate_file_extension(filename: str) -> ValidationResult:
    """Check that a filename has a supported extension."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FILE_TYPES:
        supported = ", ".join(SUPPORTED_FILE_TYPES)
        return ValidationResult(False, f"Unsupported file type '{suffix}'. Supported: {supported}")
    return ValidationResult(True)


def validate_file_size(size_bytes: int) -> ValidationResult:
    """Check that a file's size does not exceed the configured maximum."""
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return ValidationResult(
            False, f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )
    return ValidationResult(True)


def validate_upload(filename: str, size_bytes: int) -> ValidationResult:
    """Run all validation checks for an uploaded file."""
    ext_result = validate_file_extension(filename)
    if not ext_result.is_valid:
        return ext_result

    size_result = validate_file_size(size_bytes)
    if not size_result.is_valid:
        return size_result

    return ValidationResult(True)


def validate_question(question: str) -> ValidationResult:
    """Validate a user's chat question isn't empty or whitespace-only."""
    if not question or not question.strip():
        return ValidationResult(False, "Question cannot be empty.")
    if len(question.strip()) > 2000:
        return ValidationResult(False, "Question is too long (max 2000 characters).")
    return ValidationResult(True)
