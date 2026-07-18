"""
Small general-purpose helper functions used across the UI and services.
"""
from datetime import datetime


def format_bytes(size_bytes: float) -> str:
    """Format a byte count as a human-readable string (e.g. '2.3 MB')."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to a maximum length, appending an ellipsis if cut."""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def format_timestamp(dt: datetime = None) -> str:
    """Format a datetime (default: now) as 'YYYY-MM-DD HH:MM'."""
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M")


def pluralize(count: int, singular: str, plural: str = None) -> str:
    """Return '<count> <singular|plural>' with correct pluralization."""
    plural = plural or f"{singular}s"
    word = singular if count == 1 else plural
    return f"{count} {word}"
