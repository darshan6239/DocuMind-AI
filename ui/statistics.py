"""
UI: renders aggregate statistics about the document library.
"""
import streamlit as st

from services.document_service import get_total_stats


def render_statistics() -> None:
    """Render a compact row of library-wide metrics."""
    stats = get_total_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", stats["num_documents"])
    col2.metric("Chunks", stats["num_chunks"])
    col3.metric("Files on disk", stats["num_files_on_disk"])
