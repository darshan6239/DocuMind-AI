"""
UI: sidebar layout — composes the uploader, document library panel,
statistics, and model info into the app's sidebar.
"""
from typing import Optional

import streamlit as st

from config import CHAT_MODEL, EMBEDDING_MODEL
from services.document_service import is_library_empty
from ui.document_panel import render_document_panel
from ui.statistics import render_statistics
from ui.uploader import render_uploader
from ui.history import clear_history, has_messages


def render_sidebar() -> Optional[str]:
    """
    Render the full sidebar.

    Returns:
        The selected chat scope (filename) or None for "All documents".
    """
    with st.sidebar:
        st.header("📄 Documents")

        render_uploader()
        st.divider()

        selected_source = render_document_panel()

        if not is_library_empty():
            st.divider()
            render_statistics()

        st.divider()
        st.caption(f"Chat model: `{CHAT_MODEL}`")
        st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")

        if has_messages() and st.button("🧹 Clear chat"):
            clear_history()
            st.rerun()

    return selected_source
