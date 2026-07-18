"""
UI: document library panel — lists indexed documents with per-document
chunk counts and delete controls, and returns the currently selected
chat scope.
"""
from typing import Optional

import streamlit as st

from services.document_service import get_library, remove_document
from utils.helpers import pluralize


def render_document_panel() -> Optional[str]:
    """
    Render the document library panel.

    Returns:
        The selected source filename to scope chat to, or None for
        "All documents".
    """
    library = get_library()

    if not library:
        st.info("No documents indexed yet. Upload a PDF to get started.")
        return None

    st.subheader("Library")

    options = ["All documents"] + [doc.name for doc in library]
    selected = st.selectbox(
        "Chat scope",
        options=options,
        help="Restrict Q&A to a single document, or search across all.",
    )

    for doc in library:
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"📎 **{doc.name}**")
        col1.caption(pluralize(doc.chunk_count, "chunk"))
        if col2.button("🗑️", key=f"delete_{doc.name}", help=f"Remove {doc.name}"):
            remove_document(doc.name)
            st.rerun()

    return None if selected == "All documents" else selected
