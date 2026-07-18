"""
UI: renders the expandable "Sources" section under an assistant answer.
"""
from typing import List

import streamlit as st

from services.chat_service import SourceRef


def render_sources(sources: List[SourceRef]) -> None:
    """Render a list of SourceRef objects as an expandable citations block."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s.source}** — page {s.page}")
            st.caption(s.excerpt)
            st.divider()
