"""
UI: main chat interface — renders chat history and handles new
questions via the chat service.
"""
from typing import Optional

import streamlit as st

from services.chat_service import ask
from services.document_service import is_library_empty
from ui.history import add_message, get_llm_history, get_messages
from ui.sources import render_sources


def render_chat(source_filter: Optional[str] = None) -> None:
    """
    Render the chat interface: existing history plus a chat input box
    for new questions.

    Args:
        source_filter: Optional filename to scope Q&A to a single document.
    """
    if is_library_empty():
        st.info("👋 Upload one or more PDFs from the sidebar, then ask questions about them here.")
        return

    for msg in get_messages():
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            render_sources(msg.get("sources", []))

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    add_message("user", question)
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(question, history=get_llm_history(), source_filter=source_filter)

        display_text = response.error or response.answer
        st.write(display_text)
        render_sources(response.sources)

    add_message("assistant", display_text, response.sources)
