"""
UI: chat history management backed by Streamlit session state.
"""
from typing import Dict, List

import streamlit as st

from services.chat_service import SourceRef

MESSAGES_KEY = "messages"


def init_history() -> None:
    """Ensure the chat history exists in session state."""
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []


def get_messages() -> List[dict]:
    """Return all messages currently in the chat history."""
    init_history()
    return st.session_state[MESSAGES_KEY]


def add_message(role: str, content: str, sources: List[SourceRef] = None) -> None:
    """Append a message to the chat history."""
    init_history()
    st.session_state[MESSAGES_KEY].append(
        {"role": role, "content": content, "sources": sources or []}
    )


def get_llm_history() -> List[Dict[str, str]]:
    """Return history formatted as plain role/content dicts for the LLM,
    excluding the most recent (still in-flight) message."""
    messages = get_messages()
    return [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]


def clear_history() -> None:
    """Clear all chat history."""
    st.session_state[MESSAGES_KEY] = []


def has_messages() -> bool:
    """Return True if there is at least one message in history."""
    return len(get_messages()) > 0
