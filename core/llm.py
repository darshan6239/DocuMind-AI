"""
Core: Ollama chat model client (singleton).
"""
from typing import Optional

from langchain_ollama import ChatOllama

from config import CHAT_MODEL, CHAT_TEMPERATURE, OLLAMA_BASE_URL

_llm: Optional[ChatOllama] = None


def get_llm() -> ChatOllama:
    """Lazily instantiate and cache the Ollama chat model."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=CHAT_MODEL,
            temperature=CHAT_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )
    return _llm
