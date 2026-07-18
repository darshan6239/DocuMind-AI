"""
Core: retrieval-augmented generation engine. Combines vector search
with the Ollama chat model to produce grounded, cited answers.
"""
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config import TOP_K, MAX_HISTORY_TURNS
from core.llm import get_llm
from core.vector_store import similarity_search
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are DocuMind AI, an assistant that answers questions \
strictly using the provided document excerpts.

Rules:
- Only use information found in the context below to answer.
- If the answer isn't in the context, say you don't know based on the \
uploaded documents. Do not make anything up.
- When useful, mention which document / page the information came from.
- Be clear and concise.
"""


def format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks into a single context string with citations."""
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Source: {source}, page {page}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, context: str, history: List[Dict[str, str]]) -> List[dict]:
    """Build the message list sent to the chat model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = f"Context from uploaded documents:\n\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})
    return messages


@timed("rag_answer")
def generate_answer(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    k: int = TOP_K,
    source_filter: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    """
    Answer a question using retrieval-augmented generation.

    Args:
        question: The user's question.
        history: Prior conversation turns as [{"role": ..., "content": ...}].
        k: Number of chunks to retrieve.
        source_filter: Optional filename to restrict retrieval to.

    Returns:
        Tuple of (answer text, retrieved source chunks).
    """
    history = history or []
    chunks = similarity_search(question, k=k, source_filter=source_filter)

    if not chunks:
        logger.info("No relevant chunks found for question")
        return (
            "I couldn't find any relevant content in the uploaded documents "
            "to answer that question.",
            [],
        )

    context = format_context(chunks)
    messages = build_messages(question, context, history)

    llm = get_llm()
    response = llm.invoke(messages)

    return response.content, chunks
