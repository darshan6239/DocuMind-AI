"""
Chat service: validates questions and delegates to the RAG engine,
shaping the result into a UI-friendly response object.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.rag_engine import generate_answer
from utils.helpers import truncate_text
from utils.logger import get_logger
from utils.validators import validate_question

logger = get_logger(__name__)


@dataclass
class SourceRef:
    source: str
    page: int
    excerpt: str


@dataclass
class ChatResponse:
    answer: str
    sources: List[SourceRef] = field(default_factory=list)
    error: Optional[str] = None


def ask(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    source_filter: Optional[str] = None,
) -> ChatResponse:
    """
    Answer a user's question, grounded in the indexed documents.

    Args:
        question: The user's raw question text.
        history: Prior chat turns as [{"role": ..., "content": ...}].
        source_filter: Optional filename to scope retrieval to.

    Returns:
        ChatResponse with the answer text and cited source chunks.
    """
    validation = validate_question(question)
    if not validation.is_valid:
        return ChatResponse(answer="", error=validation.error)

    try:
        answer, chunks = generate_answer(question, history=history, source_filter=source_filter)
    except Exception as e:
        logger.exception("Error generating answer")
        return ChatResponse(
            answer="",
            error=(
                f"⚠️ Error while generating an answer: {e}\n\n"
                "Make sure Ollama is running locally and the required models are pulled."
            ),
        )

    sources = [
        SourceRef(
            source=c.metadata.get("source", "unknown"),
            page=c.metadata.get("page", "?"),
            excerpt=truncate_text(c.page_content, 300),
        )
        for c in chunks
    ]

    return ChatResponse(answer=answer, sources=sources)
