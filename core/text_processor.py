"""
Core text processing: PDF text extraction and chunking into
LangChain Document objects ready for embedding.
"""
from pathlib import Path
from typing import List

import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger
from utils.timer import timed

logger = get_logger(__name__)


def extract_text_from_pdf(file_path: Path) -> List[Document]:
    """
    Extract text from a PDF file, one Document per non-empty page.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        List of Document objects with page_content and metadata
        (source filename, page number).
    """
    documents: List[Document] = []
    pdf = fitz.open(file_path)

    try:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": page_num,
                    },
                )
            )
    finally:
        pdf.close()

    logger.info(f"Extracted {len(documents)} non-empty page(s) from '{file_path.name}'")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks suitable for embedding.

    Args:
        documents: List of Document objects (typically one per page).

    Returns:
        List of chunked Document objects, with a chunk_id added to metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


@timed("process_pdf")
def process_pdf(file_path: Path) -> List[Document]:
    """
    Full pipeline: extract text from a PDF and split it into chunks.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        List of chunked Document objects ready for embedding. Empty list
        if the PDF had no extractable text.
    """
    raw_documents = extract_text_from_pdf(file_path)
    if not raw_documents:
        logger.warning(f"No extractable text found in '{file_path.name}'")
        return []
    chunks = chunk_documents(raw_documents)
    logger.info(f"Split '{file_path.name}' into {len(chunks)} chunk(s)")
    return chunks


def get_page_count(file_path: Path) -> int:
    """Return the number of pages in a PDF file."""
    pdf = fitz.open(file_path)
    try:
        return pdf.page_count
    finally:
        pdf.close()
