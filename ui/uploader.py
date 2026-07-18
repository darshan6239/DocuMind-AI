"""
UI: PDF file uploader widget, wired to the upload and ingest services.
"""
import streamlit as st

from config import MAX_FILE_SIZE_MB
from services.upload_service import handle_upload
from services.ingest_service import ingest_pdf
from services.document_service import get_library


def render_uploader() -> None:
    """Render the file uploader and process any newly uploaded PDFs."""
    already_indexed = {doc.name for doc in get_library()}

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max file size: {MAX_FILE_SIZE_MB} MB per file.",
        key="pdf_uploader",
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        if uploaded_file.name in already_indexed:
            continue

        file_bytes = uploaded_file.getvalue()
        upload_result = handle_upload(uploaded_file.name, file_bytes)

        if not upload_result.success:
            st.error(f"'{uploaded_file.name}': {upload_result.error}")
            continue

        if upload_result.already_exists:
            continue

        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            ingest_result = ingest_pdf(upload_result.path)

        if ingest_result.success:
            st.success(
                f"Indexed '{ingest_result.filename}' "
                f"({ingest_result.num_chunks} chunks, {ingest_result.elapsed_seconds:.1f}s)."
            )
        else:
            st.error(f"Failed to process '{ingest_result.filename}': {ingest_result.error}")
