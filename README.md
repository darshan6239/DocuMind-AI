# DocuMind AI

Enterprise Document Intelligence Platform — upload PDFs and chat with them
using a local, fully private RAG (Retrieval-Augmented Generation) pipeline
powered by Ollama and ChromaDB.

## Features

- 📤 Upload one or more PDFs from the sidebar
- 🔍 Automatic text extraction and chunking (PyMuPDF + LangChain)
- 🧠 Local embeddings and chat via Ollama (no data leaves your machine)
- 💬 Chat interface with conversational memory
- 📎 Source citations (document + page) for every answer
- 📚 Document library — scope questions to one document or search all
- 📊 Live stats (documents, chunks, files on disk)
- 🗑️ Delete individual documents from the index and disk
- 🪵 Rotating file logging + per-operation timing

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** installed and running locally
3. Pull the required models:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Environment variables (optional, defaults shown) live in `.env`:

```
CHAT_MODEL=qwen2.5:1.5b
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
TOP_K=5
CHAT_TEMPERATURE=0.1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50
MAX_HISTORY_TURNS=6
LOG_LEVEL=INFO
```

## Run

Make sure Ollama is running (`ollama serve`, or the Ollama desktop app),
then:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project structure

```
.
├── app.py                       # Entry point — wires sidebar + chat together
├── config.py                    # Paths, model names, RAG settings
├── requirements.txt
├── .env                          # Model / runtime configuration
│
├── core/                         # Low-level engine (no Streamlit imports)
│   ├── text_processor.py        # PDF extraction + chunking
│   ├── embeddings.py            # Ollama embeddings client
│   ├── llm.py                   # Ollama chat model client
│   ├── vector_store.py          # Chroma persistence layer
│   └── rag_engine.py            # Retrieval + prompt building + generation
│
├── services/                     # Orchestration layer (business logic)
│   ├── upload_service.py        # Validate + save uploaded files
│   ├── ingest_service.py        # Save -> extract -> chunk -> embed -> store
│   ├── document_service.py      # Library listing, stats, deletion
│   └── chat_service.py          # Question validation + RAG call shaping
│
├── ui/                            # Streamlit rendering components
│   ├── sidebar.py               # Composes uploader/library/stats
│   ├── uploader.py              # File uploader widget
│   ├── document_panel.py        # Library list with delete buttons
│   ├── statistics.py            # Metrics row
│   ├── chat.py                  # Chat interface
│   ├── history.py               # Session-state chat history
│   └── sources.py               # Source citation rendering
│
├── utils/                         # Generic, dependency-light helpers
│   ├── file_utils.py            # Save/delete/list files on disk
│   ├── validators.py            # File + question validation
│   ├── logger.py                # Rotating file + console logging
│   ├── timer.py                 # Timing decorator/context manager
│   └── helpers.py               # Formatting helpers
│
├── assets/                       # Static assets (logo, icons, etc.)
├── data/uploads/                 # Uploaded PDFs (created at runtime)
├── database/chroma_db/           # Persistent vector store (created at runtime)
├── cache/                        # Scratch cache (created at runtime)
└── logs/                         # Rotating log files (created at runtime)
```

## Architecture

The app follows a simple layered design:

```
ui/  →  services/  →  core/  →  (Ollama, Chroma, PyMuPDF)
```

- **`core/`** never imports Streamlit — it's the reusable engine and could
  be reused in a CLI or API without changes.
- **`services/`** orchestrates `core/` functions into meaningful business
  operations (e.g. "ingest this PDF", "answer this question") and is what
  the UI calls.
- **`ui/`** only renders Streamlit widgets and calls into `services/` —
  it holds no business logic itself.
- **`utils/`** are small, dependency-light helpers shared across layers.

## How it works

1. **Upload** — a PDF is validated (`utils/validators.py`) and saved to
   `data/uploads/` (`services/upload_service.py`).
2. **Ingest** — `services/ingest_service.py` extracts text page-by-page
   with PyMuPDF, splits it into ~1000-character overlapping chunks, and
   embeds them with `nomic-embed-text` via Ollama, storing them in a
   local Chroma collection.
3. **Ask** — `services/chat_service.py` validates the question, retrieves
   the top-k most similar chunks (`core/vector_store.py`), and asks
   `qwen2.5:1.5b` (`core/rag_engine.py`) to answer strictly from that
   context, citing document + page.
4. **Manage** — the sidebar (`ui/sidebar.py`) shows every indexed
   document with chunk counts and lets you delete any of them, scope
   chat to a single document, or clear the conversation.

## Bulk-ingesting a large corpus (thousands to millions of PDFs)

The Streamlit upload widget is fine for a handful of files, but it
processes one PDF at a time — far too slow at scale. For large
corpora, use the bulk CLI instead:

```bash
python scripts/bulk_ingest.py /path/to/your/pdfs
```

This is dramatically faster because it:

- **Parallelizes PDF extraction** across all CPU cores (`--workers`,
  default: all logical cores) instead of one file at a time.
- **Batches embedding calls** — hundreds of chunks per call instead of
  one file per call (`--embed-batch-size`, default 256).
- **Batches vector-store writes** with precomputed embeddings instead
  of triggering an embed-and-write per file (`--write-batch-size`).
- **Skips already-ingested files** via a SHA-256 content-hash registry
  (`database/ingest_registry.sqlite3`), so re-running the same command
  after adding new files, or after a crash, only processes what's new.
- Optionally uses a **local `sentence-transformers` model** instead of
  Ollama for embeddings, which is far faster for bulk workloads and
  uses a GPU automatically if one is available.

### Enabling the fast embedding backend

```bash
pip install sentence-transformers torch
```

It's on by default (`USE_FAST_BULK_EMBEDDER=true` in `.env`) and falls
back to Ollama automatically if these packages aren't installed. The
default model is `BAAI/bge-small-en-v1.5` (`FAST_EMBED_MODEL` in
`.env`) — small, fast, and a good default; swap in any
sentence-transformers-compatible model if you prefer.

> **Important:** whichever backend is active is used for *both*
> ingestion and querying — the app enforces this and will refuse to
> start (with a clear error) if you change `USE_FAST_BULK_EMBEDDER` or
> `FAST_EMBED_MODEL`/`EMBEDDING_MODEL` after a database already has
> data in it, since mixing embedding models breaks similarity search.
> Either revert the setting or delete `database/chroma_db` to start
> fresh with the new backend.

### Tuning for your hardware

| Flag | What it controls | When to raise it |
|---|---|---|
| `--workers` | Parallel extraction processes | More CPU cores available |
| `--embed-batch-size` | Chunks per embedding call | More RAM / GPU memory |
| `--write-batch-size` | Chunks per Chroma write | Usually fine at default |

Example for a beefy machine with a GPU:
```bash
python scripts/bulk_ingest.py /data/pdfs --workers 32 --embed-batch-size 1024
```

### Rough expectations

Extraction and chunking scale near-linearly with CPU cores. Embedding
throughput is the real bottleneck at scale — a local sentence-transformers
model on GPU can embed thousands of chunks per second, versus roughly
one request at a time through Ollama. For a corpus in the hundreds of
thousands to millions of files, running this on a machine with a GPU
and enough cores to saturate it is the single biggest lever.

## Notes

- All processing is local — no external API calls are made.
- Only `.pdf` files are supported (see `SUPPORTED_FILE_TYPES` in
  `config.py`); max file size is 50 MB (`MAX_FILE_SIZE_MB`).
- Scanned/image-only PDFs with no extractable text will be rejected
  during ingestion with a clear error message.
- If answers seem off or errors occur, confirm both models are pulled
  and Ollama is reachable at `OLLAMA_BASE_URL` (default
  `http://localhost:11434`).
- Logs are written to `logs/documind.log` (rotating, 3 backups).
