<div align="center">

# 📚 DocuMind AI

**Enterprise Document Intelligence Platform**

Upload PDFs and chat with them using a local, fully private RAG
(Retrieval-Augmented Generation) pipeline — no data ever leaves your machine.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Ollama](https://img.shields.io/badge/LLM-Ollama-000000)
![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-8A2BE2)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Table of contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [Bulk-ingesting a large corpus](#bulk-ingesting-a-large-corpus-thousands-to-millions-of-pdfs)
- [Scaling & latency at 1M PDFs](#scaling--latency-at-1000000-pdfs)
- [Troubleshooting](#troubleshooting)
- [Known limits](#known-limits)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

| | |
|---|---|
| 📤 **Multi-file upload** | Drag in one or many PDFs from the sidebar |
| 🔍 **Automatic extraction + chunking** | PyMuPDF text extraction, LangChain recursive splitting |
| 🧠 **Fully local inference** | Embeddings and chat via Ollama — nothing leaves your machine |
| 💬 **Conversational chat** | Multi-turn memory, not just single-shot Q&A |
| 📎 **Cited answers** | Every response links back to document + page |
| 📚 **Document library** | Scope questions to one document or search across all |
| 📊 **Live stats** | Document count, chunk count, files on disk |
| 🗑️ **Full lifecycle management** | Delete documents from both the index and disk |
| 🪵 **Observability** | Rotating file logs + per-operation timing baked in |
| ⚡ **Bulk ingestion CLI** | Parallel extraction, batched embeddings, hash-based skip-cache — built for 100k–1M+ file corpora |

---

## Screenshots

**Empty state** — before any documents are uploaded:

<img width="1897" height="898" alt="Screenshot 2026-07-18 213239" src="https://github.com/user-attachments/assets/288fb1ac-1397-4478-b986-8805b9d9e945" />


**Chat in action** — with a document library, chat scoping, and cited sources:

<img width="1913" height="900" alt="Screenshot 2026-07-18 213058" src="https://github.com/user-attachments/assets/ad86210b-a571-4278-b458-3d234e6baef7" />


---

## Architecture

A strictly layered design — each layer only talks to the one directly below it:

```
ui/  →  services/  →  core/  →  (Ollama, Chroma, PyMuPDF)
```

![Architecture diagram](./assets/diagrams/architecture.svg)

| Layer | Responsibility | Depends on |
|---|---|---|
| `ui/` | Streamlit rendering only — zero business logic | `services/` |
| `services/` | Orchestration: upload, ingest, chat, document management | `core/` |
| `core/` | The reusable engine — extraction, embeddings, vector store, RAG | Ollama, Chroma, PyMuPDF |
| `utils/` | Dependency-light helpers (logging, validation, file ops) | nothing internal |

Because `core/` never imports Streamlit, it's directly reusable from a CLI
script (see [`scripts/bulk_ingest.py`](./scripts/bulk_ingest.py)) or a future
API layer without any changes.

---

## How it works

1. **Upload** — a PDF is validated (`utils/validators.py`) and saved to
   `data/uploads/` (`services/upload_service.py`).
2. **Ingest** — `services/ingest_service.py` extracts text page-by-page
   with PyMuPDF, splits it into ~1000-character overlapping chunks, and
   embeds them, storing the result in a local Chroma collection.
3. **Ask** — `services/chat_service.py` validates the question, retrieves
   the top-k most similar chunks (`core/vector_store.py`), and asks the
   chat model (`core/rag_engine.py`) to answer strictly from that context,
   citing document + page.
4. **Manage** — the sidebar (`ui/sidebar.py`) shows every indexed document
   with chunk counts and lets you delete any of them, scope chat to a
   single document, or clear the conversation.

> ⚠️ **One embedding backend, always.** Whichever embedding model is
> configured is used for *both* ingestion and querying — the app enforces
> this and refuses to start with a mismatched backend against an existing
> database (see [Known limits](#known-limits)).

---

## Quick start

### Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** installed and running locally
3. Pull the required models:
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```

### Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

Make sure Ollama is running (`ollama serve`, or the Ollama desktop app), then:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Configuration reference

All settings live in `.env` and are read by `config.py`. Every value below
has a working default — you only need to touch this file to customize
behavior.

| Variable | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `qwen2.5:1.5b` | Ollama chat model used to generate answers |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model (used if fast backend is off/unavailable) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `TOP_K` | `5` | Chunks retrieved per question |
| `CHAT_TEMPERATURE` | `0.1` | LLM sampling temperature (kept low for grounded answers) |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size per file (interactive UI) |
| `MAX_HISTORY_TURNS` | `6` | Prior chat turns included as LLM context |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `BULK_EXTRACT_WORKERS` | all cores | Parallel processes for bulk PDF extraction |
| `EMBED_BATCH_SIZE` | `256` | Chunks per embedding call during bulk ingestion |
| `CHROMA_WRITE_BATCH_SIZE` | `512` | Chunks per Chroma write during bulk ingestion |
| `USE_FAST_BULK_EMBEDDER` | `true` | Use local `sentence-transformers` instead of Ollama for embeddings |
| `FAST_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Model used when the fast backend is active |

---

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
│   ├── embeddings.py            # Unified embeddings backend (Ollama / fast local)
│   ├── llm.py                   # Ollama chat model client
│   ├── vector_store.py          # Chroma persistence layer + bulk writes
│   ├── rag_engine.py            # Retrieval + prompt building + generation
│   └── ingest_registry.py       # SHA-256 skip-cache for bulk ingestion
│
├── services/                     # Orchestration layer (business logic)
│   ├── upload_service.py        # Validate + save uploaded files
│   ├── ingest_service.py        # Save -> extract -> chunk -> embed -> store (single file)
│   ├── bulk_ingest_service.py   # Parallel + batched pipeline for large corpora
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
├── scripts/
│   └── bulk_ingest.py            # CLI for bulk-loading large PDF corpora
│
├── assets/
│   ├── screenshots/               # App screenshots (this README)
│   └── diagrams/                  # Architecture + latency diagrams
│
├── data/uploads/                 # Uploaded PDFs (created at runtime)
├── database/chroma_db/           # Persistent vector store (created at runtime)
├── cache/                        # Scratch cache (created at runtime)
└── logs/                         # Rotating log files (created at runtime)
```

---

## Bulk-ingesting a large corpus (thousands to millions of PDFs)

The Streamlit upload widget is fine for a handful of files, but it
processes one PDF at a time — far too slow at scale. For large corpora,
use the bulk CLI instead:

```bash
python scripts/bulk_ingest.py /path/to/your/pdfs
```

This is dramatically faster because it:

- **Parallelizes PDF extraction** across all CPU cores (`--workers`,
  default: all logical cores) instead of one file at a time.
- **Batches embedding calls** — hundreds of chunks per call instead of
  one file per call (`--embed-batch-size`, default 256).
- **Batches vector-store writes** with precomputed embeddings instead of
  triggering an embed-and-write per file (`--write-batch-size`).
- **Skips already-ingested files** via a SHA-256 content-hash registry
  (`database/ingest_registry.sqlite3`), so re-running the same command
  after adding new files, or after a crash, only processes what's new.
- Optionally uses a **local `sentence-transformers` model** instead of
  Ollama for embeddings — far faster for bulk workloads, and uses a GPU
  automatically if one is available.

### Enabling the fast embedding backend

```bash
pip install sentence-transformers torch
```

On by default (`USE_FAST_BULK_EMBEDDER=true`) and falls back to Ollama
automatically if these packages aren't installed. Default model:
`BAAI/bge-small-en-v1.5` — small, fast, good general default; swap in any
sentence-transformers-compatible model via `FAST_EMBED_MODEL`.

### Tuning for your hardware

| Flag | What it controls | When to raise it |
|---|---|---|
| `--workers` | Parallel extraction processes | More CPU cores available |
| `--embed-batch-size` | Chunks per embedding call | More RAM / GPU memory |
| `--write-batch-size` | Chunks per Chroma write | Usually fine at default |

```bash
# Example for a beefy machine with a GPU
python scripts/bulk_ingest.py /data/pdfs --workers 32 --embed-batch-size 1024
```

---

## Scaling & latency at 1,000,000 PDFs

*(Assumptions: ~10 pages/PDF, ~8 chunks/PDF average → 1M PDFs ≈ 8M chunks.
Full derivation with formulas in [`SCALING_AND_LATENCY.md`](./SCALING_AND_LATENCY.md).)*

![Estimated bulk ingestion time by hardware tier](./assets/diagrams/latency_by_hardware.png)

| Hardware profile | Estimated end-to-end time |
|---|---|
| CPU-only, Ollama embeddings | ~5–9 days |
| CPU-only, fast local embedder | ~7–13 hours |
| 32 cores + 1 GPU, fast embedder | **~2–2.5 hours** |
| 64 cores + A100, fast embedder | ~1.2–1.5 hours |

**The embedding backend is the dominant lever — not CPU count.** Switching
from Ollama to the local fast embedder is roughly a 100x speedup on its
own; adding a GPU on top gets you into the 1–3 hour range for the full
corpus. See [`SCALING_AND_LATENCY.md`](./SCALING_AND_LATENCY.md) for the
full stage-by-stage breakdown (hashing, extraction, embedding, writes),
RAM/storage requirements, and re-run/incremental-update costs.

---

## Troubleshooting

<details>
<summary><code>ModuleNotFoundError: No module named 'core.rag_engine'</code> (or any other <code>core.*</code> submodule)</summary>

<br>

This means the file genuinely didn't make it out of a zip extraction (some
Windows extractors silently drop files), not a `sys.path` issue. Check:

```powershell
dir core\rag_engine.py
dir core\text_processor.py
```

If either is missing, re-extract with 7-Zip or `Expand-Archive` rather than
Windows' built-in "Extract All", and delete stale bytecode before retrying:

```powershell
Remove-Item -Recurse -Force core\__pycache__
streamlit run app.py
```
</details>

<details>
<summary>Chat answers are wrong, irrelevant, or ignore uploaded content</summary>

<br>

Almost always an embedding backend mismatch — if you switched
`USE_FAST_BULK_EMBEDDER`, `FAST_EMBED_MODEL`, or `EMBEDDING_MODEL` after
already ingesting documents, ingestion and querying are using different
vector spaces. The app should refuse to start in this case with a clear
`RuntimeError` — if you're seeing silently bad answers instead, confirm
your config hasn't drifted, or delete `database/chroma_db` and re-ingest
from scratch with your current settings.
</details>

<details>
<summary><code>ConnectionError: Failed to connect to Ollama</code></summary>

<br>

Ollama isn't running, or isn't reachable at `OLLAMA_BASE_URL`. Start it
with `ollama serve` (or open the Ollama desktop app) and confirm the
required models are pulled:

```bash
ollama list
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```
</details>

<details>
<summary>A PDF fails to ingest with "No extractable text found"</summary>

<br>

The PDF is likely scanned/image-only with no embedded text layer.
PyMuPDF extracts text, not images — OCR is not currently implemented.
Run the file through an OCR tool first (e.g. `ocrmypdf`) to add a text
layer, then re-upload.
</details>

---

## Known limits

Being direct about where this architecture stops scaling gracefully:

- **Chroma is single-node.** Solid up to low tens of millions of vectors
  on one machine with enough RAM, but there's no built-in sharding or
  replication. Past ~10–20M chunks, consider migrating to a dedicated
  vector DB (Qdrant, Milvus, Weaviate, pgvector) — `core/vector_store.py`
  is intentionally the only file that would need to change.
- **SQLite skip-cache** handles the 1M-row registry fine for a single
  ingestion process, but isn't designed for multiple concurrent
  `bulk_ingest.py` runs against the same registry simultaneously.
- **No distributed task queue.** Everything assumes one machine's
  cores/GPU. For 10M+ files, adding Celery/Ray-style orchestration would
  likely become necessary.
- **No OCR.** Scanned/image-only PDFs are rejected rather than processed.
- **Chat-time latency is roughly constant** regardless of corpus size —
  it's dominated by the LLM call (~1–5s), not vector search (~20–100ms
  even at 8M vectors).

---

## Roadmap

- [ ] OCR fallback for scanned/image-only PDFs
- [ ] Support additional file types (`.docx`, `.txt`, `.md`)
- [ ] Pluggable vector store backend (Qdrant/pgvector) for >10M chunk corpora
- [ ] Multi-user auth and per-user document isolation
- [ ] Streaming token-by-token chat responses

---

## License

MIT — use freely, modify freely, no warranty implied.
