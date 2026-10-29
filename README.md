# PolicyPilot

AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

> **Status:** Phase 4 — Embeddings and Vector Search. PDF ingestion, page-preserving extraction, paragraph- and heading-aware chunking, dense vector embeddings (FastEmbed), persistent vector indexing (Qdrant), semantic vector search API, and search UI are operational. BM25 keyword search, hybrid retrieval (RRF), cross-encoder reranking, and LLM answer generation will be introduced in subsequent phases.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Node.js | 18+ |
| npm | 9+ |
| Python | 3.11+ |
| Git | 2.30+ |

---

## Repository Structure

```
PolicyPilot/
├── frontend/           # Next.js + TypeScript + Tailwind CSS v3
│   ├── app/            # App Router pages and layouts (Library, Chunks Viewer, Semantic Search)
│   └── ...
├── backend/            # Python / FastAPI
│   ├── app/            # Application package
│   │   ├── main.py     # FastAPI entry point & CORS configuration
│   │   ├── config.py   # Environment settings, chunking, Qdrant & embedding config
│   │   ├── models/     # (reserved)
│   │   ├── schemas/    # Pydantic schemas (Document, Chunk, Search, Index)
│   │   ├── services/   # Business logic (pdf, storage, chunking, embedding, vector, indexing, search)
│   │   └── routers/    # API routes (health, documents, search)
│   ├── tests/          # pytest test suite (test_health, test_documents, test_chunking, test_vector_search)
│   ├── data/           # Local storage for PDFs, metadata, chunks, and Qdrant (gitignored)
│   ├── requirements.txt
│   └── requirements-dev.txt
├── docs/               # Architecture and design documentation
├── .env.example        # Environment variable template (never commit real .env)
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/MohdAli056/PolicyPilot.git
cd PolicyPilot
cp .env.example .env          # Edit .env if you need non-default settings
```

### 2. Backend setup

```bash
cd backend
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 3. Frontend setup

```bash
cd frontend
npm install
```

---

## Development Commands

### Backend

```bash
cd backend

# Start the development server (auto-reload)
.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000

# Run automated tests (57 tests)
.venv\Scripts\pytest.exe -v

# Lint / format check
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
```

### Frontend

```bash
cd frontend

# Start the development server
npm run dev

# Lint
npm run lint

# Type check
npx tsc --noEmit

# Production build (verification)
npm run build
```

---

## API Endpoints

### Health Check

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service liveness check |

### Document Ingestion & Retrieval

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/documents/upload` | Ingest a PDF document, extract text, and chunk |
| GET | `/api/documents` | List all ingested documents (metadata only) |
| GET | `/api/documents/{document_id}` | Retrieve document details with raw page extractions |
| GET | `/api/documents/{document_id}/chunks` | Retrieve structured chunks with section and page attribution |
| GET | `/api/documents/{document_id}/file` | Stream or download original stored PDF file |
| POST | `/api/documents/{document_id}/index` | Generate embeddings and index document chunks into Qdrant |

### Semantic Vector Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search` | Search policy chunks by natural language semantic query |

#### POST `/api/documents/{document_id}/index` Example

```bash
curl -X POST http://localhost:8000/api/documents/72282962-7bde-404d-a821-f69d398c5c19/index
```

**Response (200 OK):**
```json
{
  "document_id": "72282962-7bde-404d-a821-f69d398c5c19",
  "indexed_chunks": 2,
  "status": "completed",
  "message": "Successfully indexed 2 chunks into vector store"
}
```

#### POST `/api/search` Example

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How many annual vacation days do employees receive?", "limit": 3}'
```

**Response (200 OK):**
```json
{
  "query": "How many annual vacation days do employees receive?",
  "total_results": 1,
  "results": [
    {
      "document_id": "72282962-7bde-404d-a821-f69d398c5c19",
      "filename": "leave_policy.pdf",
      "chunk_id": "3a0df4c5-0951-536d-b8d9-399d8b1e42fa",
      "chunk_index": 0,
      "text": "1. Annual Leave Policy\nFull-time employees receive 20 days paid annual leave per calendar year...",
      "section": "1. Annual Leave Policy",
      "page_numbers": [1],
      "character_count": 420,
      "score": 0.8124
    }
  ]
}
```

---

## Vector Search & Embedding Architecture

### 1. Vector Database: Qdrant
- **Engine:** Qdrant (`qdrant-client`).
- **Storage Mode:** Runs in embedded local disk mode by default (`data/qdrant`), requiring no Docker or external cloud services for local development. Alternatively, set `QDRANT_URL` to connect to a local container or remote cluster.
- **Collection Setup:** Default collection `policypilot_chunks` configured with Cosine distance metric and 384 vector dimensions.
- **Dimension Enforcement:** Collections automatically validate dimension compatibility upon initialization, raising a clear error if configured embedding dimensions conflict with existing stored vectors.
- **Lock Management:** Managed via a centralized singleton client to prevent Windows SQLite handle lock contention.

### 2. Embedding Provider Abstraction
- Defined in `backend/app/services/embedding_service.py` under `BaseEmbeddingProvider`.
- **FastEmbedProvider (Default):** Uses `fastembed` with the pinned model `BAAI/bge-small-en-v1.5` (384 dimensions). Runs locally via ONNX Runtime without heavy PyTorch dependencies (~67MB download on first run, cached locally).
- **MockEmbeddingProvider:** Generates reproducible, unit-normalized vectors based on deterministic text hashing for offline, sub-second test execution.

### 3. Reindexing & Idempotency
- Point IDs in Qdrant directly use the chunk's deterministic UUIDv5 identifier (`chunk_id`).
- When a document is reindexed:
  1. Prior vectors for that `document_id` are pruned to ensure no stale orphaned points remain if the chunk count changed.
  2. New vectors are upserted with deterministic point IDs.
  3. No duplicate vector records are created.

---

## Storage Architecture

- **Uploaded PDFs**: Stored at `backend/data/documents/{document_id}.pdf`.
- **Extraction Metadata**: Persisted atomically as JSON at `backend/data/metadata/{document_id}.json`.
- **Structured Chunks**: Persisted atomically as JSON at `backend/data/chunks/{document_id}.json`.
- **Qdrant Vector Database**: Stored at `backend/data/qdrant/`.
- **Git Exclusions**: `backend/data/`, `data/`, `*.pdf`, and `.fastembed_cache/` are excluded in `.gitignore` to prevent committing document or vector data.

---

## Environment Variables

See [`.env.example`](.env.example) for configuration options:

| Variable | Default | Used by | Description |
|----------|---------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000` | Backend | Allowed CORS origins (comma-separated) |
| `DEBUG` | `false` | Backend | Enable verbose debug logging |
| `DOCUMENTS_DIR` | `data/documents` | Backend | Directory for storing raw PDF files |
| `METADATA_DIR` | `data/metadata` | Backend | Directory for storing extraction JSON |
| `CHUNKS_DIR` | `data/chunks` | Backend | Directory for storing chunk JSON files |
| `TARGET_CHUNK_SIZE` | `1000` | Backend | Target chunk size in characters (must be > 0) |
| `CHUNK_OVERLAP` | `150` | Backend | Chunk overlap in characters (must be < TARGET_CHUNK_SIZE) |
| `QDRANT_PATH` | `data/qdrant` | Backend | Directory for local Qdrant embedded storage |
| `QDRANT_URL` | `None` | Backend | Optional URL for remote/container Qdrant instance |
| `QDRANT_COLLECTION` | `policypilot_chunks` | Backend | Qdrant collection name |
| `EMBEDDING_PROVIDER` | `fastembed` | Backend | Embedding provider (`fastembed` or `mock`) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Backend | Pinned embedding model identifier |
| `EMBEDDING_DIMENSION` | `384` | Backend | Vector dimension for the embedding model |
| `EMBEDDING_BATCH_SIZE` | `32` | Backend | Batch size for embedding generation |
| `DEFAULT_SEARCH_LIMIT` | `5` | Backend | Default number of search results to return |
| `MAX_SEARCH_LIMIT` | `50` | Backend | Maximum search limit allowed |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend | Backend API endpoint URL |

---

## Known Limitations

- **OCR Not Supported**: Scanned, image-only PDFs produce 0 extracted characters. Pages are preserved with `text=""` and indexing safely returns 0 indexed chunks.
- **Vector Retrieval Only**: Phase 4 retrieves ranked source passages based on semantic vector similarity. BM25 keyword matching, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and LLM answer generation will be added in subsequent phases.
- **Heading Recognition**: Plain-text regex heuristics are used; styling-based headings lacking structural markers fall back to `null`.

---

## Documentation

- [Architecture Specification](docs/architecture.md) — detailed system design, implemented vs. planned features, and open decisions.

---

## License

Private — not yet licensed for distribution.
