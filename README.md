# PolicyPilot

AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

> **Status:** Phase 3 — Document Chunking and Metadata. PDF ingestion, page-preserving extraction, paragraph- and heading-aware chunking with deterministic page attribution, and chunk inspection UI are operational. Dense embeddings, vector search (Qdrant), BM25 hybrid retrieval, reranking, and LLM answer generation are planned for future phases.

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
│   ├── app/            # App Router pages and layouts (Document Library, Chunks Viewer)
│   └── ...
├── backend/            # Python / FastAPI
│   ├── app/            # Application package
│   │   ├── main.py     # FastAPI entry point & CORS configuration
│   │   ├── config.py   # Environment-based settings & chunking configuration
│   │   ├── models/     # (reserved)
│   │   ├── schemas/    # Pydantic schemas (DocumentMetadata, DocumentDetail, DocumentChunk)
│   │   ├── services/   # Business logic (pdf_service, storage_service, chunking_service)
│   │   └── routers/    # API route modules (health, documents)
│   ├── tests/          # pytest test suite (test_health, test_documents, test_chunking)
│   ├── data/           # Local storage for PDFs, metadata, and chunks (gitignored)
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

# Run automated tests
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

**Example Response:**
```json
{
  "status": "healthy",
  "service": "PolicyPilot",
  "version": "0.1.0",
  "timestamp": "2026-10-27T16:00:00+00:00"
}
```

### Document Ingestion & Retrieval

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/documents/upload` | Ingest a PDF document (multipart/form-data), extract text, and chunk |
| GET | `/api/documents` | List all ingested documents (metadata only) |
| GET | `/api/documents/{document_id}` | Retrieve document details with raw page extractions |
| GET | `/api/documents/{document_id}/chunks` | Retrieve structured chunks with section and page attribution |
| GET | `/api/documents/{document_id}/file` | Stream or download original stored PDF file |

#### POST `/api/documents/upload` Example

```bash
curl -X POST -F "file=@leave_policy.pdf" http://localhost:8000/api/documents/upload
```

#### GET `/api/documents/{document_id}/chunks` Example

```bash
curl http://localhost:8000/api/documents/72282962-7bde-404d-a821-f69d398c5c19/chunks
```

**Response (200 OK):**
```json
{
  "document_id": "72282962-7bde-404d-a821-f69d398c5c19",
  "filename": "leave_policy.pdf",
  "total_chunks": 2,
  "chunks": [
    {
      "chunk_id": "3a0df4c5-0951-536d-b8d9-399d8b1e42fa",
      "document_id": "72282962-7bde-404d-a821-f69d398c5c19",
      "chunk_index": 0,
      "text": "1. Annual Leave Policy\nFull-time employees receive 20 days paid annual leave per calendar year. Leave accrues monthly...",
      "section": "1. Annual Leave Policy",
      "page_numbers": [1],
      "character_count": 420
    },
    {
      "chunk_id": "8fbb1a90-7d0c-512c-982d-2f16a0fce647",
      "document_id": "72282962-7bde-404d-a821-f69d398c5c19",
      "chunk_index": 1,
      "text": "2. Leave Rollover Rules\nA maximum of 5 unused leave days may be carried over into the next calendar year...",
      "section": "2. Leave Rollover Rules",
      "page_numbers": [1, 2],
      "character_count": 380
    }
  ]
}
```

---

## Chunking Architecture

The chunking engine (`backend/app/services/chunking_service.py`) prepares extracted PDF text for vector indexing and retrieval.

### 1. Algorithm & Boundary Preference
- **Unit of Measurement:** All sizes and overlaps are measured in **characters** (`character_count`), explicitly separated from token counts.
- **Boundary Priority:**
  1. Section / Heading boundaries (Markdown `#`, numbered `1.`, keyword `Section`, or `ALL-CAPS`).
  2. Paragraph boundaries (`\n\n`).
  3. Sentence boundaries (`.`, `?`, `!`).
  4. Word / character boundaries (strict fallback for oversized blocks exceeding target size).
- **Infinite Loop Protection:** Sliding window advances by at least one atomic segment per iteration (`start_idx = max(start_idx + 1, back_idx + 1)`), mathematically precluding infinite loops even with short text or large overlaps.
- **No Empty Chunks:** Strips empty or whitespace-only chunks prior to returning.

### 2. Heading-Aware Section Detection
- Detects headings using structured regex:
  - Markdown headings: `# Heading`, `## Subheading`
  - Numbered headings: `1. Introduction`, `2.1 Policy Scope`
  - Keyword headings: `Section 3: Benefits`, `Article IV`, `Chapter 1`
  - ALL-CAPS single-line headings: `HOLIDAY SCHEDULE`
- **Limitations:** Heuristic detection operates on plain extracted text without font-size or typography styling signals. Content with non-standard formatting falls back to `section: null` without fabricating phantom sections.

### 3. Page Attribution Rules
- Every chunk retains a sorted, deduplicated list of 1-based page numbers (`page_numbers: list[int]`) that contributed text to that chunk.
- Chunks spanning multiple pages explicitly record all contributing pages (e.g. `[1, 2]`), enabling accurate multi-page source citations during subsequent RAG answer generation.

### 4. Deterministic Chunk Identity
- Chunk IDs are generated using UUIDv5 based on document identity, chunk index, and SHA-256 of the chunk text:
  ```python
  uuid.uuid5(uuid.NAMESPACE_OID, f"{document_id}:{chunk_index}:{sha256(text)}")
  ```
- Guaranteed to be reproducible across runs, processes, and reloads for identical documents and settings.

---

## Storage Architecture

- **Uploaded PDFs**: Stored at `backend/data/documents/{document_id}.pdf`.
- **Extraction Metadata**: Persisted atomically as JSON at `backend/data/metadata/{document_id}.json`.
- **Structured Chunks**: Persisted atomically as JSON at `backend/data/chunks/{document_id}.json`.
- **Git Exclusions**: `backend/data/`, `data/`, and all `*.pdf` files are excluded in `.gitignore` to prevent committing document or chunk data.

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
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Backend | Maximum upload size (10 MB) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend | Backend API endpoint URL |

---

## Known Limitations

- **OCR Not Supported**: Scanned, image-only PDFs produce 0 extracted characters. Pages are preserved with `text=""` and chunking safely returns an empty chunk list.
- **Password-Protected PDFs**: Encrypted PDFs are rejected with HTTP 400.
- **Heading Recognition**: Plain-text regex heuristics are used; styling-based headings lacking structural markers fall back to `null`.
- **Retrieval Scope**: Phase 3 strictly implements document chunking and metadata. Embeddings (Qdrant), BM25 hybrid search, reranking, and LLM answer generation will be introduced in subsequent phases.

---

## Documentation

- [Architecture Specification](docs/architecture.md) — detailed system design, implemented vs. planned features, and open decisions.

---

## License

Private — not yet licensed for distribution.
