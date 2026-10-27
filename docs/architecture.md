# PolicyPilot Architecture

> Phase 3 — Document Chunking and Metadata. Last updated: 2026-10-27.

This document describes the current architecture of PolicyPilot and outlines where future capabilities will be integrated. Sections are explicitly marked **[IMPLEMENTED]** or **[PLANNED]** to distinguish working code from future design intent.

---

## 1. Overview

PolicyPilot is an AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

---

## 2. System Layers

```
┌──────────────────────────────────────────────┐
│             Frontend (Next.js 16)            │  ← User interface (Dashboard, Library, Chunks Viewer)
├──────────────────────────────────────────────┤
│             Backend  (FastAPI)               │  ← API, storage & processing pipeline
├──────────────┬───────────────┬───────┬───────┤
│ PDF Ingest   │ Chunking      │ Vector│ BM25  │  ← Ingestion & Chunking [IMPLEMENTED]
│ & Extraction │ & Metadata    │ (Qdr) │ (RRF) │  ← Retrieval & Generation [PLANNED]
└──────────────┴───────────────┴───────┴───────┘
```

---

## 3. Frontend Responsibilities — [IMPLEMENTED]

| Concern | Status |
|---------|--------|
| Landing / dashboard shell | ✅ Implemented |
| Backend health-status badge | ✅ Implemented |
| PDF file picker with validation | ✅ Implemented |
| Upload & processing states | ✅ Implemented |
| Ingested document library list (`GET /api/documents`) | ✅ Implemented |
| Document detail panel with raw page extraction viewer | ✅ Implemented |
| Structured chunk inspector (`GET /api/documents/{id}/chunks`) | ✅ Implemented |
| Stream / open original PDF (`GET /api/documents/{id}/file`) | ✅ Implemented |
| Configurable backend URL (`NEXT_PUBLIC_API_URL`) | ✅ Implemented |
| Timeout and error handling | ✅ Implemented |

**Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v3.

**Planned additions (later phases):**
- Natural language policy question input and answer display.
- Grounded citation viewer with highlightable page-level links.
- Search diagnostics panel.
- RAG benchmark and evaluation dashboard.

---

## 4. Backend Responsibilities — [IMPLEMENTED]

| Concern | Status |
|---------|--------|
| FastAPI application entry point | ✅ Implemented |
| `GET /health` liveness endpoint | ✅ Implemented |
| `POST /api/documents/upload` PDF ingestion & chunking | ✅ Implemented |
| `GET /api/documents` document library listing | ✅ Implemented |
| `GET /api/documents/{document_id}` document detail | ✅ Implemented |
| `GET /api/documents/{document_id}/chunks` chunk retrieval | ✅ Implemented |
| `GET /api/documents/{document_id}/file` original file streaming | ✅ Implemented |
| PyMuPDF page-preserving text extraction | ✅ Implemented |
| Paragraph- and heading-aware chunking engine | ✅ Implemented |
| Multi-page attribution & deterministic UUIDv5 chunk IDs | ✅ Implemented |
| Atomic local storage & JSON metadata/chunks persistence | ✅ Implemented |
| SHA-256 document hashing for audit/deduplication | ✅ Implemented |
| Path traversal protection & server-generated UUIDs | ✅ Implemented |
| Automated pytest test suite (38 tests) | ✅ Implemented |

**Stack:** Python 3.12, FastAPI, PyMuPDF, Pydantic, Uvicorn.

**Planned additions (later phases):**
- Dense vector embeddings (OpenAI / Gemini / local).
- Vector indexing with Qdrant.
- BM25 sparse keyword retrieval & Reciprocal Rank Fusion (RRF).
- Reranking model integration.
- Grounded LLM answer generation with citation verification.

---

## 5. API Communication — [IMPLEMENTED]

- Frontend → Backend over HTTP REST.
- Base URL configurable via `NEXT_PUBLIC_API_URL` (frontend) / `CORS_ORIGINS` (backend).
- All requests are standard JSON / multipart form data; no auth required in Phase 3.

### Implemented Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service liveness check |
| POST | `/api/documents/upload` | Multipart upload, storage, page extraction, and chunk generation |
| GET | `/api/documents` | Metadata listing of all ingested documents |
| GET | `/api/documents/{document_id}` | Detailed document metadata with raw page extractions |
| GET | `/api/documents/{document_id}/chunks` | Ordered document chunks with section and page attribution |
| GET | `/api/documents/{document_id}/file` | Download or stream original PDF |

---

## 6. PDF Ingestion & Extraction Architecture — [IMPLEMENTED]

- **Engine:** PyMuPDF (`pymupdf`).
- **Identity:** Each document receives a server-generated UUID (`document_id`).
- **File Storage:** Stored as `backend/data/documents/{document_id}.pdf` (gitignored).
- **Metadata Storage:** Stored atomically as `backend/data/metadata/{document_id}.json` (gitignored).
- **Page Invariant:** Page numbers are 1-based and match standard PDF viewers.
- **Image-Only Documents:** Processed safely without crashing; pages are preserved with `text=""` and status indicates OCR is unsupported.
- **Security:**
  - Filenames sanitized to avoid directory traversal.
  - Streaming chunked upload enforces `MAX_UPLOAD_SIZE_BYTES` (default 10 MB).
  - Malformed, corrupt, or encrypted files are rejected with HTTP 400 and temporary files cleaned up.

---

## 7. Document Chunking & Metadata Engine — [IMPLEMENTED]

Implemented in `backend/app/services/chunking_service.py` and persisted via `backend/app/services/storage_service.py`.

### 7.1 Chunking Strategy & Algorithm
- **Character-Based Window:** Target size `TARGET_CHUNK_SIZE` (default 1,000 characters) and overlap `CHUNK_OVERLAP` (default 150 characters).
- **Validation:** Enforced via Pydantic validator (`0 <= CHUNK_OVERLAP < TARGET_CHUNK_SIZE` and `TARGET_CHUNK_SIZE > 0`).
- **Atomic Segment Splitting:** Text across pages is segmented preserving boundary hierarchies:
  1. Headings (Markdown, numbered, section keywords, ALL-CAPS).
  2. Paragraphs (`\n\n`).
  3. Sentences (`[.!?]\s+`).
  4. Word / Character fallbacks for oversized paragraphs.
- **Accumulation & Overlap:**
  - Segments are accumulated until adding the next segment exceeds the target chunk size.
  - When splitting, the window steps back by `chunk_overlap` characters to preserve contextual continuity.
  - Forward progress is mathematically guaranteed: `start_idx = max(start_idx + 1, back_idx + 1)`, preventing infinite loops.

### 7.2 Page Attribution
- Each atomic segment preserves its original 1-based PDF `page_number`.
- Chunks record all distinct page numbers that contributed text in `page_numbers: list[int]`, sorted in ascending order.
- Multi-page spanning chunks are accurately and explicitly tracked (e.g. `[1, 2]`).

### 7.3 Section Detection & Limitations
- Current section title is tracked dynamically as segments are processed.
- Heuristic regex recognizes:
  - Markdown: `# Title`, `## Subtitle`
  - Numbered: `1. Leave Entitlement`, `2.3 Travel Expenses`
  - Keywords: `Section 4: Medical Benefits`, `Article II`, `Chapter 1`
  - ALL-CAPS single-line headings: `HOLIDAY CALENDAR`
- **Limitations:** Only structural plain-text markers are detected. Stylistic formatting (such as font size variations without structural markers) cannot be recognized from plain extracted text and safely falls back to `section: null`.

### 7.4 Chunk Identity & Persistence
- **Deterministic Identity:**
  ```python
  uuid.uuid5(uuid.NAMESPACE_OID, f"{document_id}:{chunk_index}:{sha256(text)}")
  ```
- **Storage:** Persisted atomically as JSON at `backend/data/chunks/{document_id}.json` (gitignored).

---

## 8. Embeddings, Vector Search, and Retrieval — [PLANNED]

### 8.1 Embeddings
- Embedding provider abstraction (OpenAI `text-embedding-3-small`, Gemini, or local models).
- Dense vectors generated for each chunk.

### 8.2 Vector & Hybrid Search
- **Vector search:** Qdrant collection storing embeddings and chunk metadata.
- **BM25 keyword search:** Sparse keyword matching.
- **Hybrid fusion:** Reciprocal Rank Fusion (RRF) to merge dense and sparse rankings.

### 8.3 Reranking
- Cross-encoder reranking model to re-score top-K hybrid candidates.

---

## 9. Answer Generation & Citation Flow — [PLANNED]

- Grounded LLM prompt using retrieved chunks with page attribution.
- Citations linked directly to source `document_id`, `page_numbers`, and `section`.
- Citation verification to ensure answers are strictly evidence-backed.

---

## 10. Data Flow Across Phases

```
PDF upload [IMPLEMENTED]
  → page-level text extraction (page_number preserved) [IMPLEMENTED]
    → paragraph/heading chunking (deterministic ID + section + page_numbers) [IMPLEMENTED]
      → embedding generation & Qdrant indexing [PLANNED]
        → hybrid BM25 + vector search with RRF [PLANNED]
          → cross-encoder reranking [PLANNED]
            → LLM prompt with citations [PLANNED]
              → frontend answer & highlightable page citations [PLANNED]
```

---

## 11. Testing and Verification — [IMPLEMENTED]

- **Backend Tests:** 38 automated pytest tests covering:
  - Health check & baseline endpoints.
  - Upload, multi-page extraction, blank pages, corruption rejection, oversize rejection, metadata listing, detail retrieval, file streaming, and path traversal security.
  - Short/long text chunking, overlap enforcement, paragraph boundary preference, oversized paragraph splitting, heading detection (Markdown, Numbered, Keyword, ALL-CAPS), page spanning attribution, empty text handling, invalid settings validation, deterministic chunk IDs, and API chunk retrieval.
- **Code Quality:** Ruff linter & formatter for Python; ESLint and TypeScript for Next.js.
- **Browser Verification:** Real end-to-end browser verification of document upload and page/chunk inspection tabs.

---

## 12. Open Decisions

| Decision | Notes |
|----------|-------|
| Storage backend for production | Local filesystem used for Phase 2/3; S3/GCS object storage suitable for cloud deployment. |
| Embedding model selection | Will benchmark OpenAI `text-embedding-3-small` vs Gemini embeddings in Phase 4. |
| Reranker model | Cross-encoder (e.g. `bge-reranker`) vs. API-based reranker TBD in Phase 5. |
| OCR Support | Tesseract or Cloud OCR can be plugged into `pdf_service.py` if scanned document volume requires it. |
| Multi-tenancy | Document namespace / tenant IDs can be attached to metadata schema when auth is introduced. |
