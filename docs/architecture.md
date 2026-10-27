# PolicyPilot Architecture

> Phase 2 — Document Ingestion and Text Extraction. Last updated: 2026-10-27.

This document describes the current architecture of PolicyPilot and outlines where future capabilities will be integrated. Sections are explicitly marked **[IMPLEMENTED]** or **[PLANNED]** to distinguish working code from future design intent.

---

## 1. Overview

PolicyPilot is an AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

---

## 2. System Layers

```
┌──────────────────────────────────────────────┐
│             Frontend (Next.js 16)            │  ← User interface (Dashboard, Upload, Viewer)
├──────────────────────────────────────────────┤
│             Backend  (FastAPI)               │  ← API, storage & document ingestion
├──────────────┬───────────────┬───────┬───────┤
│ PDF Ingestion│ Vector DB     │ BM25  │ LLM   │  ← Ingestion [IMPLEMENTED], RAG [PLANNED]
│ (PyMuPDF)    │ (Qdrant)      │ (RRF) │ (Rer) │
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
| Document detail panel with page-by-page text viewer | ✅ Implemented |
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
| `POST /api/documents/upload` PDF ingestion | ✅ Implemented |
| `GET /api/documents` document library listing | ✅ Implemented |
| `GET /api/documents/{document_id}` document detail | ✅ Implemented |
| `GET /api/documents/{document_id}/file` original file streaming | ✅ Implemented |
| PyMuPDF page-preserving text extraction | ✅ Implemented |
| Atomic local storage & JSON metadata persistence | ✅ Implemented |
| SHA-256 document hashing for audit/deduplication | ✅ Implemented |
| Path traversal protection & server-generated UUIDs | ✅ Implemented |
| Automated pytest test suite | ✅ Implemented |

**Stack:** Python 3.12, FastAPI, PyMuPDF, Pydantic, Uvicorn.

**Planned additions (later phases):**
- Section-aware text chunking.
- Dense vector embeddings (OpenAI / Gemini / local).
- Vector indexing with Qdrant.
- BM25 sparse keyword retrieval & Reciprocal Rank Fusion (RRF).
- Reranking model integration.
- Grounded LLM answer generation with citation verification.

---

## 5. API Communication — [IMPLEMENTED]

- Frontend → Backend over HTTP REST.
- Base URL configurable via `NEXT_PUBLIC_API_URL` (frontend) / `CORS_ORIGINS` (backend).
- All requests are standard JSON / multipart form data; no auth required in Phase 2.

### Implemented Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service liveness check |
| POST | `/api/documents/upload` | Multipart upload, storage, and page extraction |
| GET | `/api/documents` | Metadata listing of all ingested documents |
| GET | `/api/documents/{document_id}` | Detailed document metadata with page extractions |
| GET | `/api/documents/{document_id}/file` | Download or stream original PDF |

---

## 6. PDF Ingestion & Extraction Architecture — [IMPLEMENTED]

- **Engine:** PyMuPDF (`pymupdf`).
- **Identity:** Each document receives a server-generated UUID (`document_id`).
- **File Storage:** Stored as `backend/data/documents/{document_id}.pdf` (gitignored).
- **Metadata Storage:** Stored atomically as `backend/data/metadata/{document_id}.json` (gitignored).
- **Page Invariant:** Page numbers are 1-based and match standard PDF viewers.
- **Image-Only Documents:** Processed safely without crashing; pages are preserved with `text=""` and status indicates OCR is unsupported in Phase 2.
- **Security:**
  - Filenames sanitized to avoid directory traversal.
  - Streaming chunked upload enforces `MAX_UPLOAD_SIZE_BYTES` (default 10 MB).
  - Malformed, corrupt, or encrypted files are rejected with HTTP 400 and temporary files cleaned up.

---

## 7. Chunking, Embeddings, and Retrieval — [PLANNED]

### 7.1 Chunking
- Section-aware splitter: detect headings/sections from PDF structure.
- Each chunk carries metadata: `document_id`, `page_number`, `section_title`, `chunk_index`.
- Chunk size and overlap will be configurable.

### 7.2 Embeddings
- Embedding provider abstraction (swap between OpenAI, Gemini, local models).
- Chunks embedded and stored in Qdrant vector database.

### 7.3 Retrieval
- **Vector search:** Qdrant approximate nearest-neighbor search.
- **BM25 keyword search:** In-memory or index-based keyword retrieval.
- **Hybrid fusion:** Reciprocal Rank Fusion (RRF) to combine vector and keyword scores.

### 7.4 Reranking
- Reranking provider abstraction.
- Top-K hybrid results re-scored for relevance before answer generation.

---

## 8. Answer Generation — [PLANNED]

- Configurable LLM provider (Gemini, OpenAI, etc.).
- System prompt enforces grounded answers — the model must cite evidence.
- Response includes answer text + list of citations (`document_id`, `page_number`, `section`).
- If evidence is insufficient, the response explicitly says so rather than hallucinating.

---

## 9. Citation Preservation — [PLANNED]

The chain of document identity flows end-to-end:

```
PDF upload [DONE]
  → page-level text extraction (page_number preserved) [DONE]
    → section-aware chunking (document_id + page_number + section) [PLANNED]
      → embedding storage (metadata attached to vectors) [PLANNED]
        → retrieval results (metadata returned with chunks) [PLANNED]
          → LLM prompt (chunks include page refs) [PLANNED]
            → answer response (citations with page numbers) [PLANNED]
              → frontend display (clickable page links) [PLANNED]
```

---

## 10. Testing and Verification — [IMPLEMENTED]

- **Backend Tests:** 20 automated pytest tests covering upload, multi-page extraction, blank pages, corruption rejection, oversize rejection, metadata listing, detail retrieval, file streaming, and path traversal security.
- **Code Quality:** Ruff linter & formatter for Python; ESLint and TypeScript for Next.js.
- **Browser Verification:** Real end-to-end browser verification of document upload and page extraction viewer.

---

## 11. Open Decisions

| Decision | Notes |
|----------|-------|
| Storage backend for production | Local filesystem used for Phase 2; S3/GCS object storage suitable for cloud deployment. |
| Embedding model selection | Will benchmark OpenAI `text-embedding-3-small` vs Gemini embeddings in Phase 3. |
| Reranker model | Cross-encoder (e.g. `bge-reranker`) vs. API-based reranker TBD. |
| OCR Support | Tesseract or Cloud OCR can be plugged into `pdf_service.py` if scanned document volume requires it. |
| Multi-tenancy | Document namespace / tenant IDs can be attached to metadata schema when auth is introduced. |
