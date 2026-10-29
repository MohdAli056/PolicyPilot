# PolicyPilot Architecture

> Phase 4 — Embeddings and Vector Search. Last updated: 2026-10-29.

This document describes the current architecture of PolicyPilot and outlines where future capabilities will be integrated. Sections are explicitly marked **[IMPLEMENTED]** or **[PLANNED]** to distinguish working code from future design intent.

---

## 1. Overview

PolicyPilot is an AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

---

## 2. System Layers

```
┌──────────────────────────────────────────────┐
│             Frontend (Next.js 16)            │  ← User interface (Dashboard, Chunks, Semantic Search)
├──────────────────────────────────────────────┤
│             Backend  (FastAPI)               │  ← API, storage & retrieval pipeline
├──────────────┬───────────────┬───────┬───────┤
│ PDF Ingest   │ Chunking      │ Qdrant│ BM25  │  ← Ingest, Chunks, Qdrant/Vectors [IMPLEMENTED]
│ & Extraction │ & Metadata    │ Vector│ (RRF) │  ← Keyword BM25, Reranker, LLM [PLANNED]
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
| Semantic search input & ranked results display (`POST /api/search`) | ✅ Implemented |
| Document indexing action trigger (`POST /api/documents/{id}/index`) | ✅ Implemented |
| Stream / open original PDF (`GET /api/documents/{id}/file`) | ✅ Implemented |
| Configurable backend URL (`NEXT_PUBLIC_API_URL`) | ✅ Implemented |
| Timeout and error handling | ✅ Implemented |

**Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v3.

**Planned additions (later phases):**
- Natural language answer generation with grounded evidence synthesis.
- Highlightable page-level citation links in PDF viewer.
- Hybrid BM25 + dense vector search toggle and score weights.
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
| `POST /api/documents/{document_id}/index` vector indexing | ✅ Implemented |
| `POST /api/search` semantic vector retrieval | ✅ Implemented |
| `GET /api/documents/{document_id}/file` original file streaming | ✅ Implemented |
| PyMuPDF page-preserving text extraction | ✅ Implemented |
| Paragraph- and heading-aware chunking engine | ✅ Implemented |
| Multi-page attribution & deterministic UUIDv5 chunk IDs | ✅ Implemented |
| Dedicated embedding provider abstraction (`BaseEmbeddingProvider`) | ✅ Implemented |
| Local ONNX embeddings via FastEmbed (`BAAI/bge-small-en-v1.5`) | ✅ Implemented |
| Qdrant vector database collection management & cosine search | ✅ Implemented |
| Idempotent reindexing & stale vector pruning | ✅ Implemented |
| Atomic local storage & JSON metadata/chunks persistence | ✅ Implemented |
| SHA-256 document hashing for audit/deduplication | ✅ Implemented |
| Path traversal protection & server-generated UUIDs | ✅ Implemented |
| Automated pytest test suite (57 tests) | ✅ Implemented |

**Stack:** Python 3.12, FastAPI, PyMuPDF, Qdrant Client, FastEmbed (ONNX Runtime), Pydantic, Uvicorn.

**Planned additions (later phases):**
- BM25 sparse keyword retrieval & Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking model integration.
- Grounded LLM answer generation with citation verification.

---

## 5. API Communication — [IMPLEMENTED]

- Frontend → Backend over HTTP REST.
- Base URL configurable via `NEXT_PUBLIC_API_URL` (frontend) / `CORS_ORIGINS` (backend).
- All requests are standard JSON / multipart form data; no auth required in Phase 4.

### Implemented Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service liveness check |
| POST | `/api/documents/upload` | Multipart upload, storage, page extraction, and chunk generation |
| GET | `/api/documents` | Metadata listing of all ingested documents |
| GET | `/api/documents/{document_id}` | Detailed document metadata with raw page extractions |
| GET | `/api/documents/{document_id}/chunks` | Ordered document chunks with section and page attribution |
| POST | `/api/documents/{document_id}/index` | Generate embeddings and store vectors in Qdrant |
| POST | `/api/search` | Natural language semantic vector search |
| GET | `/api/documents/{document_id}/file` | Download or stream original PDF |

---

## 6. PDF Ingestion & Extraction Architecture — [IMPLEMENTED]

- **Engine:** PyMuPDF (`pymupdf`).
- **Identity:** Each document receives a server-generated UUID (`document_id`).
- **File Storage:** Stored as `backend/data/documents/{document_id}.pdf` (gitignored).
- **Metadata Storage:** Stored atomically as `backend/data/metadata/{document_id}.json` (gitignored).
- **Page Invariant:** Page numbers are 1-based and match standard PDF viewers.
- **Image-Only Documents:** Processed safely without crashing; pages are preserved with `text=""` and status indicates OCR is unsupported.

---

## 7. Document Chunking & Metadata Engine — [IMPLEMENTED]

Implemented in `backend/app/services/chunking_service.py` and persisted via `backend/app/services/storage_service.py`.

- **Character-Based Window:** Target size `TARGET_CHUNK_SIZE` (default 1,000 characters) and overlap `CHUNK_OVERLAP` (default 150 characters).
- **Atomic Segment Splitting:** Preserves headings (Markdown, numbered, section keywords, ALL-CAPS), paragraphs, and sentences.
- **Page Attribution:** Retains all 1-based contributing page numbers in `page_numbers: list[int]`, correctly attributing multi-page spanning chunks.
- **Deterministic Identity:** UUIDv5 generated from `uuid.uuid5(uuid.NAMESPACE_OID, f"{document_id}:{chunk_index}:{sha256(text)}")`.

---

## 8. Embeddings & Vector Search Architecture — [IMPLEMENTED]

Implemented in `backend/app/services/embedding_service.py`, `backend/app/services/vector_service.py`, `backend/app/services/indexing_service.py`, and `backend/app/services/search_service.py`.

### 8.1 Embedding Provider Abstraction
- **Abstract Base Class:** `BaseEmbeddingProvider` defines `embed_texts()` and `embed_query()`.
- **FastEmbedProvider (Default):**
  - Model: `BAAI/bge-small-en-v1.5` (pinned).
  - Dimension: 384 dimensions.
  - Runtime: ONNX Runtime (CPU optimized, lightweight, no PyTorch overhead).
  - Model Singleton: Loaded once and cached in memory across requests.
- **MockEmbeddingProvider:**
  - Fast, deterministic unit-normalized pseudo-vectors for offline automated test suites without requiring network access.

### 8.2 Vector Database: Qdrant
- **Engine:** `qdrant-client` 1.20+.
- **Storage Mode:** Runs in embedded local disk mode by default (`data/qdrant`), requiring zero external dependencies or Docker containers. Alternatively supports remote/container deployment via `QDRANT_URL`.
- **Collection Configuration:**
  - Collection name: `policypilot_chunks`
  - Distance metric: `Cosine`
  - Vector size: 384
- **Dimension Compatibility Validation:** The service verifies existing collection dimensions on startup and raises `ValueError` if the configured model dimension conflicts with existing vectors.

### 8.3 Vector Payloads & Reindexing Idempotency
- Each point in Qdrant uses the chunk's deterministic `chunk_id` as its Point ID.
- Payloads store:
  - `document_id`: UUID of parent document
  - `filename`: original filename
  - `chunk_id`: chunk UUIDv5
  - `chunk_index`: zero-based sequence index
  - `text`: chunk text
  - `section`: detected section or null
  - `page_numbers`: contributing 1-based page numbers
  - `character_count`: character length
  - `indexed_at`: ISO UTC timestamp
- **Reindexing Behavior:**
  1. Any prior vectors for that `document_id` are pruned first to prevent stale orphaned points if the chunk count changed.
  2. The new chunks are upserted with deterministic point IDs.
  3. Reindexing unchanged chunks is completely idempotent and produces no duplicates.

---

## 9. Data Flow Across Phases

```
PDF upload [IMPLEMENTED]
  → page-level text extraction (page_number preserved) [IMPLEMENTED]
    → paragraph/heading chunking (deterministic ID + section + page_numbers) [IMPLEMENTED]
      → embedding generation & Qdrant indexing [IMPLEMENTED]
        → semantic vector search with cosine scoring [IMPLEMENTED]
          → hybrid BM25 + dense search with RRF [PLANNED]
            → cross-encoder reranking [PLANNED]
              → LLM prompt with citations [PLANNED]
                → frontend answer & highlightable page citations [PLANNED]
```

---

## 10. Testing and Verification — [IMPLEMENTED]

- **Backend Tests:** 57 automated pytest tests covering:
  - Health check & baseline endpoints (7 tests).
  - Document upload, extraction, corruption handling, and security (13 tests).
  - Paragraph/heading chunking, page attribution, and settings validation (18 tests).
  - Embedding generation, dimension validation, Qdrant collection management, indexing idempotency, metadata preservation, empty collection search, and API search/index endpoints (19 tests).
- **Code Quality:** Ruff linter & formatter for Python; ESLint and TypeScript for Next.js.
- **Browser Verification:** Real end-to-end browser verification of document upload, indexing, semantic search, and document detail inspection.

---

## 11. Open Decisions

| Decision | Notes |
|----------|-------|
| Storage backend for production | Local filesystem & embedded Qdrant used for Phase 4; Qdrant Cloud/Cluster & S3 object storage suitable for cloud production. |
| Hybrid retrieval weights | Tuning alpha between dense cosine similarity and sparse BM25 scores in Phase 5. |
| Reranker model | Cross-encoder (e.g. `bge-reranker-base` or Cohere API) TBD in Phase 5. |
| OCR Support | Tesseract or Cloud OCR can be plugged into `pdf_service.py` if scanned document volume requires it. |
| Multi-tenancy | Document namespace / tenant IDs can be attached to metadata schema when auth is introduced. |
