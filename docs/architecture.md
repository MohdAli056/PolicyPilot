# PolicyPilot Architecture

> Phase 5 — BM25 Keyword Search + Hybrid Retrieval with Reciprocal Rank Fusion (RRF). Last updated: 2026-10-29.

This document describes the current architecture of PolicyPilot and outlines where future capabilities will be integrated. Sections are explicitly marked **[IMPLEMENTED]** or **[PLANNED]** to distinguish working code from future design intent.

---

## 1. Overview

PolicyPilot is an AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

---

## 2. System Layers

```
┌────────────────────────────────────────────────────────┐
│                  Frontend (Next.js 16)                 │  ← User interface (Library, Chunks, Search Modes)
├────────────────────────────────────────────────────────┤
│                  Backend  (FastAPI)                    │  ← API, storage & retrieval pipeline
├──────────────┬───────────────┬────────────┬────────────┤
│ PDF Ingest   │ Chunking      │ Qdrant     │ BM25 & RRF │  ← Ingest, Chunks, Vector & Hybrid [IMPLEMENTED]
│ & Extraction │ & Metadata    │ Vector     │ Hybrid     │  ← Cross-Encoder Reranker, LLM [PLANNED]
└──────────────┴───────────────┴────────────┴────────────┘
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
| Search mode toggle: Hybrid (BM25 + Vector) vs. Semantic (Vector) | ✅ Implemented |
| Hybrid search input & fused results display with RRF scores & ranks | ✅ Implemented |
| Semantic search input & cosine similarity results display | ✅ Implemented |
| Document indexing action trigger (`POST /api/documents/{id}/index`) | ✅ Implemented |
| Stream / open original PDF (`GET /api/documents/{id}/file`) | ✅ Implemented |
| Configurable backend URL (`NEXT_PUBLIC_API_URL`) | ✅ Implemented |
| Timeout, empty states, and error handling | ✅ Implemented |

**Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v3.

**Planned additions (later phases):**
- Natural language answer generation with grounded evidence synthesis (Phase 7).
- Highlightable page-level citation links in PDF viewer (Phase 7).
- Cross-encoder reranking score display (Phase 6).
- RAG benchmark and evaluation dashboard (Phase 8).

---

## 4. Backend Responsibilities — [IMPLEMENTED]

| Concern | Status |
|---------|--------|
| FastAPI application entry point with startup index sync | ✅ Implemented |
| `GET /health` liveness endpoint | ✅ Implemented |
| `POST /api/documents/upload` PDF ingestion, chunking & BM25 indexing | ✅ Implemented |
| `GET /api/documents` document library listing | ✅ Implemented |
| `GET /api/documents/{document_id}` document detail | ✅ Implemented |
| `GET /api/documents/{document_id}/chunks` chunk retrieval | ✅ Implemented |
| `POST /api/documents/{document_id}/index` vector & BM25 indexing | ✅ Implemented |
| `POST /api/search` semantic vector retrieval (backward-compatible) | ✅ Implemented |
| `POST /api/search/keyword` exact BM25 keyword retrieval | ✅ Implemented |
| `POST /api/search/hybrid` hybrid retrieval combining vector & BM25 with RRF | ✅ Implemented |
| `GET /api/documents/{document_id}/file` original file streaming | ✅ Implemented |
| PyMuPDF page-preserving text extraction | ✅ Implemented |
| Paragraph- and heading-aware chunking engine | ✅ Implemented |
| Multi-page attribution & deterministic UUIDv5 chunk IDs | ✅ Implemented |
| Dedicated embedding provider abstraction (`BaseEmbeddingProvider`) | ✅ Implemented |
| Local ONNX embeddings via FastEmbed (`BAAI/bge-small-en-v1.5`) | ✅ Implemented |
| Qdrant vector database collection management & cosine search | ✅ Implemented |
| BM25 retrieval service with Lucene-adjusted IDF & code tokenization | ✅ Implemented |
| Reciprocal Rank Fusion (RRF) algorithm with deterministic tie-breaking | ✅ Implemented |
| Keyword index synchronization with disk on startup and upload/index | ✅ Implemented |
| Idempotent reindexing & stale vector/chunk pruning | ✅ Implemented |
| Atomic local storage & JSON metadata/chunks persistence | ✅ Implemented |
| SHA-256 document hashing for audit/deduplication | ✅ Implemented |
| Path traversal protection & server-generated UUIDs | ✅ Implemented |
| Automated pytest test suite (77 tests, 100% pass) | ✅ Implemented |

**Stack:** Python 3.12, FastAPI, PyMuPDF, Qdrant Client, FastEmbed (ONNX Runtime), rank-bm25, Pydantic, Uvicorn.

**Planned additions (later phases):**
- Cross-encoder reranking model integration (Phase 6).
- Grounded LLM answer generation with citation verification (Phase 7).

---

## 5. API Communication — [IMPLEMENTED]

- Frontend → Backend over HTTP REST.
- Base URL configurable via `NEXT_PUBLIC_API_URL` (frontend) / `CORS_ORIGINS` (backend).
- All requests are standard JSON / multipart form data; no auth required in Phase 5.

### Implemented Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service liveness check |
| POST | `/api/documents/upload` | Multipart upload, storage, page extraction, chunking, and BM25 index |
| GET | `/api/documents` | Metadata listing of all ingested documents |
| GET | `/api/documents/{document_id}` | Detailed document metadata with raw page extractions |
| GET | `/api/documents/{document_id}/chunks` | Ordered document chunks with section and page attribution |
| POST | `/api/documents/{document_id}/index` | Generate embeddings and store vectors in Qdrant & sync BM25 |
| POST | `/api/search` | Natural language semantic vector search (Phase 4 backward-compatible) |
| POST | `/api/search/keyword` | Exact BM25 keyword search over indexed chunks |
| POST | `/api/search/hybrid` | Fused search combining vector & BM25 candidate lists via RRF |
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

- **FastEmbedProvider:** Uses `BAAI/bge-small-en-v1.5` (384 dimensions) via ONNX Runtime without heavy PyTorch dependencies.
- **Qdrant Vector DB:** Runs in embedded local disk mode by default (`data/qdrant`), or connects to external cluster via `QDRANT_URL`.
- **Collection Setup:** Collection `policypilot_chunks` configured with Cosine distance metric and 384 dimensions.
- **Reindexing Idempotency:** Prior vectors for `document_id` are pruned first to guarantee no stale points, then new chunks are upserted.

---

## 9. BM25 Keyword Search & Hybrid Retrieval with RRF — [IMPLEMENTED]

Implemented in `backend/app/services/bm25_service.py` and `backend/app/services/hybrid_search_service.py`.

### 9.1 BM25 Keyword Retrieval
- **Formula:** Okapi BM25 with Lucene-adjusted non-negative IDF:
  $$\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$$
  $$\text{TF}_{\text{norm}}(t, d) = \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
- **Parameters:** Default $k_1 = 1.5$, $b = 0.75$.
- **Tokenization:**
  - Case-folded to lowercase.
  - Alphanumeric regex `[a-zA-Z0-9]+(?:[.\-_][a-zA-Z0-9]+)*` preserves compound identifiers (`POL-2026-X`, `SEC-101`, `1.1`) and numbers (`750`).
  - Constituent sub-tokens are emitted so partial queries like `2026` or `POL` match `POL-2026-X`.
- **Index Synchronization & Lifecycle:**
  - Disk persistence (`data/chunks/{document_id}.json`) remains the single source of truth.
  - On application startup, `bm25_service.sync_from_storage()` scans persisted chunks and builds the index in memory in milliseconds.
  - On upload and index actions, `bm25_service.index_document_chunks()` replaces prior document chunks atomically under a thread lock.
  - Backend restarts reliably reload the persisted index without loss or stale entries.

### 9.2 Reciprocal Rank Fusion (RRF)
- **Formula:**
  $$\text{RRF\_score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$
  Default $k = 60$ ($k > 0$).
- **Candidates:** Up to `hybrid_candidate_limit` (default: 20) candidates gathered from both vector search and BM25 search.
- **Provenance:** Records 1-based ranks (`vector_rank`, `keyword_rank`) and contributing retriever names (`retrievers: list[str]`). Missing retriever ranks are represented as `None` (null), never `0`.
- **Deterministic Ordering:** Primary sort by `-rrf_score`, secondary tie-breaking by `chunk_id` ascending.

---

## 10. Data Flow Across Phases

```
PDF upload [IMPLEMENTED]
  → page-level text extraction (page_number preserved) [IMPLEMENTED]
    → paragraph/heading chunking (deterministic ID + section + page_numbers) [IMPLEMENTED]
      → embedding generation & Qdrant indexing [IMPLEMENTED]
        → BM25 keyword indexing & storage synchronization [IMPLEMENTED]
          → semantic vector search & BM25 keyword search [IMPLEMENTED]
            → Reciprocal Rank Fusion (RRF) hybrid retrieval [IMPLEMENTED]
              → cross-encoder reranking [PLANNED - Phase 6]
                → LLM prompt with citations [PLANNED - Phase 7]
                  → frontend answer & highlightable page citations [PLANNED - Phase 7]
```

---

## 11. Testing and Verification — [IMPLEMENTED]

- **Backend Tests:** 77 automated pytest tests covering:
  - Health check & baseline endpoints (7 tests).
  - Document upload, extraction, corruption handling, and security (13 tests).
  - Paragraph/heading chunking, page attribution, and settings validation (18 tests).
  - Embedding generation, dimension validation, Qdrant collection management, indexing idempotency, metadata preservation, empty collection search, and API search/index endpoints (19 tests).
  - BM25 tokenization, exact matching, case folding, numbers/codes, empty corpus, no match, limit, metadata, no duplicates, reindexing replacement, RRF formula validation, missing ranks, candidate limits, tie-breaking, API integration, and startup sync (20 tests).
- **Code Quality:** Ruff linter & formatter for Python; ESLint and TypeScript for Next.js.
- **Browser Verification:** Real end-to-end browser verification of document upload, indexing, hybrid search with RRF scores & badges, semantic search, and document detail inspection.

---

## 12. Open Decisions

| Decision | Notes |
|----------|-------|
| Storage backend for production | Local filesystem & embedded Qdrant used for Phase 5; Qdrant Cloud/Cluster & S3 object storage suitable for cloud production. |
| Reranker model | Cross-encoder (e.g. `bge-reranker-base` or Cohere API) TBD in Phase 6. |
| OCR Support | Tesseract or Cloud OCR can be plugged into `pdf_service.py` if scanned document volume requires it. |
| Multi-tenancy | Document namespace / tenant IDs can be attached to metadata schema when auth is introduced. |
