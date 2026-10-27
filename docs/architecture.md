# PolicyPilot Architecture

> Phase 1 — Foundation. Last updated: 2026-10-27.

This document describes the current architecture of PolicyPilot and outlines where future capabilities will be integrated. Sections are marked **[IMPLEMENTED]** or **[PLANNED]** to distinguish working code from design intent.

---

## 1. Overview

PolicyPilot is an AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

---

## 2. System Layers

```
┌─────────────────────────────────────┐
│           Frontend (Next.js)        │  ← User interface
├─────────────────────────────────────┤
│           Backend  (FastAPI)        │  ← API, orchestration
├──────┬──────┬───────┬───────────────┤
│ PDF  │ Vec  │ BM25  │ LLM / Rerank │  ← RAG components (future)
│ Ing  │ DB   │       │              │
└──────┴──────┴───────┴───────────────┘
```

---

## 3. Frontend Responsibilities — [IMPLEMENTED]

| Concern | Status |
|---------|--------|
| Landing / dashboard shell | ✅ Implemented |
| Backend health-status display | ✅ Implemented |
| Loading / healthy / unavailable states | ✅ Implemented |
| Configurable backend URL (env var) | ✅ Implemented |
| Timeout and error handling | ✅ Implemented |

**Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v3.

**Planned additions (later phases):**
- Policy question input and answer display.
- Citation viewer with page-level links.
- Document library / upload UI.
- Search diagnostics panel.
- Evaluation dashboard.

---

## 4. Backend Responsibilities — [IMPLEMENTED]

| Concern | Status |
|---------|--------|
| FastAPI application entry point | ✅ Implemented |
| `GET /health` endpoint | ✅ Implemented |
| Pydantic response models | ✅ Implemented |
| CORS configuration (env-driven) | ✅ Implemented |
| Automated tests (pytest) | ✅ Implemented |

**Stack:** Python 3.12, FastAPI, Pydantic, Uvicorn.

**Planned additions (later phases):**
- PDF upload and ingestion endpoints.
- Policy question / answer endpoint.
- Document library endpoints.
- Search diagnostics endpoints.

---

## 5. API Communication — [IMPLEMENTED]

- Frontend → Backend over HTTP REST.
- Base URL configurable via `NEXT_PUBLIC_API_URL` (frontend) / `CORS_ORIGINS` (backend).
- All requests are standard JSON; no auth required in Phase 1.

### Current Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service liveness check |

---

## 6. PDF Ingestion — [PLANNED]

- **Library:** PyMuPDF (`fitz`).
- **Flow:** Upload PDF → extract text per page → store raw text with page numbers.
- **Key invariant:** Page numbers are preserved from extraction through to citation display.
- Pages will be stored with a document ID, page number, and raw text.

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
PDF upload
  → page-level text extraction (page_number preserved)
    → section-aware chunking (document_id + page_number + section)
      → embedding storage (metadata attached to vectors)
        → retrieval results (metadata returned with chunks)
          → LLM prompt (chunks include page refs)
            → answer response (citations with page numbers)
              → frontend display (clickable page links)
```

---

## 10. Testing and Evaluation — [PLANNED]

### 10.1 Unit and Integration Tests
- Backend: pytest (partially implemented for `/health`).
- Frontend: TBD (likely Vitest or Jest with React Testing Library).

### 10.2 RAG Evaluation
- Benchmark dataset of question–answer pairs with expected citations.
- Metrics: retrieval recall, answer faithfulness, citation accuracy.
- Evaluation harness runnable from CLI.

---

## 11. Open Decisions

| Decision | Notes |
|----------|-------|
| Storage backend for documents/chunks | Qdrant for vectors; text/metadata storage TBD (filesystem, SQLite, or Postgres). |
| Embedding model selection | Depends on cost, latency, and quality benchmarking. |
| Reranker model | Cross-encoder vs. API-based reranker TBD. |
| Auth / multi-tenancy | Not in scope for early phases. |
| Deployment platform | Not decided; Docker Compose likely for development. |
| Frontend state management | React state sufficient for now; may add Zustand if complexity grows. |
| Chunk size / overlap | Needs empirical tuning during RAG phase. |
