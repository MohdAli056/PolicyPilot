# PolicyPilot Architecture

> Phase 7 — Grounded LLM Question Answering, Evidence-Based Citations, and Source Traceability. Last updated: 2026-10-29.

This document describes the current architecture of PolicyPilot and outlines where future capabilities will be integrated. Sections are explicitly marked **[IMPLEMENTED]** or **[PLANNED]** to distinguish working code from future design intent.

---

## 1. Overview

PolicyPilot is an AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

---

## 2. System Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Frontend (Next.js 16)                          │  ← User interface (Library, Chunks, 3 Search Modes, Ask QA)
├─────────────────────────────────────────────────────────────────────────┤
│                          Backend  (FastAPI)                             │  ← API, storage & multi-stage retrieval pipeline
├──────────────┬───────────────┬────────────┬────────────┬────────────────┤
│ PDF Ingest   │ Chunking      │ Qdrant     │ BM25 & RRF │ Cross-Encoder  │  ← Ingest, Chunks, Vector, BM25, & Rerank [IMPLEMENTED]
│ & Extraction │ & Metadata    │ Vector     │ Hybrid     │ Reranker       │
├──────────────┴───────────────┴────────────┴────────────┴────────────────┤
│  RAG Orchestration, LLM Grounding & Server-Side Citations [IMPLEMENTED]  │
└─────────────────────────────────────────────────────────────────────────┘
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
| 3-Way Search mode toggle: Reranked, Hybrid (RRF), and Semantic (Vector) | ✅ Implemented |
| Cross-encoder reranked search results with purple `Rerank: ...` and `RRF: ...` badges | ✅ Implemented |
| Hybrid search input & fused results display with green `RRF: ...` scores & ranks | ✅ Implemented |
| Semantic search input & cosine similarity results display | ✅ Implemented |
| Instant clearing of stale results when switching search modes or editing queries | ✅ Implemented |
| Ask PolicyPilot natural language question answering card (`POST /api/ask`) | ✅ Implemented |
| Grounded answer rendering with interactive citation pills linked to sources | ✅ Implemented |
| Abstention notice rendering when policies lack evidence | ✅ Implemented |
| Expandable retrieved evidence sources viewer with page and chunk metadata | ✅ Implemented |
| Document indexing action trigger (`POST /api/documents/{id}/index`) | ✅ Implemented |
| Stream / open original PDF (`GET /api/documents/{id}/file`) | ✅ Implemented |
| Configurable backend URL (`NEXT_PUBLIC_API_URL`) | ✅ Implemented |
| Timeout, empty states, and error handling | ✅ Implemented |

**Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v3.

**Planned additions (later phases):**
- RAG evaluation dashboard and metric tracking (Phase 8).

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
| `POST /api/search/rerank` cross-encoder reranked hybrid retrieval | ✅ Implemented |
| `GET /api/documents/{document_id}/file` original file streaming | ✅ Implemented |
| PyMuPDF page-preserving text extraction | ✅ Implemented |
| Paragraph- and heading-aware chunking engine | ✅ Implemented |
| Multi-page attribution & deterministic UUIDv5 chunk IDs | ✅ Implemented |
| Dedicated embedding provider abstraction (`BaseEmbeddingProvider`) | ✅ Implemented |
| Local ONNX embeddings via FastEmbed (`BAAI/bge-small-en-v1.5`) | ✅ Implemented |
| Qdrant vector database collection management & cosine search | ✅ Implemented |
| BM25 retrieval service with Lucene-adjusted IDF & code tokenization | ✅ Implemented |
| Reciprocal Rank Fusion (RRF) algorithm with deterministic tie-breaking | ✅ Implemented |
| Dedicated cross-encoder reranker abstraction (`BaseRerankerProvider`) | ✅ Implemented |
| Local ONNX cross-encoder reranking via FlashRank (`ms-marco-TinyBERT-L-2-v2`) | ✅ Implemented |
| Deterministic `MockRerankerProvider` for sub-second offline testing | ✅ Implemented |
| Retrieval quality evaluation fixture (MRR, Recall@k, nDCG@k) | ✅ Implemented |
| Keyword index synchronization with disk on startup and upload/index | ✅ Implemented |
| Idempotent reindexing & stale vector/chunk pruning | ✅ Implemented |
| Atomic local storage & JSON metadata/chunks persistence | ✅ Implemented |
| Execution latency breakdown (`retrieval_ms`, `rerank_ms`, `total_ms`) | ✅ Implemented |
| Automated pytest test suite (94 tests, 100% pass) | ✅ Implemented |

**Stack:** Python 3.12, FastAPI, PyMuPDF, Qdrant Client, FastEmbed, FlashRank (ONNX Runtime), rank-bm25, Pydantic, Uvicorn.

**Planned additions (later phases):**
- Grounded LLM answer generation with citation verification (Phase 7).

---

## 5. API Communication — [IMPLEMENTED]

- Frontend → Backend over HTTP REST.
- Base URL configurable via `NEXT_PUBLIC_API_URL` (frontend) / `CORS_ORIGINS` (backend).
- All requests are standard JSON / multipart form data; no auth required in Phase 6.

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
| POST | `/api/search/rerank` | Cross-encoder reranked hybrid retrieval with timing breakdown |
| GET | `/api/documents/{document_id}/file` | Download or stream original PDF |

---

## 6. Complete Retrieval Pipeline — [IMPLEMENTED]

```
Query
  ├── 1. Dense Semantic Retrieval  ──> FastEmbed (BAAI/bge-small-en-v1.5) ──> Qdrant Cosine Search ──┐
  │                                                                                                    ├──> RRF Rank Fusion (k=60)
  └── 2. Sparse Lexical Retrieval  ──> Custom Tokenizer (Codes/Numbers)  ──> Okapi BM25 Index     ──┘         │
                                                                                                               │
                                                                            Top M Candidates ─────────────────┘
                                                                            (e.g., candidate_limit=20)
                                                                                       │
                                                                                       ▼
                                                                     3. Cross-Encoder Reranker
                                                                     FlashRank ONNX (ms-marco-TinyBERT-L-2-v2)
                                                                                       │
                                                                                       ▼
                                                                     4. Deterministic Sort (-score, chunk_id)
                                                                     & Slice to Final Limit K (e.g. 5)
                                                                                       │
                                                                                       ▼
                                                                     Final Ranked Passages + Metadata
                                                                     (rerank_score, rrf_score, ranks, timing)
```

---

## 7. Cross-Encoder Reranking Architecture — [IMPLEMENTED]

Implemented in `backend/app/services/reranker_service.py` and `backend/app/routers/search.py`.

### 7.1 Model Selection & Rationale
- **Selected Model:** `ms-marco-TinyBERT-L-2-v2` via `flashrank` (ONNX Runtime).
- **Alternative Supported Model:** `ms-marco-MiniLM-L-12-v2`.
- **Architectural Justification:**
  - **Inference Efficiency:** Pure ONNX Runtime execution on CPU without requiring PyTorch or CUDA (~2.5GB memory and disk savings).
  - **Lightweight Footprint:** The TinyBERT model weights are approximately 3.2MB (MiniLM-L-12 is 21.6MB), downloading in under 3 seconds on standard connections.
  - **Low Latency:** Typical candidate pool inference (10-20 candidates) executes in ~15-30ms on standard CPUs.
  - **License & Compatibility:** Apache 2.0 open-source license, fully commercial and development friendly.
  - **High Relevance Quality:** Pre-trained specifically on the MS MARCO passage ranking benchmark for deep query-passage cross-attention.

### 7.2 Provider Abstraction
- `BaseRerankerProvider(ABC)`:
  - `model_name -> str`
  - `rerank(query: str, texts: list[str]) -> list[float]`
- `FlashRankRerankerProvider`: Lazily initializes the ONNX `Ranker` with thread safety. Maps scores back 1-to-1 to input candidate order.
- `MockRerankerProvider`: Deterministic lexical overlap, term frequency, and phrase matching with configurable overrides and simulated failure flags (`force_failure=True`) for offline testing without network or GPU.

### 7.3 Candidate Limits & Metadata Preservation
- **Candidate Pool Sizing:** When querying with `limit=K` and `candidate_limit=M`, the candidate pool size is set to `max(K, M)` to guarantee the reranker evaluates the intended pool without being starved by smaller final limits.
- **Tie-Breaking:** Stable sorting using `key=lambda x: (-x.rerank_score, x.chunk_id)`.
- **Preserved Metadata:** Retains `document_id`, `filename`, `chunk_id`, `chunk_index`, `section`, `page_numbers`, `character_count`, `vector_rank`, `keyword_rank`, `rrf_score`, `retrievers`, and adds `rerank_score`, `rerank_rank`, and `timing_ms`.

---

## 8. Retrieval Quality Evaluation & Benchmark — [IMPLEMENTED]

Implemented in `backend/tests/evaluate_retrieval.py`.

### 8.1 Methodology & Fixture
The evaluation fixture consists of an in-memory test corpus with 8 representative policy passages spanning travel policies, annual leave, parental leave, home office stipends, health insurance, and distracting policy clauses.

The fixture tests 5 distinct query categories:
1. **Exact Policy Identifiers:** e.g., `"POL-TRV-2024 hotel reimbursement limit"`
2. **Natural Language Questions:** e.g., `"How many weeks of fully paid leave can a new mother take off?"`
3. **Numeric & Monetary Limits:** e.g., `"What is the maximum daily per diem food allowance on trips?"`
4. **Lexical Misleading Distractors:** e.g., `"Ergonomic chair and home desk setup subsidy"`
5. **Verbatim Mismatch:** e.g., `"When does medical insurance coverage start after joining?"`

### 8.2 Measured Metrics
Metrics calculated across all 5 queries comparing dense vector search, BM25, Hybrid RRF, and Cross-Encoder Reranking:

| Pipeline | Mean MRR | Mean Recall@1 | Mean Recall@3 | Mean nDCG@3 |
|:---------|:---------|:--------------|:--------------|:------------|
| **Vector Only** (FastEmbed `bge-small-en-v1.5`) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |
| **BM25 Only** (Okapi BM25) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |
| **Hybrid RRF** ($k=60$) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |
| **Reranked** (FlashRank `ms-marco-TinyBERT-L-2-v2`) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |

*Note on Evaluation Scope:* Because this evaluation fixture is intentionally compact (8 passages, 5 queries) for rapid deterministic automated testing, all pipelines achieved high top-1 precision on these primary queries. In real end-to-end multi-page testing, the cross-encoder elevated the exact lodging reimbursement clause ($250/night) from Rank #2 to Rank #1 with a score of `0.497258` versus `0.324979`.

---

## 9. Grounded LLM Question Answering & Citation Validation — [IMPLEMENTED]

Implemented in `backend/app/services/llm_service.py`, `backend/app/services/rag_service.py`, and `backend/app/routers/qa.py`.

### 9.1 End-to-End Pipeline
For each question received at `POST /api/ask`:
1. **Candidate Retrieval & Reranking:** Reuses `RerankerService.search_and_rerank` to run hybrid retrieval (FastEmbed vector + Okapi BM25 with RRF fusion) and FlashRank cross-encoder reranking.
2. **Abstention on Empty/Irrelevant Corpus:** If 0 candidates are returned or top candidate rerank scores fall below `rag_abstention_score_threshold` (0.00001), the service immediately abstains without executing an LLM call.
3. **Context Selection & Character Budgeting:** Top $K$ passages (default 5, configurable up to 10) are bounded within `rag_context_char_budget` (default 6,000 characters). Duplicate chunks are removed, and passages are prefixed with stable source tags (`[S1]`, `[S2]`).
4. **LLM Generation:** The context and system instructions are dispatched to the configured LLM provider (`GeminiLLMProvider` using official `google-genai` SDK or `MockLLMProvider` for deterministic testing).
5. **Server-Side Citation Validation:**
   - Raw model responses cite sources using bracketed tags `[S1]`, `[S2]`.
   - The server resolves source tags against the actual retrieved chunks to extract: `document_id`, `filename`, `chunk_id`, 1-based `page_numbers`, `section`, and `excerpt`.
   - Any hallucinated source tags (e.g., `[S99]`) are stripped from the response text and omitted from citations. The LLM is never trusted to fabricate page numbers or document metadata.
   - If the model indicates insufficient evidence or all citations are invalid, the answer is flagged with `is_abstention=True`.

### 9.2 Grounding System Instructions
System instructions enforce:
- Direct, fact-based answers derived strictly from provided evidence.
- No fabricated figures, deadlines, eligibility criteria, or policy clauses.
- Treatment of retrieved policy text as untrusted reference material to prevent prompt injection attempts.
- Explicit abstention when evidence does not establish the answer.

### 9.3 LLM Provider Abstraction
- `BaseLLMProvider(ABC)`:
  - `model_name -> str`
  - `generate(prompt: str, system_instruction: str) -> str`
- `GeminiLLMProvider`: Integrates with the official `google-genai` SDK (`gemini-2.5-flash`), with thread-safe lazy initialization, timeout configuration, and sanitized error masking. Raises `LLMAuthenticationError` (mapped to HTTP 503) if `GEMINI_API_KEY` is not configured.
- `MockLLMProvider`: Deterministic offline provider for unit and regression testing, with flags for simulating timeouts, rate limits, missing authentication, malformed answers, and hallucinated citations.

---

## 10. Testing and Verification — [IMPLEMENTED]

- **Backend Tests:** 111 automated pytest tests covering:
  - Health check & baseline endpoints (7 tests).
  - Document upload, extraction, corruption handling, and security (13 tests).
  - Paragraph/heading chunking, page attribution, and settings validation (18 tests).
  - Embedding generation, Qdrant collection, and semantic search (19 tests).
  - BM25 tokenization, exact matching, RRF formula, and hybrid search (20 tests).
  - Cross-encoder reranker unit & integration tests (17 tests).
  - RAG QA, citation validation, context bounding, prompt injection mitigation, and API error mapping (17 tests).
- **Quality Benchmark:** `evaluate_retrieval.py` confirms 1.0000 MRR and Recall across 4 retrieval stages.
- **End-to-End Verification:** `backend/tests/verify_phase7_end_to_end.py` executes all real retrieval components:
  - Real PDF extraction via PyMuPDF.
  - Real paragraph-aware chunking.
  - Real BGE-small-en embeddings via FastEmbed.
  - Real in-memory Qdrant vectors.
  - Real BM25 inverted index.
  - Real FlashRank cross-encoder reranking.
  - Multi-page citation attribution with 1-based page numbers.
  - Abstention on absent facts.
  - Prompt injection neutralization.
  - 503 response without API key.
  - Backward compatibility of `/api/search`, `/api/search/keyword`, `/api/search/hybrid`, and `/api/search/rerank`.
- **Browser Verification:** Real end-to-end browser verification of document indexing, question submission, answer display with interactive citation badges, evidence inspector, and search mode stability.
- **Code Quality:** Ruff linter & formatter for Python (39 files checked and formatted, 0 errors); ESLint and TypeScript for Next.js (0 errors).

---

## 11. Open Decisions

| Decision | Notes |
|----------|-------|
| Production LLM Configuration | Gemini 2.5 Flash via `google-genai` SDK is implemented. Local LLM alternatives (e.g. Ollama) can implement `BaseLLMProvider`. |
| Cloud Production Storage | Local filesystem & embedded Qdrant used for development; Qdrant Cloud and S3-compatible storage suitable for scaled deployment. |
| Evaluation Framework (Phase 8) | RAG evaluation metrics (Ragas/TruLens style context relevance, groundedness, answer relevance) will be implemented in Phase 8. |
| OCR Support | Tesseract or Cloud OCR can be plugged into `pdf_service.py` if scanned document volume requires it. |

