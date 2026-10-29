# PolicyPilot

AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

> **Status:** Phase 5 — BM25 Keyword Search + Hybrid Retrieval with Reciprocal Rank Fusion (RRF). PDF ingestion, page-preserving extraction, paragraph/heading-aware chunking, dense vector embeddings (FastEmbed), vector storage (Qdrant), Okapi BM25 keyword search, and Reciprocal Rank Fusion (RRF) hybrid retrieval with search mode controls are operational. Cross-encoder reranking and LLM answer generation follow in subsequent phases.

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
│   ├── app/            # App Router pages and layouts (Library, Chunks Viewer, Hybrid/Semantic Search)
│   └── ...
├── backend/            # Python / FastAPI
│   ├── app/            # Application package
│   │   ├── main.py     # FastAPI entry point, lifespan startup sync & CORS
│   │   ├── config.py   # Settings (chunking, Qdrant, embeddings, BM25, RRF)
│   │   ├── models/     # (reserved)
│   │   ├── schemas/    # Pydantic schemas (Document, Chunk, Search, HybridSearch, Index)
│   │   ├── services/   # Business logic (pdf, storage, chunking, embedding, vector, indexing, bm25, hybrid)
│   │   └── routers/    # API routes (health, documents, search)
│   ├── tests/          # pytest test suite (77 tests across 5 test modules)
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

# Run automated tests (77 tests)
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
| POST | `/api/documents/upload` | Ingest PDF, extract text page-by-page, chunk, and index into BM25 |
| GET | `/api/documents` | List all ingested documents (metadata only) |
| GET | `/api/documents/{document_id}` | Retrieve document details with raw page extractions |
| GET | `/api/documents/{document_id}/chunks` | Retrieve structured chunks with section and page attribution |
| GET | `/api/documents/{document_id}/file` | Stream or download original stored PDF file |
| POST | `/api/documents/{document_id}/index` | Generate embeddings and index chunks into Qdrant & BM25 |

### Search Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search` | Pure semantic vector search (Phase 4 backward-compatible) |
| POST | `/api/search/keyword` | Exact BM25 keyword retrieval over indexed chunks |
| POST | `/api/search/hybrid` | Hybrid retrieval fusing vector and BM25 results via RRF |

#### POST `/api/search/hybrid` Example

```bash
curl -X POST http://localhost:8000/api/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "POL-2026-X equipment stipend allowance", "limit": 3, "rrf_k": 60}'
```

**Response (200 OK):**
```json
{
  "query": "POL-2026-X equipment stipend allowance",
  "total_results": 3,
  "rrf_k": 60,
  "results": [
    {
      "document_id": "aba18ae1-338b-4819-9a4d-4e7230c31f24",
      "filename": "acme_policy_2026.pdf",
      "chunk_id": "240dc73c-74a4-569c-a111-9f9392e22c94",
      "chunk_index": 0,
      "text": "ACME GLOBAL POLICY MANUAL 2026\n# Section 1: Remote Work Allowance POL-2026-X\nUnder policy identifier POL-2026-X, full-time employees working remotely are entitled to a one-time ergonomic equipment allowance of $750...",
      "section": "ACME GLOBAL POLICY MANUAL 2026",
      "page_numbers": [1, 2],
      "character_count": 515,
      "vector_rank": 1,
      "keyword_rank": 1,
      "rrf_score": 0.032787,
      "retrievers": ["vector", "bm25"]
    }
  ]
}
```

---

## Retrieval Architecture

PolicyPilot Phase 5 provides three distinct retrieval paths:

```
                  ┌──────────────────────┐
                  │    User Query Q      │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌──────────────────────┐           ┌──────────────────────┐
│  Vector Retriever    │           │    BM25 Retriever    │
│  FastEmbed BGE model │           │  Okapi BM25 + Lucene │
│  Qdrant Cosine search│           │  Tokenized chunks    │
└──────────┬───────────┘           └──────────┬───────────┘
           │ (vector_rank)                    │ (keyword_rank)
           └────────────────┬─────────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Reciprocal Rank Fusion    │
              │ RRF = Σ 1 / (k + rank)    │
              │ Tie-breaker: chunk_id     │
              └─────────────┬─────────────┘
                            ▼
                 Top-K Fused Candidates
```

### 1. BM25 Keyword Retrieval

- **Algorithm:** Okapi BM25 with Lucene-adjusted IDF:
  $$\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$$
  $$\text{TF}_{\text{norm}}(t, d) = \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
  This formulation guarantees strictly positive IDF for all corpus terms ($> 0$), avoiding zero or negative scores on small corpora.
- **Parameters:** Default $k_1 = 1.5$, $b = 0.75$ (configurable in `.env`).
- **Tokenization Strategy:**
  1. **Case Folding:** Converts input text to lowercase.
  2. **Policy Identifiers & Numbers:** Regex pattern `[a-zA-Z0-9]+(?:[.\-_][a-zA-Z0-9]+)*` captures exact compound identifiers like `POL-2026-X`, `SEC-101`, `1.1`, and monetary values like `750`.
  3. **Constituent Emission:** Emits sub-tokens for compound tokens so queries for either `POL-2026-X` or `2026` or `POL` match.
  4. **Punctuation Stripping:** Punctuation delimiters ($, %, quotes, parentheses, brackets, colons) are cleanly stripped.
- **Tokenization Limitations:** Optimized for English/alphanumeric policy documents. Non-delimited logographic scripts (e.g. Chinese, Japanese) require specialized morphological segmenters not bundled in this pipeline.
- **Index Lifecycle & Startup Synchronization:**
  - Single source of truth is disk persistence (`data/chunks/{document_id}.json`).
  - At startup, `bm25_service.sync_from_storage()` loads all chunks and rebuilds vocabulary statistics atomically in memory (<50ms).
  - Document uploads and index triggers update the in-memory BM25 index atomically under thread locks.
  - Reindexing a document replaces all prior chunks for that document, preventing stale keyword entries or duplicate hits.

### 2. Reciprocal Rank Fusion (RRF)

- **Formula:**
  $$\text{RRF\_score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$
  Where $k$ is the smoothing constant (default: 60, validated $k > 0$), and $\text{rank}_r(d)$ is the 1-based rank position in retriever $r$.
- **Provenance & Null Ranks:**
  - If a chunk is returned by only one system, the missing rank is returned as `null` (`None`), never `0`.
  - The `retrievers` array records contributing systems: `["vector", "bm25"]`, `["vector"]`, or `["bm25"]`.
- **Tie-Breaking:**
  - Candidates are sorted primarily by `rrf_score` descending.
  - Ties are deterministically broken using `chunk_id` ascending.

---

## Environment Variables

See [`.env.example`](.env.example) for all configuration options:

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
| `BM25_K1` | `1.5` | Backend | BM25 term frequency saturation parameter ($k_1 > 0$) |
| `BM25_B` | `0.75` | Backend | BM25 document length normalization parameter ($0 \le b \le 1$) |
| `RRF_K` | `60` | Backend | Reciprocal Rank Fusion smoothing constant ($k > 0$) |
| `HYBRID_CANDIDATE_LIMIT` | `20` | Backend | Number of candidate chunks retrieved per retriever |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend | Backend API endpoint URL |

---

## Roadmap Progress

- **Phase 1: Project foundation and architecture** — ✅ Complete (released)
- **Phase 2: PDF ingestion and page-preserving extraction** — ✅ Complete (released)
- **Phase 3: Page-aware chunking and metadata** — ✅ Complete (released)
- **Phase 4: Embeddings and vector search** — ✅ Complete (released)
- **Phase 5: BM25 and hybrid retrieval with RRF** — ✅ Complete (current phase: 5/8, 62.5%)
- **Phase 6: Reranking and retrieval tuning** — ⏳ Not started
- **Phase 7: Grounded LLM answers and citations** — ⏳ Not started
- **Phase 8: RAG evaluation, final hardening, and deployment** — ⏳ Not started

---

## Known Limitations

- **OCR Not Supported**: Scanned, image-only PDFs produce 0 extracted characters. Pages are preserved with `text=""` and indexing safely returns 0 indexed chunks.
- **Passage Retrieval Only**: Phase 5 returns retrieved and fused policy passages with provenance ranks. Cross-encoder reranking (Phase 6) and LLM answer generation with citation verification (Phase 7) will follow.
- **Heading Recognition**: Plain-text regex heuristics are used; styling-based headings lacking structural markers fall back to `null`.

---

## Documentation

- [Architecture Specification](docs/architecture.md) — detailed system design, implemented vs. planned features, and open decisions.

---

## License

Private — not yet licensed for distribution.
