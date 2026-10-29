# PolicyPilot

AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

> **Status:** Phase 6 — Cross-Encoder Reranking and Retrieval Quality Tuning. PDF ingestion, page-preserving extraction, paragraph/heading-aware chunking, dense vector embeddings (FastEmbed), vector storage (Qdrant), Okapi BM25 keyword search, Reciprocal Rank Fusion (RRF) hybrid retrieval, and CPU-efficient Cross-Encoder Reranking (`FlashRank` ONNX) are operational with 3-mode frontend controls. LLM answer generation follows in Phase 7.

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
│   ├── app/            # App Router pages and layouts (Library, Chunks Viewer, 3 Search Modes)
│   └── ...
├── backend/            # Python / FastAPI
│   ├── app/            # Application package
│   │   ├── main.py     # FastAPI entry point, lifespan startup sync & CORS
│   │   ├── config.py   # Settings (chunking, Qdrant, embeddings, BM25, RRF, reranker)
│   │   ├── models/     # (reserved)
│   │   ├── schemas/    # Pydantic schemas (Document, Chunk, Search, HybridSearch, RerankSearch, Index)
│   │   ├── services/   # Business logic (pdf, storage, chunking, embedding, vector, indexing, bm25, hybrid, reranker)
│   │   └── routers/    # API routes (health, documents, search)
│   ├── tests/          # pytest test suite & retrieval evaluation (94 tests across 6 modules)
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

# Run automated tests (94 tests)
.venv\Scripts\pytest.exe -v

# Run retrieval quality evaluation fixture
.venv\Scripts\python.exe tests/evaluate_retrieval.py

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
| POST | `/api/search/rerank` | Cross-encoder reranked hybrid retrieval with timing breakdown |

#### POST `/api/search/rerank` Example

```bash
curl -X POST http://localhost:8000/api/search/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "hotel lodging reimbursement rate per night", "limit": 3, "candidate_limit": 10}'
```

**Response (200 OK):**
```json
{
  "query": "hotel lodging reimbursement rate per night",
  "total_results": 3,
  "reranker_model": "ms-marco-TinyBERT-L-2-v2",
  "candidate_limit": 10,
  "results": [
    {
      "document_id": "58c00d2f-d8df-4bb0-a483-f1556354c795",
      "filename": "corporate_policy_manual_2024.pdf",
      "chunk_id": "7a36a5dd-356d-5048-b914-20e3e1b7070e",
      "chunk_index": 0,
      "text": "Section 2.0: Business Travel & Lodging (Policy POL-TRV-2024)\nDomestic business travel lodging is reimbursed up to $250 per night...",
      "section": "Section 2.0: Business Travel & Lodging",
      "page_numbers": [2],
      "character_count": 284,
      "vector_rank": 2,
      "keyword_rank": 1,
      "rrf_score": 0.032522,
      "retrievers": ["vector", "bm25"],
      "rerank_score": 0.497258,
      "rerank_rank": 1
    }
  ],
  "timing_ms": {
    "retrieval_ms": 7.18,
    "rerank_ms": 214.76,
    "total_ms": 221.94
  }
}
```

---

## Retrieval Pipeline Architecture

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
              Top M Candidates (candidate_limit)
                            │
                            ▼
              ┌───────────────────────────┐
              │ Cross-Encoder Reranker    │
              │ FlashRank ONNX TinyBERT   │
              │ Deep (query, chunk) cross │
              │ attention relevance score │
              └─────────────┬─────────────┘
                            ▼
              Sort: (-rerank_score, chunk_id)
              Slice: Top K Final Results
```

### Cross-Encoder Model Selection & Trade-Offs

- **Model:** `ms-marco-TinyBERT-L-2-v2` via `flashrank` (ONNX Runtime).
- **Alternative:** `ms-marco-MiniLM-L-12-v2` (configured via `RERANKER_MODEL`).
- **Why FlashRank ONNX?**
  - **Zero Heavy Torch/CUDA Dependencies:** Standard sentence-transformers cross-encoders require PyTorch and CUDA libraries (~2.5GB). FlashRank leverages the pre-installed ONNX Runtime, preserving a lightweight deployment footprint.
  - **Small Download Size:** The default TinyBERT model is ~3.2MB (MiniLM-L-12 is ~21.6MB), downloading in under 3 seconds.
  - **Low Inference Latency:** Evaluates candidate sets (10-20 passages) in ~15-30ms on standard CPUs.
  - **Commercial Licensing:** Apache 2.0 open-source license.
  - **Offline Testing:** Supported via `MockRerankerProvider` for sub-second offline pytest execution without model downloads or network dependencies.

### Retrieval Quality Benchmark Results

Evaluated using `backend/tests/evaluate_retrieval.py` over representative corporate policy queries across 5 categories:

| Pipeline | Mean MRR | Mean Recall@1 | Mean Recall@3 | Mean nDCG@3 |
|:---------|:---------|:--------------|:--------------|:------------|
| **Vector Only** (FastEmbed `bge-small-en-v1.5`) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |
| **BM25 Only** (Okapi BM25) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |
| **Hybrid RRF** ($k=60$) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |
| **Reranked** (FlashRank `ms-marco-TinyBERT-L-2-v2`) | 1.0000 | 1.0000 | 1.0000 | 0.9669 |

*Observation:* In real end-to-end multi-page testing, the cross-encoder elevated the exact lodging reimbursement clause ($250/night) from Rank #2 in vector search to Rank #1 with a score of `0.497258` versus `0.324979`.

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
| `RERANKER_PROVIDER` | `flashrank` | Backend | Cross-encoder reranker provider (`flashrank` or `mock`) |
| `RERANKER_MODEL` | `ms-marco-TinyBERT-L-2-v2` | Backend | Pinned cross-encoder model identifier |
| `RERANK_CANDIDATE_LIMIT` | `20` | Backend | Top RRF candidate count evaluated by reranker |
| `RERANK_BATCH_SIZE` | `16` | Backend | Batch size for cross-encoder inference |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend | Backend API endpoint URL |

---

## Roadmap Progress

- **Phase 1: Project foundation and architecture** — ✅ Complete (released)
- **Phase 2: PDF ingestion and page-preserving extraction** — ✅ Complete (released)
- **Phase 3: Page-aware chunking and metadata** — ✅ Complete (released)
- **Phase 4: Embeddings and vector search** — ✅ Complete (released)
- **Phase 5: BM25 and hybrid retrieval with RRF** — ✅ Complete (released)
- **Phase 6: Reranking and retrieval tuning** — ✅ Complete (current phase: 6/8, 75%)
- **Phase 7: Grounded LLM answers and citations** — ⏳ Not started
- **Phase 8: RAG evaluation, final hardening, and deployment** — ⏳ Not started

---

## Known Limitations

- **OCR Not Supported**: Scanned, image-only PDFs produce 0 extracted characters. Pages are preserved with `text=""` and indexing safely returns 0 indexed chunks.
- **Passage Retrieval Only**: Phase 6 returns retrieved and reranked policy passages with cross-encoder relevance scores and provenance. Natural-language LLM answer generation with citation verification follows in Phase 7.
- **Heading Recognition**: Plain-text regex heuristics are used; styling-based headings lacking structural markers fall back to `null`.

---

## Documentation

- [Architecture Specification](docs/architecture.md) — detailed system design, implemented vs. planned features, and open decisions.

---

## License

Private — not yet licensed for distribution.
