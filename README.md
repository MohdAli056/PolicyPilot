# PolicyPilot

AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

> **Status:** Phase 2 — PDF Ingestion and Page-Preserving Text Extraction. PDF ingestion, local atomic storage, and page-preserving text extraction are operational. RAG chunking, embeddings, hybrid retrieval, and answer generation will be added in subsequent phases.

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
│   ├── app/            # App Router pages and layouts
│   └── ...
├── backend/            # Python / FastAPI
│   ├── app/            # Application package
│   │   ├── main.py     # FastAPI entry point & CORS configuration
│   │   ├── config.py   # Environment-based settings
│   │   ├── models/     # (reserved)
│   │   ├── schemas/    # Pydantic schemas (DocumentMetadata, DocumentDetail, PageExtraction)
│   │   ├── services/   # Business logic (pdf_service, storage_service)
│   │   └── routers/    # API route modules (health, documents)
│   ├── tests/          # pytest test suite (test_health, test_documents)
│   ├── data/           # Local storage for PDFs and metadata (gitignored)
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
| POST | `/api/documents/upload` | Ingest a PDF document (multipart/form-data) |
| GET | `/api/documents` | List all ingested documents (metadata only) |
| GET | `/api/documents/{document_id}` | Retrieve document details with page extractions |
| GET | `/api/documents/{document_id}/file` | Stream or download original stored PDF file |

#### POST `/api/documents/upload` Example

```bash
curl -X POST -F "file=@leave_policy.pdf" http://localhost:8000/api/documents/upload
```

**Response (201 Created):**
```json
{
  "document_id": "72282962-7bde-404d-a821-f69d398c5c19",
  "filename": "leave_policy.pdf",
  "content_type": "application/pdf",
  "file_size_bytes": 1686,
  "total_pages": 2,
  "total_characters": 580,
  "sha256_hash": "f615b13d23c4bf7e7fea1e58975e4cf3f4820fab1c89436e0992d6f8a4a8c549",
  "ingestion_status": "completed",
  "uploaded_at": "2026-10-27T16:49:16.916715+00:00",
  "pages": [
    {
      "page_number": 1,
      "text": "PolicyPilot Leave Policy: Employees receive 20 days annual leave...\n",
      "character_count": 277
    },
    {
      "page_number": 2,
      "text": "Carry-over guidelines and application deadlines...\n",
      "character_count": 303
    }
  ]
}
```

---

## Storage Architecture

- **Uploaded PDFs**: Stored at `backend/data/documents/{document_id}.pdf` using server-generated UUIDs to prevent user path traversal.
- **Extracted Metadata**: Persisted atomically as JSON at `backend/data/metadata/{document_id}.json`.
- **Git Exclusions**: Both `backend/data/` and all PDF files are excluded in `.gitignore` to prevent committing document data.

---

## Environment Variables

See [`.env.example`](.env.example) for all available configuration. Key variables:

| Variable | Default | Used by | Description |
|----------|---------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000` | Backend | Allowed CORS origins (comma-separated) |
| `DEBUG` | `false` | Backend | Enable verbose debug logging |
| `DOCUMENTS_DIR` | `data/documents` | Backend | Directory for storing raw PDF files |
| `METADATA_DIR` | `data/metadata` | Backend | Directory for storing extraction JSON |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` | Backend | Maximum upload size (10 MB) |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend | Backend API endpoint URL |

---

## Known Limitations

- **OCR Not Supported**: Scanned, image-only PDFs will produce 0 extracted characters. Each page is still preserved with its 1-based page number, and the status reflects `completed (no extractable text found; OCR not supported in Phase 2)`.
- **Password-Protected PDFs**: Encrypted PDFs are rejected with HTTP 400 (`Password-protected or encrypted PDFs are not supported.`).
- **Scope Restriction**: Phase 2 strictly implements document ingestion and text extraction. Chunking, vector embeddings, hybrid search, and question answering will be introduced in subsequent phases.

---

## Documentation

- [Architecture Specification](docs/architecture.md) — system design, implemented vs. planned features, and open decisions.

---

## License

Private — not yet licensed for distribution.
