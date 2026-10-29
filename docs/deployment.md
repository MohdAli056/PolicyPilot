# PolicyPilot — Production Deployment & Operational Runbook

This guide describes how to deploy, configure, monitor, and operate **PolicyPilot** in a production or staging environment.

---

## 1. Architecture Overview

PolicyPilot is composed of two primary services:
1. **Backend**: FastAPI (Python 3.12) running under an ASGI server (Uvicorn), orchestrating local embedding generation (FastEmbed), embedded vector storage (Qdrant), in-memory BM25 index synchronization, cross-encoder reranking (FlashRank ONNX), and grounded LLM generation (Google Gemini 2.5 Flash).
2. **Frontend**: Next.js 16 (React 19) web application serving the policy assistant chat and search user interface.

```
+-------------------------------------------------------------+
|                     Next.js Frontend (Port 3000)            |
|                     Client / Browser Interface              |
+------------------------------+------------------------------+
                               | HTTP JSON API
                               v
+-------------------------------------------------------------+
|                     FastAPI Backend (Port 8000)             |
|                                                             |
|  [Liveness: /health]         [Readiness: /health/ready]     |
|                                                             |
|  +-----------------------+     +-------------------------+  |
|  | FastEmbed (Local ONNX)|     | FlashRank (Local ONNX)  |  |
|  | bge-small-en-v1.5     |     | TinyBERT Cross-Encoder  |  |
|  +-----------+-----------+     +------------+------------+  |
|              |                              |               |
|              v                              v               |
|  +-----------------------+     +-------------------------+  |
|  | Qdrant Vector Storage |     | In-Memory BM25 Index    |  |
|  | (Local / Remote)      |     | (Auto-synced at startup)|  |
|  +-----------+-----------+     +------------+------------+  |
|              |                              |               |
|              +--------------+---------------+               |
|                             |                               |
|                             v                               |
|                   [Claim Grounding Verifier]                |
|                             |                               |
|                             v                               |
|                   [Gemini 2.5 Flash LLM]                    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Persistent Storage Volume                  |
|  /app/data/documents   /app/data/metadata                   |
|  /app/data/chunks      /app/data/qdrant                     |
+-------------------------------------------------------------+
```

---

## 2. Deployment Topologies

### Topology A: Docker Compose (Single-Host Containerized)

The simplest production deployment bundles both backend and frontend into isolated containers with a shared named volume:

```bash
# 1. Clone repository
git clone https://github.com/MohdAli056/PolicyPilot.git
cd PolicyPilot

# 2. Configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and appropriate settings

# 3. Build and launch services
docker-compose up -d --build

# 4. Inspect status and logs
docker-compose ps
docker-compose logs -f backend
```

### Topology B: Bare-Metal / Process Supervision (Systemd / PM2)

For environments with native Python 3.12 and Node.js 20 runtimes:

#### Backend:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start with production Uvicorn settings
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 60
```

#### Frontend:
```bash
cd frontend
npm ci
npm run build
npm run start -- -p 3000
```

---

## 3. Persistent Storage and Volume Mounts

PolicyPilot requires persistence for documents, extracted text chunks, and vector index files. Ensure the following paths are backed by persistent storage:

| Path | Purpose | Backup Recommendation |
|------|---------|-----------------------|
| `/app/data/documents` | Uploaded raw PDF policy documents | Daily snapshots |
| `/app/data/metadata` | Document ingestion metadata JSON | Daily snapshots |
| `/app/data/chunks` | Page-extracted text chunks JSON | Daily snapshots |
| `/app/data/qdrant` | Embedded Qdrant collection vectors and payloads | Periodic snapshots |

> **Startup Synchronization**: The backend automatically reads existing document metadata and chunk files on boot to warm up its in-memory BM25 index and ensure consistency across process restarts.

---

## 4. Cold-Start Model Download Behavior

On its initial boot, the backend automatically downloads lightweight ONNX models for inference if they are not already cached:
- **FastEmbed (`BAAI/bge-small-en-v1.5`)**: ~130 MB downloaded into user cache directory.
- **FlashRank (`ms-marco-TinyBERT-L-2-v2`)**: ~20 MB downloaded into cache directory.

**First-Request Latency**: The very first search or upload request may take 2–5 seconds as models initialize and load ONNX runtimes into memory. Subsequent warm requests execute in ~3–100ms.

---

## 5. Health and Operational Monitoring Probes

PolicyPilot implements two dedicated health probes:

### 1. Liveness Probe (`GET /health`)
- **Purpose**: Verifies that the FastAPI process and event loop are responsive.
- **Status Codes**: `200 OK`.
- **Response**:
```json
{
  "status": "healthy",
  "service": "PolicyPilot",
  "version": "0.1.0",
  "timestamp": "2026-10-29T18:30:00Z"
}
```

### 2. Readiness Probe (`GET /health/ready`)
- **Purpose**: Verifies storage directory access, Qdrant vector database collection availability, embedding model readiness, cross-encoder reranker readiness, and LLM configuration state.
- **Status Codes**:
  - `200 OK`: All core subsystems ready for queries and uploads.
  - `503 Service Unavailable`: One or more subsystems are degraded or offline.
- **Response**:
```json
{
  "status": "ready",
  "service": "PolicyPilot",
  "version": "0.1.0",
  "timestamp": "2026-10-29T18:30:00Z",
  "checks": {
    "storage": {
      "status": "ready",
      "documents_dir": "/app/data/documents",
      "chunks_dir": "/app/data/chunks"
    },
    "vector_database": {
      "status": "ready",
      "provider": "qdrant",
      "collection": "policypilot_chunks",
      "dimension": 384
    },
    "embedding_provider": {
      "status": "ready",
      "provider": "fastembed",
      "model": "BAAI/bge-small-en-v1.5",
      "dimension": 384
    },
    "reranker_provider": {
      "status": "ready",
      "provider": "flashrank",
      "model": "ms-marco-TinyBERT-L-2-v2"
    },
    "llm_provider": {
      "status": "configured",
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "note": "Ready for inference"
    }
  }
}
```

---

## 6. Security Hardening Checklist

1. **Non-Root Execution**: Container runs as unprivileged user `policypilot` (UID 1001).
2. **Path Traversal Protection**: Uploaded filenames are sanitized (`re.sub(r"[^\w\.\-\s]", "_", ...)`) and UUID directory structures isolate files.
3. **PDF Magic Bytes Validation**: All uploads are verified to begin with `%PDF-` binary magic bytes.
4. **Decompression Bomb / DoS Limits**: Maximum allowed PDF pages capped at `settings.max_pdf_pages` (default 250 pages).
5. **Payload Bounds**: Query lengths strictly bound between 2 and 500 characters.
6. **Error Sanitization**: 503 and 500 error responses mask internal tracebacks and environment secrets.
7. **CORS Configuration**: Restrict `CORS_ORIGINS` to trusted frontend hostnames (e.g. `https://policy.myorg.internal`).
8. **Credential Safety**: Never log `GEMINI_API_KEY` or expose it in frontend `NEXT_PUBLIC_*` environment variables.

---

## 7. Troubleshooting Guide

| Issue | Cause | Resolution |
|-------|-------|------------|
| `/health/ready` returns 503 with storage error | Missing data directories or read-only volume mount | Ensure `/app/data` is mounted with write permissions for UID 1001. |
| `/api/ask` returns 503 with Gemini error | `GEMINI_API_KEY` is not provided or expired | Export valid `GEMINI_API_KEY` in environment or container. Retrieval endpoints continue to work without a key. |
| Cold-start timeout on first request | Initial ONNX model download over slow network | Allow 30–60s for first boot or pre-populate model cache in container image. |
| CORS block in browser | Frontend URL not included in `CORS_ORIGINS` | Add frontend domain to `CORS_ORIGINS` comma-separated variable. |

---

## 8. Verification Status

- **Configuration & Build Artifacts**: Complete and verified (`backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `docs/deployment.md`).
- **Frontend Production Build**: `npm run build` executed and verified (100% passing).
- **Backend Production Dependencies**: `requirements.txt` validated and verified.
- **Live Container Execution**: Marked as **UNVERIFIED** in local environment due to Docker Desktop background daemon being inactive on host.
