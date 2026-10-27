# PolicyPilot

AI-powered company policy assistant. Employees ask natural-language questions about internal policies (leave, insurance, travel, reimbursements) and receive evidence-grounded answers with page-level citations.

> **Status:** Phase 1 — Foundation. The backend health endpoint and frontend shell are operational. RAG capabilities will be added in subsequent phases.

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
│   │   ├── main.py     # FastAPI entry point
│   │   ├── config.py   # Environment-based settings
│   │   └── routers/    # API route modules
│   ├── tests/          # pytest test suite
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
git clone <repo-url>
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

# Run tests
.venv\Scripts\pytest.exe

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

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |

### Example: Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "service": "PolicyPilot",
  "version": "0.1.0",
  "timestamp": "2026-10-27T16:00:00+00:00"
}
```

---

## Environment Variables

See [`.env.example`](.env.example) for all available configuration. Key variables:

| Variable | Default | Used by |
|----------|---------|---------|
| `CORS_ORIGINS` | `http://localhost:3000` | Backend |
| `DEBUG` | `false` | Backend |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend |

---

## Documentation

- [Architecture](docs/architecture.md) — system design, implemented vs. planned features, and open decisions.

---

## License

Private — not yet licensed for distribution.
