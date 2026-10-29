# Multi-stage production Dockerfile for PolicyPilot Backend
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final production runtime stage
FROM python:3.12-slim AS runner

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged application user
RUN groupadd -g 1001 policypilot && \
    useradd -u 1001 -g policypilot -m -s /bin/bash policypilot

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/policypilot/.local
ENV PATH=/home/policypilot/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy backend application source
COPY app/ ./app/

# Pre-create data storage directories with correct ownership
RUN mkdir -p /app/data/documents /app/data/metadata /app/data/chunks /app/data/qdrant \
    && chown -R policypilot:policypilot /app

USER policypilot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
