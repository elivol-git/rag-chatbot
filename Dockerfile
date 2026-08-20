# ---- stage 1: build the React UI ----
FROM node:20-alpine AS frontend

WORKDIR /frontend
# Copy the manifests first so the dependency layer survives source edits.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---- stage 2: runtime ----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY --from=frontend /frontend/dist ./frontend/dist

# The corpus and the index are bind-mounted at run time; create the mount
# points here so a container started without them still boots and reports
# its state through /api/health.
RUN useradd --create-home --uid 1000 rag \
    && mkdir -p data/documents data/vector_store \
    && chown -R rag:rag /app
USER rag

EXPOSE 5000

# One worker keeps the vector store loaded once (it is ~21MB of float32 and
# every worker would hold its own copy); threads let the SSE stream in
# /api/ask/stream run without blocking health checks. Generation on CPU takes
# over a minute for a long Hebrew answer, so the default 30s timeout would
# kill answers mid-token.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--worker-class", "gthread", \
     "--threads", "8", \
     "--timeout", "600", \
     "--access-logfile", "-", \
     "src.app:app"]
