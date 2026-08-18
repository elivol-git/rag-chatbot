"""Flask API for the architecture RAG chatbot.

Routes live under /api so the React dev server (Vite proxy) and the production
build (served from frontend/dist) hit identical paths.
"""

from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .config import PROJECT_ROOT, settings
from .embeddings import EmbeddingError
from .ingest import ingest
from .llm import LLMError, chat, ollama_reachable
from .prompts import NO_CONTEXT_ANSWER, build_prompt
from .retrieval import reload_store, retrieve, store_stats
from .vector_store import DimensionMismatch

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.get("/api/health")
def health():
    try:
        stats = store_stats()
        store_ok = True
        error = None
    except Exception as exc:  # noqa: BLE001 - health must never raise
        stats = {"chunks": 0, "dimension": 0, "documents": 0, "sources": []}
        store_ok = False
        error = str(exc)

    reachable = ollama_reachable()
    return jsonify(
        {
            "status": "ok" if (store_ok and reachable and stats["chunks"]) else "degraded",
            "ollama_reachable": reachable,
            "vector_store_ok": store_ok,
            "chunks": stats["chunks"],
            "documents": stats["documents"],
            "dimension": stats["dimension"],
            "embed_model": settings.embed_model,
            "llm_model": settings.llm_model,
            "top_k": settings.top_k,
            "min_score": settings.min_score,
            "error": error,
        }
    )


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    top_k = payload.get("top_k")
    source_filter = payload.get("source") or None
    started = time.time()

    try:
        chunks = retrieve(question, top_k=top_k, source_filter=source_filter)
    except (EmbeddingError, DimensionMismatch) as exc:
        return jsonify({"error": str(exc)}), 503

    retrieval_ms = int((time.time() - started) * 1000)

    # No chunk cleared the similarity floor: refuse without spending an LLM call.
    if not chunks:
        return jsonify(
            {
                "question": question,
                "answer": NO_CONTEXT_ANSWER,
                "chunks": [],
                "grounded": False,
                "retrieval_ms": retrieval_ms,
                "llm_ms": 0,
                "total_ms": retrieval_ms,
            }
        )

    llm_started = time.time()
    try:
        answer = chat(build_prompt(question, chunks))
    except LLMError as exc:
        return jsonify({"error": str(exc)}), 503
    llm_ms = int((time.time() - llm_started) * 1000)

    return jsonify(
        {
            "question": question,
            "answer": answer,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "grounded": True,
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "total_ms": retrieval_ms + llm_ms,
        }
    )


@app.post("/api/ingest")
def run_ingest():
    payload = request.get_json(silent=True) or {}
    try:
        summary = ingest(rebuild=bool(payload.get("rebuild")), verbose=False)
    except (EmbeddingError, DimensionMismatch) as exc:
        return jsonify({"error": str(exc)}), 503
    reload_store()
    return jsonify(summary)


@app.get("/api/sources")
def sources():
    return jsonify({"sources": store_stats()["sources"]})


# ---- production: serve the built React app ----


@app.get("/")
@app.get("/<path:asset>")
def frontend(asset: str = "index.html"):
    if not FRONTEND_DIST.exists():
        return (
            jsonify(
                {
                    "error": "frontend not built",
                    "hint": "run 'npm install && npm run dev' in frontend/ "
                    "(dev) or 'npm run build' (prod)",
                }
            ),
            404,
        )
    target = Path(asset)
    if not (FRONTEND_DIST / target).is_file():
        target = Path("index.html")  # SPA fallback
    return send_from_directory(FRONTEND_DIST, str(target))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=settings.flask_port, debug=True)
