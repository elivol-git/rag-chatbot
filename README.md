# Architecture RAG Chatbot

A fully local Retrieval-Augmented Generation system over a corpus of architecture
texts (movements, architects, building elements, design theory). Every stage —
chunking, embedding, indexing, similarity search, prompt assembly — is written by
hand in plain Python. No LangChain, no managed pipeline, no cloud API keys.

```
data/documents/  ──ingest──►  chunks ──embed──►  vector store (NumPy)
                                                        │
        React UI ──► Flask /api/ask ──► retrieve() ─────┤
                                             │          │
                            MCP client ──► search_knowledge_base
                                             │
                                    prompt + context ──► Ollama (llama3.1) ──► grounded answer
```

## Stack

| Layer | Choice |
|---|---|
| LLM | `llama3.1` via local Ollama |
| Embeddings | `bge-m3` via Ollama (1024-dim, multilingual — the corpus is Hebrew and English) |
| Vector store | Hand-rolled: `embeddings.npy` (L2-normalized float32) + `chunks.json` + `manifest.json`, brute-force cosine via dot product |
| API | Flask (`/api/health`, `/api/ask`, `/api/ingest`, `/api/sources`) |
| Tool interface | MCP stdio server (`search_knowledge_base`, `knowledge_base_stats`) |
| UI | React + Vite |

Retrieval exists in exactly one place — `src/retrieval.py`. The Flask route and
the MCP tool both call `retrieve()`; neither re-implements search.

## Setup

Prerequisites: Python 3.12, Node 18+, [Ollama](https://ollama.com) running.

```bash
ollama pull llama3.1
ollama pull bge-m3

python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

cp .env.example .env            # defaults work as-is

python scripts/fetch_corpus.py  # seed data/documents (28 public-domain / CC BY-SA texts)
python -m src.ingest            # chunk + embed + index

cd frontend && npm install && npm run build && cd ..
```

## Running

```bash
python -m src.app               # http://127.0.0.1:5000  (serves the built UI)
```

Frontend development with hot reload (Flask must also be running):

```bash
cd frontend && npm run dev      # http://localhost:5173, proxies /api to Flask
```

## Ingestion

```bash
python -m src.ingest            # incremental: only changed files are re-embedded
python -m src.ingest --rebuild  # discard the index and start clean
```

Or without restarting the server: `POST /api/ingest` (the "Re-ingest documents"
button in the UI).

Incremental logic lives in `manifest.json`: each source file is keyed by sha256.
Unchanged files are skipped, changed files have their old chunks deleted before
re-embedding, and chunks of deleted files are pruned.

Adding your own material: drop `.md`, `.txt`, `.pdf`, `.docx`, or `.pptx` files
into `data/documents/` and re-ingest. Word tables and slide text are extracted;
slides keep a `[Slide n]` marker so retrieved context stays locatable. Markdown
and text files may carry a header block (`title:`, `source_url:`, `license:`,
then `---`) which becomes chunk metadata. Files that cannot be parsed are
reported and skipped rather than aborting the run.

## Corpora

| Folder | Contents |
|---|---|
| `data/documents/` | 28 public-domain / CC BY-SA texts: Vitruvius, Ruskin, Sullivan, and Wikipedia articles on movements, architects and building elements |
| `data/documents/michlala/` | 54 lecturer-authored course documents (Hebrew and English): building materials, green and desert architecture, climate, Israeli architecture |

The course documents are imported by `scripts/import_course_docs.py`, which
converts each source file to plain text and is deliberately conservative about
what it accepts:

```bash
python scripts/import_course_docs.py --dry-run   # report what would be imported
python scripts/import_course_docs.py             # write the text files
```

It skips exams, quizzes, question banks and answer keys (so the bot cannot serve
assessment material), and skips any file whose text contains an ID number or a
submission header, because those are student work and must not become
retrievable. Only extracted text is stored — the original decks run to hundreds
of megabytes of images and carry Office author metadata, and none of that is
needed to answer questions.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint — keep the literal IP, see troubleshooting |
| `LLM_MODEL` | `llama3.1` | Answer generation model |
| `EMBED_MODEL` | `bge-m3` | Embedding model (1024-dim, multilingual) |
| `DOCUMENTS_DIR` | `./data/documents` | Source documents |
| `VECTOR_STORE_DIR` | `./data/vector_store` | Index location |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Characters carried across chunk boundaries |
| `TOP_K` | `4` | Chunks retrieved per query |
| `MIN_SCORE` | `0.58` | Cosine floor; below it the bot refuses instead of guessing |
| `FLASK_PORT` | `5000` | API port |

## API

```bash
curl http://127.0.0.1:5000/api/health

curl -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does a flying buttress work?"}'
```

`/api/ask` returns the answer plus every chunk used, with its source file and
cosine score, and per-stage latency:

```json
{
  "answer": "...[1][2]",
  "grounded": true,
  "chunks": [{"text": "...", "source": "wiki-flying-buttress.md", "score": 0.71, "title": "Flying buttress"}],
  "retrieval_ms": 41, "llm_ms": 3120, "total_ms": 3161
}
```

`{"question": "...", "top_k": 8, "source": "wiki-bauhaus.md"}` overrides the
defaults and restricts retrieval to one document.

`POST /api/ask/stream` runs the same pipeline as Server-Sent Events — this is
what the UI uses. Retrieval takes milliseconds while an 8B model needs tens of
seconds, so the retrieved context arrives first and the answer follows token by
token:

```
event: meta    data: {"chunks": [...], "grounded": true, "retrieval_ms": 122}
event: token   data: {"text": "Brutalist"}
event: token   data: {"text": " architecture"}
event: done    data: {"llm_ms": 18422, "total_ms": 18544}
```

```bash
curl -N -X POST http://127.0.0.1:5000/api/ask/stream \
  -H "Content-Type: application/json" -d '{"question": "What is a passive house?"}'
```

## MCP server

```bash
python -m src.mcp_server        # stdio transport
```

Register with Claude Code:

```bash
claude mcp add arch-kb -- C:/projects/RAG_chatbot/.venv/Scripts/python.exe -m src.mcp_server
```

Tools:
- `search_knowledge_base(query, top_k=4, source="")` — returns ranked passages with scores
- `knowledge_base_stats()` — chunk/document counts, embedding dimension, source list

## Tests

```bash
pytest                          # unit tests (no Ollama needed)
pytest -m integration           # live retrieval checks; auto-skip if the index or Ollama is missing
```

Unit tests cover chunk packing/overlap/no-content-loss, vector store
save→load→search roundtrip, source filtering, dimension-mismatch guards, and
prompt assembly. Integration tests assert descending scores, that a Bauhaus
question retrieves the Bauhaus document, and that an off-topic question returns
nothing above the similarity floor.

## Validation questions

| Question | Expected |
|---|---|
| What are Vitruvius' three principles of good architecture? | firmitas, utilitas, venustas — cited from the Vitruvius text |
| What defines Brutalist architecture? | exposed béton brut, massing, 1950s–70s — cited from the Brutalism document |
| Who founded the Bauhaus and what did it teach? | Walter Gropius, unity of craft and art |
| How does a flying buttress work? | transfers vault thrust outward to a pier |
| How do I configure a Kubernetes ingress controller? | *"I don't have that in my knowledge base."* with zero chunks — the grounding control |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Query vector is N-dim but the store holds M-dim` | `EMBED_MODEL` changed after ingest. Delete `data/vector_store/` and re-ingest. |
| `Embedding request … failed` | Ollama not running, or `nomic-embed-text` not pulled. |
| `/api/health` shows `degraded` | Empty index (run ingestion) or Ollama unreachable. |
| Every answer is "I don't have that in my knowledge base." | `MIN_SCORE` too high for your corpus, or ingestion never ran. |
| Answers are vague / miss the point | Raise `TOP_K`, or raise `CHUNK_SIZE` so passages carry more context. |
| First question is slow | Ollama is loading `llama3.1` into memory; later calls are much faster. |
| Every Ollama call takes ~2s longer than it should | `OLLAMA_HOST` is set to `localhost`. On Windows that resolves to `::1` first, and Ollama binds IPv4 only, so every request pays a connect-failure timeout. Use `http://127.0.0.1:11434` — measured 2.17s → 0.08s per query embedding. |
| Retrieval returns nothing (or nonsense) right after a process starts | Ollama's first embed call in a fresh process can return an all-zero vector, which makes every cosine score 0. `src/embeddings.py` detects zero vectors, retries, and raises `EmbeddingError` rather than storing or querying with them. |
| `UnicodeEncodeError` when ingesting or evaluating | A non-UTF-8 Windows console. Both entry points call `sys.stdout.reconfigure(encoding="utf-8")`; if you wrap them in your own script, do the same. |
| `frontend not built` from `/` | Run `npm run build` in `frontend/`, or use the Vite dev server. |
| Wikipedia fetch returns 429 | The corpus script backs off and retries; re-run it to pick up missing files. |

## Design notes

**Why normalize embeddings at write time.** Both stored vectors and query vectors
are L2-normalized in `src/embeddings.py`, so cosine similarity reduces to
`embeddings @ query` — one matrix multiply, no per-query normalization, and no
scale bugs.

**Why paragraph-aware chunking.** `split_text()` packs whole paragraphs up to
`CHUNK_SIZE` rather than slicing at fixed offsets, so a chunk rarely cuts a
sentence in half. Oversized paragraphs fall back to sentence packing. The
`CHUNK_OVERLAP` tail keeps facts that straddle a boundary retrievable.

**Why a similarity floor.** Without `MIN_SCORE`, an off-topic question still
retrieves the four least-bad chunks and the LLM is tempted to answer from them.
The floor makes the refusal path explicit and skips the LLM call entirely.

**Why brute-force search.** At a few thousand chunks a NumPy dot product costs
well under a millisecond — an ANN index would add a dependency and hide the step
this project exists to show.
