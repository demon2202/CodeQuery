# CodeQuery — Local RAG over GitHub Repos

> Ask natural-language questions about any public GitHub repo, get grounded answers with exact file paths and line numbers. Everything runs locally — zero paid APIs, zero API keys.

**How it works:** Paste a repo URL → CodeQuery clones it, parses the AST, embeds every code chunk, and stores them in a local vector DB. Then ask questions like "where is auth handled?" and get answers that cite `src/auth.py:42-67` with the actual code inline.

---

## Architecture Decisions

### Why tree-sitter for chunking, not line-splitting?

Naive line-splitting breaks functions mid-way. Neither half is useful for retrieval. Tree-sitter parses code into an AST and uses function/class/method boundaries as chunk edges, so every chunk is a complete semantic unit. Fallback sliding-window chunks are marked `chunk_type="fallback"` — we don't pretend they're as good.

### Why qwen2.5-coder:7b?

- **vs CodeLlama:** Trained on 5.5T tokens of code (vs 1T). Better instruction-following for "cite file:line" instructions.
- **vs 14B+:** 14B generates ~1.5 tok/s — 30s wait per answer. 7B does 3-5 tok/s — fast enough for chat.
- **vs 3B:** Too small to reliably follow citation instructions. Hallucinates more.

### Why ChromaDB (not FAISS)?

Built-in metadata filtering (query by language, type, file path) without building a parallel index. Auto-persistence. HNSW backend with same algorithm as FAISS HNSW. The 20% slower bulk insert is a one-time cost.

### Why SQLite snapshots instead of full git clones?

After indexing, the full git clone (including `.git` folder) is replaced with a lightweight SQLite snapshot. This saves **80-95% disk space** — the `.git` folder alone is often 50-80% of a repo's size and is completely unnecessary after indexing. The snapshot contains:
- **File contents** in SQLite (fast single-query lookups for file viewer and search enrichment)
- **File tree** as JSON (instant load for file tree sidebar)
- **Repo stats** as JSON (pre-computed, no on-demand recomputation)

All features (file tree, code viewer, search enrichment, repo summary) work identically from the snapshot. If no snapshot exists (backward compatibility), endpoints fall back to reading from the filesystem.

### Why k=12 (not 3 or 20)?

"Where is auth handled?" spans middleware, routes, models — 4-6 chunks. k=3 misses cross-file answers. k=20 adds noise. k=12 balances coverage and precision. Extra chunks don't hurt — the prompt tells the LLM to only cite what's relevant.

### Why streaming tokens?

Buffering a 200-token answer takes 40-60s. Streaming the first token in ~2s and appending makes the UX feel 10x faster. The single biggest perceived-speed win for chat UIs.

### Why embed batch size = 64?

Ollama's `/api/embed` endpoint processes multiple texts per request. Batch size 64 means fewer HTTP round trips — for 404 chunks, that's ~7 requests instead of ~26 with batch size 16. The GPU processes batches efficiently; the bottleneck is HTTP overhead, not GPU compute.

---

## Setup

### Prerequisites

1. **Python 3.10+**
2. **Node.js 18+**
3. **Git** (must be in PATH — test with `git --version`)
4. **Ollama** — [install](https://ollama.ai), then:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama pull nomic-embed-text
   ollama serve   # Keep this running in a separate terminal
   ```

### Backend

**Linux/macOS:**
```bash
cd codequery/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**
```powershell
cd codequery\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or just run `start.bat` on Windows / `start.sh` on Linux — they do all of the above.

You should see:
```
INFO:     CodeQuery starting up...
INFO:     Embedding model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**⚠️ If you see "Failed to warm up Ollama model"** — that's OK if Ollama isn't running yet. Indexing will still work. Chat requires Ollama.

### Frontend

```bash
cd codequery/frontend
npm install
npm run dev
```

Open http://localhost:5173

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CQ_DATA_DIR` | `./data` | Where snapshots and ChromaDB data are stored |
| `CQ_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `CQ_OLLAMA_MODEL` | `qwen2.5-coder:7b` | Model for generation |
| `CQ_EMBEDDING_PROVIDER` | `ollama` | Embedding provider: `ollama` or `sentence-transformers` |
| `CQ_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CQ_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers fallback model |
| `CQ_RETRIEVAL_TOP_K` | `12` | Number of chunks to retrieve |
| `CQ_RETRIEVAL_MIN_SCORE` | `0.10` | Minimum similarity score |
| `CQ_HNSW_M` | `16` | ChromaDB HNSW M parameter |
| `CQ_HNSW_EF_CONSTRUCTION` | `100` | ChromaDB HNSW construction EF |
| `CQ_HNSW_EF_SEARCH` | `50` | ChromaDB HNSW search EF |
| `CQ_MAX_CHUNK_CHARS` | `2000` | Maximum chunk size in characters |
| `CQ_MAX_FILE_SIZE` | `500000` | Maximum file size in bytes |
| `CQ_EMBEDDING_BATCH_SIZE` | `64` | Embedding batch size |

**Ollama tip:** Set `OLLAMA_KEEP_ALIVE=30m` to keep the model loaded between requests.

---

## Data Directory Structure

```
data/
├── snapshots/          # Lightweight SQLite snapshots (replaces full git clones)
│   └── owner_repo/
│       ├── files.db    # SQLite: file paths, contents, languages
│       ├── tree.json   # Pre-computed file tree
│       └── stats.json  # Pre-computed repo stats
├── chromadb/           # Vector database
│   ├── chroma.sqlite3  # Main ChromaDB storage
│   └── meta_*.json     # Per-repo commit hashes
└── embed_cache.json    # Content-hash embedding cache
```

After indexing, the cloned repo directory is **deleted** — only the snapshot remains. This saves 80-95% disk space compared to keeping the full `.git` folder.

---

## Performance

Real numbers, not aspirational claims. With `nomic-embed-text` on GPU (RTX 3050 4GB):

| Repo | Files | Chunks | Clone | Parse | Embed+Store | Total |
|------|-------|--------|-------|-------|-------------|-------|
| demon2202/GreenRoute | 44 | 404 | ~2s | <1s | ~15s | ~18s |
| pallets/click | 137 | 1,617 | ~2s | ~2s | ~50s | ~55s |

Key speed factors:
- **Embed batch size 64** (was 16) — 4x fewer HTTP round trips to Ollama
- **Embedding cache** — unchanged chunks skip re-embedding on re-index
- **Pre-computed tree/stats** — file tree and summary load instantly, no on-demand filesystem walks
- **SQLite snapshots** — file content lookups are single SQL queries (~1ms)

**Retrieval:** ~0.2s per query. **First token:** ~2-4s warm, ~12-15s cold start.

---

## Limitations

### 1. Cross-file reasoning is limited

The LLM sees top-12 chunks. A call chain across 5 files (A→B→C→D→E) will likely retrieve A and E but miss B, C, D. RAG is fundamentally limited to retrieved context — call graph analysis would help but isn't implemented.

### 2. Large repos are slow to index

10K+ files = 5-10 minutes. Incremental re-indexing helps after the first run (embedding cache skips unchanged chunks).

### 3. No private repos

Only public GitHub repos via HTTPS. Deliberate v1 scope limit.

### 4. Short snippets = weak embeddings

`x = 1` produces a useless embedding. They're indexed but never show up in top-k. Fine — they don't answer questions anyway.

### 5. Config files have imprecise citations

YAML/TOML/JSON use sliding-window chunking (no AST). You get `config.yaml:1-50` instead of `config.yaml:12-15`.

---

## Project Structure

```
codequery/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Configuration (env vars)
│   │   ├── models.py            # Pydantic models
│   │   ├── routers/
│   │   │   ├── repo.py          # Indexing + file content + tree + summary endpoints
│   │   │   └── chat.py          # Chat + health + starters endpoints
│   │   ├── services/
│   │   │   ├── cloner.py        # Git clone + incremental diff
│   │   │   ├── walker.py        # File tree walker
│   │   │   ├── chunker.py       # AST-aware chunking (tree-sitter)
│   │   │   ├── embedder.py      # Batch embedding (Ollama + ST fallback + cache)
│   │   │   ├── indexer.py       # Orchestrator (clone→parse→embed→store→snapshot)
│   │   │   ├── snapshot.py      # SQLite snapshot (replaces full git clone)
│   │   │   ├── search.py        # Retrieval + threshold filtering + context enrichment
│   │   │   └── generator.py     # LLM streaming + Mermaid + Chart.js prompts
│   │   └── store/
│   │       └── chroma_store.py  # ChromaDB wrapper (HNSW, per-repo collections)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── RepoInput.jsx
│   │   │   ├── IndexingProgress.jsx
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── MessageContent.jsx  # Chart/Mermaid/code parsing
│   │   │   ├── ChartComponent.jsx  # Lazy Chart.js
│   │   │   ├── MermaidDiagram.jsx  # Lazy mermaid
│   │   │   ├── CodeCitation.jsx    # Lazy code viewer
│   │   │   ├── FileTree.jsx        # Collapsible sidebar
│   │   │   ├── RepoSummary.jsx     # Stats + LLM overview
│   │   │   └── PixelBlast.jsx      # WebGL background
│   │   └── styles/index.css
│   └── vite.config.js
├── benchmarks/bench.py
└── README.md
```
