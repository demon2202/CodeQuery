"""
CodeQuery configuration. Env vars with defaults. No API keys needed.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("CQ_DATA_DIR", str(BASE_DIR / "data")))
REPOS_DIR = DATA_DIR / "repos"
CHROMA_DIR = DATA_DIR / "chromadb"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
REPOS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# LLM Provider: "ollama" or "groq"
LLM_PROVIDER = os.getenv("CQ_LLM_PROVIDER", "ollama")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("CQ_GROQ_MODEL", "llama-3.3-70b-versatile")

# Ollama — qwen2.5-coder:7b for generation, nomic-embed-text for embeddings.
OLLAMA_BASE_URL = os.getenv("CQ_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("CQ_OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_TIMEOUT = int(os.getenv("CQ_OLLAMA_TIMEOUT", "120"))
OLLAMA_WARMUP_TIMEOUT = int(os.getenv("CQ_OLLAMA_WARMUP_TIMEOUT", "120"))

# Embedding provider: "ollama" or "sentence-transformers"
# Ollama is recommended — it uses your GPU, is code-aware, and is much faster.
# sentence-transformers is the fallback for CPU-only or no-Ollama setups.
EMBEDDING_PROVIDER = os.getenv("CQ_EMBEDDING_PROVIDER", "ollama")

# Ollama embedding model — nomic-embed-text is small (274MB), fast, and code-aware.
# Run: ollama pull nomic-embed-text
OLLAMA_EMBED_MODEL = os.getenv("CQ_OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Sentence-transformers model (used when EMBEDDING_PROVIDER=sentence-transformers,
# i.e. whenever Ollama isn't reachable — this is the path used on Render/free hosts).
#
# jina-embeddings-v2-base-code is code-aware and meaningfully better at this task
# than a general sentence model. IMPORTANT CAVEAT: it's a ~322M-param model
# (~1.3GB in memory), much heavier than all-MiniLM-L6-v2 (~22M params). On a
# memory-constrained free-tier host this may be slow to load or OOM. Test it on
# your actual deployment before relying on it — if it fails, set
# CQ_EMBEDDING_MODEL=all-MiniLM-L6-v2 to revert, or try a smaller general model
# as a middle ground (e.g. BAAI/bge-small-en-v1.5, 33M params, not code-specific
# but still better than nothing).
EMBEDDING_MODEL = os.getenv("CQ_EMBEDDING_MODEL", "jina-embeddings-v2-base-code")

# Smaller batches for the heavier jina model to reduce peak memory on CPU.
_default_batch_size = "16" if "jina" in EMBEDDING_MODEL.lower() else "64"
EMBEDDING_BATCH_SIZE = int(os.getenv("CQ_EMBEDDING_BATCH_SIZE", _default_batch_size))

# Chunking
MAX_CHUNK_CHARS = int(os.getenv("CQ_MAX_CHUNK_CHARS", "2000"))
SLIDING_WINDOW_LINES = int(os.getenv("CQ_SLIDING_WINDOW_LINES", "50"))
SLIDING_WINDOW_OVERLAP = int(os.getenv("CQ_SLIDING_WINDOW_OVERLAP", "10"))

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("CQ_RETRIEVAL_TOP_K", "12"))

# Minimum similarity score to keep a vector-search result. Different embedding
# models produce very different similarity ranges for the same content, so this
# auto-adjusts based on which model is active — UNLESS you set
# CQ_RETRIEVAL_MIN_SCORE explicitly, which always wins.
#
# These per-model numbers are reasonable starting points, not measured ground
# truth for your specific repos. If retrieval feels too strict/loose, hit
# GET /api/chat/debug/search to see real similarity scores for your queries
# and tune CQ_RETRIEVAL_MIN_SCORE from there.
if os.getenv("CQ_RETRIEVAL_MIN_SCORE") is not None:
    RETRIEVAL_MIN_SCORE = float(os.getenv("CQ_RETRIEVAL_MIN_SCORE"))
elif EMBEDDING_PROVIDER == "ollama":
    RETRIEVAL_MIN_SCORE = 0.25  # nomic-embed-text typically scores 0.40-0.70 on good matches
elif "jina" in EMBEDDING_MODEL.lower():
    RETRIEVAL_MIN_SCORE = 0.30  # estimate — verify via /api/chat/debug/search
else:
    RETRIEVAL_MIN_SCORE = 0.10  # all-MiniLM-L6-v2 and similar general models

# Hybrid search — BM25 keyword search fused with vector search via Reciprocal
# Rank Fusion. Fixes cases where embeddings miss exact identifier matches
# (e.g. "what does handleSubmit do"). Pure Python, no extra model, cheap to
# leave on. Falls back to vector-only automatically if rank_bm25 isn't installed.
HYBRID_SEARCH_ENABLED = os.getenv("CQ_HYBRID_SEARCH_ENABLED", "true").lower() == "true"
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("CQ_HYBRID_CANDIDATE_MULTIPLIER", "3"))
RRF_K = int(os.getenv("CQ_RRF_K", "60"))

# Cross-encoder reranking — extra precision pass on the fused candidates.
# Off by default: it loads a second small transformer model into memory,
# which matters on memory-constrained hosts (e.g. Render free tier already
# runs one embedding model). Turn on with CQ_RERANK_ENABLED=true if you have
# the headroom — it noticeably improves ranking quality.
RERANK_ENABLED = os.getenv("CQ_RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("CQ_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Multi-turn contextual search — folds the previous turn's question into the
# search query (not the generation prompt) so follow-ups like "where's the
# token refreshed" after "explain the auth flow" can actually retrieve
# auth-related chunks instead of searching for "token refreshed" in isolation.
# Cheap (no extra LLM call) and on by default.
CONTEXTUAL_SEARCH_ENABLED = os.getenv("CQ_CONTEXTUAL_SEARCH_ENABLED", "true").lower() == "true"

# Optional LLM-based query rewriting — asks the LLM to turn the raw question
# (+ history) into a better standalone search query before retrieval. More
# thorough than the heuristic above, but costs one extra LLM round-trip per
# question, so it's off by default.
QUERY_REWRITE_LLM_ENABLED = os.getenv("CQ_QUERY_REWRITE_LLM_ENABLED", "false").lower() == "true"

# Repo indexing limits — a public /api/repos/index endpoint with no limits is
# an open door to disk/CPU exhaustion (anyone can point it at a huge repo).
MAX_REPO_SIZE_MB = int(os.getenv("CQ_MAX_REPO_SIZE_MB", "300"))
INDEX_RATE_LIMIT_PER_MIN = int(os.getenv("CQ_INDEX_RATE_LIMIT_PER_MIN", "3"))
INDEX_RATE_LIMIT_ENABLED = os.getenv("CQ_INDEX_RATE_LIMIT_ENABLED", "true").lower() == "true"

# ChromaDB HNSW
CHROMA_HNSW_SPACE = "cosine"
CHROMA_HNSW_M = int(os.getenv("CQ_HNSW_M", "16"))
CHROMA_HNSW_EF_CONSTRUCTION = int(os.getenv("CQ_HNSW_EF_CONSTRUCTION", "100"))
CHROMA_HNSW_EF_SEARCH = int(os.getenv("CQ_HNSW_EF_SEARCH", "50"))

# File filtering
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "out", ".next", ".nuxt", ".output", "target",
    ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", ".cache", ".parcel-cache", ".turbo",
    "vendor", "third_party", ".idea", ".vscode", ".gradle", ".dart_tool",
}

AST_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
}

TEXT_EXTENSIONS = {
    ".c", ".h", ".cpp", ".cc", ".hpp", ".java", ".go", ".rs",
    ".rb", ".php", ".swift", ".kt", ".scala", ".lua",
    ".sh", ".bash", ".sql", ".graphql", ".proto",
    ".html", ".css", ".scss", ".vue", ".svelte",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".json", ".json5", ".jsonc",
    ".md", ".rst", ".txt",
}

# Pre-computed set of all source file extensions (dict keys + set merged)
ALL_SOURCE_EXTENSIONS = set(AST_EXTENSIONS.keys()) | TEXT_EXTENSIONS

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".lock",
}

MAX_FILE_SIZE_BYTES = int(os.getenv("CQ_MAX_FILE_SIZE", "500000"))