"""
CodeQuery configuration — env vars with sensible defaults.
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

# LLM — "ollama" or "groq"
LLM_PROVIDER = os.getenv("CQ_LLM_PROVIDER", "ollama")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("CQ_GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL = os.getenv("CQ_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("CQ_OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_TIMEOUT = int(os.getenv("CQ_OLLAMA_TIMEOUT", "120"))
OLLAMA_WARMUP_TIMEOUT = int(os.getenv("CQ_OLLAMA_WARMUP_TIMEOUT", "120"))

# Embedding — "ollama" (GPU, code-aware) or "sentence-transformers" (ONNX/CPU fallback)
EMBEDDING_PROVIDER = os.getenv("CQ_EMBEDDING_PROVIDER", "ollama")
OLLAMA_EMBED_MODEL = os.getenv("CQ_OLLAMA_EMBED_MODEL", "nomic-embed-text")

# fastembed model — requires full HuggingFace-style name (fastembed >= 0.4)
# CQ_EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code for better code retrieval
# (uses ~1.3GB RAM — too heavy for Render free tier)
EMBEDDING_MODEL = os.getenv("CQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_default_batch_size = "16" if "jina" in EMBEDDING_MODEL.lower() else "64"
EMBEDDING_BATCH_SIZE = int(os.getenv("CQ_EMBEDDING_BATCH_SIZE", _default_batch_size))

# Chunking
MAX_CHUNK_CHARS = int(os.getenv("CQ_MAX_CHUNK_CHARS", "2000"))
SLIDING_WINDOW_LINES = int(os.getenv("CQ_SLIDING_WINDOW_LINES", "50"))
SLIDING_WINDOW_OVERLAP = int(os.getenv("CQ_SLIDING_WINDOW_OVERLAP", "10"))

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("CQ_RETRIEVAL_TOP_K", "12"))

# Min similarity score — auto-adjusts by model, override with CQ_RETRIEVAL_MIN_SCORE
_provider_explicitly_set = os.getenv("CQ_EMBEDDING_PROVIDER") is not None

if os.getenv("CQ_RETRIEVAL_MIN_SCORE") is not None:
    RETRIEVAL_MIN_SCORE = float(os.getenv("CQ_RETRIEVAL_MIN_SCORE"))
elif _provider_explicitly_set and EMBEDDING_PROVIDER == "ollama" and "nomic" in OLLAMA_EMBED_MODEL.lower():
    RETRIEVAL_MIN_SCORE = 0.25
elif "jina" in EMBEDDING_MODEL.lower():
    RETRIEVAL_MIN_SCORE = 0.30
else:
    RETRIEVAL_MIN_SCORE = 0.10

# Hybrid search (BM25 + vector via RRF)
HYBRID_SEARCH_ENABLED = os.getenv("CQ_HYBRID_SEARCH_ENABLED", "true").lower() == "true"
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("CQ_HYBRID_CANDIDATE_MULTIPLIER", "3"))
RRF_K = int(os.getenv("CQ_RRF_K", "60"))

# Cross-encoder reranking — off by default (loads a second model into memory)
RERANK_ENABLED = os.getenv("CQ_RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("CQ_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Multi-turn contextual search
CONTEXTUAL_SEARCH_ENABLED = os.getenv("CQ_CONTEXTUAL_SEARCH_ENABLED", "true").lower() == "true"

# LLM query rewriting (costs one extra LLM round-trip per question — off by default)
QUERY_REWRITE_LLM_ENABLED = os.getenv("CQ_QUERY_REWRITE_LLM_ENABLED", "false").lower() == "true"

# Indexing limits
MAX_REPO_SIZE_MB = int(os.getenv("CQ_MAX_REPO_SIZE_MB", "300"))
INDEX_RATE_LIMIT_PER_MIN = int(os.getenv("CQ_INDEX_RATE_LIMIT_PER_MIN", "3"))
INDEX_RATE_LIMIT_ENABLED = os.getenv("CQ_INDEX_RATE_LIMIT_ENABLED", "true").lower() == "true"

# ChromaDB HNSW
CHROMA_HNSW_SPACE = "cosine"
CHROMA_HNSW_M = int(os.getenv("CQ_HNSW_M", "16"))
CHROMA_HNSW_EF_CONSTRUCTION = int(os.getenv("CQ_HNSW_EF_CONSTRUCTION", "100"))
CHROMA_HNSW_EF_SEARCH = int(os.getenv("CQ_HNSW_EF_SEARCH", "50"))

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