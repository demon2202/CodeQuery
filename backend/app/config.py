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

# Sentence-transformers model (used when EMBEDDING_PROVIDER=sentence-transformers)
# WARNING: all-MiniLM-L6-v2 is a general sentence model — poor at code retrieval.
# It produces similarity scores of 0.15-0.20 for code Q&A, barely above noise.
# Use Ollama embeddings instead if possible.
EMBEDDING_MODEL = os.getenv("CQ_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("CQ_EMBEDDING_BATCH_SIZE", "64"))

# Chunking
MAX_CHUNK_CHARS = int(os.getenv("CQ_MAX_CHUNK_CHARS", "2000"))
SLIDING_WINDOW_LINES = int(os.getenv("CQ_SLIDING_WINDOW_LINES", "50"))
SLIDING_WINDOW_OVERLAP = int(os.getenv("CQ_SLIDING_WINDOW_OVERLAP", "10"))

# Retrieval
RETRIEVAL_TOP_K = int(os.getenv("CQ_RETRIEVAL_TOP_K", "12"))
# Minimum similarity score. all-MiniLM-L6-v2 produces 0.15-0.20 for code queries,
# so 0.10 is needed. nomic-embed-text produces 0.40-0.70 for good matches, so
# 0.25 works well. Auto-adjusted based on provider in search.py if not overridden.
RETRIEVAL_MIN_SCORE = float(os.getenv("CQ_RETRIEVAL_MIN_SCORE", "0.10"))

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
