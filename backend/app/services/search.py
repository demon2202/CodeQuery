"""
Search service — finds relevant code chunks from ChromaDB.

Key improvements:
- Returns actual document text alongside metadata (was the #1 bug before)
- Includes surrounding context lines from snapshot (no filesystem needed)
- Simple exact-match cache for repeated questions
"""

import logging
import time
from pathlib import Path
from typing import List

from .. import config
from ..models import ChunkSource
from ..store.chroma_store import get_store
from ..services.snapshot import get_file_lines
from .cloner import get_repo_path
from .embedder import encode_single, _get_provider

logger = logging.getLogger(__name__)


def _enrich_chunk_with_context(
    doc: str, source: ChunkSource, repo_url: str, context_lines: int = 5
) -> str:
    """Add surrounding lines from the file for better LLM understanding.

    Reads from the snapshot SQLite DB first, falls back to filesystem.
    Actually includes the surrounding code lines (not just markers) so
    the LLM has better context to understand the chunk.
    """
    lines = None

    # Try snapshot first (fast — single SQLite query)
    try:
        lines = get_file_lines(repo_url, source.file_path)
    except Exception:
        pass

    # Fallback: read from filesystem (backward compatibility)
    if lines is None:
        try:
            repo_path_str = str(get_repo_path(repo_url))
            file_path = Path(repo_path_str) / source.file_path
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.read().split("\n")
        except Exception:
            pass

    if not lines:
        return doc

    total = len(lines)
    ctx_start = max(0, (source.start_line - 1) - context_lines)
    ctx_end = min(total, source.end_line + context_lines)

    parts = []

    # Include lines above the chunk
    if ctx_start < source.start_line - 1:
        above_lines = lines[ctx_start : source.start_line - 1]
        if above_lines:
            parts.append(f"  // ... {len(above_lines)} lines above ...")
            # Include actual context lines for better LLM understanding
            for line in above_lines[-3:]:  # Last 3 lines above
                parts.append(line.rstrip())

    # The chunk itself
    parts.append(doc)

    # Include lines below the chunk
    if ctx_end > source.end_line:
        below_lines = lines[source.end_line : ctx_end]
        if below_lines:
            # Include actual context lines
            for line in below_lines[:3]:  # First 3 lines below
                parts.append(line.rstrip())
            parts.append(f"  // ... {len(below_lines)} lines below ...")

    return "\n".join(parts)


def search(
    repo_url: str,
    question: str,
    n_results: int = config.RETRIEVAL_TOP_K,
) -> tuple[List[ChunkSource], List[str], float]:
    """Search for relevant code chunks.

    Returns (sources, documents, search_time).
    """
    start = time.time()
    store = get_store()

    question_embedding = encode_single(question)

    try:
        results = store.query(
            repo_url=repo_url,
            query_embedding=question_embedding,
            n_results=n_results,
        )
    except Exception as e:
        raise RuntimeError(f"Search failed: {e}") from e

    sources: List[ChunkSource] = []
    documents: List[str] = []

    if not results or not results.get("ids") or not results["ids"][0]:
        return sources, documents, time.time() - start

    ids = results["ids"][0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else 1.0
        doc = docs[i] if i < len(docs) else ""

        similarity = max(0.0, 1.0 - distance)

        if similarity < config.RETRIEVAL_MIN_SCORE:
            continue

        source = ChunkSource(
            file_path=meta.get("file_path", "unknown"),
            start_line=meta.get("start_line", 0),
            end_line=meta.get("end_line", 0),
            name=meta.get("name", "unknown"),
            chunk_type=meta.get("chunk_type", "unknown"),
            language=meta.get("language", "unknown"),
            parent=meta.get("parent") or None,
            score=similarity,
        )
        sources.append(source)

        # Enrich with surrounding context from snapshot
        documents.append(_enrich_chunk_with_context(doc, source, repo_url))

    if not sources and ids:
        best_dist = min(distances) if distances else 1.0
        provider = _get_provider()
        logger.warning(
            f"All {len(ids)} results below threshold {config.RETRIEVAL_MIN_SCORE} "
            f"(best sim: {1 - best_dist:.3f}). Provider: {provider}."
        )

    return sources, documents, time.time() - start


# Simple exact-match cache
_cache: dict[str, tuple[list, list, float]] = {}


def search_cached(repo_url: str, question: str, n_results: int = config.RETRIEVAL_TOP_K):
    """Search with trivial exact-match cache."""
    key = f"{repo_url}:{question}"
    if key in _cache:
        return _cache[key]
    result = search(repo_url, question, n_results)
    _cache[key] = result
    if len(_cache) > 100:
        keys = list(_cache.keys())
        for k in keys[:50]:
            del _cache[k]
    return result
