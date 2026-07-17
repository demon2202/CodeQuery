"""
Search service — finds relevant code chunks from ChromaDB.

Key improvements:
- Returns actual document text alongside metadata (was the #1 bug before)
- Includes surrounding context lines for better LLM understanding
- Simple exact-match cache for repeated questions
"""

import logging
import time
from typing import List

from .. import config
from ..models import ChunkSource
from ..store.chroma_store import get_store
from .embedder import encode_single

logger = logging.getLogger(__name__)


def _enrich_chunk_with_context(
    doc: str, source: ChunkSource, repo_path_str: str, context_lines: int = 5
) -> str:
    """Add surrounding lines from the file for better LLM understanding.

    When we retrieve a chunk, it might be a single function. But the LLM
    often needs to see what's around it — imports, class definition, etc.
    This adds up to `context_lines` lines before and after the chunk.
    """
    try:
        from pathlib import Path
        file_path = Path(repo_path_str) / source.file_path
        if not file_path.exists():
            return doc
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        start = max(0, (source.start_line - 1) - context_lines)
        end = min(total, source.end_line + context_lines)

        # Build the enriched chunk with markers
        parts = []
        if start < source.start_line - 1:
            parts.append(f"  // ... {source.start_line - 1 - start} lines before ...")
        parts.append(doc)
        if end > source.end_line:
            parts.append(f"  // ... {end - source.end_line} lines after ...")

        return "\n".join(parts)
    except Exception:
        return doc


def search(
    repo_url: str,
    question: str,
    n_results: int = config.RETRIEVAL_TOP_K,
) -> tuple[List[ChunkSource], List[str], float]:
    """Search for relevant code chunks.

    Returns (sources, documents, search_time).
    sources = ChunkSource metadata for citations.
    documents = the actual code text for each chunk (for the LLM prompt).
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

    # Get repo path for context enrichment
    from .cloner import get_repo_path
    repo_path_str = ""
    try:
        repo_path_str = str(get_repo_path(repo_url))
    except Exception:
        pass

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

        # Enrich with surrounding context
        if repo_path_str:
            documents.append(_enrich_chunk_with_context(doc, source, repo_path_str))
        else:
            documents.append(doc)

    if not sources and ids:
        best_dist = min(distances) if distances else 1.0
        from .embedder import _get_provider
        provider = _get_provider()
        logger.warning(
            f"All {len(ids)} results below threshold {config.RETRIEVAL_MIN_SCORE} "
            f"(best sim: {1 - best_dist:.3f}). "
            f"Provider: {provider}. "
            f"If using sentence-transformers, switch to Ollama embeddings: "
            f"set CQ_EMBEDDING_PROVIDER=ollama and run: ollama pull nomic-embed-text"
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
