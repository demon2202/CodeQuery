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


def _vector_candidates(
    repo_url: str, question_embedding: list[float], n_candidates: int
) -> tuple[dict[str, tuple[str, dict, float]], list[str]]:
    """Run the vector query. Returns (id -> (doc, meta, similarity), ranked_ids)."""
    store = get_store()
    try:
        results = store.query(
            repo_url=repo_url,
            query_embedding=question_embedding,
            n_results=n_candidates,
        )
    except Exception as e:
        raise RuntimeError(f"Search failed: {e}") from e

    lookup: dict[str, tuple[str, dict, float]] = {}
    ranked_ids: list[str] = []

    if not results or not results.get("ids") or not results["ids"][0]:
        return lookup, ranked_ids

    ids = results["ids"][0]
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, chunk_id in enumerate(ids):
        doc = docs[i] if i < len(docs) else ""
        meta = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else 1.0
        similarity = max(0.0, 1.0 - distance)
        lookup[chunk_id] = (doc, meta, similarity)
        ranked_ids.append(chunk_id)

    return lookup, ranked_ids


def _build_source(meta: dict, score: float) -> ChunkSource:
    return ChunkSource(
        file_path=meta.get("file_path", "unknown"),
        start_line=meta.get("start_line", 0),
        end_line=meta.get("end_line", 0),
        name=meta.get("name", "unknown"),
        chunk_type=meta.get("chunk_type", "unknown"),
        language=meta.get("language", "unknown"),
        parent=meta.get("parent") or None,
        score=score,
    )


def search(
    repo_url: str,
    question: str,
    n_results: int = config.RETRIEVAL_TOP_K,
) -> tuple[List[ChunkSource], List[str], float, List[str]]:
    """Search for relevant code chunks.

    Hybrid mode (default): fuses BM25 keyword search with vector similarity
    search via Reciprocal Rank Fusion, then optionally reranks the fused
    candidates with a cross-encoder. Falls back to pure vector search if
    hybrid is disabled or rank_bm25 isn't available.

    Any filename explicitly mentioned in the question (e.g. "package.json")
    is force-included if it exists in the repo, regardless of how it ranked —
    filenames aren't part of the embedded/BM25 text otherwise, so a specific
    file mention has no guaranteed lexical anchor to its own content without
    this. Filenames mentioned but not found anywhere in the repo are returned
    separately so the caller can tell the LLM not to invent their contents.

    Returns (sources, documents, search_time, unmatched_filenames).
    """
    start = time.time()
    question_embedding = encode_single(question)
    n_candidates = n_results * config.HYBRID_CANDIDATE_MULTIPLIER

    vector_lookup, vector_ranked = _vector_candidates(repo_url, question_embedding, n_candidates)

    sources: List[ChunkSource] = []
    documents: List[str] = []

    from . import hybrid_search

    matched_ids, unmatched_filenames = hybrid_search.find_filename_matches(repo_url, question)

    if not config.HYBRID_SEARCH_ENABLED:
        # Pure vector path (legacy behavior) — still honor filename matches
        ordered = list(dict.fromkeys(matched_ids + vector_ranked))
        for chunk_id in ordered[:n_results]:
            if chunk_id in vector_lookup:
                doc, meta, similarity = vector_lookup[chunk_id]
            else:
                continue
            if chunk_id not in matched_ids and similarity < config.RETRIEVAL_MIN_SCORE:
                continue
            source = _build_source(meta, similarity)
            sources.append(source)
            documents.append(_enrich_chunk_with_context(doc, source, repo_url))
        return sources, documents, time.time() - start, unmatched_filenames

    bm25_ranked_pairs = hybrid_search.bm25_search(repo_url, question, n_candidates)
    bm25_ranked = [cid for cid, _ in bm25_ranked_pairs]

    if not vector_ranked and not bm25_ranked and not matched_ids:
        return sources, documents, time.time() - start, unmatched_filenames

    fused_scores = hybrid_search.reciprocal_rank_fusion(
        [vector_ranked, bm25_ranked], k=config.RRF_K
    )
    fused_order = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    # Build a full id -> (doc, meta) lookup, including ids that only came from
    # BM25 or from a direct filename match (vector search may not have
    # surfaced them at all).
    bm25_entry = hybrid_search._get_bm25_index(repo_url)
    bm25_lookup: dict[str, tuple[str, dict]] = {}
    if bm25_entry:
        bm25_lookup = {
            cid: (doc, meta)
            for cid, doc, meta in zip(
                bm25_entry["ids"], bm25_entry["documents"], bm25_entry["metadatas"]
            )
        }

    def _lookup_doc_meta(cid: str) -> tuple[str, dict]:
        if cid in vector_lookup:
            doc, meta, _ = vector_lookup[cid]
            return doc, meta
        return bm25_lookup.get(cid, ("", {}))

    candidate_ids = fused_order[: max(n_results * 2, n_results)]

    if config.RERANK_ENABLED and candidate_ids:
        # Always include filename matches in what gets reranked, so they can't
        # get lost before the rerank step even considers them.
        rerank_pool = list(dict.fromkeys(matched_ids + candidate_ids))
        pairs = [(cid, _lookup_doc_meta(cid)[0]) for cid in rerank_pool if _lookup_doc_meta(cid)[0]]
        reranked = hybrid_search.rerank(question, pairs)
        import math
        final_ids_scores = [(cid, 1 / (1 + math.exp(-s))) for cid, s in reranked[:n_results]]
    else:
        final_ids_scores = []
        for cid in candidate_ids[:n_results]:
            if cid in vector_lookup:
                score = vector_lookup[cid][2]
            else:
                rank = fused_order.index(cid)
                score = 1.0 / (1 + rank)
            final_ids_scores.append((cid, score))

    # Force-include filename matches at the front, deterministically, even if
    # they didn't make the cut above. This is the guarantee: ask about a
    # specific file by name, get that file, not a ranking approximation of it.
    seen_ids = {cid for cid, _ in final_ids_scores}
    forced = [(cid, 1.0) for cid in matched_ids if cid not in seen_ids and _lookup_doc_meta(cid)[0]]
    final_ids_scores = forced + [pair for pair in final_ids_scores if pair[0] not in {f[0] for f in forced}]
    final_ids_scores = final_ids_scores[: max(n_results, len(forced))]

    for cid, score in final_ids_scores:
        doc, meta = _lookup_doc_meta(cid)
        if not doc:
            continue
        source = _build_source(meta, score)
        sources.append(source)
        documents.append(_enrich_chunk_with_context(doc, source, repo_url))

    if not sources:
        logger.warning(
            f"Hybrid search found no usable candidates for repo={repo_url}, "
            f"question={question!r}"
        )

    return sources, documents, time.time() - start, unmatched_filenames


# Simple exact-match cache
_cache: dict[str, tuple] = {}


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