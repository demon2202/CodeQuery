"""
Search service — finds relevant code chunks from ChromaDB.
Hybrid mode (BM25 + vector via RRF) with optional cross-encoder reranking.
Falls back to vector-only if hybrid is disabled.
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
from .embedder import encode_single

logger = logging.getLogger(__name__)


def _enrich_chunk_with_context(
    doc: str, source: ChunkSource, repo_url: str, context_lines: int = 5
) -> str:
    """Add surrounding lines from the file for better LLM context."""
    lines = None

    try:
        lines = get_file_lines(repo_url, source.file_path)
    except Exception:
        pass

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

    if ctx_start < source.start_line - 1:
        above_lines = lines[ctx_start: source.start_line - 1]
        if above_lines:
            parts.append(f"  // ... {len(above_lines)} lines above ...")
            for line in above_lines[-3:]:
                parts.append(line.rstrip())

    parts.append(doc)

    if ctx_end > source.end_line:
        below_lines = lines[source.end_line: ctx_end]
        if below_lines:
            for line in below_lines[:3]:
                parts.append(line.rstrip())
            parts.append(f"  // ... {len(below_lines)} lines below ...")

    return "\n".join(parts)


def _vector_candidates(
    repo_url: str, question_embedding: list[float], n_candidates: int
) -> tuple[dict[str, tuple[str, dict, float]], list[str]]:
    """Run vector query. Returns (id -> (doc, meta, similarity), ranked_ids)."""
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
    """Search for relevant code chunks. Returns (sources, documents, search_time, unmatched_filenames)."""
    start = time.time()
    question_embedding = encode_single(question)
    n_candidates = n_results * config.HYBRID_CANDIDATE_MULTIPLIER

    vector_lookup, vector_ranked = _vector_candidates(repo_url, question_embedding, n_candidates)

    sources: List[ChunkSource] = []
    documents: List[str] = []

    from . import hybrid_search

    matched_ids, unmatched_filenames = hybrid_search.find_filename_matches(repo_url, question)

    if not config.HYBRID_SEARCH_ENABLED:
        ordered = list(dict.fromkeys(matched_ids + vector_ranked))
        for chunk_id in ordered[:n_results]:
            if chunk_id not in vector_lookup:
                continue
            doc, meta, similarity = vector_lookup[chunk_id]
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

    candidate_ids = fused_order[: n_results * 2]

    if config.RERANK_ENABLED and candidate_ids:
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
        logger.warning(f"Hybrid search found no usable candidates for repo={repo_url}, question={question!r}")

    return sources, documents, time.time() - start, unmatched_filenames


_cache: dict[str, tuple] = {}


def search_cached(repo_url: str, question: str, n_results: int = config.RETRIEVAL_TOP_K):
    """Search with trivial exact-match cache. n_results is part of the key —
    slash commands fetch a wider window than regular questions."""
    key = f"{repo_url}:{n_results}:{question}"
    if key in _cache:
        return _cache[key]
    result = search(repo_url, question, n_results)
    _cache[key] = result
    if len(_cache) > 100:
        keys = list(_cache.keys())
        for k in keys[:50]:
            del _cache[k]
    return result