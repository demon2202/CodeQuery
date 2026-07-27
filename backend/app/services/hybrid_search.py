"""
Hybrid retrieval — BM25 keyword search fused with vector similarity via RRF.
Optional cross-encoder reranking pass on the fused candidates.
"""

import logging
import re
import threading
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

_bm25_cache: dict[str, dict] = {}
_bm25_lock = threading.Lock()

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tokenize(text: str) -> list[str]:
    """Split code text into lowercase tokens, breaking camelCase and snake_case."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        if "_" in raw:
            tokens.extend(p.lower() for p in raw.split("_") if p)
        camel_parts = _CAMEL_RE.split(raw)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts if p)
    return tokens


def _get_bm25_index(repo_url: str):
    """Get (or rebuild) the BM25 index for a repo. Returns None if unavailable."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed — falling back to vector-only search")
        return None

    from ..store.chroma_store import get_store

    store = get_store()
    stats = store.get_collection_stats(repo_url)
    current_count = stats["chunk_count"]
    if current_count == 0:
        return None

    with _bm25_lock:
        cached = _bm25_cache.get(repo_url)
        if cached and cached["count"] == current_count:
            return cached

        data = store.get_all_chunks(repo_url)
        ids = data["ids"]
        documents = data["documents"]
        metadatas = data["metadatas"]
        if not ids:
            return None

        # Tokenize file_path + name + content so filename queries have a lexical anchor
        tokenized = [
            _tokenize(f"{meta.get('file_path', '')} {meta.get('name', '')} {doc}")
            for doc, meta in zip(documents, metadatas)
        ]
        bm25 = BM25Okapi(tokenized)

        entry = {
            "count": current_count,
            "bm25": bm25,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        _bm25_cache[repo_url] = entry
        logger.info(f"BM25 index built for {repo_url}: {len(ids)} chunks")
        return entry


_FILENAME_TOKEN_RE = re.compile(r"[\w\-]+\.[A-Za-z0-9]{1,8}")


def find_filename_matches(repo_url: str, question: str) -> tuple[list[str], list[str]]:
    """Detect filename mentions in the question and match them against indexed chunks.

    Returns (matched_chunk_ids, unmatched_filenames).
    """
    entry = _get_bm25_index(repo_url)
    if entry is None:
        return [], []

    candidates = set()
    for m in _FILENAME_TOKEN_RE.finditer(question):
        token = m.group(0)
        ext = "." + token.rsplit(".", 1)[-1].lower()
        if ext not in config.ALL_SOURCE_EXTENSIONS and ext != ".lock":
            continue
        candidates.add(token.lower())
    if not candidates:
        return [], []

    matched_ids: list[str] = []
    matched_filenames = set()
    for cid, meta in zip(entry["ids"], entry["metadatas"]):
        basename = meta.get("file_path", "").replace("\\", "/").split("/")[-1].lower()
        for cand in candidates:
            if basename == cand or basename.endswith("/" + cand) or cand in basename:
                matched_ids.append(cid)
                matched_filenames.add(cand)
                break

    unmatched = sorted(candidates - matched_filenames)
    return matched_ids, unmatched


def invalidate(repo_url: str) -> None:
    """Drop the cached BM25 index for a repo (called after delete/re-index)."""
    with _bm25_lock:
        _bm25_cache.pop(repo_url, None)


def bm25_search(repo_url: str, question: str, n_results: int) -> list[tuple[str, float]]:
    """Keyword search. Returns [(chunk_id, bm25_score), ...] sorted best-first."""
    entry = _get_bm25_index(repo_url)
    if entry is None:
        return []

    query_tokens = _tokenize(question)
    if not query_tokens:
        return []

    scores = entry["bm25"].get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [(entry["ids"][i], float(scores[i])) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combine multiple ranked ID lists into fused scores via RRF."""
    fused: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {config.RERANK_MODEL}")
            _reranker = CrossEncoder(config.RERANK_MODEL)
        except Exception as e:
            logger.warning(f"Could not load reranker ({e}); continuing without it")
            _reranker = False
    return _reranker


def rerank(question: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Rerank (chunk_id, document_text) pairs against the question. Returns sorted best-first."""
    if not candidates:
        return []
    model = _get_reranker()
    if not model:
        return [(cid, 0.0) for cid, _ in candidates]

    pairs = [(question, doc) for _, doc in candidates]
    scores = model.predict(pairs)
    scored = list(zip([cid for cid, _ in candidates], (float(s) for s in scores)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored