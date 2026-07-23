"""
Hybrid retrieval — combines BM25 keyword search with vector similarity search.

Why hybrid: pure vector search is weak at exact identifier matching. If someone
asks "what does handleSubmit do", cosine similarity on a general embedding model
may not rank the handleSubmit chunk first — but a keyword index will, instantly.
BM25 catches exact names/strings; vectors catch semantic/conceptual matches.
The two are fused with Reciprocal Rank Fusion (RRF), which needs no score
normalization and is robust even when the two rankers disagree completely.

An optional cross-encoder reranking pass can run on the fused candidates for
extra precision. It's off by default (CQ_RERANK_ENABLED=false) because it loads
a second small transformer model into memory — fine on most machines, but worth
being deliberate about on a memory-constrained free-tier host.
"""

import logging
import re
import threading
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)

# ── BM25 keyword index, cached per repo ─────────────────────────────────────
# Rebuilt lazily whenever the underlying collection's chunk count changes
# (i.e. after indexing/re-indexing/deleting a repo). Avoids rebuilding on
# every single question.

_bm25_cache: dict[str, dict] = {}
_bm25_lock = threading.Lock()

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _tokenize(text: str) -> list[str]:
    """Split code text into lowercase tokens, breaking up camelCase/snake_case
    identifiers so 'handleSubmit' and 'handle_submit' both index as
    ['handle', 'submit'] alongside the original whole token.
    """
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
    """Get (or rebuild) the BM25 index for a repo. Returns None if the repo
    has no chunks yet or BM25 dependency isn't installed."""
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

        # Tokenize file_path + name alongside content. Filenames otherwise
        # never appear in the searchable text (documents are pure code
        # content), so a question mentioning "package.json" or "mapStore.ts"
        # by name would have no lexical anchor to its own file without this.
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
    """Detect filename-like mentions in the question (e.g. "package.json",
    "mapStore.ts") and check them against the indexed corpus.

    This exists because embeddings/BM25 match on file *content*, not file
    *names* — asking specifically about a file by name has no guaranteed
    lexical anchor to that exact file otherwise, especially for config/manifest
    files whose content shares generic vocabulary with other files.

    Returns (matched_chunk_ids, unmatched_filenames). unmatched_filenames are
    filenames the question named that don't exist anywhere in the indexed
    repo — used downstream to stop the LLM from describing a file it never
    actually saw.
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
    ranked = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:n_results]
    return [(entry["ids"][i], float(scores[i])) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """Combine multiple ranked ID lists into one fused score per ID.

    RRF score for an item = sum over lists of 1 / (k + rank_in_that_list).
    Items missing from a list simply don't contribute from it. This needs no
    score normalization between BM25 (unbounded) and cosine similarity
    ([0,1]), which is why it's the standard choice for hybrid search fusion.
    """
    fused: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


# ── Optional cross-encoder reranker ─────────────────────────────────────────

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
    """Rerank (chunk_id, document_text) pairs against the question.

    Returns [(chunk_id, rerank_score), ...] sorted best-first. Falls back to
    returning candidates unscored (score=0.0, original order) if the reranker
    can't load — callers should treat that as "no reranking happened".
    """
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