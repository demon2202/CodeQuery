"""
Embedding service for CodeQuery.

Supports two providers:
1. Ollama (RECOMMENDED) — GPU-accelerated, code-aware models like nomic-embed-text.
2. sentence-transformers (FALLBACK) — CPU-only, general-purpose models.

Handles edge cases: empty strings, overly long texts, batch failures with
automatic retry with smaller batches.

Performance: content-hash embedding cache avoids re-embedding unchanged chunks
during re-indexing. Cache is stored as a simple JSON file in the data directory.
"""

import hashlib
import json
import logging
from typing import List

import httpx

from .. import config

logger = logging.getLogger(__name__)

# ── Embedding Cache ────────────────────────────────────────────────────
# Content-hash → embedding vector cache. Stored as JSON in data directory.
# Avoids re-embedding unchanged chunks on re-index. On a typical incremental
# re-index where 90% of chunks are unchanged, this saves ~80% of embed time.

_CACHE_PATH = config.DATA_DIR / "embed_cache.json"
_cache: dict[str, list] = {}  # {content_hash: [float, ...]}
_cache_dirty = False


def _load_cache() -> None:
    """Load embedding cache from disk."""
    global _cache, _cache_dirty
    if _cache:
        return
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r") as f:
                _cache = json.load(f)
            logger.info(f"Embedding cache loaded: {len(_cache)} entries")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load embed cache: {e}")
            _cache = {}
    _cache_dirty = False


def _save_cache() -> None:
    """Save embedding cache to disk."""
    global _cache_dirty
    if not _cache_dirty:
        return
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(_cache, f)
        _cache_dirty = False
    except OSError as e:
        logger.warning(f"Failed to save embed cache: {e}")


def _content_hash(text: str) -> str:
    """Hash text content for cache lookup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# nomic-embed-text has 8192 token context. Code averages ~3.5 chars/token
# (much denser than prose). 8192 tokens * 3.5 = ~28k chars, BUT Ollama's
# BPE tokenizer can be 2.5-3 chars/token for code with many symbols/idents.
# At 2.5 chars/token, 8192 tokens = ~20k chars. We use 6000 chars to be
# very safe — some chunks are ALL identifiers/symbols which tokenize poorly.
_MAX_EMBED_TEXT_LENGTH = 6000


def _sanitize_for_embed(text: str) -> str:
    """Clean text for embedding: truncate, strip whitespace, handle empties.
    
    Important: we truncate at _MAX_EMBED_TEXT_LENGTH chars which is well
    below the 8192-token limit even for dense code (lots of symbols/idents).
    """
    if not text or not text.strip():
        return " "  # Ollama rejects empty strings — send a space instead
    text = text.strip()
    if len(text) > _MAX_EMBED_TEXT_LENGTH:
        text = text[:_MAX_EMBED_TEXT_LENGTH]
    return text


# ── Ollama Embedding ───────────────────────────────────────────────────

def _ollama_embed_single(text: str) -> List[float]:
    """Embed a single text using Ollama's /api/embed endpoint."""
    text = _sanitize_for_embed(text)
    resp = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.OLLAMA_EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    if resp.status_code != 200:
        detail = resp.text[:300] if resp.text else "no details"
        raise RuntimeError(f"Ollama embed error {resp.status_code}: {detail}")
    data = resp.json()
    return data["embeddings"][0]


def _ollama_embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts using Ollama's /api/embed endpoint.

    Handles failures by:
    1. Checking embedding cache for already-seen content
    2. Sanitizing texts (truncate, replace empty)
    3. Starting with batch_size=64, retrying with smaller batches on failure
    4. Falling back to single-item embedding if a batch keeps failing
    5. Caching new embeddings for future reuse
    """
    _load_cache()

    # Sanitize all texts
    clean_texts = [_sanitize_for_embed(t) for t in texts]

    # Check cache for each text
    all_embeddings = []
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(clean_texts):
        h = _content_hash(text)
        if h in _cache:
            all_embeddings.append(_cache[h])
        else:
            all_embeddings.append(None)  # placeholder
            uncached_indices.append(i)
            uncached_texts.append(text)

    # If everything is cached, we're done
    if not uncached_texts:
        return all_embeddings

    # Embed uncached texts
    uncached_embeddings = []
    batch_size = 64
    batch_start = 0

    while batch_start < len(uncached_texts):
        batch = uncached_texts[batch_start:batch_start + batch_size]

        try:
            resp = httpx.post(
                f"{config.OLLAMA_BASE_URL}/api/embed",
                json={"model": config.OLLAMA_EMBED_MODEL, "input": batch},
                timeout=120.0,
            )
            if resp.status_code != 200:
                detail = resp.text[:300] if resp.text else "no details"
                raise RuntimeError(f"Ollama embed error {resp.status_code}: {detail}")
            data = resp.json()
            uncached_embeddings.extend(data["embeddings"])
            batch_start += batch_size
        except Exception as e:
            if batch_size <= 1:
                # Even single items fail — log and use a zero vector
                logger.error(f"Embedding failed for single item at index {batch_start}: {e}")
                uncached_embeddings.append([0.0] * 768)
                batch_start += 1
            else:
                # Retry with smaller batch size
                logger.warning(f"Batch embed failed (size {batch_size}), retrying with size {batch_size // 4}: {e}")
                batch_size = max(1, batch_size // 4)
                # Don't advance — retry this batch with smaller size

    # Fill in uncached results and update cache
    global _cache_dirty
    for j, idx in enumerate(uncached_indices):
        emb = uncached_embeddings[j]
        all_embeddings[idx] = emb
        # Cache it
        h = _content_hash(uncached_texts[j])
        _cache[h] = emb
        _cache_dirty = True

    # Periodically save cache (every 100 new entries or so)
    if _cache_dirty and len(uncached_texts) >= 10:
        _save_cache()

    return all_embeddings


def _check_ollama_embed_available() -> bool:
    """Check if Ollama embedding model is available."""
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        embed_model = config.OLLAMA_EMBED_MODEL
        return any(
            embed_model in name or name.startswith(embed_model.split(":")[0])
            for name in model_names
        )
    except Exception:
        return False


# ── Sentence-Transformers Embedding ───────────────────────────────────

_st_model = None

def _get_st_model():
    """Get or load the sentence-transformers model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformers model: {config.EMBEDDING_MODEL}...")
        kwargs = {}
        if "jina" in config.EMBEDDING_MODEL.lower():
            kwargs["trust_remote_code"] = True
        _st_model = SentenceTransformer(config.EMBEDDING_MODEL, **kwargs)
        logger.info("Sentence-transformers model loaded.")
    return _st_model


def _st_encode_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch using sentence-transformers."""
    import numpy as np
    model = _get_st_model()
    embeddings = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return [e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings]


def _st_encode_single(text: str) -> List[float]:
    """Embed a single text using sentence-transformers."""
    import numpy as np
    model = _get_st_model()
    embedding = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    return list(embedding)


# ── Unified API ────────────────────────────────────────────────────────

_provider = None

def _get_provider() -> str:
    """Determine the embedding provider. Auto-detects on first call."""
    global _provider
    if _provider is not None:
        return _provider

    requested = config.EMBEDDING_PROVIDER.lower().strip()
    if requested == "ollama":
        if _check_ollama_embed_available():
            logger.info(f"Using Ollama embeddings with {config.OLLAMA_EMBED_MODEL}")
            _provider = "ollama"
        else:
            logger.warning(
                f"Ollama embedding model '{config.OLLAMA_EMBED_MODEL}' not found. "
                f"Run: ollama pull {config.OLLAMA_EMBED_MODEL}"
            )
            logger.warning("Falling back to sentence-transformers (poor code retrieval)")
            _provider = "sentence-transformers"
    else:
        logger.info(f"Using sentence-transformers embeddings with {config.EMBEDDING_MODEL}")
        _provider = "sentence-transformers"

    return _provider


def warm_up() -> None:
    """Pre-load the embedding model and verify it works."""
    _load_cache()  # Load cache on startup
    provider = _get_provider()
    if provider == "ollama":
        try:
            result = _ollama_embed_single("warm up")
            dim = len(result)
            logger.info(f"Ollama embeddings ready ({dim}-dim)")
        except Exception as e:
            logger.error(f"Ollama embedding test failed: {e}")
            logger.warning("Falling back to sentence-transformers")
            global _provider
            _provider = "sentence-transformers"
            _get_st_model()
            _st_encode_single("warm up")
            logger.info("Sentence-transformers ready (fallback)")
    else:
        _get_st_model()
        _st_encode_single("warm up")
        logger.info("Sentence-transformers warmed up")


def encode_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts. Returns list of embedding vectors."""
    if not texts:
        return []
    provider = _get_provider()
    if provider == "ollama":
        return _ollama_embed_batch(texts)
    return _st_encode_batch(texts)


def encode_single(text: str) -> List[float]:
    """Embed a single text (typically a user's question)."""
    provider = _get_provider()
    if provider == "ollama":
        return _ollama_embed_single(text)
    return _st_encode_single(text)


def flush_cache() -> None:
    """Force-save the embedding cache to disk. Call after indexing completes."""
    _save_cache()
