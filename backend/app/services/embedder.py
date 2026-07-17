"""
Embedding service for CodeQuery.

Supports two providers:
1. Ollama (RECOMMENDED) — GPU-accelerated, code-aware models like nomic-embed-text.
2. sentence-transformers (FALLBACK) — CPU-only, general-purpose models.

Handles edge cases: empty strings, overly long texts, batch failures with
automatic retry with smaller batches.
"""

import logging
from typing import List, Optional

import httpx

from .. import config

logger = logging.getLogger(__name__)

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
        # Truncate with a note so the LLM knows content was cut
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
    1. Sanitizing texts (truncate, replace empty)
    2. Starting with batch_size=64, retrying with smaller batches on failure
    3. Falling back to single-item embedding if a batch keeps failing
    """
    # Sanitize all texts
    clean_texts = [_sanitize_for_embed(t) for t in texts]

    all_embeddings = []
    batch_size = 64

    i = 0
    while i < len(clean_texts):
        batch = clean_texts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(clean_texts))))

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
            all_embeddings.extend(data["embeddings"])
            i += batch_size
        except Exception as e:
            if batch_size <= 1:
                # Even single items fail — log and use a zero vector
                logger.error(f"Embedding failed for single item at index {i}: {e}")
                # Return a zero vector as fallback so indexing doesn't crash
                all_embeddings.append([0.0] * 768)
                i += 1
            else:
                # Retry with smaller batch size
                logger.warning(f"Batch embed failed (size {batch_size}), retrying with size {batch_size // 4}: {e}")
                batch_size = max(1, batch_size // 4)
                # Don't advance i — retry this batch with smaller size

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
        import numpy as np
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


def get_embedding_dim() -> int:
    """Get the dimension of the embedding vectors."""
    provider = _get_provider()
    if provider == "ollama":
        try:
            test = _ollama_embed_single("dim test")
            return len(test)
        except Exception:
            return 768
    else:
        model = _get_st_model()
        return model.get_sentence_embedding_dimension()
