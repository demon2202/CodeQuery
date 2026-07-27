import hashlib
import json
import logging
from typing import List

import httpx

from .. import config

logger = logging.getLogger(__name__)

_CACHE_PATH = config.DATA_DIR / "embed_cache.json"
_cache: dict[str, list] = {}
_cache_dirty = False

_MAX_EMBED_TEXT_LENGTH = 6000


def _load_cache() -> None:
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
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sanitize_for_embed(text: str) -> str:
    if not text or not text.strip():
        return " "
    text = text.strip()
    if len(text) > _MAX_EMBED_TEXT_LENGTH:
        text = text[:_MAX_EMBED_TEXT_LENGTH]
    return text


def _ollama_embed_single(text: str) -> List[float]:
    text = _sanitize_for_embed(text)
    resp = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/embed",
        json={"model": config.OLLAMA_EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    if resp.status_code != 200:
        detail = resp.text[:300] if resp.text else "no details"
        raise RuntimeError(f"Ollama embed error {resp.status_code}: {detail}")
    return resp.json()["embeddings"][0]


def _ollama_embed_batch(texts: List[str]) -> List[List[float]]:
    _load_cache()

    clean_texts = [_sanitize_for_embed(t) for t in texts]
    all_embeddings = []
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(clean_texts):
        h = _content_hash(text)
        if h in _cache:
            all_embeddings.append(_cache[h])
        else:
            all_embeddings.append(None)
            uncached_indices.append(i)
            uncached_texts.append(text)

    if not uncached_texts:
        return all_embeddings

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
            uncached_embeddings.extend(resp.json()["embeddings"])
            batch_start += batch_size
        except Exception as e:
            if batch_size <= 1:
                logger.error(f"Embedding failed for single item at index {batch_start}: {e}")
                uncached_embeddings.append([0.0] * 768)
                batch_start += 1
            else:
                logger.warning(f"Batch embed failed (size {batch_size}), retrying with size {batch_size // 4}: {e}")
                batch_size = max(1, batch_size // 4)

    global _cache_dirty
    for j, idx in enumerate(uncached_indices):
        emb = uncached_embeddings[j]
        all_embeddings[idx] = emb
        _cache[_content_hash(uncached_texts[j])] = emb
        _cache_dirty = True

    if _cache_dirty and len(uncached_texts) >= 10:
        _save_cache()

    return all_embeddings


def _check_ollama_embed_available() -> bool:
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


_st_model = None

# fastembed requires fully-namespaced model IDs (e.g. "sentence-transformers/all-MiniLM-L6-v2"),
# not bare names like "all-MiniLM-L6-v2". This maps common bare/legacy names people put in
# config or env vars to the correct fastembed model ID, so a misconfigured EMBEDDING_MODEL
# doesn't crash indexing in prod.
_MODEL_NAME_ALIASES = {
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}


def _resolve_model_name(name: str) -> str:
    if "/" in name:
        return name
    resolved = _MODEL_NAME_ALIASES.get(name.lower())
    if resolved:
        logger.warning(
            f"EMBEDDING_MODEL='{name}' is not a valid fastembed model ID; "
            f"using '{resolved}' instead. Update config/env to avoid this warning."
        )
        return resolved
    return name


def _get_st_model():
    """Get or load the fastembed ONNX model."""
    global _st_model
    if _st_model is None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise RuntimeError("fastembed is not installed. Run: pip install fastembed")

        model_name = _resolve_model_name(config.EMBEDDING_MODEL)

        try:
            supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        except Exception:
            supported = None

        if supported is not None and model_name not in supported:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not supported by fastembed. "
                f"Supported models: {sorted(supported)}"
            )

        logger.info(f"Loading FastEmbed model: {model_name}...")
        cache_dir = str(config.DATA_DIR / "fastembed_cache")
        _st_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        logger.info("FastEmbed model loaded.")
    return _st_model


def _st_encode_batch(texts: List[str]) -> List[List[float]]:
    import numpy as np
    model = _get_st_model()
    clean = [_sanitize_for_embed(t) for t in texts]
    embeddings = list(model.embed(clean, batch_size=config.EMBEDDING_BATCH_SIZE))
    return [e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings]


def _st_encode_single(text: str) -> List[float]:
    import numpy as np
    model = _get_st_model()
    text = _sanitize_for_embed(text)
    embeddings = list(model.embed([text]))
    e = embeddings[0]
    return e.tolist() if isinstance(e, np.ndarray) else list(e)


_provider = None


def _get_provider() -> str:
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
            logger.warning("Falling back to FastEmbed")
            _provider = "sentence-transformers"
    else:
        logger.info(f"Using FastEmbed (ONNX) embeddings with {config.EMBEDDING_MODEL}")
        _provider = "sentence-transformers"

    return _provider


def warm_up() -> None:
    _load_cache()
    provider = _get_provider()
    if provider == "ollama":
        try:
            result = _ollama_embed_single("warm up")
            logger.info(f"Ollama embeddings ready ({len(result)}-dim)")
        except Exception as e:
            logger.error(f"Ollama embedding test failed: {e}")
            logger.warning("Falling back to FastEmbed")
            global _provider
            _provider = "sentence-transformers"
            _get_st_model()
            _st_encode_single("warm up")
            logger.info("FastEmbed ready (fallback from Ollama)")
    else:
        _get_st_model()
        _st_encode_single("warm up")
        logger.info("FastEmbed (ONNX) warmed up")


def encode_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    provider = _get_provider()
    if provider == "ollama":
        return _ollama_embed_batch(texts)
    return _st_encode_batch(texts)


def encode_single(text: str) -> List[float]:
    provider = _get_provider()
    if provider == "ollama":
        return _ollama_embed_single(text)
    return _st_encode_single(text)


def flush_cache() -> None:
    _save_cache()