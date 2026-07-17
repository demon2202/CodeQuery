"""
Indexing orchestrator. Clone → walk → chunk → embed → store.

Simplified from 150 lines. Key changes:
- Removed ProcessPoolExecutor for tree-sitter parsing. It adds complexity
  (pickling issues, hard to debug) for a problem that doesn't exist —
  tree-sitter parsing is ~1ms per file. Even 500 files = 0.5s. The real
  bottleneck is embedding, which is already batched.
- Inline the progress events — no need for _progress_event/_error_event helpers.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncGenerator

from .. import config
from ..store.chroma_store import get_store
from .cloner import clone_repo, get_repo_path, get_changed_files
from .walker import walk_source_files
from .chunker import chunk_file, CodeChunk
from .embedder import encode_batch

logger = logging.getLogger(__name__)


async def index_repo(repo_url: str) -> AsyncGenerator[dict, None]:
    """Index a GitHub repo. Yields SSE-progress events."""
    start = time.time()
    store = get_store()
    loop = asyncio.get_event_loop()

    # 1. Clone
    yield {"type": "progress", "stage": "cloning", "message": f"Cloning {repo_url}..."}
    try:
        repo_path, commit_hash = await clone_repo(repo_url)
    except Exception as e:
        yield {"type": "error", "message": f"Clone failed: {e}"}
        return
    yield {"type": "progress", "stage": "cloning", "message": f"Cloned (commit {commit_hash[:8]})"}

    # 2. Check for incremental update
    old_hash = store.get_commit_hash(repo_url)
    changed_files = None
    if old_hash and old_hash != commit_hash:
        changed_files = await get_changed_files(repo_path, old_hash, commit_hash)
        if changed_files:
            yield {"type": "progress", "stage": "incremental",
                   "message": f"{len(changed_files)} changed files (incremental)"}
            # Delete old chunks for changed files
            for f in changed_files:
                try:
                    store.delete_chunks_by_file(repo_url, f)
                except Exception:
                    pass
        else:
            changed_files = None  # Fall through to full index
    elif old_hash == commit_hash:
        yield {"type": "progress", "stage": "incremental",
               "message": "Already up to date (same commit)"}

    # 3. Walk files
    yield {"type": "progress", "stage": "walking", "message": "Scanning files..."}
    file_lang_pairs = walk_source_files(repo_path, changed_files)
    total_files = len(file_lang_pairs)
    if total_files == 0:
        yield {"type": "error", "message": "No source files found"}
        return
    yield {"type": "progress", "stage": "walking",
           "message": f"Found {total_files} source files", "total": total_files}

    # 4. Parse files into chunks
    yield {"type": "progress", "stage": "parsing", "current": 0, "total": total_files}
    all_chunks: list[CodeChunk] = []
    for i, (fp, lang) in enumerate(file_lang_pairs):
        try:
            all_chunks.extend(chunk_file(fp, repo_path, lang))
        except Exception as e:
            logger.warning(f"Parse error {fp}: {e}")
        yield {"type": "progress", "stage": "parsing",
               "current": i + 1, "total": total_files}

    if not all_chunks:
        yield {"type": "error", "message": "No chunks extracted from repo"}
        return
    yield {"type": "progress", "stage": "parsing",
           "message": f"{total_files} files → {len(all_chunks)} chunks"}

    # 5. Embed in batches
    total_chunks = len(all_chunks)
    yield {"type": "progress", "stage": "embedding", "current": 0, "total": total_chunks}
    all_embeddings: list[list[float]] = []
    # Use smaller batch for progress granularity — the embedder will
    # internally batch/retry, but we report per-batch progress.
    embed_batch = 16  # Small enough for progress updates, embedder handles internal batching

    for i in range(0, total_chunks, embed_batch):
        batch = [c.content for c in all_chunks[i:i + embed_batch]]
        try:
            embeddings = await loop.run_in_executor(None, encode_batch, batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            yield {"type": "error", "message": f"Embedding failed: {e}"}
            return
        done = min(i + embed_batch, total_chunks)
        yield {"type": "progress", "stage": "embedding",
               "current": done, "total": total_chunks}

    # 6. Store in ChromaDB
    yield {"type": "progress", "stage": "storing", "current": 0, "total": total_chunks}
    store_batch = 500
    for i in range(0, total_chunks, store_batch):
        batch_c = all_chunks[i:i + store_batch]
        batch_e = all_embeddings[i:i + store_batch]
        try:
            await loop.run_in_executor(
                None,
                store.add_chunks,
                repo_url,
                [c.chunk_id for c in batch_c],
                [c.content for c in batch_c],
                batch_e,
                [c.to_metadata() for c in batch_c],
            )
        except Exception as e:
            yield {"type": "error", "message": f"Storage failed: {e}"}
            return
        yield {"type": "progress", "stage": "storing",
               "current": min(i + store_batch, total_chunks), "total": total_chunks}

    # Save commit hash for incremental re-indexing
    try:
        store.set_commit_hash(repo_url, commit_hash)
    except Exception as e:
        logger.warning(f"Commit hash save failed: {e}")

    yield {
        "type": "complete",
        "files_indexed": total_files,
        "chunks_created": total_chunks,
        "time_seconds": round(time.time() - start, 1),
        "commit_hash": commit_hash,
        "is_incremental": changed_files is not None,
    }
