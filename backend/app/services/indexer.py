"""
Indexing orchestrator: clone → walk → chunk → embed → store → snapshot → cleanup.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncGenerator

from .. import config
from ..store.chroma_store import get_store
from .cloner import clone_repo_streaming, get_repo_path, get_changed_files
from .walker import walk_source_files
from .chunker import chunk_file, CodeChunk
from .embedder import encode_batch

try:
    from .embedder import flush_cache as _flush_embed_cache
except ImportError:
    _flush_embed_cache = None
from .snapshot import create_snapshot

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


def _build_file_tree(repo_path: Path) -> dict:
    skip_dirs = config.SKIP_DIRS
    skip_exts = config.SKIP_EXTENSIONS
    all_exts = config.ALL_SOURCE_EXTENSIONS

    def build_tree(path: Path, rel: str = "") -> dict:
        result = {"name": path.name, "type": "dir", "path": rel, "children": []}
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return result

        for entry in entries:
            if entry.name.startswith(".") and entry.name != ".github":
                continue
            if entry.is_dir():
                if entry.name in skip_dirs:
                    continue
                child_rel = f"{rel}/{entry.name}" if rel else entry.name
                child = build_tree(entry, child_rel)
                if child["children"]:
                    result["children"].append(child)
            elif entry.is_file():
                ext = entry.suffix.lower()
                if ext in skip_exts or ext not in all_exts:
                    continue
                try:
                    if entry.stat().st_size > config.MAX_FILE_SIZE_BYTES:
                        continue
                except OSError:
                    continue
                file_rel = str(entry.relative_to(repo_path))
                lang = config.AST_EXTENSIONS.get(ext, "text")
                result["children"].append({
                    "name": entry.name,
                    "type": "file",
                    "path": file_rel,
                    "language": lang,
                })
        return result

    return build_tree(repo_path)


def _build_repo_stats(
    file_lang_pairs: list[tuple[Path, str]],
    chunks: list[CodeChunk],
    chunk_count: int,
) -> dict:
    lang_counts: dict[str, int] = {}
    file_count = 0
    func_count = 0

    for _, lang in file_lang_pairs:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        file_count += 1

    for c in chunks:
        if c.chunk_type in ("function", "method"):
            func_count += 1

    total_files = max(file_count, 1)
    languages = [
        {"name": lang, "files": count, "pct": round(count / total_files * 100)}
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "files": file_count,
        "chunks": chunk_count,
        "functions": func_count,
        "languages": languages[:8],
    }


async def index_repo(repo_url: str) -> AsyncGenerator[dict, None]:
    """Index a GitHub repo. Yields SSE-progress events."""
    start = time.time()
    store = get_store()
    loop = asyncio.get_event_loop()

    # 1. Clone
    yield {"type": "progress", "stage": "cloning", "message": f"Cloning {repo_url}..."}
    repo_path = None
    commit_hash = None
    try:
        async for item in clone_repo_streaming(repo_url):
            if isinstance(item, tuple):
                repo_path, commit_hash = item
            else:
                yield {"type": "progress", "stage": "cloning", "message": item}
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
            yield {
                "type": "progress",
                "stage": "incremental",
                "message": f"{len(changed_files)} changed files (incremental)",
            }
            for f in changed_files:
                try:
                    store.delete_chunks_by_file(repo_url, f)
                except Exception:
                    pass
        else:
            logger.info("Incremental diff failed — doing full re-index")
            changed_files = None
    elif old_hash == commit_hash:
        yield {"type": "progress", "stage": "incremental", "message": "Already up to date (same commit)"}

    # 3. Walk files
    yield {"type": "progress", "stage": "walking", "message": "Scanning files..."}

    if not repo_path.exists():
        yield {"type": "error", "message": f"Repo directory not found: {repo_path}"}
        return

    total_all_files = sum(1 for _ in repo_path.rglob("*") if _.is_file())
    if total_all_files == 0:
        yield {"type": "error", "message": f"Repo directory is empty: {repo_path}"}
        return

    file_lang_pairs = walk_source_files(repo_path, changed_files)
    total_files = len(file_lang_pairs)
    if total_files == 0:
        yield {
            "type": "error",
            "message": (
                f"No source files found. Repo has {total_all_files} files but none match "
                "supported extensions (.py, .js, .ts, .jsx, .tsx, .c, .h, .java, .go, .rs, "
                ".rb, .php, .html, .css, .md, etc.)."
            ),
        }
        return
    yield {"type": "progress", "stage": "walking", "message": f"Found {total_files} source files", "total": total_files}

    # 4. Build file tree (while repo is on disk)
    tree = _build_file_tree(repo_path)

    # 5. Parse into chunks
    yield {"type": "progress", "stage": "parsing", "current": 0, "total": total_files}
    all_chunks: list[CodeChunk] = []
    for i, (fp, lang) in enumerate(file_lang_pairs):
        try:
            all_chunks.extend(chunk_file(fp, repo_path, lang))
        except Exception as e:
            logger.warning(f"Parse error {fp}: {e}")
        yield {"type": "progress", "stage": "parsing", "current": i + 1, "total": total_files}

    if not all_chunks:
        yield {"type": "error", "message": "No chunks extracted from repo"}
        return
    yield {"type": "progress", "stage": "parsing", "message": f"{total_files} files → {len(all_chunks)} chunks"}

    # 6. Pre-compute stats
    stats = _build_repo_stats(file_lang_pairs, all_chunks, len(all_chunks))

    # 7. Embed
    total_chunks = len(all_chunks)
    yield {"type": "progress", "stage": "embedding", "current": 0, "total": total_chunks}
    all_embeddings: list[list[float]] = []

    for i in range(0, total_chunks, EMBED_BATCH_SIZE):
        chunk_batch = all_chunks[i: i + EMBED_BATCH_SIZE]
        batch = [f"{c.file_path}\n{c.content}" for c in chunk_batch]
        try:
            embeddings = await loop.run_in_executor(None, encode_batch, batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            yield {"type": "error", "message": f"Embedding failed: {e}"}
            return
        yield {"type": "progress", "stage": "embedding", "current": min(i + EMBED_BATCH_SIZE, total_chunks), "total": total_chunks}

    # 8. Store in ChromaDB
    yield {"type": "progress", "stage": "storing", "current": 0, "total": total_chunks}
    store_batch = 500
    for i in range(0, total_chunks, store_batch):
        batch_c = all_chunks[i: i + store_batch]
        batch_e = all_embeddings[i: i + store_batch]
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
        yield {"type": "progress", "stage": "storing", "current": min(i + store_batch, total_chunks), "total": total_chunks}

    try:
        store.set_commit_hash(repo_url, commit_hash)
    except Exception as e:
        logger.warning(f"Commit hash save failed: {e}")

    if _flush_embed_cache:
        try:
            _flush_embed_cache()
        except Exception as e:
            logger.warning(f"Embed cache flush failed: {e}")

    # 9. Snapshot + cleanup
    yield {"type": "progress", "stage": "storing", "message": "Saving snapshot & cleaning up..."}
    try:
        try:
            db_stats = store.get_collection_stats(repo_url)
            stats["chunks"] = db_stats["chunk_count"]
        except Exception:
            pass
        create_snapshot(repo_url, repo_path, tree, stats)
    except Exception as e:
        logger.warning(f"Snapshot creation failed (non-critical): {e}")

    yield {
        "type": "complete",
        "files_indexed": total_files,
        "chunks_created": total_chunks,
        "time_seconds": round(time.time() - start, 1),
        "commit_hash": commit_hash,
        "is_incremental": changed_files is not None,
    }