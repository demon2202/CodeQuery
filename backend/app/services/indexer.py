"""
Indexing orchestrator: clone → walk → streaming chunk & embed → store → snapshot.
Refactored for ultra-low memory (<250MB) and keep-alive SSE heartbeats.
"""

import asyncio
import concurrent.futures
import gc
import logging
import time
from pathlib import Path
from typing import AsyncGenerator

from .. import config
from ..store.chroma_store import get_store
from .cloner import clone_repo_streaming, get_repo_path, get_changed_files, force_remove
from .walker import walk_source_files
from .chunker import chunk_file, CodeChunk
from .embedder import encode_batch
from .graph_builder import extract_imports, build_dependency_graph, GRAPH_LANGS

try:
    from .embedder import flush_cache as _flush_embed_cache
except ImportError:
    _flush_embed_cache = None
from .snapshot import create_snapshot, get_graph as get_saved_graph, save_graph

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = max(1, config.EMBEDDING_BATCH_SIZE)


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


async def index_repo(repo_url: str) -> AsyncGenerator[dict, None]:
    """Index a GitHub repo with streaming chunking & embedding to minimize memory usage."""
    start = time.time()
    store = get_store()
    loop = asyncio.get_event_loop()

    # Clone → walk, with ONE automatic recovery if a previous deletion left a
    # corrupt/partial checkout behind (the classic "Repo has 3 files" bug on
    # Windows: read-only git files survive rmtree, the fetch path then works
    # on a skeleton directory).
    repo_path = None
    commit_hash = None
    changed_files = None
    file_lang_pairs: list = []
    total_all_files = 0

    for attempt in range(2):
        # 1. Clone with heartbeats
        yield {"type": "progress", "stage": "cloning", "message": f"Cloning {repo_url}..."}
        try:
            async for item in clone_repo_streaming(repo_url):
                if isinstance(item, tuple):
                    repo_path, commit_hash = item
                else:
                    yield {"type": "progress", "stage": "cloning", "message": item}
        except Exception as e:
            if attempt == 0 and repo_path is not None:
                logger.warning(f"Clone attempt 1 failed ({e}); wiping and retrying once")
                yield {"type": "progress", "stage": "cloning", "message": "Clone hiccup — wiping and retrying once..."}
                force_remove(repo_path)
                continue
            yield {"type": "error", "message": f"Clone failed: {e}"}
            return

        yield {"type": "progress", "stage": "cloning", "message": f"Cloned (commit {commit_hash[:8] if commit_hash else 'HEAD'})"}

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
        file_lang_pairs = walk_source_files(repo_path, changed_files)

        if file_lang_pairs:
            break

        # Zero source files on disk but the repo should have code: the
        # checkout is corrupt — wipe cache + directory and re-clone ONCE.
        if attempt == 0:
            logger.warning(
                f"Checkout for {repo_url} yielded 0 source files "
                f"({total_all_files} files on disk) — forcing clean re-clone"
            )
            yield {"type": "progress", "stage": "cloning", "message": "Checkout looks empty/corrupt — wiping and re-cloning cleanly..."}
            try:
                store.delete_collection(repo_url)
            except Exception:
                pass
            force_remove(repo_path)
            continue

    if not file_lang_pairs:
        yield {
            "type": "error",
            "message": (
                f"No source files found. Repo has {total_all_files} files but none match "
                "supported extensions (.py, .js, .ts, .jsx, .tsx, .c, .h, .java, .go, .rs, "
                ".rb, .php, .html, .css, .md, etc.)."
            ),
        }
        return

    total_files = len(file_lang_pairs)
    yield {"type": "progress", "stage": "walking", "message": f"Found {total_files} source files", "total": total_files}

    # 4. Build file tree
    tree = _build_file_tree(repo_path)

    # 5, 6, 7 & 8: Streaming Parse → Embed → Store Pipeline
    # Chunks are processed in small batches and released immediately to keep RAM < 250MB.
    yield {"type": "progress", "stage": "parsing", "current": 0, "total": total_files}

    lang_counts: dict[str, int] = {}
    func_count = 0
    total_chunks_indexed = 0
    chunk_buffer: list[CodeChunk] = []

    for _, lang in file_lang_pairs:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    last_heartbeat = time.time()
    parse_start = time.time()
    collected_imports: dict[str, list[str]] = {}

    async def _flush_chunk_buffer():
        nonlocal chunk_buffer, total_chunks_indexed, last_heartbeat
        if not chunk_buffer:
            return

        texts = [f"{c.file_path}\n{c.content}" for c in chunk_buffer]
        try:
            embeddings = await loop.run_in_executor(None, encode_batch, texts)
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")
            raise

        try:
            await loop.run_in_executor(
                None,
                store.add_chunks,
                repo_url,
                [c.chunk_id for c in chunk_buffer],
                [c.content for c in chunk_buffer],
                embeddings,
                [c.to_metadata() for c in chunk_buffer],
            )
        except Exception as e:
            logger.error(f"Storage batch failed: {e}")
            raise

        total_chunks_indexed += len(chunk_buffer)
        chunk_buffer = []
        del texts, embeddings
        gc.collect()

    def _parse_one(fp: Path, lang: str):
        """Chunk one file and harvest its imports for the dependency graph.

        Runs in a worker thread. tree-sitter parsing releases the GIL, and the
        chunker creates a fresh Parser per call, so this is thread-safe. RAM
        cost is one file's text in flight per worker — safe even on 512 MB.
        """
        file_chunks = chunk_file(fp, repo_path, lang)
        imports: list[str] = []
        if lang in GRAPH_LANGS:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    imports = extract_imports(f.read(), lang)
            except OSError:
                pass
        return fp, file_chunks, imports

    def _record_parsed(fp: Path, file_chunks, imports):
        nonlocal func_count
        for c in file_chunks:
            if c.chunk_type in ("function", "method"):
                func_count += 1
            chunk_buffer.append(c)
        if imports:
            try:
                collected_imports[str(fp.relative_to(repo_path))] = imports
            except ValueError:
                pass

    try:
        parse_workers = max(1, config.PARSE_WORKERS)
        window = max(parse_workers * 8, 16)
        processed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=parse_workers) as executor:
            for w_start in range(0, total_files, window):
                file_window = file_lang_pairs[w_start:w_start + window]
                futures = {
                    executor.submit(_parse_one, fp, lang): (fp, lang)
                    for fp, lang in file_window
                }
                for future in concurrent.futures.as_completed(futures):
                    fp, _ = futures[future]
                    processed += 1
                    try:
                        _, file_chunks, imports = future.result()
                        _record_parsed(fp, file_chunks, imports)
                        if len(chunk_buffer) >= EMBED_BATCH_SIZE:
                            await _flush_chunk_buffer()
                    except Exception as e:
                        logger.warning(f"Parse error in {fp}: {e}")

                    # Send progress update and keep-alive heartbeats
                    now = time.time()
                    if processed % 5 == 0 or (now - last_heartbeat) > 3.0 or processed == total_files:
                        last_heartbeat = now
                        elapsed = max(now - parse_start, 0.001)
                        chunks_so_far = total_chunks_indexed + len(chunk_buffer)
                        yield {
                            "type": "progress",
                            "stage": "parsing",
                            "current": processed,
                            "total": total_files,
                            "chunks_per_sec": round(chunks_so_far / elapsed, 1),
                            "message": f"Processed {processed}/{total_files} files ({chunks_so_far} chunks)",
                        }

        # Flush any remaining chunks
        if chunk_buffer:
            await _flush_chunk_buffer()

    except Exception as e:
        yield {"type": "error", "message": f"Indexing pipeline error: {e}"}
        return

    if total_chunks_indexed == 0:
        yield {"type": "error", "message": "No code chunks extracted from repository."}
        return

    yield {
        "type": "progress",
        "stage": "embedding",
        "current": total_chunks_indexed,
        "total": total_chunks_indexed,
        "message": f"Embedded and stored {total_chunks_indexed} chunks successfully",
    }

    try:
        store.set_commit_hash(repo_url, commit_hash)
    except Exception as e:
        logger.warning(f"Commit hash save failed: {e}")

    if _flush_embed_cache:
        try:
            _flush_embed_cache()
        except Exception as e:
            logger.warning(f"Embed cache flush failed: {e}")

    # Build final stats
    languages = [
        {"name": lang, "files": count, "pct": round(count / max(total_files, 1) * 100)}
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])
    ]
    stats = {
        "files": total_files,
        "chunks": total_chunks_indexed,
        "functions": func_count,
        "languages": languages[:8],
    }

    # 9. Build the dependency graph from the imports harvested while parsing
    graph_payload = None
    try:
        merged_imports: dict[str, list[str]] = {}
        if changed_files is not None:
            # Incremental update: keep the previous graph edges for unchanged files.
            existing = get_saved_graph(repo_url)
            if existing:
                merged_imports.update(existing.get("imports_of", {}))
        merged_imports.update(collected_imports)

        if merged_imports:
            full_pairs = walk_source_files(repo_path)
            all_paths = [str(fp.relative_to(repo_path)) for fp, _ in full_pairs]
            lang_map = {str(fp.relative_to(repo_path)): lang for fp, lang in full_pairs}
            graph_payload = build_dependency_graph(merged_imports, all_paths, lang_map)
    except Exception as e:
        logger.warning(f"Dependency graph build failed (non-critical): {e}")

    # 10. Snapshot & final storage
    yield {"type": "progress", "stage": "storing", "message": "Saving snapshot & finalizing index..."}
    try:
        create_snapshot(repo_url, repo_path, tree, stats)
    except Exception as e:
        logger.warning(f"Snapshot creation failed (non-critical): {e}")

    if graph_payload:
        try:
            save_graph(repo_url, graph_payload)
        except Exception as e:
            logger.warning(f"Dependency graph save failed (non-critical): {e}")

    gc.collect()

    yield {
        "type": "complete",
        "files_indexed": total_files,
        "chunks_created": total_chunks_indexed,
        "time_seconds": round(time.time() - start, 1),
        "commit_hash": commit_hash,
        "is_incremental": changed_files is not None,
        "graph_edges": graph_payload.get("num_edges", 0) if graph_payload else 0,
    }