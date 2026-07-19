"""
Repo API endpoints — uses lightweight snapshots instead of full git clones.

After indexing, repos are replaced with SQLite snapshots that contain:
- All source file contents (for file viewer and search enrichment)
- File tree structure (for file tree sidebar)
- Pre-computed stats (for repo summary)

The snapshot approach saves 80-95% disk space by removing the .git folder
and git objects, which are unnecessary after indexing.

Fallback: If no snapshot exists (backward compatibility), endpoints still
read from the filesystem.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..models import IndexRequest, RepoStatus, RepoStatusList, FileContentResponse
from .. import config
from ..services.indexer import index_repo
from ..services.cloner import get_repo_path
from ..services.snapshot import (
    snapshot_exists,
    get_file_content as get_snapshot_file,
    get_file_tree as get_snapshot_tree,
    get_repo_stats as get_snapshot_stats,
    delete_snapshot,
    get_dir_names,
)
from ..store.chroma_store import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/repos", tags=["repos"])

LANG_MAP = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".php": "php",
    ".c": "c", ".cpp": "cpp", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".sql": "sql", ".sh": "bash",
}


@router.post("/index")
async def index_repository(request: IndexRequest):
    """Index a repo. Returns SSE stream with real progress."""
    async def event_stream():
        try:
            async for event in index_repo(request.repo_url):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Indexing error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/status", response_model=RepoStatusList)
async def list_repos():
    store = get_store()
    repos = store.list_repos()
    return RepoStatusList(repos=[
        RepoStatus(
            repo_url=r["repo_url"],
            commit_hash=r.get("commit_hash", ""),
            files_indexed=0,
            chunks_created=r.get("chunks", 0),
            indexed_at="",
        ) for r in repos
    ])


@router.get("/file", response_model=FileContentResponse)
async def get_file_content(
    repo_url: str = Query(...),
    file_path: str = Query(...),
    start_line: Optional[int] = Query(None, ge=1),
    end_line: Optional[int] = Query(None, ge=1),
):
    """Get file content for citation expansion. Reads from snapshot."""
    # First-line defense: reject path traversal attempts
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Try snapshot first (fast — single SQLite query)
    data = get_snapshot_file(repo_url, file_path)

    if data is None:
        # Fallback: read from filesystem (backward compatibility)
        repo_path = get_repo_path(repo_url).resolve()
        full_path = (repo_path / file_path).resolve()
        try:
            full_path.relative_to(repo_path)
        except ValueError:
            raise HTTPException(status_code=400, detail="File path escapes repo directory")

        if not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            total_lines = content.count("\n") + 1
            lang = LANG_MAP.get(full_path.suffix.lower(), "text")
            data = {"content": content, "language": lang, "total_lines": total_lines}
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Read error: {e}")

    if data is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Apply line range
    total_lines = data["total_lines"]
    content = data["content"]
    language = data["language"]

    if start_line or end_line:
        lines = content.split("\n")
        s = max(0, (start_line or 1) - 1)
        e = min(len(lines), end_line or len(lines))
        content = "\n".join(lines[s:e])

    return FileContentResponse(
        file_path=file_path,
        content=content,
        language=language,
        total_lines=total_lines,
    )


@router.delete("/index")
async def delete_repo_index(repo_url: str = Query(...), delete_files: bool = Query(True)):
    """Delete a repo's index data, snapshot, and any remaining repo files."""
    store = get_store()
    try:
        store.delete_collection(repo_url)
    except Exception as e:
        logger.warning(f"Failed to delete collection: {e}")

    # Delete snapshot
    try:
        delete_snapshot(repo_url)
    except Exception as e:
        logger.warning(f"Failed to delete snapshot: {e}")

    # Also delete any remaining repo files on disk (for backward compat)
    if delete_files:
        repo_path = get_repo_path(repo_url)
        if repo_path.exists():
            try:
                shutil.rmtree(repo_path, ignore_errors=True)
                logger.info(f"Deleted repo files: {repo_path}")
            except Exception as e:
                logger.warning(f"Failed to delete repo files: {e}")

    return {"status": "deleted", "files_removed": delete_files}


@router.get("/tree")
async def get_file_tree(repo_url: str = Query(...)):
    """Get the file tree for a repo. Reads from snapshot."""
    # Try snapshot first (instant — pre-computed JSON)
    tree = get_snapshot_tree(repo_url)

    if tree is None:
        # Fallback: build from filesystem (backward compatibility)
        repo_path = get_repo_path(repo_url)
        if not repo_path.exists():
            raise HTTPException(status_code=404, detail="Repo not cloned and no snapshot found")

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
                    child_rel = f"{rel}/{entry.name}" if rel else entry.name
                    lang = LANG_MAP.get(ext, config.AST_EXTENSIONS.get(ext, "text"))
                    result["children"].append({
                        "name": entry.name,
                        "type": "file",
                        "path": child_rel,
                        "language": lang,
                    })
            return result

        tree = build_tree(repo_path)

    return tree


@router.get("/summary")
async def get_repo_summary(repo_url: str = Query(...)):
    """Generate a repo summary with stats and LLM overview. Uses pre-computed snapshot."""
    store = get_store()

    # Get collection stats
    try:
        db_stats = store.get_collection_stats(repo_url)
        chunk_count = db_stats["chunk_count"]
    except Exception:
        chunk_count = 0

    # Try pre-computed stats from snapshot first
    snap_stats = get_snapshot_stats(repo_url)

    if snap_stats:
        # Use pre-computed stats — update chunk_count from live DB
        languages = snap_stats.get("languages", [])
        file_count = snap_stats.get("files", 0)
        func_count = snap_stats.get("functions", 0)
    else:
        # Fallback: compute from filesystem (backward compatibility)
        repo_path = get_repo_path(repo_url)
        lang_counts = {}
        file_count = 0
        func_count = 0

        if repo_path.exists():
            for fp in repo_path.rglob("*"):
                if not fp.is_file():
                    continue
                if any(p.startswith(".") for p in fp.relative_to(repo_path).parts if p.startswith(".")):
                    continue
                if any(p in config.SKIP_DIRS for p in fp.relative_to(repo_path).parts):
                    continue
                ext = fp.suffix.lower()
                lang = config.AST_EXTENSIONS.get(ext) or ("text" if ext in config.TEXT_EXTENSIONS else None)
                if lang:
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    file_count += 1

        total_files = max(file_count, 1)
        languages = []
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            languages.append({
                "name": lang,
                "files": count,
                "pct": round(count / total_files * 100),
            })

    # Count functions from chunks metadata
    if chunk_count > 0 and func_count == 0:
        try:
            collection = store._get_collection(repo_url)
            sample = collection.peek(limit=100)
            metas = sample.get("metadatas", [])
            for m in metas:
                if m.get("chunk_type") in ("function", "method"):
                    func_count += 1
            if len(metas) < chunk_count and len(metas) > 0:
                func_count = int(func_count * chunk_count / len(metas))
        except Exception:
            pass

    result = {
        "stats": {
            "files": file_count,
            "chunks": chunk_count,
            "functions": func_count,
            "languages": languages[:8],
        },
        "text": None,
    }

    # Try to get LLM summary
    if chunk_count > 0:
        try:
            collection = store._get_collection(repo_url)
            sample = collection.peek(limit=30)
            metas = sample.get("metadatas", [])
            docs = sample.get("documents", [])

            file_list = sorted(set(m.get("file_path", "") for m in metas))[:20]
            func_list = [m.get("name", "") for m in metas if m.get("chunk_type") in ("function", "method") and m.get("name")][:15]
            class_list = [m.get("name", "") for m in metas if m.get("chunk_type") == "class" and m.get("name")][:10]

            lang_summary = ", ".join(f"{l['name']} ({l['pct']}%)" for l in languages[:5])

            prompt = f"""Give a brief 3-4 sentence overview of this codebase. Focus on what it does and how it's structured. Be specific.

Languages: {lang_summary}
Files: {', '.join(file_list)}
Functions: {', '.join(func_list)}
Classes: {', '.join(class_list)}

Overview:"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{config.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 300},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "").strip()
                    if text:
                        result["text"] = text
        except Exception as e:
            logger.debug(f"LLM summary failed (non-critical): {e}")

    return result
