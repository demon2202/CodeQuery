"""
Repo API endpoints — fixed path traversal vulnerability.

OLD BUG: The /file endpoint did `repo_path / file_path` where file_path
came from user input. Even with the relative_to check, the resolve() call
could be fooled by symlinks inside the repo.

FIX: Resolve both paths, then check that the resolved path is under the
resolved repo root. Also reject any path containing `..` segments directly
as a first-line defense.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..models import IndexRequest, RepoStatus, RepoStatusList, FileContentResponse
from ..services.indexer import index_repo
from ..services.cloner import get_repo_path
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
    """Get file content for citation expansion."""
    # First-line defense: reject path traversal attempts
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")

    repo_path = get_repo_path(repo_url).resolve()
    full_path = (repo_path / file_path).resolve()

    # Verify resolved path is under the repo root
    try:
        full_path.relative_to(repo_path)
    except ValueError:
        raise HTTPException(status_code=400, detail="File path escapes repo directory")

    if not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")

    total_lines = len(lines)
    s = max(0, (start_line or 1) - 1)
    e = min(total_lines, end_line or total_lines)
    content = "".join(lines[s:e])

    return FileContentResponse(
        file_path=file_path,
        content=content,
        language=LANG_MAP.get(full_path.suffix.lower(), "text"),
        total_lines=total_lines,
    )


@router.delete("/index")
async def delete_repo_index(repo_url: str = Query(...), delete_files: bool = Query(True)):
    """Delete a repo's index data. Optionally also delete the cloned repo files."""
    import shutil
    store = get_store()
    try:
        store.delete_collection(repo_url)
    except Exception as e:
        logger.warning(f"Failed to delete collection: {e}")

    # Also delete the cloned repo files from disk
    if delete_files:
        repo_path = get_repo_path(repo_url)
        if repo_path.exists():
            try:
                shutil.rmtree(repo_path, ignore_errors=True)
                logger.info(f"Deleted repo files: {repo_path}")
            except Exception as e:
                logger.warning(f"Failed to delete repo files: {e}")

    return {"status": "deleted", "files_removed": delete_files}
