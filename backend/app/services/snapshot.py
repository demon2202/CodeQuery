"""
Lightweight repo snapshot — replaces full git clone on disk.

After indexing, we save:
1. File contents in SQLite (no .git folder, no git objects)
2. File tree as JSON
3. Repo stats (language counts, file counts) as JSON

Then delete the cloned repo directory. This saves 80-95% disk space
(.git folder alone is often 50-80% of a repo's size) while keeping
all features working (file tree, file viewer, search enrichment, etc).

SQLite is already a dependency of ChromaDB, so no new packages needed.
"""

import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .. import config

logger = logging.getLogger(__name__)


def _snapshot_dir(repo_url: str) -> Path:
    """Get the snapshot directory path for a repo URL."""
    from .cloner import get_repo_path
    repo_path = get_repo_path(repo_url)
    return config.SNAPSHOT_DIR / repo_path.name


def snapshot_exists(repo_url: str) -> bool:
    """Check if a snapshot exists for a repo."""
    snap_dir = _snapshot_dir(repo_url)
    return (snap_dir / "files.db").exists()


def create_snapshot(
    repo_url: str,
    repo_path: Path,
    tree: dict,
    stats: dict,
) -> None:
    """Create a lightweight snapshot from the cloned repo.

    Saves file contents to SQLite, tree as JSON, stats as JSON.
    Then deletes the cloned repo directory.
    """
    snap_dir = _snapshot_dir(repo_url)
    snap_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save file contents to SQLite
    db_path = snap_dir / "files.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE files (path TEXT PRIMARY KEY, content TEXT, language TEXT, total_lines INTEGER)"
    )
    conn.execute("CREATE INDEX idx_files_path ON files(path)")

    # Walk source files and save to DB
    from .walker import walk_source_files

    file_lang_pairs = walk_source_files(repo_path)

    batch = []
    for fp, lang in file_lang_pairs:
        try:
            rel_path = str(fp.relative_to(repo_path))
        except ValueError:
            rel_path = fp.name

        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            if lines == 0:
                lines = content.count("\n") + 1
            batch.append((rel_path, content, lang, lines))
        except OSError:
            continue

        # Insert in batches of 100
        if len(batch) >= 100:
            conn.executemany(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)", batch
            )
            batch = []

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)", batch
        )

    conn.commit()
    conn.close()

    # 2. Save file tree as JSON
    with open(snap_dir / "tree.json", "w") as f:
        json.dump(tree, f)

    # 3. Save stats as JSON
    with open(snap_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    logger.info(f"Snapshot created: {snap_dir} ({len(file_lang_pairs)} files)")

    # 4. Delete the cloned repo directory — the snapshot replaces it
    # On Windows, files may be locked so we try multiple times
    deleted = False
    for attempt in range(3):
        try:
            shutil.rmtree(repo_path)
            deleted = True
            logger.info(f"Deleted repo clone: {repo_path}")
            break
        except Exception as e:
            if attempt < 2:
                import time as _t
                _t.sleep(0.5)
            else:
                logger.warning(f"Failed to delete repo clone after 3 attempts: {e}")
                logger.warning("Repo files will remain on disk. You can manually delete them later.")


# ── Read operations ──────────────────────────────────────────────────


def get_file_content(repo_url: str, file_path: str) -> Optional[dict]:
    """Get file content from the snapshot.

    Returns dict with: content, language, total_lines
    or None if not found.
    """
    snap_dir = _snapshot_dir(repo_url)
    db_path = snap_dir / "files.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT content, language, total_lines FROM files WHERE path = ?",
            (file_path,),
        ).fetchone()
        if row:
            return {
                "content": row["content"],
                "language": row["language"],
                "total_lines": row["total_lines"],
            }
        return None
    finally:
        conn.close()


def get_file_lines(repo_url: str, file_path: str) -> Optional[list[str]]:
    """Get file lines from the snapshot. Returns list of lines or None."""
    data = get_file_content(repo_url, file_path)
    if data is None:
        return None
    return data["content"].split("\n")


def get_file_tree(repo_url: str) -> Optional[dict]:
    """Get the file tree from the snapshot."""
    snap_dir = _snapshot_dir(repo_url)
    tree_path = snap_dir / "tree.json"
    if not tree_path.exists():
        return None
    try:
        with open(tree_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_repo_stats(repo_url: str) -> Optional[dict]:
    """Get the repo stats from the snapshot."""
    snap_dir = _snapshot_dir(repo_url)
    stats_path = snap_dir / "stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_all_file_paths(repo_url: str) -> list[str]:
    """Get all file paths from the snapshot."""
    snap_dir = _snapshot_dir(repo_url)
    db_path = snap_dir / "files.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT path FROM files").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_dir_names(repo_url: str) -> set[str]:
    """Get all directory names from the snapshot (for starter generation)."""
    snap_dir = _snapshot_dir(repo_url)
    db_path = snap_dir / "files.db"
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT path FROM files").fetchall()
        dirs = set()
        for (path,) in rows:
            parts = path.split("/")
            for part in parts[:-1]:
                dirs.add(part.lower())
        return dirs
    finally:
        conn.close()


def delete_snapshot(repo_url: str) -> None:
    """Delete a repo's snapshot."""
    snap_dir = _snapshot_dir(repo_url)
    if snap_dir.exists():
        shutil.rmtree(snap_dir, ignore_errors=True)
        logger.info(f"Deleted snapshot: {snap_dir}")
