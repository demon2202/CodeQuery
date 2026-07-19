"""
File tree walker. Walks repo, filters to source files, skips junk.

Enhanced with diagnostics — logs total files found, skipped reasons,
so "No source files found" errors are debuggable.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from .. import config

logger = logging.getLogger(__name__)


def walk_source_files(
    repo_path: Path,
    changed_files: list[str] | None = None,
) -> List[Tuple[Path, str]]:
    """Walk repo and return (file_path, language) pairs.

    If changed_files is set (incremental re-indexing), only return those files.
    """
    repo_path = repo_path.resolve()
    result = []

    if not repo_path.exists():
        logger.error(f"Repo path does not exist: {repo_path}")
        return result

    if changed_files is not None:
        for rel in changed_files:
            fp = repo_path / rel
            if fp.is_file():
                lang = _detect_lang(fp)
                if lang:
                    result.append((fp, lang))
        return result

    total_files = 0
    skipped_no_ext = 0
    skipped_extension = 0
    skipped_binary = 0
    skipped_size = 0

    for root, dirs, files in os.walk(repo_path):
        # Filter dirs in-place so os.walk skips them
        dirs[:] = sorted([d for d in dirs if not _skip_dir(d)])

        for fname in files:
            total_files += 1
            fp = Path(root) / fname

            # Quick extension check before expensive file I/O
            ext = fp.suffix.lower()
            if not ext:
                skipped_no_ext += 1
                continue
            if ext in config.SKIP_EXTENSIONS:
                skipped_extension += 1
                continue
            if ext not in config.ALL_SOURCE_EXTENSIONS:
                skipped_extension += 1
                continue

            if _skip_file(fp):
                # Determine why it was skipped for diagnostics
                try:
                    size = fp.stat().st_size
                    if size == 0 or size > config.MAX_FILE_SIZE_BYTES:
                        skipped_size += 1
                    else:
                        skipped_binary += 1
                except OSError:
                    skipped_binary += 1
                continue

            lang = _detect_lang(fp)
            if lang:
                result.append((fp, lang))

    # Log diagnostics so we can debug "No source files found"
    if result:
        logger.info(
            f"Walked {total_files} files → {len(result)} source files "
            f"(skipped: {skipped_no_ext} no-ext, {skipped_extension} wrong-ext, "
            f"{skipped_binary} binary, {skipped_size} size)"
        )
    else:
        logger.warning(
            f"Walked {total_files} files but found 0 source files! "
            f"(skipped: {skipped_no_ext} no-ext, {skipped_extension} wrong-ext, "
            f"{skipped_binary} binary, {skipped_size} size). "
            f"Repo path: {repo_path}"
        )
        # List what extensions ARE in the repo for debugging
        ext_counts = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not _skip_dir(d)]
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext:
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
        if ext_counts:
            top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:15]
            logger.info(f"Extensions found in repo: {top_exts}")

    return result


def _skip_dir(name: str) -> bool:
    """Skip hidden dirs, known junk dirs."""
    if name in config.SKIP_DIRS:
        return True
    if name.startswith(".") and name != ".github":
        return True
    return False


def _skip_file(fp: Path) -> bool:
    """Skip binary, generated, huge, or empty files."""
    ext = fp.suffix.lower()
    if ext in config.SKIP_EXTENSIONS:
        return True
    # Generated file patterns
    lower = fp.name.lower()
    if any(lower.endswith(p) for p in (".min.js", ".min.css", ".bundle.js",
                                        "_pb2.py", "_pb2_grpc.py")):
        return True
    if lower in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml",
                 "go.sum", "cargo.lock", "poetry.lock"):
        return True
    try:
        size = fp.stat().st_size
        if size == 0 or size > config.MAX_FILE_SIZE_BYTES:
            return True
    except OSError:
        return True
    # Quick binary check — null bytes in first 4KB
    try:
        with open(fp, "rb") as f:
            if b"\x00" in f.read(4096):
                return True
    except OSError:
        return True
    return False


def _detect_lang(fp: Path) -> str | None:
    ext = fp.suffix.lower()
    if ext in config.AST_EXTENSIONS:
        return config.AST_EXTENSIONS[ext]
    if ext in config.TEXT_EXTENSIONS:
        return "text"
    return None
