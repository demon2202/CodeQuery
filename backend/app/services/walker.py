"""
File tree walker. Walks repo, filters to source files, skips junk.

Simplified from the 120-line version with hardcoded skip lists for every
framework under the sun. A few heuristics catch 99% of junk.
"""

import os
from pathlib import Path
from typing import List, Tuple

from .. import config


def walk_source_files(
    repo_path: Path,
    changed_files: list[str] | None = None,
) -> List[Tuple[Path, str]]:
    """Walk repo and return (file_path, language) pairs.

    If changed_files is set (incremental re-indexing), only return those files.
    """
    repo_path = repo_path.resolve()
    result = []

    if changed_files is not None:
        for rel in changed_files:
            fp = repo_path / rel
            if fp.is_file():
                lang = _detect_lang(fp)
                if lang:
                    result.append((fp, lang))
        return result

    for root, dirs, files in os.walk(repo_path):
        # Filter dirs in-place so os.walk skips them
        dirs[:] = [d for d in dirs if not _skip_dir(d)]

        for fname in files:
            fp = Path(root) / fname
            if _skip_file(fp):
                continue
            lang = _detect_lang(fp)
            if lang:
                result.append((fp, lang))

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
