"""
File tree walker. Walks repo, filters to source files, skips junk.

Enhanced with diagnostics, expansive language support, and intelligent text fallback.
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
    skipped_extension = 0
    skipped_binary = 0
    skipped_size = 0

    for root, dirs, files in os.walk(repo_path):
        # Filter dirs in-place so os.walk skips them
        dirs[:] = sorted([d for d in dirs if not _skip_dir(d)])

        for fname in files:
            total_files += 1
            fp = Path(root) / fname

            # Skip binary, generated, huge, or empty files
            if _skip_file(fp):
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
            else:
                skipped_extension += 1

    # Log diagnostics so we can debug "No source files found"
    if result:
        logger.info(
            f"Walked {total_files} files → {len(result)} source files "
            f"(skipped: {skipped_extension} unsupported, {skipped_binary} binary, {skipped_size} size/empty)"
        )
    else:
        logger.warning(
            f"Walked {total_files} files but found 0 source files! "
            f"(skipped: {skipped_extension} unsupported, {skipped_binary} binary, {skipped_size} size/empty). "
            f"Repo path: {repo_path}"
        )
        ext_counts = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not _skip_dir(d)]
            for fname in files:
                ext = Path(fname).suffix.lower() or "(no-ext)"
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

    # Quick binary check
    return _is_binary(fp)


def _is_binary(fp: Path) -> bool:
    """Check if file contains null bytes or non-text content in first 4KB."""
    try:
        with open(fp, "rb") as f:
            chunk = f.read(4096)
            if not chunk:
                return False
            if b"\x00" in chunk:
                return True
            try:
                chunk.decode("utf-8")
                return False
            except UnicodeDecodeError:
                # Try latin-1 / ascii fallback check
                # Check ratio of non-printable characters
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
                non_text = chunk.translate(None, text_chars)
                return len(non_text) / len(chunk) > 0.30
    except OSError:
        return True


def _detect_lang(fp: Path) -> str | None:
    """Detect file language with AST, text extension, known filename, and text fallback."""
    ext = fp.suffix.lower()
    fname_lower = fp.name.lower()

    if ext in config.AST_EXTENSIONS:
        return config.AST_EXTENSIONS[ext]

    if ext in config.TEXT_EXTENSIONS:
        return "text"

    known_names = getattr(config, "KNOWN_SOURCE_FILENAMES", set())
    if fname_lower in known_names:
        return "text"

    # Intelligent text fallback: if it's readable text and not in skip extensions, index as text
    if ext not in config.SKIP_EXTENSIONS and not _is_binary(fp):
        return "text"

    return None
