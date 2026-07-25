"""
Git clone service — fixed command injection + better Windows error handling.

SECURITY: Validate the URL strictly before passing to git. Only allow
https://github.com/owner/repo format. No shell metacharacters.

WINDOWS FIX: Check if git is available before trying to clone.
Give clear error messages when git is missing or clone fails.
Also capture stdout alongside stderr for better diagnostics.
"""

import asyncio
import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Tuple

from .. import config

logger = logging.getLogger(__name__)

_GIT_EXE: str = "git"  # default — overridden below if we can find it

_git_candidate = shutil.which("git")
if _git_candidate:
    _GIT_EXE = _git_candidate
else:
    # Common Windows installation paths
    _windows_fallbacks = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
    ]
    for _fb in _windows_fallbacks:
        if os.path.isfile(_fb):
            _GIT_EXE = _fb
            break

logger.info(f"Git executable resolved to: {_GIT_EXE}")

# Strict URL validation — only allow safe GitHub URLs
# Prevents command injection via shell metacharacters in repo_url
_SAFE_URL_RE = re.compile(
    r"^https?://(www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$"
)


def validate_repo_url(url: str) -> str:
    """Validate and normalize a GitHub repo URL. Raises ValueError if invalid."""
    url = url.strip().rstrip("/")
    if not _SAFE_URL_RE.match(url):
        raise ValueError(
            f"Invalid repo URL: {url}. Only public GitHub URLs are supported "
            f"(e.g., https://github.com/owner/repo)"
        )
    return url


def get_repo_path(repo_url: str) -> Path:
    """Get the local directory path for a repo URL."""
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    # Use hash for uniqueness, keep readable prefix
    safe_name = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else hashlib.sha256(url.encode()).hexdigest()[:16]
    return config.REPOS_DIR / safe_name


async def _run_git(*args: str, cwd: str | None = None) -> Tuple[str, str, int]:
    """Run a git command. Returns (stdout, stderr, returncode)."""
    # Replace the first token ("git") with the resolved absolute path so the
    # subprocess finds git even when PATH is incomplete (common on Windows).
    resolved_args = (_GIT_EXE,) + args[1:]
    try:
        proc = await asyncio.create_subprocess_exec(
            *resolved_args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode().strip(), stderr.decode().strip(), proc.returncode or 0
    except FileNotFoundError:
        raise RuntimeError(
            f"git not found at '{_GIT_EXE}'. "
            "Please install Git: https://git-scm.com/downloads"
        )


async def _run_git_streaming(*args: str, cwd: str | None = None):
    """Run a git command and yield stderr lines in real-time. Returns (returncode, all_stderr).

    git writes progress to stderr (--progress), so we stream stderr line-by-line
    so callers can forward progress events to the frontend without blocking.
    Yields individual (non-empty) stderr lines as strings.
    After iteration completes, the caller can check the return code from the
    process object returned as the last yielded item (a special sentinel).
    """
    resolved_args = (_GIT_EXE,) + args[1:]
    try:
        proc = await asyncio.create_subprocess_exec(
            *resolved_args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"git not found at '{_GIT_EXE}'. "
            "Please install Git: https://git-scm.com/downloads"
        )

    stderr_chunks: list[str] = []
    stdout_chunks: list[str] = []

    # Stream stderr line-by-line (git clone --progress writes here)
    # We must also drain stdout to avoid the process blocking on a full pipe.
    async def _drain_stdout():
        assert proc.stdout is not None
        raw = await proc.stdout.read()
        stdout_chunks.append(raw.decode(errors="replace"))

    # Start draining stdout in background
    stdout_task = asyncio.create_task(_drain_stdout())

    assert proc.stderr is not None
    # git uses \r for in-place progress updates (e.g. "Receiving objects: 42%")
    # readline() only splits on \n. Read in chunks and split on both.
    buf = b""
    while True:
        chunk = await proc.stderr.read(256)
        if not chunk:
            break
        buf += chunk
        # Split on both \r and \n
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()  # keep last incomplete fragment
        for part in parts:
            line = part.decode(errors="replace").strip()
            if line:
                stderr_chunks.append(line)
                yield line

    # Flush any remaining buffer
    if buf:
        line = buf.decode(errors="replace").strip()
        if line:
            stderr_chunks.append(line)
            yield line

    await stdout_task
    await proc.wait()

    # Yield a sentinel tuple so caller can get returncode + full output
    full_stderr = "\n".join(stderr_chunks)
    full_stdout = "".join(stdout_chunks)
    yield (proc.returncode or 0, full_stdout, full_stderr)


async def check_git_available() -> str:
    """Check if git is installed and return its version. Raises if not found."""
    try:
        stdout, _, rc = await _run_git("git", "--version")
        if rc == 0:
            return stdout
        raise RuntimeError("git command failed")
    except RuntimeError as e:
        if "not found" in str(e) or "not installed" in str(e):
            raise
        raise RuntimeError(
            f"git not found at '{_GIT_EXE}'. "
            "Please install Git: https://git-scm.com/downloads"
        )


def _dir_size_mb(path: Path) -> float:
    """Total size of a directory in MB, following regular files only."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


async def clone_repo_streaming(repo_url: str):
    """Clone a repo (shallow) or update if already exists, streaming progress.

    Yields progress strings during the git clone, then finally yields a tuple
    (repo_path, commit_hash) when done. Raises RuntimeError on failure.

    Usage:
        async for item in clone_repo_streaming(url):
            if isinstance(item, tuple):
                repo_path, commit_hash = item
            else:
                # item is a progress string — forward to the frontend
                yield_to_sse(item)
    """
    repo_url = validate_repo_url(repo_url)
    repo_path = get_repo_path(repo_url)

    # Check git is available before doing anything
    await check_git_available()

    if (repo_path / ".git").exists():
        # Update existing clone — fetch + reset instead of pull
        # (shallow clones often can't `git pull` due to missing tracking info)
        yield "Updating existing clone..."
        stdout, stderr, rc = await _run_git("git", "fetch", "--depth=1", "--quiet", cwd=str(repo_path))
        if rc == 0:
            await _run_git("git", "reset", "--hard", "FETCH_HEAD", "--quiet", cwd=str(repo_path))
        else:
            # Fetch failed — log it but keep existing clone usable
            logger.warning(f"git fetch failed for {repo_url}: {stderr or stdout}")
            # If the repo directory exists but is corrupted, try a fresh clone
            if not (repo_path / ".git" / "HEAD").exists():
                logger.warning("Repo appears corrupted, removing for fresh clone")
                shutil.rmtree(repo_path, ignore_errors=True)
                # Fall through to fresh clone below
            else:
                # Keep existing clone — it's still usable
                pass

    if not (repo_path / ".git").exists():
        # Fresh shallow clone — stream progress from stderr
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

        rc = 0
        full_stderr = ""
        full_stdout = ""

        async for item in _run_git_streaming(
            "git", "clone", "--depth", "1", "--progress", repo_url, str(repo_path)
        ):
            if isinstance(item, tuple):
                # Sentinel: (returncode, stdout, stderr)
                rc, full_stdout, full_stderr = item
            else:
                # Progress line from stderr — forward to caller
                yield item

        if rc != 0:
            shutil.rmtree(repo_path, ignore_errors=True)
            # Build a helpful error message
            parts = ["git clone failed"]
            if full_stderr:
                parts.append(full_stderr)
            if full_stdout:
                parts.append(full_stdout)
            msg = ": ".join(parts) if len(parts) > 1 else parts[0]

            # Add helpful hints for common failures
            lower_msg = (full_stderr + " " + full_stdout).lower()
            if "could not resolve" in lower_msg or "network" in lower_msg:
                msg += "\n\nPossible causes:\n- No internet connection\n- GitHub is unreachable\n- URL is incorrect"
            elif "repository not found" in lower_msg or "not found" in lower_msg:
                msg += "\n\nThe repository doesn't exist or is private. Only public repos are supported."
            elif "permission denied" in lower_msg:
                msg += "\n\nPermission denied. Check if the repo directory is accessible."
            elif "timeout" in lower_msg:
                msg += "\n\nNetwork timeout. Try again or check your connection."

            raise RuntimeError(msg)

    # Get current commit hash
    stdout, stderr, rc = await _run_git("git", "rev-parse", "HEAD", cwd=str(repo_path))
    if rc != 0:
        # rev-parse failed — repo might be in a weird state
        logger.warning(f"git rev-parse failed: {stderr or stdout}")
        commit_hash = "unknown"
    else:
        commit_hash = stdout

    # Verify the clone actually has files (not empty or corrupted)
    has_files = any(True for _ in repo_path.rglob("*") if _.is_file() and not _.name.startswith("."))
    if not has_files:
        logger.error(f"Cloned repo appears empty: {repo_path}")
        shutil.rmtree(repo_path, ignore_errors=True)
        raise RuntimeError("Git clone succeeded but the repo directory is empty. The repository might be empty on GitHub.")

    # Enforce size limit — reject oversized repos to protect disk/CPU on shared hosts
    size_mb = _dir_size_mb(repo_path)
    if size_mb > config.MAX_REPO_SIZE_MB:
        logger.warning(f"Repo {repo_url} is {size_mb:.0f}MB, exceeds limit of {config.MAX_REPO_SIZE_MB}MB. Rejecting.")
        shutil.rmtree(repo_path, ignore_errors=True)
        raise RuntimeError(
            f"Repository is too large ({size_mb:.0f}MB, even shallow-cloned). "
            f"This deployment supports repos up to {config.MAX_REPO_SIZE_MB}MB. "
            f"Try a smaller repo or a specific subdirectory's repo."
        )

    # Final result — yield the tuple so the caller knows we're done
    yield (repo_path, commit_hash)


async def clone_repo(repo_url: str) -> Tuple[Path, str]:
    """Clone a repo (shallow) or update if already exists.
    Returns (repo_path, commit_hash). Raises on failure.

    Non-streaming wrapper around clone_repo_streaming for backward compat.
    """
    result = None
    async for item in clone_repo_streaming(repo_url):
        if isinstance(item, tuple):
            result = item
    if result is None:
        raise RuntimeError("clone_repo_streaming completed without yielding a result")
    return result


async def get_changed_files(repo_path: Path, old_hash: str, new_hash: str) -> list[str]:
    """Get files changed between two commits (for incremental re-indexing)."""
    stdout, _, rc = await _run_git(
        "git", "diff", "--name-only", old_hash, new_hash, cwd=str(repo_path)
    )
    if rc != 0:
        return []
    return [f for f in stdout.split("\n") if f]