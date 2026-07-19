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
import re
import shutil
from pathlib import Path
from typing import Tuple

from .. import config

logger = logging.getLogger(__name__)

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
    # On Windows, git might not be found without shell=True in some edge cases,
    # but shell=True is a security risk. Instead, try 'git' directly first.
    # If that fails with FileNotFoundError, we know git isn't installed.
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode().strip(), stderr.decode().strip(), proc.returncode or 0
    except FileNotFoundError:
        raise RuntimeError(
            "git is not installed or not in PATH. "
            "Please install Git: https://git-scm.com/downloads"
        )


async def check_git_available() -> str:
    """Check if git is installed and return its version. Raises if not found."""
    try:
        stdout, _, rc = await _run_git("git", "--version")
        if rc == 0:
            return stdout
        raise RuntimeError("git command failed")
    except RuntimeError as e:
        if "not installed" in str(e):
            raise
        raise RuntimeError(
            "git is not installed or not in PATH. "
            "Please install Git: https://git-scm.com/downloads"
        )


async def clone_repo(repo_url: str) -> Tuple[Path, str]:
    """Clone a repo (shallow) or update if already exists.
    Returns (repo_path, commit_hash). Raises on failure.
    """
    repo_url = validate_repo_url(repo_url)
    repo_path = get_repo_path(repo_url)

    # Check git is available before doing anything
    await check_git_available()

    if (repo_path / ".git").exists():
        # Update existing clone — fetch + reset instead of pull
        # (shallow clones often can't `git pull` due to missing tracking info)
        stdout, stderr, rc = await _run_git("git", "fetch", "--depth=1", "--quiet", cwd=str(repo_path))
        if rc == 0:
            await _run_git("git", "reset", "--hard", "FETCH_HEAD", "--quiet", cwd=str(repo_path))
        else:
            # Fetch failed — log it but keep existing clone usable
            logger.warning(f"git fetch failed for {repo_url}: {stderr or stdout}")
            # If the repo directory exists but is corrupted, try a fresh clone
            if not (repo_path / ".git" / "HEAD").exists():
                logger.warning(f"Repo appears corrupted, removing for fresh clone")
                shutil.rmtree(repo_path, ignore_errors=True)
                # Fall through to fresh clone below
            else:
                # Keep existing clone — it's still usable
                pass
    if not (repo_path / ".git").exists():
        # Fresh shallow clone
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

        stdout, stderr, rc = await _run_git(
            "git", "clone", "--depth", "1", repo_url, str(repo_path)
        )
        if rc != 0:
            shutil.rmtree(repo_path, ignore_errors=True)
            # Build a helpful error message
            parts = ["git clone failed"]
            if stderr:
                parts.append(stderr)
            if stdout:
                parts.append(stdout)
            msg = ": ".join(parts) if len(parts) > 1 else parts[0]

            # Add helpful hints for common failures
            lower_msg = (stderr + " " + stdout).lower()
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

    return repo_path, commit_hash


async def get_changed_files(repo_path: Path, old_hash: str, new_hash: str) -> list[str]:
    """Get files changed between two commits (for incremental re-indexing)."""
    stdout, _, rc = await _run_git(
        "git", "diff", "--name-only", old_hash, new_hash, cwd=str(repo_path)
    )
    if rc != 0:
        return []
    return [f for f in stdout.split("\n") if f]
