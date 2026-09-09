import asyncio
import hashlib
import time
import logging
import os
import re
import shutil
from pathlib import Path
import subprocess
from typing import Tuple

from .. import config

logger = logging.getLogger(__name__)

def _get_git_exe() -> str:
    candidate = shutil.which("git")
    if candidate:
        return candidate
    fallbacks = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
    ]
    for fb in fallbacks:
        if os.path.isfile(fb):
            return fb
    return "git"


_SAFE_URL_RE = re.compile(
    r"^https?://(www\.)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$"
)


def force_remove(path: Path, attempts: int = 5) -> bool:
    """Recursively delete a path, surviving read-only/locked files.

    Plain shutil.rmtree(ignore_errors=True) silently leaves half-deleted git
    checkouts behind (Windows file locks, read-only packfiles). The next clone
    then AJAX fetches into a skeleton directory with a handful of dotfiles
    and indexing fails with "Repo has N files but none match". This helper
    chmods undeletable entries and retries; returns True if the path is gone.
    """
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, 0o700 if os.path.isdir(p) else 0o600)
            func(p)
        except Exception:
            pass

    for _ in range(attempts):
        if not path.exists():
            return True
        try:
            shutil.rmtree(path, onerror=_onerror)
        except Exception:
            pass
        time.sleep(0.3)
    if path.exists():
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
    return not path.exists()


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
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    safe_name = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else hashlib.sha256(url.encode()).hexdigest()[:16]
    return config.REPOS_DIR / safe_name


def _sync_run_git(
    args: list[str],
    cwd: str | None = None,
    timeout: float | None = None,
) -> Tuple[str, str, int]:
    git_exe = _get_git_exe()
    resolved_args = [git_exe] + list(args[1:])
    command_timeout = timeout or config.GIT_COMMAND_TIMEOUT
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        res = subprocess.run(
            resolved_args,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=max(1.0, command_timeout),
            errors="replace",
        )
        return (res.stdout.strip(), res.stderr.strip(), res.returncode)
    except subprocess.TimeoutExpired:
        return (
            "",
            f"git command timed out after {command_timeout:.0f}s: {' '.join(resolved_args[1:3])}",
            124,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"git not found at '{git_exe}'. "
            "Please install Git: https://git-scm.com/downloads"
        )
    except Exception as e:
        return ("", str(e), 1)


async def _run_git(
    *args: str,
    cwd: str | None = None,
    timeout: float | None = None,
) -> Tuple[str, str, int]:
    """Run a non-interactive git command safely using thread-backed subprocess."""
    return await asyncio.to_thread(_sync_run_git, list(args), cwd, timeout)


async def _run_git_with_heartbeats(
    *args: str,
    cwd: str | None = None,
    timeout: float | None = None,
    status: str = "",
):
    """Yield status heartbeats while a git command runs, then its result tuple."""
    task = asyncio.create_task(_run_git(*args, cwd=cwd, timeout=timeout))
    loop = asyncio.get_running_loop()
    started = loop.time()

    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=5.0)
            if task in done:
                yield task.result()
                return
            elapsed = int(loop.time() - started)
            yield f"{status} ({elapsed}s elapsed)..."
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _run_git_streaming(*args: str, cwd: str | None = None):
    """Run a git command streaming stderr line-by-line, then yields a final (returncode, stdout, stderr) tuple."""
    git_exe = _get_git_exe()
    resolved_args = [git_exe] + list(args[1:])
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    def _start_proc():
        return subprocess.Popen(
            resolved_args,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
        )

    try:
        proc = await asyncio.to_thread(_start_proc)
    except FileNotFoundError:
        raise RuntimeError(
            f"git not found at '{git_exe}'. "
            "Please install Git: https://git-scm.com/downloads"
        )

    stderr_chunks: list[str] = []

    def _read_stderr_line():
        return proc.stderr.readline()

    def _read_stdout():
        return proc.stdout.read()

    while True:
        line = await asyncio.to_thread(_read_stderr_line)
        if not line:
            break
        cleaned = line.strip()
        if cleaned:
            stderr_chunks.append(cleaned)
            yield cleaned

    stdout_text = await asyncio.to_thread(_read_stdout)
    await asyncio.to_thread(proc.wait)

    yield (proc.returncode or 0, stdout_text or "", "\n".join(stderr_chunks))


async def check_git_available() -> str:
    """Check if git is installed and return its version string. Raises if not found."""
    git_exe = _get_git_exe()
    stdout, stderr, rc = await _run_git("git", "--version")
    if rc == 0 and stdout:
        return stdout
    raise RuntimeError(
        f"git not working at '{git_exe}': {stderr or stdout or f'exit code {rc}'}"
    )


def _dir_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


async def clone_repo_streaming(repo_url: str):
    """Clone (shallow) or update an existing clone, streaming progress strings.

    Yields progress strings during the clone, then a final (repo_path, commit_hash) tuple.
    """
    repo_url = validate_repo_url(repo_url)
    repo_path = get_repo_path(repo_url)

    await check_git_available()

    if (repo_path / ".git").exists():
        # Check if directory has working files or if .git is corrupted
        non_git_files = sum(1 for p in repo_path.iterdir() if p.name != ".git")
        if non_git_files == 0:
            logger.info("Found .git without working tree files. Removing for fresh clone.")
            force_remove(repo_path)
        else:
            yield "Updating existing clone..."
            fetch_result = None
            async for item in _run_git_with_heartbeats(
                "git",
                "fetch",
                "--depth=1",
                "--quiet",
                cwd=str(repo_path),
                timeout=config.GIT_COMMAND_TIMEOUT,
                status="Still updating the existing clone",
            ):
                if isinstance(item, tuple):
                    fetch_result = item
                else:
                    yield item

            if fetch_result is None:
                raise RuntimeError("git fetch ended without a result")
            stdout, stderr, rc = fetch_result
            if rc == 0:
                _, reset_stderr, reset_rc = await _run_git(
                    "git",
                    "reset",
                    "--hard",
                    "FETCH_HEAD",
                    "--quiet",
                    cwd=str(repo_path),
                )
                if reset_rc != 0:
                    logger.warning(f"git reset failed: {reset_stderr}. Removing for fresh clone.")
                    force_remove(repo_path)
            else:
                logger.warning(f"git fetch failed for {repo_url}: {stderr or stdout}. Removing for fresh clone.")
            force_remove(repo_path)

    if not (repo_path / ".git").exists():
        if repo_path.exists():
            force_remove(repo_path)

        yield "Cloning repository (may take 30-60s for large repos)..."

        clone_result = None
        async for item in _run_git_with_heartbeats(
            "git",
            "clone",
            "--depth",
            "1",
            "--progress",
            repo_url,
            str(repo_path),
            timeout=config.GIT_CLONE_TIMEOUT,
            status="Still cloning from GitHub",
        ):
            if isinstance(item, tuple):
                clone_result = item
            else:
                yield item

        if clone_result is None:
            raise RuntimeError("git clone ended without a result")
        stdout, stderr, rc = clone_result

        if rc != 0:
            force_remove(repo_path)
            parts = ["git clone failed"]
            if stderr:
                parts.append(stderr)
            if stdout:
                parts.append(stdout)
            msg = ": ".join(parts) if len(parts) > 1 else parts[0]

            lower_msg = (stderr + " " + stdout).lower()
            if "could not resolve" in lower_msg or "network" in lower_msg:
                msg += "\n\nPossible causes:\n- No internet connection\n- GitHub is unreachable\n- URL is incorrect"
            elif "repository not found" in lower_msg or "not found" in lower_msg:
                msg += "\n\nThe repository doesn't exist or is private. Only public repos are supported."
            elif "permission denied" in lower_msg:
                msg += "\n\nPermission denied. Check if the repo directory is accessible."
            elif "timeout" in lower_msg or "timed out" in lower_msg:
                msg += "\n\nNetwork timeout. Try again or check your connection."

            raise RuntimeError(msg)

        yield "Clone complete."

    stdout, stderr, rc = await _run_git("git", "rev-parse", "HEAD", cwd=str(repo_path))
    if rc != 0:
        logger.warning(f"git rev-parse failed: {stderr or stdout}")
        commit_hash = "unknown"
    else:
        commit_hash = stdout

    has_files = any(True for _ in repo_path.rglob("*") if _.is_file() and not _.name.startswith("."))
    if not has_files:
        logger.error(f"Cloned repo appears empty: {repo_path}")
        force_remove(repo_path)
        raise RuntimeError("Git clone succeeded but the repo directory is empty.")

    size_mb = _dir_size_mb(repo_path)
    if size_mb > config.MAX_REPO_SIZE_MB:
        logger.warning(f"Repo {repo_url} is {size_mb:.0f}MB, exceeds limit of {config.MAX_REPO_SIZE_MB}MB")
        force_remove(repo_path)
        raise RuntimeError(
            f"Repository is too large ({size_mb:.0f}MB, even shallow-cloned). "
            f"This deployment supports repos up to {config.MAX_REPO_SIZE_MB}MB."
        )

    yield (repo_path, commit_hash)


async def clone_repo(repo_url: str) -> Tuple[Path, str]:
    """Non-streaming wrapper around clone_repo_streaming."""
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