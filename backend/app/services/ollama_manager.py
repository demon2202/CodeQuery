"""Start a local Ollama server when CodeQuery needs one and it is offline."""

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .. import config

logger = logging.getLogger(__name__)

_started_process: asyncio.subprocess.Process | None = None


def _is_local_ollama_url() -> bool:
    """Return True only for loopback Ollama URLs that are safe to auto-start."""
    try:
        host = (urlparse(config.OLLAMA_BASE_URL).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


def _find_ollama_executable() -> str | None:
    """Find Ollama on PATH or in its common Windows installation folders."""
    found = shutil.which("ollama")
    if found:
        return found

    candidates: list[Path] = []
    local_app_data = os.getenv("LOCALAPPDATA")
    program_files = os.getenv("ProgramFiles")
    if local_app_data:
        candidates.extend(
            [
                Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe",
                Path(local_app_data) / "Ollama" / "ollama.exe",
            ]
        )
    if program_files:
        candidates.append(Path(program_files) / "Ollama" / "ollama.exe")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


async def _ollama_api_is_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


async def ensure_ollama_running() -> bool:
    """Ensure a required local Ollama API is running.

    Returns True when the API is reachable. It never launches a process for a
    non-loopback CQ_OLLAMA_URL, and it does nothing when neither configured
    provider uses Ollama.
    """
    needs_ollama = (
        config.LLM_PROVIDER.lower().strip() == "ollama"
        or config.EMBEDDING_PROVIDER.lower().strip() == "ollama"
    )
    if not needs_ollama:
        return False

    if await _ollama_api_is_ready():
        logger.info("Ollama API is already running.")
        return True

    if not _is_local_ollama_url():
        logger.warning(
            "Ollama is unreachable at %s. Auto-start is restricted to localhost.",
            config.OLLAMA_BASE_URL,
        )
        return False

    if not config.OLLAMA_AUTO_START:
        logger.warning("Ollama is offline and CQ_OLLAMA_AUTO_START is disabled.")
        return False

    executable = _find_ollama_executable()
    if not executable:
        logger.warning(
            "Ollama is offline and its executable was not found. Install Ollama "
            "or add 'ollama' to PATH."
        )
        return False

    global _started_process
    logger.info("Ollama is offline; starting: %s serve", executable)

    creation_kwargs = {}
    if os.name == "nt":
        # Keep Ollama from opening an extra console window on Windows.
        creation_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        _started_process = await asyncio.create_subprocess_exec(
            executable,
            "serve",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **creation_kwargs,
        )
    except (OSError, ValueError) as exc:
        logger.warning("Could not start Ollama: %s", exc)
        return False

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(1, config.OLLAMA_START_TIMEOUT)
    while loop.time() < deadline:
        if await _ollama_api_is_ready():
            logger.info("Ollama API started successfully.")
            return True
        if _started_process.returncode is not None:
            logger.warning(
                "The Ollama process exited before its API became ready (code %s).",
                _started_process.returncode,
            )
            return False
        await asyncio.sleep(0.5)

    logger.warning(
        "Ollama did not become ready within %ss. CodeQuery will continue and "
        "FastEmbed may be used as the embedding fallback.",
        config.OLLAMA_START_TIMEOUT,
    )
    return False
