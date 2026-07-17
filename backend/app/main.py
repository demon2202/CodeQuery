"""
CodeQuery — FastAPI application entry point.

Sets up the FastAPI app with:
- API routers for repos and chat
- CORS for the React frontend
- Gzip compression for API responses
- Startup/shutdown lifecycle for model warm-up
- Proper async handling (no blocking the event loop)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import config
from .routers import repo, chat
from .services.embedder import warm_up as warm_up_embedder
from .services.generator import warm_up_model
from .services.cloner import check_git_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup and shutdown."""
    logger.info("CodeQuery starting up...")
    logger.info(f"Ollama model: {config.OLLAMA_MODEL}")
    logger.info(f"Embedding provider: {config.EMBEDDING_PROVIDER}")
    if config.EMBEDDING_PROVIDER == "ollama":
        logger.info(f"Embedding model: {config.OLLAMA_EMBED_MODEL} (via Ollama)")
    else:
        logger.info(f"Embedding model: {config.EMBEDDING_MODEL} (sentence-transformers)")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"ChromaDB directory: {config.CHROMA_DIR}")

    # Check git is available (needed for cloning repos)
    logger.info("Checking git availability...")
    try:
        git_version = await check_git_available()
        logger.info(f"Git found: {git_version}")
    except RuntimeError as e:
        logger.error(f"Git check failed: {e}")
        logger.error("You MUST install Git to use CodeQuery: https://git-scm.com/downloads")

    # Warm up the embedding model (loads into memory)
    logger.info("Warming up embedding model...")
    try:
        warm_up_embedder()
        logger.info("Embedding model ready.")
    except Exception as e:
        logger.error(f"Failed to warm up embedding model: {e}")
        logger.error("The app will still start, but first query will be slow.")

    # Warm up the Ollama model (keeps it loaded in memory)
    # This can take 30-60s on first load (model needs to be loaded into GPU memory)
    logger.info("Warming up Ollama model (this may take 30-60s on first load)...")
    try:
        await warm_up_model()
        logger.info("Ollama model ready.")
    except Exception as e:
        logger.warning(f"Could not warm up Ollama model: {e}")
        logger.warning("First chat response will be slow while the model loads into GPU memory.")
        logger.warning("Make sure Ollama is running: ollama serve")
        logger.warning(f"Make sure the model is pulled: ollama pull {config.OLLAMA_MODEL}")

    yield

    logger.info("CodeQuery shutting down.")


app = FastAPI(
    title="CodeQuery",
    description="Local RAG over GitHub repos — ask natural-language questions about any codebase",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────

# CORS — allow the React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Common React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression — reduces API response size by ~70% for JSON
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(repo.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint — basic API info."""
    return {
        "name": "CodeQuery",
        "version": "1.0.0",
        "description": "Local RAG over GitHub repos",
        "docs": "/docs",
    }
