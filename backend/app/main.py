import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import config
from .routers import repo, chat
from .services.embedder import warm_up as warm_up_embedder
from .services.generator import warm_up_model
from .services.cloner import check_git_available
from .services.ollama_manager import ensure_ollama_running

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CodeQuery starting up...")
    logger.info(f"Ollama model: {config.OLLAMA_MODEL}")
    logger.info(f"Embedding provider: {config.EMBEDDING_PROVIDER}")
    if config.EMBEDDING_PROVIDER == "ollama":
        logger.info(f"Embedding model: {config.OLLAMA_EMBED_MODEL} (via Ollama)")
    else:
        logger.info(f"Embedding model: {config.EMBEDDING_MODEL} (sentence-transformers)")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"ChromaDB directory: {config.CHROMA_DIR}")

    logger.info("Checking git availability...")
    try:
        git_version = await check_git_available()
        logger.info(f"Git found: {git_version}")
    except RuntimeError as e:
        logger.error(f"Git check failed: {e}")

    # If a configured provider needs a local Ollama API, start it automatically
    # before checking the embedding model. Remote Ollama URLs are never launched.
    await ensure_ollama_running()

    logger.info("Warming up embedding model...")
    try:
        warm_up_embedder()
        logger.info("Embedding model ready.")
    except Exception as e:
        logger.error(f"Failed to warm up embedding model: {e}")
        logger.error("The app will still start, but first query will be slow.")

    llm_warmup_task = None
    if config.LLM_PROVIDER.lower().strip() == "ollama":
        if config.OLLAMA_WARMUP_ON_STARTUP:
            # Do not keep the API offline for minutes while a large model loads.
            # The background task warms it after Uvicorn begins accepting traffic.
            logger.info("Warming up Ollama LLM in the background...")
            llm_warmup_task = asyncio.create_task(warm_up_model())
        else:
            logger.info(
                "Ollama LLM startup warm-up skipped; the model will load on the first chat request."
            )
    else:
        # Groq warm-up only validates configuration and returns immediately.
        await warm_up_model()

    yield

    if llm_warmup_task is not None and not llm_warmup_task.done():
        llm_warmup_task.cancel()
        try:
            await llm_warmup_task
        except asyncio.CancelledError:
            pass

    logger.info("CodeQuery shutting down.")


app = FastAPI(
    title="CodeQuery",
    description="Local RAG over GitHub repos — ask natural-language questions about any codebase",
    version="1.0.0",
    lifespan=lifespan,
)

_default_origins = (
    "http://localhost:5173,"
    "http://localhost:3000,"
    "http://127.0.0.1:5173,"
    "http://127.0.0.1:3000"
)
_raw_origins = os.getenv("ALLOWED_ORIGINS", _default_origins)
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in _raw_origins.split(",") if o.strip()]
logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(repo.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "name": "CodeQuery",
        "version": "1.0.0",
        "description": "Local RAG over GitHub repos",
        "docs": "/docs",
    }
