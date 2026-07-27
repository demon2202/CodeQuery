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

    logger.info("Warming up embedding model...")
    try:
        warm_up_embedder()
        logger.info("Embedding model ready.")
    except Exception as e:
        logger.error(f"Failed to warm up embedding model: {e}")
        logger.error("The app will still start, but first query will be slow.")

    logger.info("Warming up Ollama model...")
    try:
        await warm_up_model()
        logger.info("Ollama model ready.")
    except Exception as e:
        logger.warning(f"Could not warm up Ollama model: {e}")

    yield

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
