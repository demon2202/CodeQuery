"""
Chat endpoint with dynamic starters and Mermaid diagram support.
"""

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import config
from ..models import ChatRequest, HealthResponse
from ..services.search import search_cached, search
from ..services.generator import generate_answer, check_ollama_health
from ..services.cloner import check_git_available, get_repo_path
from ..store.chroma_store import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Cache health check results to avoid hammering Ollama on every poll.
# Frontend polls every 3-10s, and each call hit /api/tags — very wasteful.
_health_cache = {"result": None, "timestamp": 0}
_HEALTH_CACHE_TTL = 15.0  # seconds


# ── File-pattern based starter questions ───────────────────────────────

_PATTERNS = [
    # (directory/file pattern, questions)
    ("auth", [
        "How does authentication work?",
        "What auth middleware is used?",
    ]),
    ("route", [
        "How are routes defined?",
        "What does the routing structure look like?",
    ]),
    ("model", [
        "What are the main data models?",
        "How is the database schema structured?",
    ]),
    ("config", [
        "What configuration options are available?",
        "How is the app configured?",
    ]),
    ("test", [
        "How is testing set up?",
        "What testing framework is used?",
    ]),
    ("middleware", [
        "What middleware is applied?",
        "How does the request pipeline work?",
    ]),
    ("api", [
        "What API endpoints are available?",
        "How is the API structured?",
    ]),
    ("util", [
        "What utility functions exist?",
        "What helper modules are available?",
    ]),
    ("component", [
        "What are the main UI components?",
        "How is the component hierarchy organized?",
    ]),
    ("service", [
        "What services does the app use?",
        "How are services structured?",
    ]),
    ("db", [
        "How is the database connected?",
        "What ORM or query layer is used?",
    ]),
    ("handler", [
        "How are requests handled?",
        "What are the main handler functions?",
    ]),
    ("app.", [
        "What does the main entry point do?",
        "How is the application initialized?",
    ]),
    ("main.", [
        "What does the main entry point do?",
        "How is the application bootstrapped?",
    ]),
    ("index.", [
        "What does the main entry point do?",
        "How is the app started?",
    ]),
]


def _generate_file_based_starters(repo_url: str) -> list[str]:
    """Generate starter questions based on the repo's file structure."""
    try:
        repo_path = get_repo_path(repo_url)
        if not repo_path.exists():
            return _default_starters()

        # Walk the repo to find directory/file names
        found_dirs = set()
        found_files = set()
        for item in repo_path.rglob("*"):
            if item.is_dir():
                found_dirs.add(item.name.lower())
            elif item.is_file():
                found_files.add(item.name.lower())

        all_names = found_dirs | {f.split(".")[0] for f in found_files}

        questions = []
        seen = set()
        for pattern, qs in _PATTERNS:
            # Check if any dir or file matches the pattern
            if any(pattern in name for name in all_names):
                for q in qs:
                    if q not in seen:
                        questions.append(q)
                        seen.add(q)
                        if len(questions) >= 4:
                            break
            if len(questions) >= 4:
                break

        # Always add a generic question
        if len(questions) < 3:
            questions.append("What does the main entry point do?")

        return questions[:5]

    except Exception:
        return _default_starters()


def _default_starters() -> list[str]:
    return [
        "What does the main entry point do?",
        "How are errors handled?",
        "What is the overall architecture?",
    ]


@router.get("/starters")
async def get_starters(repo_url: str = Query(..., description="Repository URL")):
    """Get dynamic starter questions for a repo.

    Returns file-based questions immediately. If Ollama is running,
    also returns LLM-generated questions (may be slower).
    """
    file_starters = _generate_file_based_starters(repo_url)

    result = {
        "starters": file_starters,
        "llm_starters": None,
    }

    # Try to get LLM-generated starters (fast — just one call)
    try:
        import httpx
        store = get_store()
        stats = store.get_collection_stats(repo_url)

        if stats["chunk_count"] > 0:
            # Get a sample of chunk metadata to understand the repo
            collection = store._get_collection(repo_url)
            sample = collection.peek(limit=20)
            metas = sample.get("metadatas", [])

            # Build a repo summary
            file_set = set()
            func_names = []
            class_names = []
            for m in metas:
                fp = m.get("file_path", "")
                if fp:
                    file_set.add(fp.split("/")[0] if "/" in fp else fp)
                name = m.get("name", "")
                if m.get("chunk_type") == "class" and name:
                    class_names.append(name)
                elif m.get("chunk_type") == "function" and name:
                    func_names.append(name)

            summary = f"Files: {', '.join(sorted(file_set)[:15])}"
            if class_names:
                summary += f"\nClasses: {', '.join(class_names[:10])}"
            if func_names:
                summary += f"\nFunctions: {', '.join(func_names[:15])}"

            prompt = f"""Based on this codebase summary, suggest exactly 4 specific, interesting questions someone would want to ask about this code. Make them specific to the actual code, not generic. One per line, no numbering, no quotes.

{summary}

Questions:"""

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{config.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 200},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "").strip()
                    llm_qs = [q.strip().lstrip("0123456789.-) ") for q in text.split("\n") if q.strip()]
                    llm_qs = [q for q in llm_qs if len(q) > 5 and len(q) < 150][:4]
                    if llm_qs:
                        result["llm_starters"] = llm_qs
    except Exception as e:
        logger.debug(f"LLM starters failed (non-critical): {e}")

    return result


@router.post("")
async def chat(request: ChatRequest):
    """Ask a question about an indexed repository. Returns SSE stream."""
    store = get_store()
    stats = store.get_collection_stats(request.repo_url)
    if stats["chunk_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Repo not indexed: {request.repo_url}. Index it first.",
        )

    async def event_stream():
        try:
            sources, documents, _ = search_cached(
                repo_url=request.repo_url,
                question=request.question,
            )

            if not sources:
                yield f"data: {json.dumps({'type': 'sources', 'chunks': []})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'answer': 'I could not find relevant code for that question. Try rephrasing or asking about a different part of the codebase.'})}\n\n"
                return

            async for event in generate_answer(
                question=request.question,
                sources=sources,
                documents=documents,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.exception("Chat error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    # Use cached result if fresh enough
    now = time.time()
    cached = _health_cache["result"]
    if cached and (now - _health_cache["timestamp"]) < _HEALTH_CACHE_TTL:
        return cached

    ollama_status = await check_ollama_health()
    git_ok = True
    try:
        await check_git_available()
    except RuntimeError:
        git_ok = False
    result = HealthResponse(
        status="healthy" if ollama_status.get("running") and git_ok else "degraded",
        ollama_running=ollama_status.get("running", False),
        ollama_model=config.OLLAMA_MODEL,
        model_available=ollama_status.get("model_available", False),
        embedding_model=config.EMBEDDING_MODEL,
        git_available=git_ok,
    )
    _health_cache["result"] = result
    _health_cache["timestamp"] = now
    return result


@router.get("/debug/search")
async def debug_search(
    repo_url: str = Query(..., description="Repository URL"),
    question: str = Query(..., description="Question to search for"),
):
    """Debug endpoint — shows raw search scores."""
    try:
        sources, documents, search_time = search(repo_url, question)
        return {
            "question": question,
            "repo_url": repo_url,
            "search_time": round(search_time, 3),
            "num_results": len(sources),
            "min_score_threshold": config.RETRIEVAL_MIN_SCORE,
            "top_k": config.RETRIEVAL_TOP_K,
            "results": [
                {
                    "file_path": s.file_path,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "name": s.name,
                    "chunk_type": s.chunk_type,
                    "score": round(s.score, 4),
                    "above_threshold": s.score >= config.RETRIEVAL_MIN_SCORE,
                    "content_preview": documents[i][:200] if i < len(documents) else "",
                }
                for i, s in enumerate(sources)
            ],
        }
    except Exception as e:
        return {"error": str(e)}
