"""
LLM answer generation service for CodeQuery.

Uses Ollama for local generation with qwen2.5-coder:7b.
Streams tokens from Ollama to the frontend via SSE.

The prompt engineering is the most critical part of this service.
Key design decisions:
1. System prompt enforces structured, grounded answers with proper citations
2. Chunks are formatted with clear headers so the LLM knows what each chunk is
3. Citation format is strict: one citation per claim, inline in the text
4. Mermaid diagrams must be detailed and specific (not generic 4-box diagrams)
5. Chart JSON must be valid — strict format with example provided
6. The LLM is told to ANALYZE all chunks first, then answer — not just dump the first one
"""

import json
import logging
from typing import AsyncGenerator, List

import httpx

from .. import config
from ..models import ChunkSource

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CodeQuery, an expert code analyst. You answer questions about codebases using ONLY the provided code chunks. You are thorough, specific, and produce well-structured answers with detailed diagrams.

## ABSOLUTE RULES

1. ONLY use information from the provided code chunks. Never invent code, files, or logic.
2. If chunks don't fully answer the question, state what you CAN determine, then say "I don't have enough context to determine [specific thing]."
3. NEVER dump all citations at the top of your answer. Integrate citations naturally inline where they're relevant.
4. NEVER dump an entire code chunk verbatim. Extract and show only the relevant lines/snippets.

## CITATION FORMAT

Cite sources inline using this format: **`filepath:line-range`** — for example:
- "The Express server is initialized in **`server/index.js:15-30`**"
- "The `handleLogin` function at **`src/auth.js:42-67`** validates credentials"

One citation per claim. Don't stack citations together. Don't list all sources at the top.

## ANSWER STRUCTURE

For OVERVIEW/ARCHITECTURE questions:
1. Start with a 2-3 sentence summary of what the project does
2. Describe the tech stack with file citations
3. Explain the architecture layer by layer (frontend → API → database)
4. Include a DETAILED Mermaid diagram showing all major components and their relationships
5. If relevant, include a chart showing file/module distribution

For CODE-LEVEL questions ("how does X work?", "where is Y handled?"):
1. Direct answer with the key file and function
2. Walk through the logic step by step with line citations
3. Show only the relevant code snippets (not entire chunks)
4. Include a sequence diagram if there's a multi-step flow

For COMPARISON questions:
1. Compare side by side
2. Include a chart with real data from the chunks
3. Cite specific differences

## CODE SNIPPETS

When showing code:
- ONLY show the relevant lines, not the entire chunk
- Use ```language code blocks
- Add a comment showing the file:line source
- Example:

```javascript
// src/auth.js:45-52
const token = jwt.sign(
  { userId: user._id, role: user.role },
  process.env.JWT_SECRET,
  { expiresIn: '7d' }
);
```

## MERMAID DIAGRAMS

Your Mermaid diagrams MUST be DETAILED and SPECIFIC:
- Include ACTUAL file names, function names, class names from the code
- Show ALL major components, not just 3-4 boxes
- Use specific labels, not generic ones like "Client" or "Server"
- For flowcharts: show the actual function names on edges
- For class diagrams: show actual methods and their return types
- For sequence diagrams: show actual API endpoints and function calls

BAD (too generic):
```mermaid
graph TD
  A[Client] --> B[Server]
  B --> C[Database]
```

GOOD (detailed and specific):
```mermaid
graph TD
  subgraph Frontend ["React Client (client/src/)"]
    App["App.js — Router"]
    Terr["Territories.js — Map + Socket"]
    LB["Leaderboard.js — Rankings"]
    Settings["Settings.js — User prefs"]
  end
  subgraph Backend ["Express Server (server/)"]
    AuthMW["authMiddleware — JWT verify"]
    TerrRoutes["routes/territory.js — CRUD + capture"]
    SocketHandler["Socket.IO — cellCaptured, activityCreated"]
  end
  subgraph Database ["MongoDB"]
    UserCol["users collection"]
    TerrCol["territories collection"]
    ActCol["activities collection"]
  end
  App --> Terr
  App --> LB
  App --> Settings
  Terr -- "Socket.IO events" --> SocketHandler
  Terr -- "REST /api/territories" --> TerrRoutes
  TerrRoutes --> AuthMW
  TerrRoutes --> TerrCol
  SocketHandler --> ActCol
  AuthMW --> UserCol
```

## CHARTS

When including a chart, you MUST use a ```chart code block with VALID JSON. No YAML, no Python syntax — only valid JSON.

Format:
```chart
{"type": "pie", "title": "Language Distribution", "labels": ["JavaScript", "CSS", "JSON"], "datasets": [{"label": "Files", "data": [12, 5, 3]}]}
```

RULES:
- The entire chart spec must be a SINGLE valid JSON object
- "type" must be one of: pie, bar, line, doughnut, radar, polarArea
- "labels" is an array of strings
- "datasets" is an array of objects with "label" (string) and "data" (array of numbers)
- NEVER use YAML, Python dicts, or any non-JSON syntax
- KEEP IT SIMPLE — one chart per answer, compact JSON on as few lines as possible

When to include charts:
- Repo structure/composition questions → pie or doughnut chart
- Module/file size comparisons → bar chart
- Counting things (functions per file, routes per module) → bar chart
"""


CHUNK_PROMPT_TEMPLATE = """You have retrieved {count} code chunks from the repository. Study ALL of them carefully before answering.

{chunks}

---

{history_section}Question: {question}

INSTRUCTIONS:
1. Read ALL chunks above — don't just use the first one
2. Synthesize information across chunks to give a complete answer
3. Cite specific file:line ranges inline as you make each claim
4. Show only relevant code snippets, not entire chunks
5. If explaining architecture or structure, include a DETAILED Mermaid diagram with actual file/function names
6. If showing distributions or comparisons, include a chart with valid JSON

Answer:"""


def _format_history(history: list[dict] | None) -> str:
    """Format conversation history for the prompt."""
    if not history:
        return ""
    recent = history[-4:]
    parts = []
    for h in recent:
        q = h.get("question", "")
        a = h.get("answer", "")
        if q and a:
            if len(a) > 500:
                a = a[:500] + "..."
            parts.append(f"Previous Q: {q}\nPrevious A: {a}")
    if not parts:
        return ""
    return "CONVERSATION HISTORY (use this for follow-up context):\n" + "\n".join(parts) + "\n\n"


def _format_chunks_for_prompt(sources: List[ChunkSource], documents: List[str]) -> str:
    """Format retrieved chunks into the prompt with clear, structured headers."""
    # Build a file summary first so the LLM has context
    file_set = set()
    for source in sources:
        file_set.add(source.file_path)

    parts = []

    # File overview — helps the LLM understand the repo structure
    if file_set:
        sorted_files = sorted(file_set)
        parts.append("FILES IN RETRIEVED CHUNKS:")
        for f in sorted_files:
            chunks_in_file = [s for s in sources if s.file_path == f]
            names = []
            for c in chunks_in_file:
                if c.chunk_type == "class":
                    names.append(f"class {c.name}")
                elif c.chunk_type == "function":
                    names.append(f"fn {c.name}")
                elif c.chunk_type == "method":
                    parent = c.parent or ""
                    names.append(f"fn {parent}.{c.name}")
                elif c.chunk_type == "module":
                    names.append("module-level")
                else:
                    names.append(f"{c.chunk_type} {c.name}")
            parts.append(f"  {f}: {', '.join(names)}")
        parts.append("")

    # Individual chunks with clear headers
    for i, (source, doc) in enumerate(zip(sources, documents)):
        location = f"{source.file_path}:{source.start_line}-{source.end_line}"

        if source.chunk_type == "class":
            header = f"[{i+1}] class {source.name} ({location})"
        elif source.chunk_type == "method":
            header = f"[{i+1}] {source.parent}.{source.name}() ({location})"
        elif source.chunk_type == "function":
            header = f"[{i+1}] {source.name}() ({location})"
        elif source.chunk_type == "module":
            header = f"[{i+1}] module-level code ({location})"
        else:
            header = f"[{i+1}] {source.name} ({location})"

        parts.append(f"{header}\n```{source.language}\n{doc}\n```")

    return "\n\n".join(parts)


async def generate_answer(
    question: str,
    sources: List[ChunkSource],
    documents: List[str],
    history: list[dict] | None = None,
) -> AsyncGenerator[dict, None]:
    """Generate an answer using Ollama, streaming tokens.

    Yields:
        {"type": "sources", "chunks": [...]} — the retrieved source chunks
        {"type": "token", "text": "..."} — streamed answer tokens
        {"type": "done", "answer": "..."} — complete answer
        {"type": "error", "message": "..."} — if generation fails
    """

    # Send sources first so the UI can show citations immediately
    yield {
        "type": "sources",
        "chunks": [s.model_dump() for s in sources],
    }

    # If no sources or all below threshold, skip LLM call
    if not sources:
        yield {
            "type": "done",
            "answer": "I couldn't find this in the codebase. The retrieval didn't return any relevant code chunks for your question.",
        }
        return

    # Build the prompt
    chunks_text = _format_chunks_for_prompt(sources, documents)
    history_section = _format_history(history)
    prompt = CHUNK_PROMPT_TEMPLATE.format(
        count=len(sources),
        chunks=chunks_text,
        question=question,
        history_section=history_section,
    )

    # Call Ollama streaming API
    full_answer = []

    try:
        async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 4096,
                    },
                },
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    yield {
                        "type": "error",
                        "message": f"Ollama returned status {response.status_code}: {error_body.decode()[:200]}",
                    }
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("done"):
                        break

                    token = data.get("response", "")
                    if token:
                        full_answer.append(token)
                        yield {"type": "token", "text": token}

    except httpx.ConnectError:
        yield {
            "type": "error",
            "message": "Ollama is not running. Start it with: ollama serve",
        }
        return
    except httpx.TimeoutException:
        yield {
            "type": "error",
            "message": f"Ollama request timed out after {config.OLLAMA_TIMEOUT}s. The model may be loading — try again.",
        }
        return
    except Exception as e:
        yield {
            "type": "error",
            "message": f"Generation failed: {e}",
        }
        return

    yield {"type": "done", "answer": "".join(full_answer)}


async def check_ollama_health() -> dict:
    """Check if Ollama is running and the model is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return {
                    "running": False,
                    "model_available": False,
                    "error": f"Ollama returned status {response.status_code}",
                }

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            model_available = any(
                config.OLLAMA_MODEL in name or name.startswith(config.OLLAMA_MODEL.split(":")[0])
                for name in model_names
            )

            return {
                "running": True,
                "model_available": model_available,
                "models": model_names,
            }
    except httpx.ConnectError:
        return {
            "running": False,
            "model_available": False,
            "error": "Cannot connect to Ollama",
        }
    except Exception as e:
        return {
            "running": False,
            "model_available": False,
            "error": str(e),
        }


async def warm_up_model() -> None:
    """Send a warm-up request to Ollama so the model stays loaded."""
    try:
        async with httpx.AsyncClient(timeout=config.OLLAMA_WARMUP_TIMEOUT) as client:
            await client.post(
                f"{config.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": "ready",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            logger.info(f"Ollama model {config.OLLAMA_MODEL} warmed up")
    except Exception as e:
        logger.warning(f"Failed to warm up Ollama model: {e}")
