"""
LLM answer generation service for CodeQuery.

Uses Ollama for local generation with qwen2.5-coder:7b.
Streams tokens from Ollama to the frontend via SSE.

The prompt engineering is critical — it forces the LLM to:
1. Only answer based on the provided code chunks (grounded answers)
2. Cite file:line_start-line_end for every claim
3. Admit when it can't find the answer instead of hallucinating

Key design decisions:
- Streaming: Ollama supports streaming natively. We forward tokens
  to the frontend immediately — this is the #1 perceived-speed win
  for chat UIs. Most student projects buffer the entire response.
- Warm model: We send a keep-alive request on startup so Ollama
  keeps the model loaded. Cold-start reload adds 5-15 seconds of
  latency that users notice.
- Citation format: We use `file_path:line_start-line_end` which
  matches the chunk metadata exactly, making citations clickable.
"""

import json
import logging
from typing import AsyncGenerator, List

import httpx

from .. import config
from ..models import ChunkSource

logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CodeQuery, an expert code analysis assistant. You answer questions about codebases using ONLY the provided code chunks.

CRITICAL RULES:
1. Base your answer ONLY on the provided code chunks. Do not invent, assume, or hallucinate code that isn't shown.
2. For EVERY factual claim, cite the source using this exact format: `file_path:line_start-line_end` (e.g., `src/auth.py:42-67`).
3. If the provided chunks don't contain enough information to answer the question fully, say what you CAN determine from the chunks, then explicitly say "I don't have enough context to determine X." Do NOT guess.
4. When explaining code, be specific — refer to function names, variable names, and line numbers from the chunks.
5. If multiple chunks are relevant, synthesize them into a coherent answer.

ANSWER STRUCTURE:
- Start with a direct, concise answer to the question
- Then provide details with code references
- Use **bold** for file names and important terms
- Use `backticks` for function/method/variable names and code references
- Use bullet points for lists
- Use numbered steps for processes/flows
- Use ```language code blocks for code examples

DIAGRAMS:
- When explaining architecture, data flow, class hierarchies, or component relationships, include a Mermaid diagram in a ```mermaid code block.
- Use flowchart for control flow, classDiagram for class hierarchies, sequenceDiagram for API call sequences.

CHARTS — USE THESE ACTIVELY:
When your answer involves comparisons, distributions, or measurements, include a chart. This is important — charts make answers much more useful.

Use a ```chart code block with a JSON object. Format:
```chart
{
  "type": "pie|bar|line|doughnut|radar|polarArea",
  "title": "Chart title",
  "labels": ["Label1", "Label2", ...],
  "datasets": [
    {"label": "Dataset name", "data": [10, 20, ...]}
  ]
}
```

When to use each chart type:
- **pie/doughnut**: Proportions — e.g., "What fraction of files are Python vs JS?" or "How is the codebase split by language?"
- **bar**: Comparisons — e.g., "How many functions per file?" or "Lines of code per module?"
- **line**: Trends or sequences — e.g., "How does function complexity grow across modules?"
- **radar**: Multi-dimensional comparison — e.g., "Compare modules on complexity, size, dependencies"

ALWAYS include a chart when:
- The user asks about repo structure, composition, or distribution
- You're comparing sizes/counts/proportions of different components
- You're showing complexity or metrics across files/modules

You can include BOTH a Mermaid diagram AND a chart in the same answer if appropriate.
"""

CHUNK_PROMPT_TEMPLATE = """Here are the relevant code chunks from the repository:

{chunks}

---

Question: {question}

Provide a clear, structured answer based on the code chunks above. Cite file:line for every claim. Include Mermaid diagrams for architecture and ```chart blocks for any comparisons, distributions, or metrics."""


def _format_chunks_for_prompt(sources: List[ChunkSource], documents: List[str]) -> str:
    """Format retrieved chunks into the prompt, with clear file:line markers."""
    parts = []
    for i, (source, doc) in enumerate(zip(sources, documents)):
        location = f"{source.file_path}:{source.start_line}-{source.end_line}"
        header = f"[{i+1}] {location}"
        if source.parent:
            header += f" (method {source.name} in class {source.parent})"
        elif source.chunk_type == "class":
            header += f" (class {source.name})"
        elif source.chunk_type == "function":
            header += f" (function {source.name})"
        elif source.chunk_type == "module":
            header += f" (module code)"
        else:
            header += f" ({source.chunk_type} {source.name})"
        
        parts.append(f"{header}\n```{source.language}\n{doc}\n```")
    
    return "\n\n".join(parts)


async def generate_answer(
    question: str,
    sources: List[ChunkSource],
    documents: List[str],
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
    prompt = CHUNK_PROMPT_TEMPLATE.format(chunks=chunks_text, question=question)
    
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
                        "temperature": 0.1,  # Low temp for factual, grounded answers
                        "num_predict": 4096,  # Need room for Mermaid + charts + detailed text
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
    """Check if Ollama is running and the model is available.
    
    Returns dict with status information. Used for the health endpoint.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            response = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return {
                    "running": False,
                    "model_available": False,
                    "error": f"Ollama returned status {response.status_code}",
                }
            
            # Check if our model is available
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Ollama model names may or may not include the tag
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
    """Send a warm-up request to Ollama so the model stays loaded.
    
    Ollama unloads models after a period of inactivity (default 5 minutes).
    Cold-start reload takes 5-15 seconds. By sending a lightweight request,
    we keep the model warm for faster first responses.
    
    The OLLAMA_KEEP_ALIVE env var (set on the Ollama side) controls how
    long the model stays loaded. We recommend setting it to at least 30m.
    """
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
