import json
import logging
from typing import AsyncGenerator, List

import httpx

from .. import config
from ..models import ChunkSource

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CodeQuery, a senior principal software architect and codebase analysis engine. You answer questions using ONLY the provided code chunks with utmost precision, crisp typography, and publication-grade technical diagrams.

## ABSOLUTE CONSTRAINTS

1. NO EMOJIS OR FLUFF: Never use emojis, decorative symbols, or AI buzzwords ("seamless", "elevate", "unleash", "delve"). Use clean, technical, minimal language.
2. CITATIONS: Ground every technical assertion with inline file:line citations: **`filepath:line-range`** (e.g., **`backend/app/main.py:15-30`**).
3. CODE SNIPPETS: Only show relevant lines with language tags and file headers. Never dump entire chunks verbatim.
4. HONESTY: If chunks do not contain enough information, state exactly what is missing and cite what is known.

## SLASH COMMAND PROTOCOLS

When the user query begins with a slash command, follow these specialized structures:

### `/grill-me [topic]`
1. State the core architectural dilemma, trade-off, or execution path for the given topic in 2-3 precise paragraphs.
2. Reference the exact files and functions governing this behavior.
3. ALWAYS conclude with an interactive ```options ``` block offering 3-4 distinct technical exploration forks.

### `/architecture`
1. Provide a concise 2-sentence system overview.
2. Output a DETAILED, clean Mermaid architecture diagram (`graph TD` or `graph LR`) using subgraphs for each layer (Client, API, Services, Database/Storage).
3. Walk through each subsystem with direct file citations.
4. Conclude with an ```options ``` block.

### `/dataflow [feature]`
1. Trace the end-to-end event or request flow.
2. Output a Mermaid `sequenceDiagram` with `autonumber`, clear participants, and activation lifecycles.
3. Conclude with an ```options ``` block.

### `/summary`
1. **Core Purpose**: 1-2 paragraphs summarizing the codebase.
2. **Key Entrypoints & Modules**: Bullet points with file citations.
3. **Execution Flow**: Step-by-step lifecycle.
4. Conclude with an ```options ``` block.

## REFINED MERMAID DIAGRAM RULES

Your Mermaid diagrams MUST render cleanly without syntax errors:
1. **ALWAYS enclose node labels in double quotes**: `NodeID["Clean Descriptive Title (file/path)"]`. Never use unquoted brackets or parentheses inside labels.
2. **Use Structured Subgraphs**: Group components logically (`subgraph ClientLayer ["Client Layer (frontend/src)"] ... end`).
3. **Edge Labels**: Use quotes on edges: `A -- "POST /api/chat" --> B`.
4. **Sequence Diagrams**: Use `sequenceDiagram`, `autonumber`, and explicit actor/participant names.

### Example Architecture:
```mermaid
graph TD
  subgraph Client ["Client Layer (frontend/src/)"]
    UI["App.jsx — State & Shell"]
    Chat["ChatInterface.jsx — Stream Consumer"]
  end
  subgraph Backend ["API Layer (backend/app/)"]
    Router["routers/chat.py — SSE Endpoint"]
    Indexer["services/indexer.py — Pipeline"]
  end
  subgraph Storage ["Storage Layer"]
    VectorDB["ChromaDB — Embeddings"]
    Disk["SQLite — Snapshots"]
  end

  UI --> Chat
  Chat -- "POST /api/chat" --> Router
  Router --> Indexer
  Indexer --> VectorDB
  Indexer --> Disk
```

## INTERACTIVE OPTIONS BLOCK (```options)

Whenever presenting next steps, architectural forks, or responding to `/grill-me`, append an interactive ```options block:

```options
{
  "question": "Select a component to inspect in detail:",
  "options": [
    "Inspect the authentication middleware and session token lifecycle",
    "Analyze the vector indexing and memory streaming pipeline",
    "Review error handling, retries, and network timeout fallbacks"
  ]
}
```

RULES:
- Valid JSON inside a ```options ... ``` code fence.
- "question" is a clean, direct prompt.
- "options" contains 2 to 4 actionable, specific technical inquiries.
"""


CHUNK_PROMPT_TEMPLATE = """Retrieved {count} code chunks from the repository:

{chunks}

---

{history_section}User Query: {question}

DIRECTIVES:
1. Synthesize information across all provided chunks.
2. Ground every claim with inline file:line citations: **`filepath:start-end`**.
3. For architectural or multi-step questions, include a clean, quote-wrapped Mermaid diagram.
4. Avoid emojis and generic fluff — keep tone crisp, technical, and minimal.
5. End with an interactive ```options ``` block offering 3-4 specific technical paths.

Answer:"""



def _format_history(history: list[dict] | None) -> str:
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
    file_set = {source.file_path for source in sources}
    parts = []

    key_counts: dict[tuple, int] = {}
    for s in sources:
        key = (s.file_path, s.name, s.chunk_type)
        key_counts[key] = key_counts.get(key, 0) + 1
    key_seen: dict[tuple, int] = {}

    if file_set:
        parts.append("FILES IN RETRIEVED CHUNKS:")
        for f in sorted(file_set):
            chunks_in_file = [s for s in sources if s.file_path == f]
            names = []
            seen_names = set()
            for c in chunks_in_file:
                if c.chunk_type == "class":
                    label = f"class {c.name}"
                elif c.chunk_type == "function":
                    label = f"fn {c.name}"
                elif c.chunk_type == "method":
                    parent = c.parent or ""
                    label = f"fn {parent}.{c.name}"
                elif c.chunk_type == "module":
                    label = "module-level"
                else:
                    label = f"{c.chunk_type} {c.name}"
                if label not in seen_names:
                    names.append(label)
                    seen_names.add(label)
            parts.append(f"  {f}: {', '.join(names)}")
        parts.append("")

    for i, (source, doc) in enumerate(zip(sources, documents)):
        location = f"{source.file_path}:{source.start_line}-{source.end_line}"
        key = (source.file_path, source.name, source.chunk_type)
        total_parts = key_counts[key]
        key_seen[key] = key_seen.get(key, 0) + 1
        part_num = key_seen[key]
        continuation_note = ""
        if total_parts > 1:
            continuation_note = f" — PART {part_num}/{total_parts} OF THE SAME {source.chunk_type.upper()}, NOT A SEPARATE ONE"

        if source.chunk_type == "class":
            header = f"[{i+1}] class {source.name} ({location}){continuation_note}"
        elif source.chunk_type == "method":
            header = f"[{i+1}] {source.parent}.{source.name}() ({location}){continuation_note}"
        elif source.chunk_type == "function":
            header = f"[{i+1}] {source.name}() ({location}){continuation_note}"
        elif source.chunk_type == "module":
            header = f"[{i+1}] module-level code ({location}){continuation_note}"
        else:
            header = f"[{i+1}] {source.name} ({location}){continuation_note}"

        parts.append(f"{header}\n```{source.language}\n{doc}\n```")

    return "\n\n".join(parts)


async def generate_answer(
    question: str,
    sources: List[ChunkSource],
    documents: List[str],
    history: list[dict] | None = None,
    unmatched_filenames: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """Generate an answer using Ollama or Groq, streaming tokens.

    Yields:
        {"type": "sources", "chunks": [...]}
        {"type": "token", "text": "..."}
        {"type": "done", "answer": "..."}
        {"type": "error", "message": "..."}
    """
    yield {"type": "sources", "chunks": [s.model_dump() for s in sources]}

    if not sources:
        yield {
            "type": "done",
            "answer": "I couldn't find this in the codebase. The retrieval didn't return any relevant code chunks for your question.",
        }
        return

    chunks_text = _format_chunks_for_prompt(sources, documents)
    history_section = _format_history(history)

    unmatched_note = ""
    if unmatched_filenames:
        unmatched_note = (
            f"\n\nIMPORTANT: the question mentions {', '.join(unmatched_filenames)}, but no file "
            f"matching that name exists anywhere in this indexed repository (confirmed by exact "
            f"filename search, not just semantic search). Do NOT describe what that file "
            f"probably contains based on general/training knowledge of typical projects. "
            f"Explicitly tell the user this file was not found in the indexed repo."
        )

    prompt = CHUNK_PROMPT_TEMPLATE.format(
        count=len(sources),
        chunks=chunks_text,
        question=question,
        history_section=history_section,
    ) + unmatched_note

    full_answer = []

    if config.LLM_PROVIDER == "groq" or (config.GROQ_API_KEY and config.LLM_PROVIDER != "ollama"):
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "stream": True,
        }
        # gpt-oss models burn hidden reasoning tokens unless told otherwise;
        # "low" saves free-tier TPM budget and cuts latency. Conditional because
        # non-gpt-oss models reject this param with a 400.
        if "gpt-oss" in config.GROQ_MODEL:
            payload["reasoning_effort"] = "low"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield {
                            "type": "error",
                            "message": f"Groq API error {response.status_code}: {error_body.decode()[:200]}",
                        }
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                delta = data["choices"][0]["delta"].get("content", "")
                                if delta:
                                    full_answer.append(delta)
                                    yield {"type": "token", "text": delta}
                            except Exception:
                                continue
            yield {"type": "done", "answer": "".join(full_answer)}
            return
        except Exception as e:
            yield {"type": "error", "message": f"Groq generation failed: {e}"}
            return

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
                    "options": {"temperature": 0.1, "num_predict": 4096},
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
        yield {"type": "error", "message": "Ollama is not running. Start it with: ollama serve"}
        return
    except httpx.TimeoutException:
        yield {"type": "error", "message": f"Ollama request timed out after {config.OLLAMA_TIMEOUT}s. The model may be loading — try again."}
        return
    except Exception as e:
        yield {"type": "error", "message": f"Generation failed: {e}"}
        return

    yield {"type": "done", "answer": "".join(full_answer)}


async def check_ollama_health() -> dict:
    """Check if LLM provider (Ollama or Groq) is ready."""
    if config.LLM_PROVIDER == "groq" or (config.GROQ_API_KEY and config.LLM_PROVIDER != "ollama"):
        if config.GROQ_API_KEY:
            return {"running": True, "model_available": True, "provider": "groq", "models": [config.GROQ_MODEL]}
        return {"running": False, "model_available": False, "error": "GROQ_API_KEY is not set"}

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

            return {"running": True, "model_available": model_available, "models": model_names}
    except httpx.ConnectError:
        return {"running": False, "model_available": False, "error": "Cannot connect to Ollama"}
    except Exception as e:
        return {"running": False, "model_available": False, "error": str(e)}


async def warm_up_model() -> None:
    """Warm up the LLM model."""
    if config.LLM_PROVIDER == "groq" or (config.GROQ_API_KEY and config.LLM_PROVIDER != "ollama"):
        logger.info(f"Using Groq API model: {config.GROQ_MODEL}")
        return

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


async def generate_simple(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    """Generate a non-streaming text response from Groq or Ollama."""
    if config.LLM_PROVIDER == "groq" or (config.GROQ_API_KEY and config.LLM_PROVIDER != "ollama"):
        headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if "gpt-oss" in config.GROQ_MODEL:
            payload["reasoning_effort"] = "low"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:200]}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": max_tokens},
            },
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:200]}")