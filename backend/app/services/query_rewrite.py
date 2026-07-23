"""
Query rewriting for retrieval — separate from the question shown to the LLM
for answer generation.

Problem this solves: a follow-up question like "where's the token refreshed"
after "explain the auth flow" has zero shared vocabulary with the auth code
if searched in isolation. The user's intent depends on the previous turn, but
search() only ever saw the raw current-turn text.

Two strategies, independently toggleable:
1. Heuristic contextual query (cheap, on by default): folds the previous
   user question into the search string.
2. LLM query rewrite (costs one extra round-trip, off by default): asks the
   LLM to produce a standalone search query given the full history.
"""

import logging

import httpx

from .. import config

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions into standalone search queries for a code "
    "search engine. Given the recent conversation and a new question, output ONLY "
    "the rewritten standalone query — no preamble, no explanation, no quotes. "
    "If the question is already standalone, output it unchanged. Keep it short: "
    "a phrase or single sentence, not a full explanation."
)


def contextual_query(question: str, history: list[dict] | None) -> str:
    """Cheap heuristic: prepend the previous user question for extra context.

    This changes only the text used for retrieval (embedding + BM25), never
    the text shown to the LLM for the final answer.
    """
    if not config.CONTEXTUAL_SEARCH_ENABLED or not history:
        return question

    last_question = history[-1].get("question", "").strip()
    if not last_question or last_question.lower() == question.lower():
        return question

    return f"{last_question} {question}"


async def llm_rewrite_query(question: str, history: list[dict] | None) -> str:
    """Ask the LLM to produce a standalone search query from question + history.

    Falls back to the heuristic contextual_query() (or the raw question) on
    any failure — this must never block or break search.
    """
    if not config.QUERY_REWRITE_LLM_ENABLED or not history:
        return contextual_query(question, history)

    recent = history[-3:]
    convo = "\n".join(
        f"Q: {h.get('question', '')}\nA: {h.get('answer', '')[:200]}" for h in recent
    )
    prompt = f"Recent conversation:\n{convo}\n\nNew question: {question}\n\nStandalone search query:"

    try:
        if config.LLM_PROVIDER == "groq" or (config.GROQ_API_KEY and config.LLM_PROVIDER != "ollama"):
            headers = {
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 60,
                "stream": False,
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip()
                    if text:
                        return text
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{config.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "prompt": prompt,
                        "system": _REWRITE_SYSTEM_PROMPT,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 60},
                    },
                )
                if resp.status_code == 200:
                    text = resp.json().get("response", "").strip()
                    if text:
                        return text
    except Exception as e:
        logger.debug(f"LLM query rewrite failed (non-critical): {e}")

    return contextual_query(question, history)
