"""
Deterministic dependency graph — built at index time from ACTUAL import
statements in the parsed files (not LLM guesses).

Pipeline:
    extract_imports(text, language)        -> raw import specifiers per file
    build_dependency_graph(imports_of, all_files, lang_map)
        -> {"imports_of", "edges", "hub_files", "num_files", "num_edges"}
    graph_context_text(graph)              -> compressed adjacency for LLM prompts
    graph_to_mermaid(edges)                -> exact Mermaid `graph TD` source

Resolving import specifiers to in-repo file paths is heuristic by design:
* python: dotted modules (+ relative-dot handling) -> a/b/c.py, a/b/c/__init__.py,
  with a suffix match so `app.services.x` still resolves in repos where the
  package root is nested (e.g. `backend/app/services/x.py`).
* js/ts:  ./, ../ relative specifiers -> normalization + extension probing
  (.js .jsx .ts .tsx .mjs .cjs, + /index.*).
Anything that doesn't resolve to a file inside the repo is treated as an
external dependency and dropped (external deps add noise, not architecture).
"""

import logging
import re
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

GRAPH_LANGS = {"python", "javascript", "typescript"}

_PY_FROM_RE = re.compile(r"^\s*from\s+([\.\w]+)\s+import\s+([\w\*\s,\(\)]+)", re.MULTILINE)
_PY_IMPORT_RE = re.compile(r"^\s*import\s+([\.\w]+)", re.MULTILINE)
_JS_FROM_RE = re.compile(r"""(?:import|export)\s+(?:[\w\*\s{},]*?\s+from\s+)?['"]([^'"]+)['"]""")
_JS_REQUIRE_RE = re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""")

_JS_EXTS = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]


def extract_imports(text: str, language: str) -> list[str]:
    """Return raw import specifiers found in a source file."""
    if not text:
        return []
    specs: list[str] = []
    try:
        if language == "python":
            for m in _PY_FROM_RE.finditer(text):
                base = m.group(1)
                # For "from pkg import mod" also record base.mod — the base
                # alone often only resolves to an __init__ we don't care about.
                specs.append(base)
                names = [n.strip() for n in re.split(r"[,\s]+", m.group(2)) if n and n.strip() != "*"]
                for name in names[:8]:
                    specs.append(f"{base}.{name}")
            for m in _PY_IMPORT_RE.finditer(text):
                specs.append(m.group(1))
        elif language in ("javascript", "typescript"):
            specs.extend(_JS_FROM_RE.findall(text))
            specs.extend(_JS_REQUIRE_RE.findall(text))
    except Exception:
        # Import extraction must never break indexing.
        return specs
    return specs


def _resolve_python(spec: str, src: str, file_set: set[str]) -> Optional[str]:
    src_dir = PurePosixPath(src).parent

    # Relative import: ".x" / "..x" — leading dots climb package levels.
    if spec.startswith("."):
        dots = len(spec) - len(spec.lstrip("."))
        rest = spec.lstrip(".")
        target_dir = src_dir
        for _ in range(dots - 1):
            target_dir = target_dir.parent
        prefix = "" if str(target_dir) in (".", "") else str(target_dir).replace("/", ".") + "."
        spec = f"{prefix}{rest}"

    if not spec:
        return None

    path = spec.replace(".", "/")
    candidates = [path + ".py", path + "/__init__.py"]
    for c in candidates:
        if c in file_set:
            return c
    # Package root may be nested (backend/app/..., src/...): suffix match,
    # prefer the shortest (least nested) hit and skip obvious tests of ours.
    for c in candidates:
        hits = [f for f in file_set if f.endswith(c) or f.endswith("/" + c)]
        if hits:
            return min(hits, key=len)
    return None


def _resolve_js(spec: str, src: str, file_set: set[str]) -> Optional[str]:
    if not (spec.startswith("./") or spec.startswith("../") or spec.startswith("/")):
        return None  # bare specifier = node module, not an in-repo file
    src_dir = PurePosixPath(src).parent
    joined = (src_dir / spec) if not spec.startswith("/") else PurePosixPath("." ) / spec.lstrip("/")
    try:
        norm = str(Path(str(joined))).replace("\\", "/")
    except Exception:
        norm = str(joined)
    norm = str(PurePosixPath(norm))
    # PurePosixPath keeps "a/../b" literal; normalize it:
    parts: list[str] = []
    for part in norm.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part in (".", ""):
            continue
        else:
            parts.append(part)
    base = "/".join(parts)
    if not base:
        return None

    candidates = [base + ext for ext in _JS_EXTS]
    candidates += [base + "/index" + ext for ext in _JS_EXTS]
    if PurePosixPath(base).suffix in _JS_EXTS:
        candidates.insert(0, base)
    for c in candidates:
        if c in file_set:
            return c
    for c in candidates:
        hits = [f for f in file_set if f.endswith(c) or f.endswith("/" + c)]
        if hits:
            return min(hits, key=len)
    return None


def resolve_import(spec: str, src: str, language: str, file_set: set[str]) -> Optional[str]:
    if language == "python":
        return _resolve_python(spec, src, file_set)
    if language in ("javascript", "typescript"):
        return _resolve_js(spec, src, file_set)
    return None


def build_dependency_graph(
    imports_of: dict[str, list[str]],
    all_files: list[str],
    lang_map: dict[str, str],
) -> dict:
    """Resolve raw import specifiers into in-repo file-to-file edges."""
    file_set = set(all_files)
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    inbound: dict[str, int] = {}

    for src, specs in imports_of.items():
        if src not in file_set:
            continue
        language = lang_map.get(src, "")
        for spec in specs:
            try:
                dst = resolve_import(spec, src, language, file_set)
            except Exception:
                dst = None
            if not dst or dst == src:
                continue
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": src, "to": dst})
            inbound[dst] = inbound.get(dst, 0) + 1

    hub_files = [
        {"file": f, "inbound": c}
        for f, c in sorted(inbound.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    ]

    return {
        "imports_of": imports_of,
        "edges": edges,
        "hub_files": hub_files,
        "num_files": len(all_files),
        "num_edges": len(edges),
    }


def graph_context_text(graph: dict, max_edges: int = 40) -> str:
    """Compress the graph into a compact, exact adjacency list for LLM prompts."""
    edges = graph.get("edges", [])
    hubs = graph.get("hub_files", [])
    parts = [
        "MODULE DEPENDENCY GRAPH (computed at index time from real import statements — EXACT. "
        "a -> b means \"a imports b\". Only files and edges listed here exist; never invent others.):"
    ]
    if hubs:
        parts.append("\nHUB FILES (most depended-upon — likely core architecture):")
        for h in hubs[:10]:
            parts.append(f"- {h['file']} (imported by {h['inbound']} file{'s' if h['inbound'] != 1 else ''})")
    shown = edges[:max_edges]
    parts.append(f"\nEDGES (showing {len(shown)} of {len(edges)} total):")
    for e in shown:
        parts.append(f"- {e['from']} -> {e['to']}")
    return "\n".join(parts)


def _mermaid_id(path: str, mapping: dict[str, str]) -> str:
    if path not in mapping:
        mapping[path] = f"n{len(mapping)}"
    return mapping[path]


def graph_to_mermaid(edges: list[dict], max_edges: int = 80) -> str:
    """Render the exact edge set as a Mermaid graph TD source string."""
    shown = edges[:max_edges]
    lines = ["graph TD"]
    if len(edges) > max_edges:
        lines.append(f"  %% {len(edges) - max_edges} edges hidden (rendering cap)")
    id_map: dict[str, str] = {}
    for e in shown:
        src, dst = e.get("from", ""), e.get("to", "")
        if not src or not dst:
            continue
        sid, did = _mermaid_id(src, id_map), _mermaid_id(dst, id_map)
        lines.append(f'  {sid}["{src}"] --> {did}["{dst}"]')
    if len(lines) <= 2 and not shown:
        lines.append('  empty["No in-repo imports detected"]')
    return "\n".join(lines)
