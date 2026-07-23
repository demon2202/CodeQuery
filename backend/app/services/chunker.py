"""
AST-aware code chunker.

Key design decisions:
- Each function/class = one chunk. Independently retrievable.
- Methods inside classes: separate chunks with parent=ClassName.
- NO "preamble" or "epilogue" chunks — these are usually just comments,
  imports, or whitespace between functions. They pollute search results.
  Instead, imports and module-level code get a single "module" chunk
  at the top of the file IF it has meaningful content (not just comments).
- Skip chunks that are only comments/whitespace — these are noise for retrieval.
- Fallback: sliding-window for unsupported languages.
"""

import os
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser, Node

from .. import config


@dataclass
class CodeChunk:
    content: str
    file_path: str
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed, inclusive
    name: str
    chunk_type: str  # function, class, method, module, fallback
    language: str
    parent: Optional[str] = None

    @property
    def chunk_id(self) -> str:
        key = f"{self.file_path}:{self.start_line}:{self.end_line}:{self.name}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]

    def to_metadata(self) -> dict:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "name": self.name,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "parent": self.parent or "",
        }


# Language configs
_LANG_CONFIGS = {
    "python": {
        "lang": lambda: Language(tspython.language()),
        "chunk_types": {"function_definition", "class_definition"},
        "decorator_wrapper": "decorated_definition",
        "body_types": {"block"},
    },
    "javascript": {
        "lang": lambda: Language(tsjavascript.language()),
        "chunk_types": {
            "function_declaration", "class_declaration", "method_definition",
            "arrow_function", "variable_declarator",
        },
        "decorator_wrapper": None,
        "body_types": {"class_body", "statement_block"},
    },
    "typescript": {
        "lang": lambda: Language(tstypescript.language_typescript()),
        "chunk_types": {
            "function_declaration", "class_declaration", "method_definition",
            "interface_declaration", "type_alias_declaration", "enum_declaration",
            "arrow_function", "variable_declarator", "public_field_definition",
        },
        "decorator_wrapper": None,
        "body_types": {"class_body", "statement_block", "interface_body"},
    },
}


def _is_only_comments(text: str) -> bool:
    """Check if text is only comments, whitespace, and separator lines."""
    # Strip common patterns: # ===, # ---, blank lines, single-line comments
    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Python/JS/TS comment lines
        if stripped.startswith("#") or stripped.startswith("//"):
            # Skip separator lines like # ==== or # ----
            content = stripped.lstrip("#/").strip()
            if re.match(r'^[=\-_*~]+$', content):
                continue
            # It's a real comment with actual text — that's OK, keep it
            # But if the ENTIRE chunk is just one-liner comments, still skip
            continue
        # If we reach here, there's actual code — don't skip
        return False
    return True  # All lines are comments/whitespace/separators


def _has_real_code(text: str) -> bool:
    """Check if text has actual code, not just comments/whitespace."""
    lines = text.strip().split("\n")
    real_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            content = stripped.lstrip("#/").strip()
            if re.match(r'^[=\-_*~]+$', content):
                continue  # separator line
            continue  # comment line
        real_lines += 1
    return real_lines >= 2  # At least 2 lines of actual code


def chunk_file(file_path: Path, repo_root: Path, language: str) -> List[CodeChunk]:
    """Parse a file into chunks. Returns empty list on errors."""
    try:
        rel_path = str(file_path.relative_to(repo_root))
    except ValueError:
        rel_path = file_path.name

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError:
        return []

    if not source.strip():
        return []

    if language in _LANG_CONFIGS:
        return _chunk_ast(source, rel_path, language)
    return _chunk_sliding_window(source, rel_path, language)


def _split_long_chunk(chunk: CodeChunk, max_chars: int = 2000) -> List[CodeChunk]:
    """Split a chunk that exceeds max_chars into smaller pieces.
    
    Tries to split at line boundaries for cleaner chunks.
    """
    if len(chunk.content) <= max_chars:
        return [chunk]
    
    lines = chunk.content.split("\n")
    pieces = []
    current_lines = []
    current_start = chunk.start_line
    
    for i, line in enumerate(lines):
        current_lines.append(line)
        current_text = "\n".join(current_lines)
        
        # If adding this line pushes us over the limit, flush
        if len(current_text) > max_chars and len(current_lines) > 1:
            # Remove the last line, flush the rest
            current_lines.pop()
            pieces.append(CodeChunk(
                content="\n".join(current_lines),
                file_path=chunk.file_path,
                start_line=current_start,
                end_line=current_start + len(current_lines) - 1,
                name=chunk.name,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                parent=chunk.parent,
            ))
            current_start = current_start + len(current_lines)
            current_lines = [line]
    
    # Flush remaining
    if current_lines:
        remaining_text = "\n".join(current_lines)
        if remaining_text.strip() and not _is_only_comments(remaining_text):
            pieces.append(CodeChunk(
                content=remaining_text,
                file_path=chunk.file_path,
                start_line=current_start,
                end_line=chunk.end_line,
                name=chunk.name,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                parent=chunk.parent,
            ))
    
    return pieces if pieces else [chunk]


def _chunk_ast(source: str, rel_path: str, language: str) -> List[CodeChunk]:
    """Tree-sitter AST-aware chunking."""
    cfg = _LANG_CONFIGS[language]
    ts_lang = cfg["lang"]()
    parser = Parser(ts_lang)

    try:
        tree = parser.parse(bytes(source, "utf8"))
    except Exception:
        return _chunk_sliding_window(source, rel_path, language)

    if tree.root_node is None:
        return _chunk_sliding_window(source, rel_path, language)

    lines = source.split("\n")
    chunks: List[CodeChunk] = []
    chunk_types = cfg["chunk_types"]
    deco_wrapper = cfg.get("decorator_wrapper")

    # Track where the last structural node ended
    last_end = 0
    first_structural_found = False

    for child in tree.root_node.children:
        # Unwrap decorator wrapper if present
        actual = child
        if deco_wrapper and child.type == deco_wrapper:
            for sub in child.children:
                if sub.type in chunk_types:
                    actual = sub
                    break

        if actual.type not in chunk_types:
            continue

        # Code before this node (module-level: imports, constants, etc.)
        node_start = child.start_point[0]
        if not first_structural_found and node_start > 0:
            preamble = "\n".join(lines[0:node_start])
            # Only add module chunk if it has real code (not just comments)
            if _has_real_code(preamble):
                chunks.append(CodeChunk(
                    content=preamble, file_path=rel_path,
                    start_line=1, end_line=node_start,
                    name=os.path.splitext(os.path.basename(rel_path))[0],
                    chunk_type="module", language=language,
                ))
            first_structural_found = True
        elif first_structural_found and node_start > last_end:
            # Code between two structural nodes — skip if just comments
            gap = "\n".join(lines[last_end:node_start])
            if _has_real_code(gap):
                chunks.append(CodeChunk(
                    content=gap, file_path=rel_path,
                    start_line=last_end + 1, end_line=node_start,
                    name=os.path.splitext(os.path.basename(rel_path))[0],
                    chunk_type="module", language=language,
                ))

        name = _node_name(actual) or _derive_name(rel_path, actual.type)
        start = child.start_point[0] + 1
        end = child.end_point[0] + 1
        text = "\n".join(lines[child.start_point[0]:child.end_point[0] + 1])

        # Skip chunks that are only comments/whitespace
        if _is_only_comments(text):
            last_end = child.end_point[0] + 1
            continue

        # If it's a class, also extract methods as separate chunks
        if actual.type in ("class_definition", "class_declaration"):
            # First: add class header (declaration + docstring, before first method)
            header_end = _class_header_end(actual)
            if header_end > actual.start_point[0]:
                header_text = "\n".join(lines[actual.start_point[0]:header_end])
                if not _is_only_comments(header_text):
                    chunks.append(CodeChunk(
                        content=header_text, file_path=rel_path,
                        start_line=actual.start_point[0] + 1, end_line=header_end,
                        name=name, chunk_type="class", language=language,
                    ))
            # Then: extract each method
            for body in actual.children:
                if body.type in cfg["body_types"]:
                    for member in body.children:
                        member_actual = member
                        if deco_wrapper and member.type == deco_wrapper:
                            for sub in member.children:
                                if sub.type in chunk_types:
                                    member_actual = sub
                                    break
                        if member_actual.type in chunk_types:
                            m_name = _node_name(member_actual) or f"{name}_anon"
                            m_start = member.start_point[0] + 1
                            m_end = member.end_point[0] + 1
                            m_text = "\n".join(lines[member.start_point[0]:member.end_point[0] + 1])
                            if not _is_only_comments(m_text):
                                chunks.append(CodeChunk(
                                    content=m_text, file_path=rel_path,
                                    start_line=m_start, end_line=m_end,
                                    name=m_name, chunk_type="method",
                                    language=language, parent=name,
                                ))
        else:
            # Function — one chunk
            chunks.append(CodeChunk(
                content=text, file_path=rel_path,
                start_line=start, end_line=end,
                name=name, chunk_type="function", language=language,
            ))

        last_end = child.end_point[0] + 1

    # Trailing code after last structural node — only if it has real code
    if last_end < len(lines):
        trailing = "\n".join(lines[last_end:])
        if _has_real_code(trailing):
            chunks.append(CodeChunk(
                content=trailing, file_path=rel_path,
                start_line=last_end + 1, end_line=len(lines),
                name=os.path.splitext(os.path.basename(rel_path))[0],
                chunk_type="module", language=language,
            ))

    # If tree-sitter parsed but found nothing structural, fall back
    if not chunks and source.strip():
        return _chunk_sliding_window(source, rel_path, language)

    # Enforce max chunk size — split oversized chunks at line boundaries
    max_chars = config.MAX_CHUNK_CHARS
    final_chunks = []
    for c in chunks:
        if len(c.content) > max_chars:
            final_chunks.extend(_split_long_chunk(c, max_chars))
        else:
            final_chunks.append(c)

    return final_chunks


def _class_header_end(node: Node) -> int:
    """Find end line of class declaration + docstring (before first method)."""
    for body in node.children:
        if body.type in ("block", "class_body", "declaration_list"):
            for child in body.children:
                if child.type in ("comment", "pass_statement", "expression_statement"):
                    continue
                return child.start_point[0]
            break
    return min(node.end_point[0], node.start_point[0] + 15)


def _node_name(node: Node) -> str | None:
    """Extract name from a function/class node."""
    name_node = node.child_by_field_name("name")
    return name_node.text.decode("utf8") if name_node else None


def _derive_name(rel_path: str, node_type: str) -> str:
    basename = os.path.splitext(os.path.basename(rel_path))[0]
    return f"{basename}:{node_type}"


def _chunk_sliding_window(source: str, rel_path: str, language: str) -> List[CodeChunk]:
    """Fallback for unsupported languages."""
    lines = source.split("\n")
    size = config.SLIDING_WINDOW_LINES
    overlap = config.SLIDING_WINDOW_OVERLAP
    chunks = []
    is_code = language not in ("text", "unknown")

    start = 0
    while start < len(lines):
        end = min(start + size, len(lines))
        text = "\n".join(lines[start:end])
        if text.strip() and not _is_only_comments(text):
            chunks.append(CodeChunk(
                content=text, file_path=rel_path,
                start_line=start + 1, end_line=end,
                name=_derive_name(rel_path, f"L{start+1}-{end}"),
                chunk_type="fallback" if is_code else "module",
                language=language,
            ))
        if end >= len(lines):
            break
        start += size - overlap

    return chunks