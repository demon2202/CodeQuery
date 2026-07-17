"""
Pydantic models for CodeQuery API requests and responses.
"""

from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


# ── Request models ───────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    """Request to index a GitHub repository."""
    repo_url: str = Field(
        ...,
        description="GitHub repository URL (e.g., https://github.com/user/repo)",
        examples=["https://github.com/pallets/click"],
    )


class ChatRequest(BaseModel):
    """Request to ask a question about an indexed repository."""
    repo_url: str = Field(
        ...,
        description="Repository URL (must be already indexed)",
    )
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question about the codebase",
    )


class FileContentRequest(BaseModel):
    """Request to view a file's content."""
    repo_url: str
    file_path: str = Field(..., description="Relative path from repo root")
    start_line: Optional[int] = Field(None, ge=1, description="Start line (1-indexed)")
    end_line: Optional[int] = Field(None, ge=1, description="End line (1-indexed)")


# ── Response models ──────────────────────────────────────────────────────────

class RepoStatus(BaseModel):
    """Status of an indexed repository."""
    repo_url: str
    commit_hash: str
    files_indexed: int
    chunks_created: int
    indexed_at: str


class RepoStatusList(BaseModel):
    """List of all indexed repositories."""
    repos: list[RepoStatus]


class ChunkSource(BaseModel):
    """A source chunk used to answer a question — this is the citation."""
    file_path: str
    start_line: int
    end_line: int
    name: str = Field(..., description="Function/class/method name")
    chunk_type: str = Field(..., description="function, class, method, or module")
    language: str
    parent: Optional[str] = Field(None, description="Parent class name for methods")
    score: float = Field(..., description="Retrieval similarity score (0-1)")


class FileContentResponse(BaseModel):
    """Content of a file (or a line range within it)."""
    file_path: str
    content: str
    language: str
    total_lines: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_running: bool
    ollama_model: str
    model_available: bool
    embedding_model: str
    git_available: bool = True
