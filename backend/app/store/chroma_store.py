"""
ChromaDB vector store wrapper for CodeQuery.

Handles persistence, collection management, and CRUD operations for code chunks.
One collection per repo, named by URL hash for isolation.

Why ChromaDB over FAISS (detailed justification):
1. Native metadata filtering — query chunks by language, type, file path without
   building a separate metadata index. FAISS requires a parallel metadata store.
2. HNSW backend — same algorithm as FAISS HNSW, comparable speed at scale.
   Brute-force is O(n); HNSW is O(log n). For <1000 chunks, no difference.
   For 50K+ chunks, HNSW is 10-100x faster at ~5% recall cost.
3. Auto-persistence — ChromaDB saves to disk on every write. FAISS needs
   explicit save/load logic that's easy to forget on crash.
4. Collection-per-repo — trivial to delete/update a repo's chunks during
   incremental re-indexing without touching other repos.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .. import config

logger = logging.getLogger(__name__)


def _repo_collection_name(repo_url: str) -> str:
    """Generate a valid ChromaDB collection name from a repo URL.
    
    ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens/underscores,
    start/end with alphanumeric. We hash the URL to get a stable, valid name.
    """
    h = hashlib.sha256(repo_url.encode()).hexdigest()[:20]
    return f"repo_{h}"


def _repo_meta_path(repo_url: str) -> Path:
    """Path to the per-repo metadata file (commit hash, etc.)."""
    h = hashlib.sha256(repo_url.encode()).hexdigest()[:20]
    return config.CHROMA_DIR / f"meta_{h}.json"


class ChromaStore:
    """Persistent vector store for code chunks."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # No telemetry — fully local
            ),
        )
        self._collections: dict[str, chromadb.Collection] = {}
        self._repo_meta: dict[str, dict] = {}  # In-memory cache of per-repo metadata

    def _load_repo_meta(self, repo_url: str) -> dict:
        """Load per-repo metadata from JSON file."""
        if repo_url in self._repo_meta:
            return self._repo_meta[repo_url]
        meta_path = _repo_meta_path(repo_url)
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    self._repo_meta[repo_url] = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._repo_meta[repo_url] = {}
        else:
            self._repo_meta[repo_url] = {}
        return self._repo_meta[repo_url]

    def _save_repo_meta(self, repo_url: str) -> None:
        """Save per-repo metadata to JSON file."""
        meta_path = _repo_meta_path(repo_url)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(self._repo_meta.get(repo_url, {}), f)

    def _get_collection(self, repo_url: str) -> chromadb.Collection:
        """Get or create a collection for a repo. Cached for connection pooling."""
        name = _repo_collection_name(repo_url)
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={
                    "hnsw:space": config.CHROMA_HNSW_SPACE,
                    "hnsw:M": config.CHROMA_HNSW_M,
                    "hnsw:construction_ef": config.CHROMA_HNSW_EF_CONSTRUCTION,
                    "hnsw:search_ef": config.CHROMA_HNSW_EF_SEARCH,
                    "repo_url": repo_url,
                },
            )
        return self._collections[name]

    def add_chunks(
        self,
        repo_url: str,
        chunk_ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add chunks to the collection. Each chunk gets a unique ID for updates."""
        collection = self._get_collection(repo_url)
        # ChromaDB will raise if IDs conflict — use upsert for incremental safety
        collection.upsert(
            ids=chunk_ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_chunks_by_file(self, repo_url: str, file_path: str) -> None:
        """Delete all chunks belonging to a specific file. Used for incremental re-indexing."""
        collection = self._get_collection(repo_url)
        # ChromaDB supports 'where' filtering for deletion
        collection.delete(
            where={"file_path": file_path}
        )

    def delete_collection(self, repo_url: str) -> None:
        """Delete an entire repo's collection."""
        name = _repo_collection_name(repo_url)
        try:
            self._client.delete_collection(name=name)
        except (chromadb.errors.NotFoundError, chromadb.errors.InvalidArgumentError):
            pass  # Collection doesn't exist — nothing to delete
        self._collections.pop(name, None)
        # Also clean up metadata file
        meta_path = _repo_meta_path(repo_url)
        if meta_path.exists():
            meta_path.unlink()
        self._repo_meta.pop(repo_url, None)

    def query(
        self,
        repo_url: str,
        query_embedding: list[float],
        n_results: int = config.RETRIEVAL_TOP_K,
        where: Optional[dict] = None,
    ) -> dict:
        """Query for similar chunks. Returns documents, metadatas, distances."""
        collection = self._get_collection(repo_url)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    def get_collection_stats(self, repo_url: str) -> dict:
        """Get stats about a repo's collection."""
        collection = self._get_collection(repo_url)
        count = collection.count()
        metadata = collection.metadata
        return {
            "chunk_count": count,
            "metadata": metadata,
        }

    def get_commit_hash(self, repo_url: str) -> Optional[str]:
        """Get the last indexed commit hash for a repo, if any.
        
        Stored in a separate JSON file, not in ChromaDB collection metadata,
        because ChromaDB's modify(metadata=) doesn't reliably support updating
        metadata on collections that have HNSW configuration.
        """
        meta = self._load_repo_meta(repo_url)
        return meta.get("commit_hash")

    def set_commit_hash(self, repo_url: str, commit_hash: str) -> None:
        """Update the commit hash for a repo's collection.
        
        Uses a separate JSON file instead of ChromaDB collection metadata
        because ChromaDB's modify() has issues with HNSW collection metadata
        mutation ("Changing the distance function of a collection once it is
        created is not supported currently" error).
        """
        meta = self._load_repo_meta(repo_url)
        meta["commit_hash"] = commit_hash
        meta["repo_url"] = repo_url
        self._save_repo_meta(repo_url)

    def list_repos(self) -> list[dict]:
        """List all indexed repos with their stats."""
        try:
            collections = self._client.list_collections()
        except Exception:
            return []
        repos = []
        for col_info in collections:
            try:
                # list_collections returns Collection objects or dicts depending on version
                if isinstance(col_info, str):
                    name = col_info
                    collection = self._client.get_collection(name=name)
                elif hasattr(col_info, 'name'):
                    name = col_info.name
                    collection = col_info
                else:
                    continue
                meta = collection.metadata or {}
                repo_url = meta.get("repo_url", "")
                count = collection.count()
                if repo_url:
                    # Get commit hash from our metadata file
                    commit_hash = self.get_commit_hash(repo_url) or ""
                    repos.append({
                        "repo_url": repo_url,
                        "commit_hash": commit_hash,
                        "chunks": count,
                    })
            except Exception:
                continue
        return repos


# Singleton instance — connection-pooled, shared across requests
_store: Optional[ChromaStore] = None


def get_store() -> ChromaStore:
    """Get the shared ChromaStore instance."""
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
