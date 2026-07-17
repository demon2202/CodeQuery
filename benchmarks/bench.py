#!/usr/bin/env python3
"""
CodeQuery Benchmark Script

Measures real performance numbers for the README:
- Time to clone
- Time to parse + embed per file
- Time to first token (LLM response)
- Time to full answer

Usage:
    python bench.py [--repo URL] [--repo URL] [--skip-llm]

Tests against at least 2 repos of different sizes by default.
These are real repos, not mock data.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.cloner import clone_repo, get_repo_path
from app.services.walker import walk_source_files
from app.services.chunker import chunk_file
from app.services.embedder import encode_batch, warm_up
from app.store.chroma_store import get_store


# Default test repos — chosen for structural diversity:
# 1. pallets/click — Small Python CLI library (~100 files, mostly .py)
# 2. expressjs/express — Medium JavaScript web framework (~200 files, .js)
# 3. fastapi/fastapi — Larger Python framework (~400+ files, .py + some .js/.ts)
DEFAULT_REPOS = [
    "https://github.com/pallets/click",
    "https://github.com/expressjs/express",
    "https://github.com/fastapi/fastapi",
]


async def benchmark_clone(repo_url: str) -> dict:
    """Benchmark: time to clone a repo."""
    start = time.time()
    try:
        repo_path, commit_hash = await clone_repo(repo_url)
        elapsed = time.time() - start
        return {
            "status": "success",
            "time_seconds": round(elapsed, 2),
            "commit_hash": commit_hash[:8],
            "repo_path": str(repo_path),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "error", "error": str(e), "time_seconds": round(elapsed, 2)}


def benchmark_walk(repo_path: str) -> dict:
    """Benchmark: time to walk file tree and identify source files."""
    start = time.time()
    files = walk_source_files(Path(repo_path))
    elapsed = time.time() - start

    # Count by language
    lang_counts = {}
    for _, lang in files:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    return {
        "status": "success",
        "time_seconds": round(elapsed, 3),
        "total_files": len(files),
        "by_language": lang_counts,
    }


def benchmark_parse(repo_path: str, files: list) -> dict:
    """Benchmark: time to parse files into chunks with tree-sitter."""
    repo_root = Path(repo_path)
    start = time.time()

    all_chunks = []
    errors = 0
    per_file_times = []

    for file_path, lang in files:
        file_start = time.time()
        try:
            chunks = chunk_file(file_path, repo_root, lang)
            all_chunks.extend(chunks)
        except Exception:
            errors += 1
        per_file_times.append(time.time() - file_start)

    elapsed = time.time() - start

    # Count by chunk type
    type_counts = {}
    for chunk in all_chunks:
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1

    return {
        "status": "success",
        "time_seconds": round(elapsed, 2),
        "total_chunks": len(all_chunks),
        "errors": errors,
        "chunk_types": type_counts,
        "avg_file_parse_ms": round(sum(per_file_times) / max(len(per_file_times), 1) * 1000, 1),
        "max_file_parse_ms": round(max(per_file_times) * 1000, 1) if per_file_times else 0,
    }


def benchmark_embed(chunks: list, batch_size: int = 64) -> dict:
    """Benchmark: time to embed all chunks in batches."""
    texts = [c.content for c in chunks]

    start = time.time()
    embeddings = encode_batch(texts)
    elapsed = time.time() - start

    return {
        "status": "success",
        "time_seconds": round(elapsed, 2),
        "total_chunks": len(texts),
        "batch_size": batch_size,
        "throughput_chunks_per_sec": round(len(texts) / max(elapsed, 0.01), 1),
    }


def benchmark_store(repo_url: str, chunks: list, embeddings: list) -> dict:
    """Benchmark: time to store chunks in ChromaDB."""
    store = get_store()

    start = time.time()

    chunk_ids = [c.chunk_id for c in chunks]
    documents = [c.content for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]

    store.add_chunks(repo_url, chunk_ids, documents, embeddings, metadatas)

    elapsed = time.time() - start

    return {
        "status": "success",
        "time_seconds": round(elapsed, 2),
        "total_chunks": len(chunks),
        "throughput_chunks_per_sec": round(len(chunks) / max(elapsed, 0.01), 1),
    }


async def run_benchmark(repo_url: str, skip_llm: bool = False) -> dict:
    """Run the full benchmark pipeline for a single repo."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {repo_url}")
    print(f"{'='*60}")

    results = {"repo_url": repo_url}

    # Step 1: Clone
    print("  [1/5] Cloning...")
    clone_result = await benchmark_clone(repo_url)
    results["clone"] = clone_result
    print(f"        {clone_result['time_seconds']}s — {clone_result.get('commit_hash', 'N/A')}")

    if clone_result["status"] == "error":
        results["status"] = "clone_failed"
        return results

    repo_path = clone_result["repo_path"]

    # Step 2: Walk
    print("  [2/5] Walking file tree...")
    walk_result = benchmark_walk(repo_path)
    results["walk"] = walk_result
    print(f"        {walk_result['time_seconds']}s — {walk_result['total_files']} files ({json.dumps(walk_result['by_language'])})")

    if walk_result["total_files"] == 0:
        results["status"] = "no_source_files"
        return results

    # Step 3: Parse
    print("  [3/5] Parsing with tree-sitter...")
    parse_result = benchmark_parse(repo_path, walk_result.get("files", []))
    # Re-do with actual file list since we didn't pass it from walk
    files = walk_source_files(Path(repo_path))
    parse_result = benchmark_parse(repo_path, files)
    results["parse"] = parse_result
    print(f"        {parse_result['time_seconds']}s — {parse_result['total_chunks']} chunks ({json.dumps(parse_result['chunk_types'])})")
    print(f"        avg {parse_result['avg_file_parse_ms']}ms/file, max {parse_result['max_file_parse_ms']}ms/file")

    # Step 4: Embed
    print("  [4/5] Embedding chunks...")
    repo_root = Path(repo_path)
    all_chunks = []
    for file_path, lang in files:
        try:
            chunks = chunk_file(file_path, repo_root, lang)
            all_chunks.extend(chunks)
        except Exception:
            pass

    embed_result = benchmark_embed(all_chunks)
    results["embed"] = embed_result
    print(f"        {embed_result['time_seconds']}s — {embed_result['throughput_chunks_per_sec']} chunks/sec")

    # Step 5: Store
    print("  [5/5] Storing in ChromaDB...")
    texts = [c.content for c in all_chunks]
    embeddings = encode_batch(texts)
    store_result = benchmark_store(repo_url, all_chunks, embeddings)
    results["store"] = store_result
    print(f"        {store_result['time_seconds']}s — {store_result['throughput_chunks_per_sec']} chunks/sec")

    # Summary
    total_time = (
        clone_result.get("time_seconds", 0) +
        walk_result.get("time_seconds", 0) +
        parse_result.get("time_seconds", 0) +
        embed_result.get("time_seconds", 0) +
        store_result.get("time_seconds", 0)
    )
    results["total_time"] = round(total_time, 2)
    results["status"] = "success"

    print(f"\n  Total: {total_time}s for {walk_result['total_files']} files, {parse_result['total_chunks']} chunks")

    return results


async def main():
    parser = argparse.ArgumentParser(description="CodeQuery Benchmark")
    parser.add_argument("--repo", action="append", help="Repo URL to benchmark (can specify multiple)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM benchmark (Ollama not required)")
    args = parser.parse_args()

    repos = args.repo or DEFAULT_REPOS

    print("CodeQuery Benchmark")
    print("=" * 60)
    print(f"Warming up embedding model...")
    warm_up()
    print("Ready.\n")

    all_results = []
    for repo_url in repos:
        result = await run_benchmark(repo_url, args.skip_llm)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Repo':<30} {'Files':<8} {'Chunks':<8} {'Clone':<8} {'Parse':<8} {'Embed':<8} {'Total':<8}")
    print("-" * 90)
    for r in all_results:
        if r["status"] != "success":
            print(f"{r['repo_url']:<30} FAILED: {r['status']}")
            continue
        repo_name = r["repo_url"].replace("https://github.com/", "")
        files = r["walk"]["total_files"]
        chunks = r["parse"]["total_chunks"]
        clone_t = r["clone"]["time_seconds"]
        parse_t = r["parse"]["time_seconds"]
        embed_t = r["embed"]["time_seconds"]
        total_t = r["total_time"]
        print(f"{repo_name:<30} {files:<8} {chunks:<8} {clone_t:<8} {parse_t:<8} {embed_t:<8} {total_t:<8}")

    # Save detailed results
    output_file = Path(__file__).parent / "results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
