"""
embed_and_index.py — ADTC 2026 Agriculture RAG: embed chunks + build FAISS index

This script now supports both:
  1. the existing server-based embedding flow via llama-server, and
  2. a fully local offline flow using llama-cpp-python and the GGUF model.

It can build a FAISS index, retrieve the most relevant chunks for a query,
and generate an answer from those chunks using the same local model.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    import faiss
except ImportError:
    print("faiss not installed. Run: pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None


def wait_for_server_ready(server_url: str, timeout: int = 300, interval: int = 2) -> None:
    """Poll the llama.cpp health endpoint until the embedding server is ready."""
    health_url = f"{server_url}/health"
    deadline = time.time() + timeout
    last_error = "unknown"

    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                return
            last_error = f"{response.status_code}: {response.text.strip()}"
        except requests.RequestException as exc:
            last_error = str(exc)

        time.sleep(interval)

    raise RuntimeError(f"Embedding server did not become ready: {last_error}")


def get_embedding(server_url: str, text: str, retries: int = 6) -> list[float]:
    """Get an embedding from a llama-server /embedding endpoint."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{server_url}/embedding",
                json={"content": text},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data[0]["embedding"]
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            return embedding
        except (requests.RequestException, KeyError, IndexError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to embed after {retries} attempts: {exc}") from exc
            time.sleep(2.0 * (attempt + 1))


def build_embedding_model(model_path: str, n_threads: int | None = None) -> Any:
    if Llama is None:
        raise RuntimeError("llama-cpp-python is not installed. Install it to use the offline flow.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    threads = n_threads or max(1, min(4, os.cpu_count() or 1))
    return Llama(model_path=model_path, embedding=True, n_ctx=512, n_threads=threads, verbose=False)


def build_generation_model(model_path: str, n_threads: int | None = None) -> Any:
    if Llama is None:
        raise RuntimeError("llama-cpp-python is not installed. Install it to use the offline flow.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    threads = n_threads or max(1, min(4, os.cpu_count() or 1))
    return Llama(model_path=model_path, n_ctx=1024, n_threads=threads, verbose=False)


def _normalize_embedding_payload(payload: Any) -> list[float]:
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return payload[0]
    if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data:
            return _normalize_embedding_payload(data[0].get("embedding", []))
    raise ValueError(f"Unsupported embedding payload shape: {type(payload)!r}")


def embed_texts(
    texts: list[str],
    model_path: str | None = None,
    server_url: str | None = None,
    retries: int = 6,
    model: Any | None = None,
) -> list[list[float]]:
    if model is not None:
        embeddings = []
        for text in texts:
            response = model.create_embedding(text)
            embeddings.append(_normalize_embedding_payload(response))
        return embeddings

    if model_path:
        model = build_embedding_model(model_path)
        embeddings = []
        for text in texts:
            response = model.create_embedding(text)
            embeddings.append(_normalize_embedding_payload(response))
        return embeddings

    if server_url:
        wait_for_server_ready(server_url)
        return [get_embedding(server_url, text, retries=retries) for text in texts]

    raise ValueError("Either model_path or server_url must be provided")


def embed_query(query: str, model_path: str | None = None, server_url: str | None = None, model: Any | None = None) -> list[float]:
    embeddings = embed_texts([query], model_path=model_path, server_url=server_url, model=model)
    return embeddings[0]


def load_chunks(chunks_path: str | Path) -> list[dict[str, Any]]:
    with open(chunks_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Chunk file must contain a JSON array")
    return data


def build_index_from_chunks(
    chunks: list[dict[str, Any]],
    model_path: str | None = None,
    server_url: str | None = None,
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[str, str]:
    embeddings = []
    metadata = []
    failed = []
    embedding_model = build_embedding_model(model_path) if model_path else None

    for i, chunk in enumerate(chunks):
        try:
            vec = embed_texts([chunk["text"]], model_path=None, server_url=server_url, model=embedding_model)[0]
            embeddings.append(vec)
            metadata.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            })
        except Exception as exc:  # pragma: no cover - runtime path
            failed.append((chunk["id"], str(exc)))

    if not embeddings:
        raise RuntimeError("No embeddings could be created")

    matrix = np.array(embeddings, dtype="float32")
    dim = matrix.shape[1]
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    if index_path is None:
        raise ValueError("index_path is required")
    if metadata_path is None:
        raise ValueError("metadata_path is required")

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return str(index_path), str(metadata_path)


def _normalize_query_terms(query: str) -> set[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if token not in {"the", "and", "for", "with", "how", "what", "in", "to", "of"}]
    expanded = set(tokens)
    synonym_map = {
        "maize": {"maize", "corn"},
        "corn": {"maize", "corn"},
        "soil": {"soil", "land"},
        "prep": {"prep", "preparation", "prepare", "preparing"},
        "preparation": {"prep", "preparation", "prepare", "preparing"},
        "prepare": {"prep", "preparation", "prepare", "preparing"},
        "preparing": {"prep", "preparation", "prepare", "preparing"},
    }
    for token in list(tokens):
        expanded.update(synonym_map.get(token, {token}))
    return expanded


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 4,
    vector_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    query_terms = _normalize_query_terms(query)
    if not query_terms:
        return candidates[:top_k]

    scored = []
    for index, candidate in enumerate(candidates):
        text = (candidate.get("text") or "").lower()
        source = (candidate.get("source") or "").lower()
        terms = set(re.findall(r"[a-z0-9]+", text))
        overlap = len(query_terms & terms)
        exact_phrase_bonus = 1 if query.lower() in text else 0
        source_bonus = 1 if any(term in source for term in query_terms) else 0
        vector_score = vector_scores[index] if vector_scores and index < len(vector_scores) else 0.0
        score = overlap * 3 + exact_phrase_bonus + source_bonus + vector_score
        scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [candidate for _, candidate in scored[:top_k]]


def retrieve_top_k(
    index_path: str | Path,
    metadata_path: str | Path,
    query_vector: list[float] | np.ndarray,
    top_k: int = 4,
    query_text: str | None = None,
) -> list[dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)

    query_array = np.asarray(query_vector, dtype="float32")
    if query_array.ndim == 1:
        query = query_array.reshape(1, -1)
    else:
        query = query_array.reshape(1, -1)

    faiss.normalize_L2(query)
    _, indices = index.search(query, min(top_k * 4, len(metadata)))

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(metadata[int(idx)])
    return rerank_candidates(query_text or "", results, top_k=top_k)


def generate_answer(query: str, context_chunks: list[dict[str, Any]], model_path: str, max_tokens: int = 220) -> str:
    if not context_chunks:
        raise ValueError("At least one context chunk is required")

    context_text = "\n\n".join(
        f"Source: {chunk.get('source', 'unknown')} (page {chunk.get('page', 'unknown')})\n{chunk.get('text', '')}"
        for chunk in context_chunks
    )
    prompt = (
        "You are a helpful agricultural assistant. Use only the provided context to answer the user's question.\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer briefly and clearly."
    )

    model = build_generation_model(model_path)
    output = model(prompt, max_tokens=max_tokens, temperature=0.2, stop=["\n\nQuestion:"])
    return output["choices"][0]["text"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--index", required=True, help="Output path for FAISS index")
    parser.add_argument("--metadata", required=True, help="Output path for chunk metadata JSON")
    parser.add_argument("--server", default=None, help="llama-server base URL (optional)")
    parser.add_argument("--model", default=None, help="Path to a GGUF model for local embedding/generation")
    parser.add_argument("--query", default=None, help="Optional query to retrieve context and generate an answer")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks to include")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional limit for indexing only the first N chunks")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    if not chunks:
        print("No chunks found in input file.", file=sys.stderr)
        sys.exit(1)
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    if args.model is None and args.server is None:
        print("Either --model or --server must be provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Building embeddings for {len(chunks)} chunks...")
    index_path, metadata_path = build_index_from_chunks(
        chunks,
        model_path=args.model,
        server_url=args.server,
        index_path=args.index,
        metadata_path=args.metadata,
    )
    print(f"Indexed {len(chunks)} chunks")
    print(f"FAISS index: {index_path}")
    print(f"Metadata:    {metadata_path}")

    if args.query:
        query_vector = embed_query(args.query, model_path=args.model, server_url=args.server)
        retrieved = retrieve_top_k(index_path, metadata_path, query_vector, top_k=args.top_k, query_text=args.query)
        answer = generate_answer(args.query, retrieved, model_path=args.model)
        print("\nRetrieved context:")
        for chunk in retrieved:
            print(f"- {chunk.get('source')} page {chunk.get('page')}")
        print("\nAnswer:")
        print(answer)
    main()
