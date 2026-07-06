"""
embed_and_index.py — ADTC 2026 Agriculture RAG: embed chunks + build FAISS index

WHAT THIS DOES (plain English version):
  Takes the chunks.json from pdf_to_chunks.py, sends each chunk's text to
  your running llama-server (the --embedding one you already tested with
  curl), and gets back a vector — a list of numbers that represents the
  *meaning* of that chunk. It stacks all those vectors into a FAISS index,
  which is basically a fast "find me the most similar vectors" lookup
  structure. Alongside the index, it saves a small JSON file mapping each
  vector's position back to the original chunk text and metadata, because
  FAISS itself only knows positions/numbers, not what they mean.

  At query time (the next script, not this one): you embed the user's
  question the same way, ask FAISS for the closest chunk vectors, and feed
  those chunks' original text into the model as context. That's the "R" in
  RAG — retrieval before generation.

WHY THIS STAYS LLAMA.CPP-ONLY:
  Embeddings come from your own tiny-aya-earth-q4_k_m.gguf via llama-server's
  /embedding endpoint — no torch, no sentence-transformers, no second model.
  Same runtime end to end, which matters for the ADTC "llama.cpp only" rule
  and keeps your dependency footprint (and RAM budget) small.

PREREQUISITE:
  llama-server must already be running with --embedding, e.g.:
    ~/llama.cpp/build/bin/llama-server -m model/tiny-aya-earth-q4_k_m.gguf \
        --embedding -c 512 --port 8080

USAGE:
  python embed_and_index.py --chunks corpus/chunks.json \
      --index corpus/index.faiss --metadata corpus/metadata.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

try:
    import faiss
except ImportError:
    print("faiss not installed. Run: pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)


def get_embedding(server_url: str, text: str, retries: int = 3) -> list[float]:
    """
    Post one chunk's text to llama-server's /embedding endpoint and return
    the vector. Retries on transient failures (e.g. server still warming up
    a batch) since these can happen mid-run on constrained hardware, not
    just at startup.

    Note on the response shape: llama-server returns a nested list —
    [{"index": 0, "embedding": [[...]]}]. The outer list is per-request,
    the inner one is per-token-position for some modes, but with a single
    short prompt and pooling enabled you get one row back. We grab that
    row directly rather than assuming a flat list, since the exact nesting
    has changed across llama.cpp versions.
    """
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{server_url}/embedding",
                json={"content": text},
                # 90s, not 30s: server logs show single requests taking
                # 20-30s on Codespace's shared CPU even for a mid-size
                # chunk. 30s was cutting off requests that were slow but
                # still working, not actually stuck.
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data[0]["embedding"]
            # Unwrap one level of nesting if present (see docstring above).
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            return embedding
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Failed to embed after {retries} attempts: {e}"
                ) from e
            time.sleep(1.5 * (attempt + 1))  # brief backoff before retry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--index", required=True, help="Output path for FAISS index")
    parser.add_argument("--metadata", required=True, help="Output path for chunk metadata JSON")
    parser.add_argument("--server", default="http://localhost:8080", help="llama-server base URL")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"Chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(chunks)} chunks via {args.server}...")

    embeddings = []
    metadata = []
    failed = []

    for i, chunk in enumerate(chunks):
        try:
            vec = get_embedding(args.server, chunk["text"])
            embeddings.append(vec)
            metadata.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            })
        except RuntimeError as e:
            print(f"  [{i}] FAILED ({chunk['id']}): {e}", file=sys.stderr)
            failed.append(chunk["id"])
            continue

        if (i + 1) % 25 == 0 or (i + 1) == len(chunks):
            print(f"  {i + 1}/{len(chunks)} embedded")

    if not embeddings:
        print("All embeddings failed. Is llama-server running with --embedding?", file=sys.stderr)
        sys.exit(1)

    # Stack into a single matrix FAISS can index. dtype must be float32 —
    # FAISS silently misbehaves or errors on float64, which is numpy's
    # default, so this cast is not optional.
    matrix = np.array(embeddings, dtype="float32")
    dim = matrix.shape[1]

    # IndexFlatIP = exact inner-product search, no approximation. For a
    # corpus this size (a few hundred chunks from 6 PDFs), exact search is
    # fast enough that there's no reason to trade accuracy for the
    # approximate-index speedup — that trade only pays off at 100k+ vectors.
    # Inner product (not L2) because it pairs naturally with normalized
    # embeddings for cosine-similarity-style retrieval.
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    index_path = Path(args.index)
    metadata_path = Path(args.metadata)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nIndexed {len(embeddings)} chunks (dim={dim})")
    if failed:
        print(f"Failed: {len(failed)} chunks -> {failed}")
    print(f"FAISS index: {index_path}")
    print(f"Metadata:    {metadata_path}")


if __name__ == "__main__":
    main()
