#!/usr/bin/env python3
"""Simple benchmark harness for retrieval and reranking behavior."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embed_and_index import evaluate_retrieval_variants


def load_prompts(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        prompts = payload.get("test_prompts", [])
    elif isinstance(payload, list):
        prompts = payload
    else:
        raise ValueError("Prompt file must be a JSON object or list")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    for item in prompts:
        prompt = item.get("prompt") or item.get("text") or ""
        if not prompt:
            continue
        result = evaluate_retrieval_variants(
            prompt,
            args.index,
            args.metadata,
            top_k=args.top_k,
            model_path=args.model,
        )
        print(f"PROMPT: {prompt}")
        print(f"  raw: {[chunk.get('id') for chunk in result['raw_results']]}")
        print(f"  reranked: {[chunk.get('id') for chunk in result['reranked_results']]}")
        print()


if __name__ == "__main__":
    main()
