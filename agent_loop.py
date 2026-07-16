"""
agent_loop.py — ADTC 2026 Agriculture RAG: lightweight tool-use loop

WHAT THIS DOES (plain English):
  Takes a farmer's question and decides which of three paths to take:
    1. CALCULATE — a deterministic rate/area scaling question (see calc_tool.py).
       Detected by regex, answered without any model call at all: zero risk
       of a hallucinated number.
    2. RAG — the question needs grounded, corpus-backed information (planting
       dates, pest identification, fertilization guidance). Retrieves from
       the FAISS index and generates an answer using retrieved context.
    3. DIRECT — the question doesn't need the corpus at all (a greeting, a
       general non-agricultural question, small talk). Answered directly by
       the model with no retrieval step, saving the retrieval overhead.

WHY THIS STAYS "LIGHTWEIGHT":
  The calculation path costs nothing at all (pure regex). The RAG-vs-DIRECT
  decision is ALSO a cheap deterministic check, not a model call: testing
  showed the 3.35B model is unreliable at this classification when asked
  directly (it answered DIRECT to "what is the best time to plant maize in
  South Africa" - the exact question this whole RAG system exists to answer
  correctly). Rather than trust a small model's judgment about whether to
  use its own grounding, this defaults to RAG for anything that isn't
  recognizably small talk.

USAGE:
  python agent_loop.py --query "..." --index corpus/index.faiss \
      --metadata corpus/metadata.json --model model/tiny-aya-earth-q4_k_m.gguf
"""

import argparse
import re
import sys
from pathlib import Path

from calc_tool import try_calculate

sys.path.insert(0, str(Path(__file__).parent))
try:
    from embed_and_index import (
        build_generation_model,
        embed_query,
        retrieve_top_k,
        generate_answer,
    )
except ImportError:
    build_generation_model = embed_query = retrieve_top_k = generate_answer = None


SMALL_TALK_MAX_LENGTH = 40

SMALL_TALK_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|"
    r"how are you|what'?s up|thanks|thank you|bye|goodbye|"
    r"who are you|what can you do|what is your name)\b",
    re.IGNORECASE,
)


def classify_route(query: str, model=None) -> str:
    """
    Deterministic routing: DIRECT only for short, unambiguous small talk,
    RAG for everything else.
    """
    stripped = query.strip()
    is_short_greeting = (
        len(stripped) <= SMALL_TALK_MAX_LENGTH and SMALL_TALK_PATTERNS.match(stripped)
    )
    return "DIRECT" if is_short_greeting else "RAG"


def run_agent(
    query: str,
    index_path: str,
    metadata_path: str,
    model_path: str,
    top_k: int = 4,
) -> dict:
    calc_result = try_calculate(query)
    if calc_result is not None:
        return {"route": "CALCULATE", "answer": calc_result}

    route = classify_route(query)

    if route == "RAG":
        query_vector = embed_query(query, model_path=model_path)
        retrieved = retrieve_top_k(index_path, metadata_path, query_vector, top_k=top_k, query_text=query)
        answer = generate_answer(query, retrieved, model_path=model_path)
        return {
            "route": "RAG",
            "answer": answer,
            "sources": [f"{c.get('source')} p.{c.get('page')}" for c in retrieved],
        }
    else:
        model = build_generation_model(model_path)
        output = model(query, max_tokens=220, temperature=0.3)
        return {"route": "DIRECT", "answer": output["choices"][0]["text"].strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="The farmer's question")
    parser.add_argument("--index", required=True, help="Path to FAISS index")
    parser.add_argument("--metadata", required=True, help="Path to chunk metadata JSON")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--top-k", type=int, default=4, help="Chunks to retrieve for RAG path")
    args = parser.parse_args()

    result = run_agent(args.query, args.index, args.metadata, args.model, top_k=args.top_k)

    print(f"[Route: {result['route']}]")
    if result.get("sources"):
        print(f"Sources: {', '.join(result['sources'])}")
    print(f"\n{result['answer']}")


if __name__ == "__main__":
    main()
