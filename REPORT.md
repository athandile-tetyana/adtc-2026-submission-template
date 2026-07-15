# Technical Report — Offline Agricultural Advisory Assistant

**Team ID:** athandile-tetyana
**Domain:** agriculture
**Model:** tiny-aya-earth-q4_k_m (3.35B, Q4_K_M)

---

## Problem

Farmers and agricultural extension officers across Southern Africa often work in areas with limited or unreliable internet access, ruling out cloud-hosted AI advisory tools. This project builds an offline, on-device advisory assistant for crop, vegetable, and livestock guidance, targeted at smallholder farmers and extension workers in South Africa.

The system answers practical questions — planting dates, fertilization rates, pest and disease diagnosis, soil preparation — by retrieving grounded answers from real South African agricultural extension documents (ARC and NDA production guides), rather than relying purely on the base model's training data. This matters because a base LLM can be confidently wrong on region-specific facts: during development, the base tiny-aya-earth model was tested un-augmented and stated maize should be planted April-August in South Africa — the correct window is October-December. This single test result was the motivating case for building retrieval-augmented generation (RAG) into the system rather than shipping the base model alone.

The assistant also supports isiXhosa and isiZulu, reflecting the language scope of the target users and the source documents themselves, several of which include local-language crop names.

---

## Design Decisions

- **Base model:** Tiny Aya Earth (CohereLabs), 3.35B parameters, GGUF Q4_K_M quantization (2.14GB). Chosen because it is an Africa-specific variant of Cohere's Aya model family, explicitly designed for on-device deployment, supports 70+ languages including Zulu, Xhosa, Swahili, Yoruba, and Shona, and was directly recommended by ADTC organizers as a challenge resource.
- **Quantization:** Q4_K_M selected over Q8_0 (too large for comfortable headroom within the 7GB efficiency budget) and Q2_K (quality degradation risk on a model already small at 3.35B parameters was judged too aggressive for reliable agricultural advice).
- **Architecture:** Offline RAG pipeline — PDF corpus → chunking → embedding via llama.cpp → FAISS vector index → retrieval → lexical/metadata reranking → generation, all via llama.cpp with GGUF weights end to end.
- **Embedding approach:** Deliberately uses the same Tiny Aya Earth GGUF model for embeddings (via llama.cpp's native `--embedding` mode) rather than a separate embedding model through `sentence-transformers`/`torch`. This keeps the entire pipeline on a single llama.cpp runtime, avoiding a heavier non-llama.cpp dependency stack, in line with the challenge's llama.cpp-only requirement for the runtime and keeping the RAM footprint minimal.
- **Chunking strategy:** Section-heading-aware chunking (splitting on detected headers in the source PDFs) rather than fixed-size windows, since the government/ARC guides have clear per-crop section structure. A hard character cap of 700 characters per chunk was set after discovering during development that the embedding server would crash on inputs crossing roughly 512 tokens — chunk size was deliberately kept well under that threshold for reliability, not just context economy.
- **Retrieval:** FAISS `IndexFlatIP` (exact inner-product search over normalized embeddings) chosen over an approximate index, since the corpus size (511 chunks) is small enough that exact search costs nothing in speed while guaranteeing correctness — approximate indexing only pays off at a much larger scale.
- **Reranking:** A lexical + metadata-aware reranker sits on top of raw FAISS retrieval. This was added after evaluation showed raw vector search alone was insufficiently discriminating on this corpus (see Evaluation below) — reranking corrects cases where FAISS returns generic, broadly-worded chunks that superficially match many different queries.
- **Alternatives considered:** An earlier iteration used a metadata pre-filter that hard-truncated the candidate pool before reranking; this was found to actively harm retrieval quality on production-shaped data (real chunks only carry `id/source/page/text`, no `crop/topic/region` tags), since the filter would silently fall back to alphabetical-by-ID ordering and discard FAISS's similarity ranking entirely. This was identified via code review, fixed, and covered by a regression test using production-shaped fixtures so the bug class cannot silently reappear.

---

## Constraints

- **Target hardware:** ADTC Standard Laptop — Intel i5 10th-12th gen / AMD Ryzen 5, 8GB DDR4 RAM, integrated graphics only, Ubuntu 22.04 LTS.
- **No GPU acceleration** — pure CPU inference via llama.cpp, `--parallel 1` (single inference slot) to minimize KV-cache memory overhead rather than trading RAM for concurrency the target use case doesn't need.
- **100% offline at query time** — no network dependency once the model and index are in place; all corpus building (PDF sourcing, chunking, embedding) happens as a one-time offline preprocessing step, and the resulting FAISS index (`corpus/index.faiss`) and metadata (`corpus/metadata.json`) are committed directly to the repository so the deployed application never needs network access, including for index construction.
- **Language scope:** English, isiZulu (zu), isiXhosa (xh) — validated by direct model testing in both African languages with coherent responses.
- **Data availability:** Two of the six source PDFs (ARC Winter Vegetables Guide, Climate-Smart Agriculture Manual) were intermittently unreachable via automated fetch from ARC's server; both were retrieved successfully via direct browser download, suggesting a bot-blocking or rate-limiting policy on ARC's side rather than the documents being unavailable.

---

## Benchmarks

| Metric | Value |
|---|---|
| Corpus | 6 South African agricultural extension PDFs (ARC, NDA), 511 chunks after heading-aware chunking |
| Model size | 2.14 GB (Q4_K_M) |
| Estimated efficiency score | `(7GB − 2.14GB) / 7GB × 100 ≈ 69%` at n_ctx=1024 baseline |
| Generation context | n_ctx=4096, raised from an initial 1024 after measurement showed real query+context prompts (~950-1,100 tokens for 4 retrieved chunks + instructions + question) regularly exceeded 1024, causing prompt-overflow failures |
| Context window RAM cost | Analytical estimate (see note below): approximately +220-300 MiB RSS moving from n_ctx=1024 to 4096, roughly a 9-12% increase in peak memory — a deliberate accuracy-over-efficiency tradeoff, since n_ctx=1024 guaranteed generation failures on real queries |
| Retrieval corpus build time | 511 chunks embedded in well under one minute on a 28-core cloud CPU instance; multi-hour runs were observed on memory-constrained development containers, illustrating the sensitivity of embedding throughput to available RAM/CPU headroom rather than model size alone |

**Note on RAM benchmarking:** A dedicated benchmarking script (`scripts/bench_rss.py`) was built to directly measure peak RSS at different `n_ctx` settings. Direct measurement was not possible on the development container used partway through this project due to a per-process memory ceiling below the model's own weight size (every load attempt was terminated by the environment, confirmed via a raw-allocation control test independent of the model itself). The RAM figures above are therefore analytical, derived from the GGUF header (architecture cohere2, 36 layers, 4 KV heads via GQA, head_dim 128, f16 KV cache) rather than measured. `scripts/bench_rss.py` remains in the repository and is intended to be run on the actual ADTC reference laptop or equivalent unconstrained hardware for a final measured figure before submission.

These are self-reported development benchmarks. Official scores are measured by the ADTC profiler on the standard evaluation machine.

---

## Retrieval Evaluation

Raw FAISS retrieval (dense vector similarity only) was compared against the full pipeline (FAISS retrieval + lexical/metadata reranking) on real corpus data using two representative test prompts:

**Prompt 1:** *"What is the best time to plant maize in the Eastern Cape province of South Africa, and what soil preparation is needed?"*
- Raw FAISS top-4: `7-Vegetable production_0015, _0014, _0013, _0012`
- Reranked top-4: `7-Vegetable production_0003, _0014, _0013, _0008`

**Prompt 2:** *"My tomato leaves are turning yellow from the bottom up. What could be causing this and how do I treat it?"*
- Raw FAISS top-4: `7-Vegetable production_0015, _0014, _0013, _0012` (**identical to Prompt 1**)
- Reranked top-4: `7-Vegetable production_0009, _0012, _0008, _0006`

**Finding:** Raw FAISS retrieval returned the *same* top-4 chunks for two unrelated agricultural questions. Inspection showed these four chunks are all general climate/water-use passages from the Climate-Smart Agriculture manual — content that is broadly worded enough to sit close, in embedding space, to almost any agricultural query. This is a genuine, measurable limitation of single-model dense retrieval on a modest-sized corpus: generic content can crowd out topically specific content purely on vector proximity.

The reranking layer corrected this in both cases, producing different, more topically appropriate top-4 sets for each distinct query. This result is the concrete justification for including a reranking stage rather than relying on vector search alone, and is treated here as a known limitation with a working mitigation, rather than a fully solved problem — a dedicated smaller embedding model (rather than reusing the generative Tiny Aya Earth model for embeddings) is the most likely further improvement, noted as a stretch goal given remaining time.

---

## Known Limitations

- RAM benchmarks for the generation context window are analytical, not directly measured, due to a development environment memory constraint (see Benchmarks section). `scripts/bench_rss.py` is included for a follow-up measured run on reference-equivalent hardware.
- Raw dense retrieval alone shows a measurable weakness on generic/broadly-worded corpus content (see Retrieval Evaluation); the reranking layer mitigates but does not eliminate this.
- Embeddings and generation currently share a single 3.35B model; a dedicated smaller embedding model is a likely quality improvement not yet implemented.
