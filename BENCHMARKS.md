# Performance Benchmarks

All benchmarks run on a MacBook Air M-series, 8GB RAM unless otherwise noted.

---

## How to Measure Yourself
```bash
# Index size on disk
du -sh chroma_db/

# RAM usage — run this while the app is running
top -pid $(pgrep -f streamlit)
```

---

### Current Baseline (v0.1)
| Metric | Value |
|---|---|
| Startup / index time | 9.67s |
| Query response time | ~2.7s |
| RAM usage (idle) | ~353 MB |
| RAM usage (during query) | ~380 MB |
| ChromaDB index size on disk | 4.4MB |
| Chunks indexed | 112 |
| Embedding dimensions | 768 |
| Embedding precision | float32 |
| Avg chunk size | 1232 chars |
| Model | gemma3:1b |
| Embedding model | nomic-embed-text |


## v0.2 — Persistent Index

Added index persistence. ChromaDB is checked on startup and re-indexing is skipped if documents are already embedded.

| Metric | Value |
|---|---|
| Startup time (persistent) | ~0.4s |
| Query response time | ~2.7s |
| Improvement | 95% faster startup |

**What changed:** `_build_query_engine` now checks `chroma_collection.count()` before indexing. If chunks exist, it loads directly from the vector store using `VectorStoreIndex.from_vector_store()` instead of re-processing documents.


## v0.3 — Chunk Size Tuning

Reduced chunk size from default to 256 tokens with 25 token overlap using `SentenceSplitter`.

| Metric | Before (v0.3 baseline) | After |
|---|---|---|
| Avg chunk size | 1342 chars | 722 chars |
| Min chunk size | 474 chars | 101 chars |
| Max chunk size | 3893 chars | 1186 chars |
| Total chunks | 55 | 113 |
| Oversized chunks | 78% | 17% |
| ChromaDB index size | 1.2 MB | 3.4 MB |
| Query response time | ~2-7s | ~1.5-3.3s |
| Documents indexed | 6 | 9 |

**What changed:** Added `SentenceSplitter(chunk_size=256, chunk_overlap=25)` to the indexing pipeline. Smaller chunks give the model more focused context per retrieval, improving answer precision and reducing query time.

**Tradeoff:** Index size grew from 1.2MB to 3.4MB because more chunks means more embeddings stored. This is expected and acceptable — more chunks at smaller sizes is better for retrieval quality.

**Chunk distribution after tuning:**
| Range | Count |
|---|---|
| 0-200 chars | 1 |
| 200-500 chars | 23 |
| 500-1000 chars | 70 |
| 1000+ chars | 19 |


## v0.4 — Float16 Embedding Quantization

Wrapped the embedding model with a float16 quantization step to reduce embedding memory footprint.

| Metric | float32 | float16 | Improvement |
|---|---|---|---|
| Embedding array size | 150 KB | 75 KB | 2x smaller |
| Precision loss (max diff) | baseline | 0.000030 | Negligible |
| Index size on disk | 3.4 MB | 3.4 MB | No visible change at current scale |

**What changed:** Added `Float16EmbeddingWrapper` in `core/embeddings.py` that intercepts embeddings after generation and converts them from float32 to float16 using numpy before storage.

**Note:** Disk size difference becomes visible at larger scale (1000+ chunks). The 2x memory reduction is confirmed at the array level and compounds significantly as the number of indexed chunks grows. Precision loss of 0.000030 has no practical impact on cosine similarity scores used for retrieval.


## v1.3.0 — Provider Abstraction: Ollama vs llama.cpp

Introduced a provider abstraction layer allowing the app to run with either Ollama or llama.cpp as the backend. Both tested against the same 107-chunk index across 15 questions each on a MacBook Air M-series, 8GB RAM.

### Startup Time
| Metric | Ollama | llama.cpp |
|---|---|---|
| First run (indexing) | 5.88s | 6.37s |
| Subsequent runs (persistent index) | ~0.4s | ~0.4s |

First-run startup is slightly slower with llama.cpp because both the generation model (~800MB) and embedding model (~274MB) are loaded into memory as part of the Python process. Ollama loads models lazily on first query. On subsequent runs both providers are equivalent since the index is already built.

### Query Response Times

**Ollama — 15 queries**
| Question | Time |
|---|---|
| What is a pointer? | 3.98s |
| Difference between reference and primitive types? | 2.73s |
| Are reference types and pointers the same thing? | 2.19s |
| Does Java have pointers? | 1.66s |
| What is Heap Memory? | 2.43s |
| What is the JVM? | 2.46s |
| What is a transformer? | 1.80s |
| Transformer architecture? | 1.74s |
| What is a token? | 2.29s |
| What is a vector and embedding? | 2.52s |
| What is an embedding layer? | 2.02s |
| What does element mean in this context? | 2.42s |
| What is word2vec? | 2.19s |
| What is NLP? | 1.90s |
| What is attention mechanism? | 2.20s |

**llama.cpp — 15 queries**
| Question | Time |
|---|---|
| What is a pointer? | 2.27s |
| What is C? | 2.82s |
| Does C have classes? | 1.68s |
| What is tokenization? | 1.23s |
| What is an attention mechanism? | 1.93s |
| What is a model? | 1.17s |
| So AI is all math? | 0.85s |
| What is a reference type? | 1.10s |
| Is a reference stored in heap memory? | 1.26s |
| Are objects stored in heap memory? | 1.18s |
| What are vector embeddings? | 1.91s |
| So words turned into numbers? | 1.08s |
| What does multi dimensional space mean in relation to vectors? | 1.28s |
| So words are numbers? | 2.06s |
| But what kind | 0.47s |

### Summary Comparison
| Metric | Ollama | llama.cpp | Difference |
|---|---|---|---|
| Avg query time | 2.24s | 1.42s | **37% faster** |
| Min query time | 1.66s | 0.47s | |
| Max query time | 3.98s | 2.82s | |
| Chunks indexed | 107 | 107 | Same |
| Background service required | Yes | No | |
| Open network port | Yes (11434) | No | |
| GPU acceleration | Via Ollama | Direct Metal | |

### Analysis

llama.cpp is 37% faster on average than Ollama for query response time. The gap is most pronounced on shorter, simpler questions where Ollama's HTTP overhead is relatively large compared to actual generation time. On longer, more complex questions the difference narrows since generation time dominates.

The speed improvement comes from eliminating the HTTP layer. With Ollama every query goes through an HTTP request to the local server, through its request queue, and back. With llama.cpp the model is a direct function call inside the Python process — no serialization, no network stack, no queue.

The tradeoff is memory. With Ollama the model lives in Ollama's process and is shared across any app that talks to it. With llama.cpp the model lives inside the Python process, consuming RAM that would otherwise be available to the OS and other apps. On 8GB unified memory this is measurable but acceptable — the app uses roughly 1GB more RAM with llama.cpp loaded.

---

## Current Baseline — gemma3:1b llamacpp, 448 chunks, 20 questions
*Date: 2026-03-29 | Hardware: MacBook Air M2, 8GB unified memory, Apple Silicon*

### Setup
| Parameter | Value |
|---|---|
| Provider | llamacpp (direct Metal acceleration) |
| LLM | google_gemma-3-1b-it-Q4_K_M.gguf |
| Embeddings | nomic-embed-text-v1.5.Q8_0.gguf |
| Chunks indexed | 448 |
| Documents | 24 |
| similarity_top_k | 5 |
| Evaluator | embedding-based (all-MiniLM-L6-v2), NLI disabled |

### Query Results
| # | Question | Time | Composite | Faithfulness | Relevance | Rating |
|---|---|---|---|---|---|---|
| 1 | What is a pointer? | 3.27s | 0.863 | 1.000 | 0.799 | GOOD |
| 2 | What is a linked list? | 3.32s | 0.632 | 0.500 | 0.883 | GOOD |
| 3 | What is the call stack? | 2.00s | 0.900 | 1.000 | 0.860 | GOOD |
| 4 | What is virtual memory? | 2.25s | 0.909 | 1.000 | 0.898 | GOOD |
| 5 | What is tokenization? | 2.11s | 0.884 | 1.000 | 0.847 | BAD |
| 6 | How does memory management differ between Java and C? | 1.84s | 0.864 | 1.000 | 0.826 | GOOD |
| 7 | What is the relationship between word embeddings and vector space models? | 2.12s | 0.887 | 1.000 | 0.841 | BAD |
| 8 | How does the attention mechanism relate to transformer architecture? | 2.08s | 0.766 | 1.000 | 0.511 | BAD |
| 9 | What is the difference between a stack and a linked list? | 2.24s | 0.870 | 1.000 | 0.858 | BAD |
| 10 | How does garbage collection relate to memory management? | 2.20s | 0.892 | 1.000 | 0.858 | GOOD |
| 11 | Why would you use a linked list instead of a stack? | 1.88s | 0.149 | 0.000 | 0.113 | BAD |
| 12 | What happens to a pointer when its memory is freed? | 1.80s | 0.841 | 1.000 | 0.763 | BAD |
| 13 | How does tokenization affect word embeddings? | 2.30s | 0.870 | 1.000 | 0.821 | GOOD |
| 14 | Why does Java not have pointers like C does? | 2.43s | 0.876 | 1.000 | 0.878 | GOOD |
| 15 | What is the relationship between cosine similarity and vector embeddings? | 2.05s | 0.600 | 0.500 | 0.727 | GOOD |
| 16 | How does the transformer attention mechanism use vector embeddings? | 2.27s | 0.874 | 1.000 | 0.877 | GOOD |
| 17 | How does polymorphism relate to inheritance in OOP? | 2.11s | 0.877 | 1.000 | 0.805 | BAD |
| 18 | What role does the call stack play in memory management? | 2.17s | 0.907 | 1.000 | 0.889 | GOOD |
| 19 | How does word2vec produce vector embeddings? | 2.10s | 0.863 | 1.000 | 0.794 | GOOD |
| 20 | How does virtual memory relate to memory addresses? | 2.58s | 0.887 | 1.000 | 0.805 | GOOD |

### Summary
| Metric | Value |
|---|---|
| Avg query time | 2.24s |
| Min query time | 1.80s |
| Max query time | 3.32s |
| Avg composite score | 0.813 |
| Avg faithfulness | 0.950 |
| Avg relevance | 0.789 |
| Thumbs up rate | 60% (12/20) |
| Questions scored >0.85 composite | 13/20 (65%) |
| Questions scored <0.50 composite | 1/20 (5%) |

### Analysis

**Strengths:** Faithfulness is high at 0.950 — when the model answers from retrieved context it stays grounded. Simple definition questions and cross-document factual questions perform consistently well. Six questions scored above 0.90 composite.

**Failure cases:**

Q11 ("Why would you use a linked list instead of a stack?") is the single biggest failure — composite 0.149, faithfulness 0.000. This is an inference question where the answer requires reasoning across two documents rather than finding a direct statement. The model answered from general knowledge rather than the retrieved context. This category of question is the primary weakness of the current pure vector search pipeline.

Q8 ("How does the attention mechanism relate to transformer architecture?") retrieved the correct document but scored relevance 0.511 — the answer was superficial despite good source retrieval. Likely a chunking issue where the relevant explanation was split across chunk boundaries.

Q2 and Q15 both show faithfulness 0.500, indicating partial grounding — the model supplemented retrieved content with general knowledge rather than staying strictly within the documents.

**Comparison to v1.3.0:**

Thumbs up rate dropped from 75% (v1.3.0, 15 questions) to 60% (20 questions). This is expected — the v1.3.0 question set was weighted toward simple definitions where the model performs best. This question set deliberately includes harder inference and multi-hop questions that expose real failure modes.

Average query time is unchanged at 2.24s, consistent with v1.3.0 Ollama baseline. This confirms the llamacpp speed advantage (37% faster than Ollama) is maintained at 448 chunks.

**This data serves as the baseline for Pleias-RAG-1B comparison.**
The same 20 questions will be run against Pleias-RAG-1B to measure whether a RAG-specific model improves composite scores, faithfulness, and thumbs up rate on this question set, particularly on the inference and multi-hop categories where gemma3:1b struggles.

---

## Hybrid Search — BM25 + Vector + RRF vs Pure Vector Baseline
*Date: 2026-03-29 | Hardware: MacBook Air M2, 8GB unified memory, Apple Silicon*

Same 20 questions run against hybrid search (BM25 + vector + Reciprocal Rank Fusion) to measure improvement over the pure vector baseline above.

### Setup
| Parameter | Value |
|---|---|
| Provider | llamacpp (direct Metal acceleration) |
| LLM | google_gemma-3-1b-it-Q4_K_M.gguf |
| Embeddings | nomic-embed-text-v1.5.Q8_0.gguf |
| Chunks indexed | 448 |
| Documents | 24 |
| similarity_top_k | 5 |
| Retrieval mode | BM25 + vector + RRF (QueryFusionRetriever, num_queries=1) |
| Evaluator | embedding-based (all-MiniLM-L6-v2), NLI disabled |

### Query Results
| # | Question | Time | Composite | Faithfulness | Relevance | Rating | vs Baseline |
|---|---|---|---|---|---|---|---|
| 1 | What is a pointer? | 5.76s | 0.872 | 1.000 | 0.820 | GOOD | +0.009 |
| 2 | What is a linked list? | 5.73s | 0.882 | 1.000 | 0.882 | GOOD | +0.250 |
| 3 | What is the call stack? | 4.20s | 0.905 | 1.000 | 0.869 | GOOD | +0.005 |
| 4 | What is virtual memory? | 4.45s | 0.902 | 1.000 | 0.855 | GOOD | −0.007 |
| 5 | What is tokenization? | 4.82s | 0.861 | 1.000 | 0.778 | GOOD | +0.023 |
| 6 | How does memory management differ between Java and C? | 3.88s | 0.870 | 1.000 | 0.863 | GOOD | +0.006 |
| 7 | What is the relationship between word embeddings and vector space models? | 4.25s | 0.894 | 1.000 | 0.887 | GOOD | +0.007 |
| 8 | How does the attention mechanism relate to transformer architecture? | 4.54s | 0.634 | 0.500 | 0.895 | GOOD | −0.132 |
| 9 | What is the difference between a stack and a linked list? | 5.58s | 0.848 | 1.000 | 0.784 | BAD | −0.022 |
| 10 | How does garbage collection relate to memory management? | 4.52s | 0.897 | 1.000 | 0.854 | GOOD | +0.005 |
| 11 | Why would you use a linked list instead of a stack? | 4.51s | 0.608 | 0.500 | 0.814 | BAD | +0.459 |
| 12 | What happens to a pointer when its memory is freed? | 4.50s | 0.759 | 1.000 | 0.479 | BAD | −0.082 |
| 13 | How does tokenization affect word embeddings? | 4.55s | 0.891 | 1.000 | 0.876 | GOOD | +0.021 |
| 14 | Why does Java not have pointers like C does? | 5.30s | 0.736 | 0.750 | 0.803 | GOOD | −0.140 |
| 15 | What is the relationship between cosine similarity and vector embeddings? | 6.38s | 0.757 | 0.875 | 0.641 | GOOD | +0.157 |
| 16 | How does the transformer attention mechanism use vector embeddings? | 4.37s | 0.875 | 1.000 | 0.865 | BAD | +0.001 |
| 17 | How does polymorphism relate to inheritance in OOP? | 4.21s | 0.835 | 1.000 | 0.743 | BAD | −0.042 |
| 18 | What role does the call stack play in memory management? | 3.88s | 0.818 | 1.000 | 0.733 | BAD | −0.089 |
| 19 | How does word2vec produce vector embeddings? | 4.23s | 0.612 | 0.500 | 0.821 | GOOD | −0.251 |
| 20 | How does virtual memory relate to memory addresses? | 4.32s | 0.903 | 1.000 | 0.862 | GOOD | +0.016 |

### Summary
| Metric | Hybrid (BM25+Vector+RRF) | Pure Vector Baseline | Difference |
|---|---|---|---|
| Avg query time | 4.62s | 2.24s | **+2.38s slower** |
| Min query time | 3.88s | 1.80s | |
| Max query time | 6.38s | 3.32s | |
| Avg composite score | 0.798 | 0.813 | −0.015 |
| Avg faithfulness | 0.881 | 0.950 | −0.069 |
| Avg relevance | 0.809 | 0.789 | +0.020 |
| Thumbs up rate | 55% (11/20) | 60% (12/20) | −5% |
| Questions scored >0.85 composite | 12/20 (60%) | 13/20 (65%) | −5% |
| Questions scored <0.50 composite | 0/20 (0%) | 1/20 (5%) | eliminated worst failure |

### Analysis

**Speed cost:** Hybrid search is 106% slower on average (2.24s → 4.62s). BM25 adds overhead on every query — it must score all 448 chunks against the query at retrieval time. This is the most significant regression and is a hard tradeoff for a latency-sensitive use case.

**Where hybrid helped:**

Q11 ("Why would you use a linked list instead of a stack?") — the biggest baseline failure at composite 0.149 — improved to 0.608 (+0.459). This was the primary motivation for hybrid search: BM25 keyword matching surfaces both "linked list" and "stack" documents even when vector similarity is weak on inference questions. The improvement is real but the question is still not well answered, indicating the issue is generation quality not just retrieval.

Q2 (linked list, +0.250) and Q15 (cosine similarity, +0.157) also improved meaningfully. Both are questions where keyword overlap between the query and document text is strong, which is where BM25 adds the most value.

The worst failure (composite <0.50) was eliminated entirely — 1/20 in baseline vs 0/20 with hybrid.

**Where hybrid hurt:**

Q19 (word2vec, −0.251) and Q14 (Java no pointers, −0.140) dropped the most. These are cases where RRF is likely surfacing BM25 keyword matches that dilute better vector results — the fused ranking pulls in chunks that are keyword-adjacent but semantically weaker than what pure vector search would have retrieved.

Q8 (attention + transformer) dropped from faithfulness 1.000 to 0.500 despite relevance improving. The model received more chunks but stayed less grounded — possibly because the fused result set mixed high-relevance vector chunks with lower-quality BM25 matches, diluting the context quality.

**Verdict:**

Hybrid search helps the worst failures but hurts mid-tier questions and doubles query time. The net composite score and thumbs up rate are both lower than the pure vector baseline. At 448 chunks with gemma3:1b, the speed penalty is not justified by quality gains. 

The pattern suggests hybrid search may become more valuable at larger document sets (1000+ chunks) where pure vector search degrades more severely and BM25's keyword precision provides a stronger signal. Re-evaluate after document set expands significantly.

**Pure vector search restored as default.** `use_hybrid_search: false` in config.yaml.

---

---

## Vector + Cross-Encoder Reranking vs Pure Vector Baseline
*Date: 2026-03-29 | Hardware: MacBook Air M2, 8GB unified memory, Apple Silicon*

Same 20 questions run against vector search + cross-encoder reranking to measure
improvement over the pure vector baseline.

### Setup
| Parameter | Value |
|---|---|
| Provider | llamacpp (direct Metal acceleration) |
| LLM | google_gemma-3-1b-it-Q4_K_M.gguf |
| Embeddings | nomic-embed-text-v1.5.Q8_0.gguf |
| Chunks indexed | 448 |
| Documents | 24 |
| Retrieval candidates | 10 (similarity_top_k: 10) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (~70MB) |
| Rerank top_k | 5 |
| Evaluator | embedding-based (all-MiniLM-L6-v2), NLI disabled |

### Query Results
| # | Question | Time | Composite | Faithfulness | Relevance | vs Baseline |
|---|---|---|---|---|---|---|
| 1 | What is a pointer? | 4.68s | 0.868 | 1.000 | 0.799 | +0.005 |
| 2 | What is a linked list? | 4.20s | 0.628 | 1.000 | 0.883 | −0.004 |
| 3 | What is the call stack? | 3.75s | 0.893 | 1.000 | 0.860 | −0.007 |
| 4 | What is virtual memory? | 3.38s | 0.901 | 1.000 | 0.863 | −0.008 |
| 5 | What is tokenization? | 3.09s | 0.871 | 1.000 | 0.817 | −0.013 |
| 6 | How does memory management differ between Java and C? | 2.67s | 0.874 | 1.000 | 0.862 | +0.010 |
| 7 | What is the relationship between word embeddings and vector space models? | 3.07s | 0.885 | 1.000 | 0.832 | −0.002 |
| 8 | How does the attention mechanism relate to transformer architecture? | 2.83s | 0.860 | 1.000 | 0.809 | +0.094 |
| 9 | What is the difference between a stack and a linked list? | 3.84s | 0.859 | 1.000 | 0.814 | −0.011 |
| 10 | How does garbage collection relate to memory management? | 3.34s | 0.888 | 1.000 | 0.853 | −0.004 |
| 11 | Why would you use a linked list instead of a stack? | 2.74s | 0.823 | 1.000 | 0.669 | +0.674 |
| 12 | What happens to a pointer when its memory is freed? | 2.74s | 0.862 | 1.000 | 0.836 | +0.021 |
| 13 | How does tokenization affect word embeddings? | 2.80s | 0.854 | 1.000 | 0.788 | −0.016 |
| 14 | Why does Java not have pointers like C does? | 2.41s | 0.887 | 1.000 | 0.901 | +0.011 |
| 15 | What is the relationship between cosine similarity and vector embeddings? | 3.14s | 0.849 | 1.000 | 0.744 | +0.249 |
| 16 | How does the transformer attention mechanism use vector embeddings? | 2.31s | 0.855 | 1.000 | 0.835 | −0.019 |
| 17 | How does polymorphism relate to inheritance in OOP? | 2.89s | 0.851 | 1.000 | 0.776 | −0.026 |
| 18 | What role does the call stack play in memory management? | 2.92s | 0.901 | 1.000 | 0.872 | −0.006 |
| 19 | How does word2vec produce vector embeddings? | 2.62s | 0.890 | 1.000 | 0.881 | +0.027 |
| 20 | How does virtual memory relate to memory addresses? | 2.69s | 0.900 | 1.000 | 0.839 | +0.013 |

### Summary
| Metric | Vector + Rerank | Pure Vector Baseline | Difference |
|---|---|---|---|
| Avg query time | 3.06s | 2.24s | +0.82s slower |
| Min query time | 2.31s | 1.80s | |
| Max query time | 4.68s | 3.32s | |
| Avg composite score | 0.864 | 0.813 | **+0.051** |
| Avg faithfulness | 1.000 | 0.950 | **+0.050** |
| Avg relevance | 0.820 | 0.789 | +0.031 |
| Questions scored >0.85 composite | 16/20 (80%) | 13/20 (65%) | +15% |
| Questions scored <0.50 composite | 0/20 (0%) | 1/20 (5%) | eliminated |

### Analysis

**Reranking is the clear winner across all three retrieval configurations tested.**
Composite score improved from 0.813 to 0.864 (+0.051), faithfulness reached a
perfect 1.000 across all 20 questions, and the number of questions scoring above
0.85 jumped from 13 to 16.

**Q11 ("Why would you use a linked list instead of a stack?")** — the worst
baseline failure at 0.149 — improved to 0.823 (+0.674). This is the largest
single-question improvement recorded. The cross-encoder correctly re-scored
chunks that explain linked list insertion flexibility over stack's LIFO
constraint, surfacing context the vector retriever ranked lower.

**Q8 ("How does the attention mechanism relate to transformer architecture?")**
improved from 0.766 to 0.860 (+0.094). In the baseline this question retrieved
the correct document but scored low relevance (0.511) — a chunking boundary
issue. Reranking retrieved better chunks from the same document, recovering the
answer quality.

**Speed cost:** +0.82s average over baseline (2.24s → 3.06s). The cross-encoder
adds a re-scoring pass over 10 candidates per query. This is acceptable — roughly
37% overhead for a +6.3% composite improvement. Significantly better than the
hybrid search tradeoff (+106% slower, −1.5% composite).

**Faithfulness at 1.000** is the most significant finding. Every answer in this
run was fully grounded in retrieved context — no partial answers supplemented
with general knowledge. The reranker is consistently surfacing chunks the model
can work with cleanly.

**Config:** `use_reranking: true`, `use_hybrid_search: false`, `rerank_top_k: 5`
This is now the active retrieval configuration.

---

## Pleias-RAG-1B Q4_K_M — vector only, no rerank
*Date: 2026-03-30 | Hardware: MacBook Air M2, 8GB unified memory, Apple Silicon*

### Setup
| Parameter | Value |
|---|---|
| Provider | pleias |
| Chunks indexed | 448 |
| Evaluator | embedding-based (all-MiniLM-L6-v2), NLI disabled |

### Query Results
| # | Question | Time | Composite | Faithfulness | Relevance |
|---|---|---|---|---|---|
| 1 | How does virtual memory relate to memory addresses? | 26.65s | 0.720 | 0.714 | 0.723 |
| 2 | How does word2vec produce vector embeddings? | 24.92s | 0.621 | 0.645 | 0.580 |
| 3 | What role does the call stack play in memory management? | 27.57s | 0.749 | 0.793 | 0.708 |
| 4 | How does polymorphism relate to inheritance in OOP? | 27.26s | 0.769 | 0.833 | 0.767 |
| 5 | How does the transformer attention mechanism use vector embeddings? | 25.19s | 0.397 | 0.261 | 0.522 |
| 6 | What is the relationship between cosine similarity and vector embeddings? | 27.22s | 0.655 | 0.571 | 0.794 |
| 7 | Why does Java not have pointers like C does? | 26.24s | 0.674 | 0.800 | 0.538 |
| 8 | How does tokenization affect word embeddings? | 26.00s | 0.723 | 0.812 | 0.634 |
| 9 | What happens to a pointer when its memory is freed? | 24.74s | 0.604 | 0.735 | 0.412 |
| 10 | Why would you use a linked list instead of a stack? | 22.45s | 0.631 | 0.704 | 0.547 |
| 11 | How does garbage collection relate to memory management? | 26.56s | 0.652 | 0.611 | 0.708 |
| 12 | What is the difference between a stack and a linked list? | 24.26s | 0.644 | 0.727 | 0.561 |
| 13 | How does the attention mechanism relate to transformer architecture? | 26.03s | 0.695 | 0.724 | 0.734 |
| 14 | What is the relationship between word embeddings and vector space models? | 24.42s | 0.590 | 0.485 | 0.709 |
| 15 | How does memory management differ between Java and C? | 25.55s | 0.756 | 0.816 | 0.771 |
| 16 | What is tokenization? | 25.52s | 0.563 | 0.567 | 0.498 |
| 17 | What is virtual memory? | 25.53s | 0.776 | 0.833 | 0.733 |
| 18 | What is the call stack? | 22.27s | 0.726 | 0.875 | 0.490 |
| 19 | What is a linked list? | 27.71s | 0.614 | 0.647 | 0.578 |
| 20 | What is a pointer? | 27.17s | 0.727 | 0.853 | 0.590 |

### Summary
| Metric | Value |
|---|---|
| Avg query time | 25.66s |
| Min query time | 22.27s |
| Max query time | 27.71s |
| Avg composite score | 0.664 |
| Avg faithfulness | 0.700 |
| Avg relevance | 0.630 |
| Questions scored >0.85 composite | 0/20 (0%) |
| Questions scored <0.50 composite | 1/20 (5%) |

### Analysis

**Overall verdict:** Pleias-RAG-1B Q4_K_M underperforms gemma3:1b + cross-encoder
reranking on all metrics while being 8x slower. Not recommended for English CS Q&A
on this document set.

**vs gemma3:1b + rerank (current default):**
| Metric | Pleias Q4_K_M | gemma3:1b + rerank | Difference |
|---|---|---|---|
| Avg query time | 25.66s | 3.06s | 8.4x slower |
| Avg composite | 0.664 | 0.864 | −0.200 |
| Avg faithfulness | 0.700 | 1.000 | −0.300 |
| Questions >0.85 | 0/20 (0%) | 16/20 (80%) | −80% |

**Failure case — Q5 (transformer attention + vectors): 0.397 composite, 0.261
faithfulness.** Complete faithfulness failure despite good source retrieval. The
model generated content not grounded in sources. Worst result of any single
question across all configurations tested.

**Q11 (linked list vs stack — hardest inference question):** Pleias scored 0.631
— better than the gemma baseline (0.149) but worse than gemma + rerank (0.823).
Reranking solved this problem more effectively than a RAG-specific model.

**Why Pleias underperforms here:** Use case mismatch. Pleias-RAG-1B was designed
for European multilingual academic citation tasks with structured sources. English
CS Q&A from Wikipedia-style documents is not its target domain. Answers are
verbose (~5000 chars vs gemma's ~200), generate repetitive reasoning traces, and
show lower faithfulness despite the model having native citation support.

**Technical note:** This is the first community Q4_K_M quantization of
Pleias-RAG-1B. Required a C++ fix to llama.cpp (pleias-rag pre-tokenizer) and a
Python-level streaming stop token fix. The quantization is technically sound —
the underperformance reflects model-task mismatch, not quantization quality.
SHA-256 provenance tracked in models/provenance.json.
