
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