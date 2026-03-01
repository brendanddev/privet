
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

---