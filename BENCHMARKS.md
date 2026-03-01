
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

---