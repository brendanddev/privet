
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

---