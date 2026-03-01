
# Local RAG Assistant

A fully local, private document assistant powered by Ollama, LlamaIndex, and ChromaDB. Ask questions about your documents... nothing leaves your machine.

---

## Models
- **LLM:** `gemma3:1b` — lightweight local language model via Ollama
- **Embeddings:** `nomic-embed-text` — local embedding model via Ollama

---

## Setup

### Installation
```bash
git clone https://github.com/brendanddev/local-rag-assistant.git
cd local-rag-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pull required models
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### Add your documents
Drop any PDF files into the `docs/` folder.

---

## Performance Benchmarks

All benchmarks run on a MacBook Air M-series, 8GB RAM.

### Current Baseline (v0.1)
| Metric | Value |
|---|---|
| Startup / index time | _s |
| Query response time | _s |
| RAM usage (idle) | _ MB |
| RAM usage (during query) | _ MB |
| ChromaDB index size on disk | _ MB |
| Embedding dimensions | 768 |
| Embedding precision | float32 |
| Avg chunk size | 1232 chars |
| Model | gemma3:1b |
| Embedding model | nomic-embed-text |

### How to Measure Yourself
\```bash
# Index size on disk
du -sh chroma_db/

# RAM usage — run this while the app is running
top -pid $(pgrep -f streamlit)
\```

---

