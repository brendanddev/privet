
# Local RAG Assistant

A fully local, private document assistant powered by Ollama, LlamaIndex, and ChromaDB. Ask questions about your documents... nothing leaves your machine.

---

## Philosophy

Most AI document tools send your data to the cloud. This project is built around three core principles:

**Privacy first.** Every part of the pipeline runs on your machine. Your documents, your embeddings, your queries — none of it is transmitted to any external server. No OpenAI, no cloud APIs, no tracking.

**Hardware optimized.** Built to run efficiently on consumer hardware with limited RAM. The goal is maximum performance from minimum resources — smaller indexes, quantized models, efficient retrieval.

**Ease of use.** A local AI tool should be as simple to use as any cloud alternative. No technical setup beyond the initial install, no command line required to use it day to day.

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

### How to Measure Yourself
\```bash
# Index size on disk
du -sh chroma_db/

# RAM usage — run this while the app is running
top -pid $(pgrep -f streamlit)
\```

---

