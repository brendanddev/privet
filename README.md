
# Local RAG Assistant

A fully local, private document assistant powered by Ollama, LlamaIndex, and ChromaDB. Ask questions about your documents... nothing leaves your machine.

---

## Philosophy

Most AI document tools send your data to the cloud. This project is built around three core principles:

**Privacy first.** Every part of the pipeline runs on your machine. Your documents, your embeddings, your queries — none of it is transmitted to any external server. No OpenAI, no cloud APIs, no tracking.

**Hardware optimized.** Built to run efficiently on consumer hardware with limited RAM. The goal is maximum performance from minimum resources — smaller indexes, quantized models, efficient retrieval.

**Ease of use.** A local AI tool should be as simple to use as any cloud alternative. No technical setup beyond the initial install, no command line required to use it day to day.

---


## Features

- **Fully local pipeline** — every component runs on your machine, no external API calls
- **PDF document ingestion** — drop any PDF into the `docs/` folder and ask questions about it
- **Persistent index** — documents are only embedded once, startup is near-instant on subsequent runs
- **Source citations** — every answer includes the source document and page number it was pulled from
- **Relevance scoring** — see how confident the retrieval was for each source chunk
- **Debug sidebar** — live panel showing engine stats, collection info, and last query sources
- **Query + startup benchmarking** — response times logged and displayed in the UI
- **Structured logging** — all engine activity written to daily log files in `logs/`
- **RAG Debugger** — standalone inspection tool to analyze chunks, embeddings, similarity scores, and trace queries

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

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full performance data and optimization history.

---

