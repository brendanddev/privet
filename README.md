
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
