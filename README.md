
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
- **Multi-format document ingestion** — supports PDF, TXT, DOCX, and CSV via folder drop or direct UI upload
- **Persistent index** — documents are only embedded once, startup is near-instant on subsequent runs
- **Document management** — add and remove documents through the UI, index stays in sync
- **Source citations** — every answer includes the source document and page number it was pulled from
- **Relevance scoring** — see how confident the retrieval was for each source chunk
- **Model switcher** — swap LLM and embedding models from the sidebar without restarting
- **Float16 embedding quantization** — embeddings stored at half precision, 2x memory reduction with negligible quality loss
- **Chunk size tuning** — documents split at 256 tokens for precise retrieval
- **Debug sidebar** — live panel with engine stats, chunk distribution chart, index health indicators, and retrieval confidence meters
- **Structured logging** — all engine activity written to daily log files in `logs/`
- **RAG Debugger** — standalone inspection tool to analyze chunks, embeddings, similarity scores, and trace queries
- **Docker support** — run the entire stack with a single command, no manual setup required

---

## Stack

| Component | Library | Purpose |
|---|---|---|
| LLM | Ollama | Runs the language model locally |
| RAG Framework | LlamaIndex 0.14.15 | Document ingestion and retrieval |
| Vector Store | ChromaDB 1.5.2 | Local embedding storage |
| UI | Streamlit 1.54.0 | Browser-based chat interface |
| Containers | Docker + Compose | Single command deployment |

---

## Models
- **LLM:** `gemma3:1b` — lightweight local language model via Ollama
- **Embeddings:** `nomic-embed-text` — local embedding model via Ollama

---

## Setup

### Option 1 — Docker (recommended)

The easiest way to run the app. No Python setup required.
```bash
git clone https://github.com/brendanddev/local-rag-assistant.git
cd local-rag-assistant
docker compose up --build
```

Then open your browser to `http://localhost:8501`

On first run Docker will pull the Ollama image and download the required models automatically. This takes a few minutes. Every run after is fast.

**GPU support:**
- **Linux + NVIDIA:** Works automatically
- **Mac:** Ollama uses Apple Metal automatically, no extra config needed. Remove the `deploy` block from `docker-compose.yml`
- **Windows:** Requires WSL2 with NVIDIA drivers

**To stop:**
```bash
docker compose down
```

**To stop and remove all data including downloaded models:**
```bash
docker compose down -v
```

---

### Option 2 — Manual Setup

**Prerequisites:**
- [Ollama](https://ollama.com) installed and running
- Python 3.11+

**Install:**
```bash
git clone https://github.com/brendanddev/local-rag-assistant.git
cd local-rag-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Pull required models:**
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

**Add your documents:**
Drop any supported file (PDF, TXT, DOCX, CSV) into the `docs/` folder, or upload directly through the UI.

**Run:**
```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — start the app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Run the debugger:**
```bash
python3 core/rag_debugger.py
```

---

## Project Structure
```
local-rag-assistant/
├── core/
│   ├── rag_engine.py       # RAG pipeline logic
│   ├── rag_debugger.py     # Inspection and debugging tools
│   └── embeddings.py       # Float16 embedding quantization wrapper
├── ui/
│   └── dashboard.py        # Sidebar debug panel
├── utils/
│   └── logger.py           # Logging configuration
├── app.py                  # Streamlit UI entry point
├── Dockerfile              # App container definition
├── docker-compose.yml      # Multi-container orchestration
├── entrypoint.sh           # Container startup script
├── docs/                   # Drop your documents here
├── chroma_db/              # Vector store (auto-generated)
├── logs/                   # Daily log files (auto-generated)
├── BENCHMARKS.md           # Performance data and optimization history
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full performance data and optimization history.

---

