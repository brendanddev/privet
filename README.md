
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
- **Provider abstraction** — swap between Ollama and llama.cpp via a single config value, no code changes required
- **llama.cpp support** — run GGUF models directly with no background service, no open ports, direct Metal acceleration on Apple Silicon
- **Multi-format document ingestion** — supports PDF, TXT, DOCX, and CSV via folder drop or direct UI upload
- **Persistent index** — documents are only embedded once, startup is near-instant on subsequent runs
- **Document management** — add and remove documents through the UI, index stays in sync
- **Streaming responses** — answers stream token by token for immediate feedback
- **Source citations** — every answer includes the source document and page number it was pulled from
- **Relevance scoring** — see how confident the retrieval was for each source chunk
- **Conversation memory** — the model remembers previous exchanges in the same session
- **Feedback system** — rate answers with thumbs up or down, feedback logged locally for future analysis and fine-tuning
- **Model switcher** — swap LLM and embedding models from the sidebar without restarting, provider-aware
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
| LLM (Ollama) | Ollama | Runs language models locally via HTTP |
| LLM (llama.cpp) | llama-cpp-python | Runs GGUF models directly, no service required |
| RAG Framework | LlamaIndex 0.14.15 | Document ingestion and retrieval |
| Vector Store | ChromaDB 1.5.2 | Local embedding storage |
| UI | Streamlit 1.54.0 | Browser-based chat interface |
| Containers | Docker + Compose | Single command deployment |

---

## Models

### Ollama provider
- **LLM:** `gemma3:1b` — lightweight local language model
- **Embeddings:** `nomic-embed-text` — dedicated local embedding model

### llama.cpp provider
- **LLM:** `google_gemma-3-1b-it-Q4_K_M.gguf` — quantized Gemma 3 1B, ~800MB
- **Embeddings:** `nomic-embed-text-v1.5.Q8_0.gguf` — quantized nomic embed, ~274MB

Both GGUF models available from [Hugging Face](https://huggingface.co). Place them in the `models/` folder.

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

## Feedback System

Every answer can be rated with a thumbs up or down. Ratings are stored locally in `feedback/feedback.jsonl` — one JSON entry per rating including the question, answer, sources retrieved, model used, and query time.

This data is used to identify patterns in answer quality and will serve as training data for future fine-tuning.

To disable feedback collection set `collect_feedback: false` in `config.yaml`.

---

## Project Structure
```
local-rag-assistant/
├── core/
│   ├── rag_engine.py           # RAG pipeline logic
│   ├── rag_debugger.py         # Inspection and debugging tools
│   ├── embeddings.py           # Float16 embedding quantization wrapper
│   └── providers/
│       ├── base.py             # Abstract provider interface
│       ├── ollama.py           # Ollama provider implementation
│       ├── llamacpp.py         # llama.cpp provider implementation
│       └── factory.py          # Provider selection logic
├── ui/
│   └── dashboard.py            # Sidebar debug panel
├── utils/
│   ├── logger.py               # Logging configuration
│   ├── config.py               # YAML config loader
│   └── feedback.py             # Feedback logging utility
├── app.py                      # Streamlit UI entry point
├── config.yaml                 # Runtime configuration (gitignored)
├── config.example.yaml         # Configuration template
├── Dockerfile                  # App container definition
├── docker-compose.yml          # Multi-container orchestration
├── entrypoint.sh               # Container startup script
├── models/                     # GGUF model files (gitignored)
├── docs/                       # Drop your documents here
├── chroma_db/                  # Vector store (auto-generated)
├── logs/                       # Daily log files (auto-generated)
├── feedback/                   # Feedback data (auto-generated, gitignored)
├── BENCHMARKS.md               # Performance data and optimization history
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Version History

| Version | Changes |
|---|---|
| v1.0.0 | Initial release — Docker, float16 quantization, persistent index |
| v1.1.0 | Conversation memory |
| v1.2.0 | Streaming responses |
| v1.3.0 | llama.cpp provider — direct GGUF loading, Metal acceleration, provider abstraction |

---

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full performance data and optimization history.

---

