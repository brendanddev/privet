# Privet

Your notes stay on your laptop. Ask questions about them. Nothing leaves your machine — and you can prove it.

---

## The Problem

Every AI study tool sends your data to the cloud. Your lecture notes, assignments, research papers — uploaded to servers you don't control, processed by companies whose privacy policies change. ChatGPT has no memory of what you uploaded last week. Every session starts from zero.

Privet runs entirely on your machine. Your documents never leave. And unlike every other local AI tool that just *says* your data is private, Privet *proves* it — with a cryptographic audit log you can verify yourself.

---

## Philosophy

**Privacy first.** Every component runs locally. Your documents, embeddings, and queries never touch an external server. No OpenAI, no cloud APIs, no tracking. A built-in SHA-256 hash-chained audit log records every query session and produces a signed report proving zero data left your machine.

**Hardware optimized.** Built for consumer hardware with limited RAM. Privet auto-detects your hardware tier at setup and configures itself accordingly — from 4GB machines to 16GB+ workstations. Runs on the laptop you already own.

**Ease of use.** One setup command. No command line required day to day. Drop documents in a folder, ask questions, get answers with source citations.

---

## What Makes Privet Different

Every other local RAG tool says "your data is private." Privet proves it.

- **Cryptographic audit log** — every query session is logged with SHA-256 hash chaining. Tamper with any entry and verification fails. Run `python3 -m utils.verify_audit_log` at any time to produce a signed report showing zero external bytes across all sessions.
- **Network monitor** — live sidebar panel measures external interface bytes before and after every query. Distinguishes real outbound traffic from OS background noise (mDNS, Bonjour).
- **Hardware auto-configuration** — detects RAM, CPU, and GPU (Metal/CUDA/ROCm) at setup and writes a tuned `config.yaml` automatically. Four hardware tiers.
- **Cross-encoder reranking** — retrieves 10 candidates, re-scores top 5 with a local cross-encoder model. Meaningfully improves answer quality over pure vector search — benchmarked, not claimed.
- **Automatic quality evaluation** — every answer is scored for faithfulness and relevance using local embedding models. Results tracked over time.

---

## Benchmarks

All benchmarks run on MacBook Air M2, 8GB unified memory, Apple Silicon. Same 20 questions across all configurations.

| Setup | Avg Time | Composite | Faithfulness | Thumbs Up |
|---|---|---|---|---|
| gemma3:1b · vector only | 2.24s | 0.813 | 0.950 | 60% |
| gemma3:1b · hybrid (BM25 + vector + RRF) | 4.62s | 0.798 | 0.881 | 55% |
| gemma3:1b · vector + rerank | 3.06s | 0.864 | 1.000 | — |

Vector + cross-encoder reranking is the current default. Faithfulness hits 1.000 — every answer fully grounded in retrieved context.

See [BENCHMARKS.md](BENCHMARKS.md) for full per-question results and analysis.

---

## Features

- **Fully local pipeline** — every component runs on your machine, zero external API calls
- **Provider abstraction** — swap between Ollama and llama.cpp via a single config value
- **llama.cpp support** — run GGUF models directly, no background service, no open ports, Metal acceleration on Apple Silicon
- **Cross-encoder reranking** — retrieve 10 candidates, rerank to top 5 with ms-marco-MiniLM-L-6-v2
- **Multi-format ingestion** — PDF, TXT, DOCX, CSV via folder drop or UI upload
- **Persistent index** — documents embedded once, near-instant startup on subsequent runs
- **Streaming responses** — answers stream token by token
- **Source citations** — every answer shows source document, page number, and relevance score
- **Conversation memory** — model remembers previous exchanges within a session
- **Feedback system** — thumbs up/down logged locally for future fine-tuning
- **Privacy audit log** — tamper-evident JSONL log with SHA-256 hash chaining
- **Audit verification CLI** — `python3 -m utils.verify_audit_log` produces a printable privacy report
- **Network monitor** — live per-query external byte measurement with noise filtering
- **Hardware profiler** — auto-detects tier and GPU backend, generates tuned config at setup
- **RAG evaluator** — automatic scoring on faithfulness and relevance, tracked over time
- **Model quantization pipeline** — `python3 -m utils.quantize` converts and quantizes any HuggingFace model to GGUF with SHA-256 provenance record
- **Debug sidebar** — chunk distribution, index health, retrieval confidence meters
- **Docker support** — full stack with a single command

---

## Stack

| Component | Library | Purpose |
|---|---|---|
| LLM (Ollama) | Ollama | Local language models via HTTP |
| LLM (llama.cpp) | llama-cpp-python | GGUF models directly, no service required |
| RAG Framework | LlamaIndex 0.14.15 | Document ingestion and retrieval |
| Vector Store | ChromaDB 1.5.2 | Local embedding storage |
| Reranker | sentence-transformers | Cross-encoder reranking |
| UI | Streamlit 1.54.0 | Browser-based chat interface |
| Evaluation | sentence-transformers | Local answer quality scoring |
| Containers | Docker + Compose | Single command deployment |

---

## Models

### Ollama provider
- **LLM:** `gemma3:1b` — lightweight local language model
- **Embeddings:** `nomic-embed-text` — dedicated local embedding model

### llama.cpp provider
- **LLM:** `google_gemma-3-1b-it-Q4_K_M.gguf` — quantized Gemma 3 1B, ~800MB
- **Embeddings:** `nomic-embed-text-v1.5.Q8_0.gguf` — quantized nomic embed, ~274MB

Both GGUF models available from [Hugging Face](https://huggingface.co). Place them in `models/`.

To quantize your own models from HuggingFace weights:
```bash
```

---

## Hardware Tiers

Privet auto-detects your hardware at setup and configures itself accordingly.

| Tier | RAM | n_ctx | Model |
|---|---|---|---|
| High | 16GB+ | 8192 | gemma3:4b recommended |
| Standard | 8-15GB | 4096 | gemma3:1b Q4_K_M |
| Low | 4-7GB | 1024 | gemma3:1b Q4_K_S, embed unloads after indexing |
| Minimal | <4GB | 512 | CPU only, conservative settings |

To re-run hardware detection:
```bash
python3 -m utils.hardware
```

---

## Setup

### Option 1 — Automated (recommended)

```bash
git clone https://github.com/brendanddev/privet.git
cd privet
bash setup.sh
```

The setup script will:
1. Check system dependencies (Python 3.11+, Ollama if needed)
2. Ask which provider to use: Ollama or llama.cpp
3. Create a virtual environment and install dependencies
4. Run hardware detection and auto-generate a tuned `config.yaml`
5. Pull Ollama models or print llama.cpp download instructions

Then start the app:
```bash
source venv/bin/activate
python3 -m streamlit run app.py
```

Open `http://localhost:8501`

---

### Option 2 — Docker

```bash
git clone https://github.com/brendanddev/privet.git
cd privet
docker compose up --build
```

Open `http://localhost:8501`

**GPU support:**
- **Linux + NVIDIA:** Works automatically
- **Mac:** Metal used automatically. Remove the `deploy` block from `docker-compose.yml`
- **Windows:** Requires WSL2 with NVIDIA drivers

---

### Option 3 — Manual

**Prerequisites:** Python 3.11+, Ollama installed and running

```bash
git clone https://github.com/brendanddev/privet.git
cd privet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull gemma3:1b
ollama pull nomic-embed-text
cp config.example.yaml config.yaml
```

Start:
```bash
ollama serve                    # terminal 1
python3 -m streamlit run app.py # terminal 2
```

---

## Verifying Privacy

Run at any time to verify the audit log integrity and see a session summary:

```bash
python3 -m utils.verify_audit_log
```

Output:
```
╔══════════════════════════════════════╗
║   PRIVET — PRIVACY AUDIT REPORT      ║
╚══════════════════════════════════════╝
  Chain integrity:  VALID
  Period:           2026-01-01  to  2026-03-29
  Total queries:    412
  External bytes:   0
  Data left device: NO
  Verification:     ALL ENTRIES VALID
```

The report is saved to `logs/privacy_report_{date}.txt`.

For stronger verification, use OS-level tools the application cannot influence:
```bash
# Per-process network activity (macOS)
nettop -P -p $(pgrep -f streamlit) -l 1

# Kernel-level syscall tracing (macOS)
sudo dtrace -n 'syscall::connect*:entry /pid == $1/ { ustack(); }' -p $(pgrep -f streamlit)
```

---

## Feedback and Fine-tuning

Every answer can be rated thumbs up or down. Ratings are stored in `feedback/feedback.jsonl` — one JSON entry per rating including the question, answer, sources, model, and query time.

This data is the foundation for a planned fine-tuning pipeline using KTO (Kahneman-Tversky Optimization), which works with unpaired feedback. The goal: a model shaped entirely by your own usage, stored locally, owned by you.

To disable feedback collection: set `collect_feedback: false` in `config.yaml`.

---

## Project Structure

```
privet/
├── core/
│   ├── rag_engine.py           # RAG pipeline
│   ├── rag_debugger.py         # Standalone inspection CLI
│   ├── embeddings.py           # Float16 embedding wrapper
│   └── providers/
│       ├── base.py             # Abstract provider interface
│       ├── ollama.py           # Ollama provider
│       ├── llamacpp.py         # llama.cpp provider
│       └── factory.py          # Provider selection
├── ui/
│   ├── dashboard.py            # Debug sidebar
│   ├── privacy_panel.py        # Network monitor panel
│   ├── eval_panel.py           # Quality scores panel
│   └── hardware_panel.py       # Hardware stats panel
├── utils/
│   ├── config.py               # YAML config loader
│   ├── logger.py               # Daily log files
│   ├── feedback.py             # Feedback logging
│   ├── hardware.py             # Hardware profiler
│   ├── network_monitor.py      # External byte monitoring
│   ├── rag_evaluator.py        # Automatic answer scoring
│   ├── privacy_audit_log.py    # Hash-chained audit log
│   ├── verify_audit_log.py     # Audit verification CLI
│   └── quantize.py             # Model quantization pipeline
├── app.py                      # Streamlit entry point
├── setup.sh                    # One-command installer
├── config.example.yaml         # Config template
├── Dockerfile
├── docker-compose.yml
├── models/                     # GGUF files (gitignored)
├── models/provenance.json      # SHA-256 model provenance records
├── documents/                  # Your documents go here
├── chroma_db/                  # Vector store (auto-generated)
├── logs/                       # Logs + audit log + eval DB
└── feedback/                   # Feedback data (gitignored)
```

---

## Version History

| Version | Changes |
|---|---|
| v1.0.0 | Initial release — Docker, float16 quantization, persistent index |
| v1.1.0 | Conversation memory |
| v1.2.0 | Streaming responses |
| v1.3.0 | llama.cpp provider, provider abstraction, feedback system |

---

## Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for full performance data, per-question results, and retrieval configuration comparisons.