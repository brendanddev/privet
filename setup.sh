#!/usr/bin/env bash

# =============================================================================
# setup.sh - Local RAG Assistant installer
#
# Supports macOS (primary) and Linux.
# Run from the project root:
#
#   bash setup.sh
#
# Or via one-liner from GitHub (once repo is public):
#
#   curl -fsSL https://raw.githubusercontent.com/brendanddev/local-rag-assistant/main/setup.sh | bash
#
# What this script does:
#   1. Checks for required system dependencies (Homebrew, Python, Ollama)
#   2. Asks which provider to use: Ollama or llama.cpp
#   3. Creates a Python venv and installs requirements
#   4. Runs hardware detection to auto-generate a tuned config.yaml
#   5. Pulls Ollama models (if Ollama provider chosen)
#   6. Prints llama.cpp model download instructions (if llamacpp chosen)
#   7. Creates required directories
#   8. Prints a start command
# =============================================================================

set -euo pipefail

### Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

### Helpers
info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# Confirm the script is run from the project root
if [ ! -f "config.example.yaml" ]; then
    error "Run this script from the local-rag-assistant project root."
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

### Banner
echo ""
echo -e "${BOLD}=================================================${RESET}"
echo -e "${BOLD}   Local RAG Assistant — Setup${RESET}"
echo -e "${BOLD}=================================================${RESET}"
echo ""
info "OS: $OS | Arch: $ARCH"
echo ""

### Step 1 — Provider choice
header "Step 1 — Choose a provider"
echo ""
echo "  [1] Ollama    — Easiest setup. Runs as a background service."
echo "                  Models download automatically."
echo ""
echo "  [2] llama.cpp — No background service. Faster inference."
echo "                  Requires downloading GGUF model files manually."
echo ""

while true; do
    read -rp "  Enter 1 or 2: " PROVIDER_CHOICE
    case "$PROVIDER_CHOICE" in
        1) PROVIDER="ollama";   break ;;
        2) PROVIDER="llamacpp"; break ;;
        *) warn "Please enter 1 or 2." ;;
    esac
done

success "Provider selected: $PROVIDER"

### Step 2 — System dependencies
header "Step 2 — Checking system dependencies"

# Disk space — require 2 GB free, warn below 4 GB for Ollama
MIN_FREE_KB=2097152   # 2 GB hard minimum
WARN_FREE_KB=4194304  # 4 GB recommended (Ollama models ~1.5 GB)
FREE_KB=$(df -Pk . | awk 'NR==2 {print $4}')
if [ "$FREE_KB" -lt "$MIN_FREE_KB" ]; then
    error "Not enough disk space. Need at least 2 GB free — only $(( FREE_KB / 1024 )) MB available."
fi
if [ "$PROVIDER" = "ollama" ] && [ "$FREE_KB" -lt "$WARN_FREE_KB" ]; then
    warn "Ollama models require ~1.5 GB. Only $(( FREE_KB / 1024 )) MB free — may run out of space."
fi
success "Disk space OK: $(( FREE_KB / 1024 )) MB free"

# curl (Linux only — macOS ships with curl)
if [ "$OS" = "Linux" ]; then
    if ! command -v curl &>/dev/null; then
        info "curl not found — installing..."
        sudo apt-get update -qq && sudo apt-get install -y curl
        success "curl installed"
    else
        success "curl found: $(curl --version | head -1)"
    fi
fi

# Homebrew (macOS only)
if [ "$OS" = "Darwin" ]; then
    if ! command -v brew &>/dev/null; then
        info "Homebrew not found — installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for Apple Silicon
        if [ "$ARCH" = "arm64" ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        success "Homebrew installed"
    else
        success "Homebrew found: $(brew --version | head -1)"
    fi
fi

# Python 3.11+
PYTHON=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c 'import sys; print(sys.version_info[:2])')
        # Accept (3, 11) or higher
        if "$cmd" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    info "Python 3.11+ not found — installing via Homebrew..."
    if [ "$OS" = "Darwin" ]; then
        brew install python@3.11
        PYTHON="python3.11"
    elif [ "$OS" = "Linux" ]; then
        sudo apt-get update -qq && sudo apt-get install -y python3.11 python3.11-venv
        PYTHON="python3.11"
    else
        error "Python 3.11+ is required. Please install it from https://python.org and re-run this script."
    fi
fi

success "Python found: $($PYTHON --version)"

# python3-venv (Linux only — not installed by default on Debian/Ubuntu)
if [ "$OS" = "Linux" ]; then
    if ! "$PYTHON" -m venv --help &>/dev/null; then
        info "python3-venv not found — installing..."
        PY_VER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        sudo apt-get install -y "python${PY_VER}-venv" 2>/dev/null || \
            sudo apt-get install -y python3-venv
        success "python3-venv installed"
    fi
fi

# Ollama (only if Ollama provider chosen)
if [ "$PROVIDER" = "ollama" ]; then
    if ! command -v ollama &>/dev/null; then
        info "Ollama not found — installing..."
        if [ "$OS" = "Darwin" ]; then
            brew install ollama
        elif [ "$OS" = "Linux" ]; then
            curl -fsSL https://ollama.com/install.sh | sh
        else
            error "Please install Ollama manually from https://ollama.com and re-run this script."
        fi
        success "Ollama installed"
    else
        success "Ollama found: $(ollama --version 2>/dev/null || echo 'version unknown')"
    fi
fi

### Step 3 — Python virtual environment
header "Step 3 — Setting up Python environment"

if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv venv
    success "Virtual environment created"
else
    success "Virtual environment already exists"
fi

# Activate
# shellcheck disable=SC1091
source venv/bin/activate
success "Virtual environment activated"

info "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip --quiet

# Use uv for fast dependency installs if available; fall back to pip
if ! command -v uv &>/dev/null; then
    info "Installing uv for faster package installation..."
    pip install uv --quiet
fi

if command -v uv &>/dev/null; then
    uv pip install -r requirements.txt --quiet
else
    pip install -r requirements.txt --quiet
fi
success "Dependencies installed"

### Step 4 — Hardware detection + config generation
header "Step 4 — Detecting hardware and generating config"

if [ -f "config.yaml" ]; then
    warn "config.yaml already exists — skipping generation."
    warn "Delete config.yaml and re-run setup.sh to regenerate."
else
    info "Running hardware profiler..."

    # Run hardware.py and capture JSON output for config generation
    # We call a small inline Python script that imports HardwareProfiler,
    # gets recommendations, and prints them as shell-sourceable KEY=VALUE pairs
    HARDWARE_ENV=$(python3 - <<'PYEOF'
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from utils.hardware import HardwareProfiler
    p = HardwareProfiler()
    profile = p.profile()
    recs = p.get_recommendations()

    print(f"HW_TIER={profile.tier}")
    print(f"HW_TOTAL_RAM={profile.total_ram_gb}")
    print(f"HW_FREE_RAM={profile.available_ram_gb}")
    print(f"HW_GPU={profile.gpu.backend}")
    print(f"HW_APPLE_SILICON={str(profile.is_apple_silicon).lower()}")
    print(f"REC_N_CTX={recs['n_ctx']}")
    print(f"REC_CHUNK_SIZE={recs['chunk_size']}")
    print(f"REC_TOP_K={recs['similarity_top_k']}")
    print(f"REC_N_GPU_LAYERS={recs['n_gpu_layers']}")
    print(f"REC_LLM_MODEL={recs['llm_model']}")
    print(f"REC_EMBED_MODEL={recs['embed_model']}")
    print(f"REC_UNLOAD_EMBED={str(recs['unload_embed_after_index']).lower()}")
except Exception as e:
    print(f"HW_TIER=standard", file=sys.stderr)
    print(f"HW_ERROR={e}", file=sys.stderr)
    # Safe defaults
    print("HW_TIER=standard")
    print("HW_TOTAL_RAM=8.0")
    print("HW_FREE_RAM=4.0")
    print("HW_GPU=none")
    print("HW_APPLE_SILICON=false")
    print("REC_N_CTX=4096")
    print("REC_CHUNK_SIZE=256")
    print("REC_TOP_K=3")
    print("REC_N_GPU_LAYERS=-1")
    print("REC_LLM_MODEL=gemma3:1b")
    print("REC_EMBED_MODEL=nomic-embed-text")
    print("REC_UNLOAD_EMBED=false")
PYEOF
)

    # Source the hardware env vars into shell
    eval "$HARDWARE_ENV"

    success "Hardware detected | Tier: $HW_TIER | RAM: ${HW_TOTAL_RAM}GB | GPU: $HW_GPU"

    # Write config.yaml with hardware-tuned values
    cat > config.yaml <<YAML
# Local RAG Assistant Configuration
# Auto-generated by setup.sh on $(date)
# Hardware tier: $HW_TIER | RAM: ${HW_TOTAL_RAM}GB | GPU: $HW_GPU

# Provider: ollama or llamacpp
provider: $PROVIDER

# Model settings
llm_model: $REC_LLM_MODEL
embed_model: $REC_EMBED_MODEL

# Ollama settings
ollama_host: http://localhost:11434
request_timeout: 120.0

# Document processing
# Tuned for hardware tier: $HW_TIER
chunk_size: $REC_CHUNK_SIZE
chunk_overlap: 25

# Retrieval settings
similarity_top_k: $REC_TOP_K

# Conversation memory
history_length: 6

# Paths
docs_path: ./documents
chroma_path: ./chroma_db
collection_name: documents
logs_path: ./logs

# Feedback collection
collect_feedback: true
feedback_path: ./feedback/feedback.jsonl

# llama.cpp settings (only used when provider is llamacpp)
model_path: ./models/google_gemma-3-1b-it-Q4_K_M.gguf
embed_model_path: ./models/nomic-embed-text-v1.5.Q8_0.gguf
n_gpu_layers: $REC_N_GPU_LAYERS
n_ctx: $REC_N_CTX
n_threads: null

# Hardware profiler
auto_apply_hardware_profile: false
show_hardware_panel: true
YAML

    success "config.yaml generated (tier: $HW_TIER, n_ctx: $REC_N_CTX, chunk_size: $REC_CHUNK_SIZE, top_k: $REC_TOP_K)"
fi

### Step 5 — Create required directories
header "Step 5 — Creating directories"

mkdir -p documents chroma_db logs feedback models
success "Directories ready: documents/ chroma_db/ logs/ feedback/ models/"

### Step 6 — Provider-specific setup
header "Step 6 — Provider setup"

if [ "$PROVIDER" = "ollama" ]; then

    # Start Ollama service if not already running
    if ! pgrep -x "ollama" > /dev/null 2>&1; then
        info "Starting Ollama service..."
        if [ "$OS" = "Darwin" ]; then
            # On Mac, Ollama runs as an app or via CLI
            ollama serve &>/dev/null &
            OLLAMA_PID=$!
            sleep 3
            success "Ollama service started (PID: $OLLAMA_PID)"
        elif [ "$OS" = "Linux" ]; then
            sudo systemctl start ollama 2>/dev/null || ollama serve &>/dev/null &
            sleep 3
            success "Ollama service started"
        fi
    else
        success "Ollama service already running"
    fi

    info "Pulling LLM model: gemma3:1b (this will take a few minutes on first run)..."
    ollama pull gemma3:1b
    success "gemma3:1b ready"

    info "Pulling embedding model: nomic-embed-text..."
    ollama pull nomic-embed-text
    success "nomic-embed-text ready"

elif [ "$PROVIDER" = "llamacpp" ]; then

    echo ""
    echo -e "${YELLOW}${BOLD}  llama.cpp requires GGUF model files downloaded manually.${RESET}"
    echo ""
    echo "  Download these two files from Hugging Face and place them in models/"
    echo ""
    echo -e "  ${BOLD}LLM model (~800MB):${RESET}"
    echo "  https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF"
    echo "  File: google_gemma-3-1b-it-Q4_K_M.gguf"
    echo ""
    echo -e "  ${BOLD}Embedding model (~274MB):${RESET}"
    echo "  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
    echo "  File: nomic-embed-text-v1.5.Q8_0.gguf"
    echo ""
    echo "  Place both files in:  $(pwd)/models/"
    echo ""

    # Check if models are already present
    LLM_GGUF="models/google_gemma-3-1b-it-Q4_K_M.gguf"
    EMBED_GGUF="models/nomic-embed-text-v1.5.Q8_0.gguf"

    if [ -f "$LLM_GGUF" ] && [ -f "$EMBED_GGUF" ]; then
        success "Both GGUF model files found in models/"
    else
        [ ! -f "$LLM_GGUF" ]   && warn "Missing: $LLM_GGUF"
        [ ! -f "$EMBED_GGUF" ] && warn "Missing: $EMBED_GGUF"
        warn "Add the model files to models/ before starting the app."
    fi

fi

### Done
echo ""
echo -e "${BOLD}=================================================${RESET}"
echo -e "${GREEN}${BOLD}   Setup complete!${RESET}"
echo -e "${BOLD}=================================================${RESET}"
echo ""
echo -e "  ${BOLD}Hardware tier:${RESET} $HW_TIER (${HW_TOTAL_RAM}GB RAM)"
echo -e "  ${BOLD}Provider:${RESET}      $PROVIDER"
echo -e "  ${BOLD}GPU:${RESET}           $HW_GPU"
echo ""
echo -e "  ${BOLD}To start the app:${RESET}"
echo ""

if [ "$PROVIDER" = "ollama" ]; then
    echo "    source venv/bin/activate"
    echo "    ollama serve          # in a separate terminal if not already running"
    echo "    streamlit run app.py"
else
    echo "    source venv/bin/activate"
    echo "    streamlit run app.py"
fi

echo ""
echo -e "  ${BOLD}To run hardware report:${RESET}"
echo "    source venv/bin/activate && python3 -m utils.hardware"
echo ""
echo -e "  ${BOLD}To run the debugger:${RESET}"
echo "    source venv/bin/activate && python3 -m core.rag_debugger"
echo ""
echo -e "  Drop documents into ${BOLD}documents/${RESET} or upload via the app UI."
echo ""
