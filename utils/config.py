
import yaml
import os
from utils.logger import setup_logger

logger = setup_logger()


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Falls back to sensible defaults if a key is missing so the app never crashes due to a missing config value. 
    This means you can add new config keys without breaking existing installs that don't have them yet.

    Args:
        config_path (str): Path to the YAML config file

    Returns:
        dict: Configuration values
    """
    defaults = {
        "provider": "ollama",
        "llm_model": "gemma3:1b",
        "embed_model": "nomic-embed-text",
        "ollama_host": "http://localhost:11434",
        "request_timeout": 120.0,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "similarity_top_k": 5,
        "history_length": 6,
        "docs_path": "./docs",
        "chroma_path": "./chroma_db",
        "collection_name": "documents",
        "logs_path": "./logs",
        "collect_feedback": True,
        "feedback_path": "./feedback/feedback.jsonl",
        "model_path": "./models/google_gemma-3-1b-it-Q4_K_M.gguf",
        "embed_model_path": "./models/nomic-embed-text-v1.5.Q8_0.gguf",
        "n_gpu_layers": -1,
        "n_ctx": 4096,
        "n_threads": None,
        "privacy_audit_path": "./logs/privacy_audit.jsonl",
        "show_hardware_panel": True,
        "use_hybrid_search": False,
        "use_reranking": False,
        "rerank_top_k": 5,
        "models_path": "./models",
        "llamacpp_path": None,
        "pleias_model_path": "./models/Pleias-RAG-1B-Q4_K_M.gguf",
    }

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path} — using defaults")
        return defaults

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    if user_config is None:
        logger.warning("Config file is empty — using defaults")
        return defaults

    # Merge user config over defaults so missing keys fall back gracefully
    merged = {**defaults, **user_config}
    logger.info(f"Config loaded from {config_path}")
    return merged
