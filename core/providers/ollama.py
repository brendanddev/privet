
import os
from typing import Generator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from core.providers.base import BaseProvider
from core.embeddings import Float16EmbeddingWrapper
from utils.logger import setup_logger

logger = setup_logger()


class OllamaProvider(BaseProvider):
    """
    LLM provider using Ollama as the backend.

    Ollama runs as a separate service and exposes a local HTTP API.
    This is the default provider, easiest to set up and supports
    automatic model management via ollama pull.

    The OLLAMA_HOST environment variable takes priority over config
    so Docker can override without changing config.yaml.
    """

    def __init__(self, config: dict):
        self.llm_model = config["llm_model"]
        self.embed_model_name = config["embed_model"]

        ollama_host = os.environ.get(
            "OLLAMA_HOST",
            config.get("ollama_host", "http://localhost:11434")
        )
        request_timeout = config.get("request_timeout", 120.0)

        self.llm = Ollama(
            model=self.llm_model,
            request_timeout=request_timeout,
            base_url=ollama_host
        )

        base_embed = OllamaEmbedding(
            model_name=self.embed_model_name,
            base_url=ollama_host
        )
        self.embed_model = Float16EmbeddingWrapper(base_embed)

        logger.info(f"OllamaProvider initialized | LLM: {self.llm_model} | Host: {ollama_host}")

    def generate(self, prompt: str) -> str:
        return str(self.llm.complete(prompt))

    def stream(self, prompt: str) -> Generator[str, None, None]:
        for token in self.llm.stream_complete(prompt):
            yield token.delta

    def get_embeddings(self, text: str) -> list:
        return self.embed_model.get_text_embedding(text)