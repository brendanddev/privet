
import os
import numpy as np
from typing import Generator, List
from llama_cpp import Llama
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import PromptTemplate
from core.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger()


QA_PROMPT = PromptTemplate(
    "<start_of_turn>system\n"
    "You are a helpful assistant. Answer the user's question using only the provided context.\n"
    "Rules:\n"
    "- Answer in 2-3 sentences maximum\n"
    "- Do not repeat the question\n"
    "- Do not include file paths or metadata\n"
    "- Do not simulate a conversation or add fake follow-up questions\n"
    "- If the context does not contain the answer, say 'I don't have enough information to answer that'\n"
    "<end_of_turn>\n"
    "<start_of_turn>user\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


class LlamaCppEmbedding(BaseEmbedding):
    """
    LlamaIndex compatible embedding wrapper around a llama.cpp model.

    LlamaIndex requires Settings.embed_model to be a BaseEmbedding subclass.
    This wrapper bridges the llama.cpp embedding API to the LlamaIndex
    interface so the rest of the app works without any changes.

    The embed_llm is stored via object.__setattr__ to bypass Pydantic
    validation, LlamaIndex models use Pydantic internally and don't
    allow arbitrary attributes without this approach.
    """

    _embed_llm: object = None

    def __init__(self, embed_llm):
        super().__init__(model_name="nomic-embed-text-llamacpp")
        object.__setattr__(self, '_embed_llm', embed_llm)

    def _embed(self, text: str) -> List[float]:
        """
        Generate an embedding and apply float16 quantization.

        Quantizing to float16 halves storage size with negligible
        quality loss, same approach as Float16EmbeddingWrapper.
        """
        raw = self._embed_llm.create_embedding(text)["data"][0]["embedding"]
        return np.array(raw, dtype=np.float16).astype(np.float32).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)


class LlamaCppLLM(CustomLLM):
    """
    LlamaIndex compatible LLM wrapper around a llama.cpp model.

    LlamaIndex requires Settings.llm to be a BaseLLM subclass.
    This wrapper bridges the llama.cpp generation API to the
    LlamaIndex interface so the rest of the app works without changes.

    The model is stored via object.__setattr__ to bypass Pydantic
    validation, same approach as LlamaCppEmbedding.
    """

    _model: object = None
    context_window: int = 4096
    num_output: int = 512
    model_name: str = "llamacpp"

    def __init__(self, model):
        super().__init__()
        object.__setattr__(self, '_model', model)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self._model(
            prompt,
            max_tokens=self.num_output,
            echo=False
        )
        return CompletionResponse(text=response["choices"][0]["text"].strip())

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        def gen():
            full = ""
            for chunk in self._model(
                prompt,
                max_tokens=self.num_output,
                stream=True,
                echo=False
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    full += token
                    yield CompletionResponse(text=token, delta=token)
        return gen()
    

class LlamaCppProvider(BaseProvider):
    """
    LLM provider using llama.cpp directly via llama-cpp-python.

    Runs models from local GGUF files with no external dependencies.
    No Ollama required, the model loads directly into memory as a
    library call inside the Python process.
    """

    def __init__(self, config: dict):
        """
        Initialize the llama.cpp provider from config.

        Args:
            config (dict): App config dict from config.yaml
        """
        model_path = config.get("model_path", "./models/google_gemma-3-1b-it-Q4_K_M.gguf")
        embed_model_path = config.get("embed_model_path", "./models/nomic-embed-text-v1.5.Q8_0.gguf")
        n_gpu_layers = config.get("n_gpu_layers", -1)
        n_ctx = config.get("n_ctx", 4096)
        n_threads = config.get("n_threads", None)

        logger.info(f"Loading llama.cpp generation model | Path: {model_path} | GPU layers: {n_gpu_layers}")

        # Generation model, handles text generation
        _llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        self.llm = LlamaCppLLM(_llm)

        logger.info(f"Loading llama.cpp embedding model | Path: {embed_model_path}")

        # Embedding model - separate instance with embedding mode enabled
        # Uses smaller context since embeddings don't need long context
        _embed_llm = Llama(
            model_path=embed_model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=512,
            embedding=True,
            verbose=False
        )

        # Wrap in LlamaIndex compatible interface so Settings.embed_model works
        self.embed_model = LlamaCppEmbedding(_embed_llm)

        logger.info("llama.cpp provider ready | Generation and embedding models loaded")

    def generate(self, prompt: str) -> str:
        return self.llm.complete(prompt).text

    def stream(self, prompt: str) -> Generator[str, None, None]:
        for chunk in self.llm.stream_complete(prompt):
            yield chunk.delta

    def get_embeddings(self, text: str) -> list:
        return self.embed_model.get_text_embedding(text)