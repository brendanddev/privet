
from abc import ABC, abstractmethod
from typing import Generator


class BaseProvider(ABC):
    """
    Abstract base class that all LLM providers must implement.

    Any provider: Ollama, llama.cpp, OpenAI, must implement these methods. 
    The rest of the app only talks to this interface, never to the provider 
    directly. Swap providers without touching any other code.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> list:
        pass