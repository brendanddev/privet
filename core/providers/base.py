
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
        """
        Generate a complete response for the given prompt.

        Args:
            prompt (str): The input prompt

        Returns:
            str: The generated response text
        """
        pass

    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Stream a response token by token for the given prompt.

        Args:
            prompt (str): The input prompt

        Yields:
            str: Each token as it is generated
        """
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> list:
        """
        Generate an embedding vector for the given text.

        Args:
            text (str): The text to embed

        Returns:
            list: Embedding vector as a list of floats
        """
        pass