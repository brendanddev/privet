from abc import ABC, abstractmethod
from typing import Generator, List

class BaseProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        pass