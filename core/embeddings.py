
import numpy as np
from typing import List
from llama_index.core.embeddings import BaseEmbedding


class Float16EmbeddingWrapper(BaseEmbedding):
    """
    Wraps an existing embedding model and converts all output embeddings from float32 
    to float16 before they are stored in ChromaDB.

    This cuts embedding storage size roughly in half with negligible quality loss, 
    float16 has enough precision for cosine similarity calculations used in retrieval.
    """

    _base: BaseEmbedding = None

    def __init__(self, base_model: BaseEmbedding):
        super().__init__(
            model_name=base_model.model_name,
        )
        object.__setattr__(self, '_base', base_model)

    def _quantize(self, embedding: List[float]) -> List[float]:
        """
        Convert a float32 embedding to float16 and back to a Python list.

        ChromaDB expects a list of floats so we convert back after
        quantizing -the values are stored at float16 precision.

        Args:
            embedding: float32 embedding vector

        Returns:
            list: float16 quantized embedding as a Python list
        """
        return np.array(embedding, dtype=np.float16).astype(np.float32).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._quantize(self._base._get_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._quantize(self._base._get_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._quantize(e) for e in self._base._get_text_embeddings(texts)]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._quantize(await self._base._aget_query_embedding(query))

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._quantize(await self._base._aget_text_embedding(text))