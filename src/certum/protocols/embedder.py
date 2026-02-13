# certum/protocols/embedder.py

from typing import Protocol, List
import numpy as np


class Embedder(Protocol):
    """
    Computes embeddings (may internally use backend).
    """

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns (n, d) float32 embeddings.
        """

    def dimension(self) -> int:
        """
        Embedding dimensionality.
        """
