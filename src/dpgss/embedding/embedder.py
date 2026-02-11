from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder(ABC):

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns (n, d) array. For single text: embed([text])[0] gives (d,)."""
        pass
    
    @abstractmethod
    def dimension(self) -> int:
        pass

class HFEmbedder(Embedder):
    name: str = "HFEmbedder"
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        # Suppress harmless warning about position_ids
        import warnings
        warnings.filterwarnings("ignore", message=".*embeddings.position_ids.*")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # CRITICAL: Ensure output is (n, d) with float32
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize manually for control
            show_progress_bar=False
        ).astype(np.float32)
        
        # Handle edge cases
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {vecs.shape}")
        
        return vecs
    
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()