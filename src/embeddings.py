import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

