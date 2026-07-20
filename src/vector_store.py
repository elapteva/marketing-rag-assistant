from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    index: int

class InMemoryVectorStore:
    def __init__(self) -> None:
        self.texts = []
        self.embeddings = None

    def add(self, texts: list[str], embeddings: np.ndarray) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("Texts and embeddings must have the same length.")
        self.texts = list(texts)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        if self.embeddings is None:
            return []
        scores = self.embeddings @ np.asarray(query_embedding, dtype=np.float32)
        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.texts[int(i)], float(scores[int(i)]), int(i)) for i in indices]

