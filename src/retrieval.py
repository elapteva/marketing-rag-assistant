from embeddings import EmbeddingService
from vector_store import InMemoryVectorStore, SearchResult

def retrieve_relevant_chunks(
    question: str,
    embedding_service: EmbeddingService,
    vector_store: InMemoryVectorStore,
    top_k: int = 3,
) -> list[SearchResult]:
    return vector_store.search(embedding_service.embed_query(question), top_k=top_k)

