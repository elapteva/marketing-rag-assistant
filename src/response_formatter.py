from vector_store import SearchResult

def format_retrieval_summary(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant sections were found."
    return f"Retrieved {len(results)} section(s). Best similarity score: {results[0].score:.3f}."

