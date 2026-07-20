from vector_store import SearchResult

def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n---\n\n".join(
        f"Source {i} (similarity {r.score:.3f}):\n{r.text}"
        for i, r in enumerate(results, start=1)
    )
    return f"""You are a marketing report analysis assistant.
Answer using only the supplied report context. Do not invent facts.
If the context is insufficient, say so.

REPORT CONTEXT:
{context}

USER QUESTION:
{question}
"""

