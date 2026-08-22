import statistics
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from chunking import split_text
from document_processor import extract_text
from embeddings import EmbeddingService
from llm import generate_answer
from prompt_builder import build_rag_prompt
from retrieval import retrieve_relevant_chunks
from vector_store import InMemoryVectorStore


QUESTIONS = [
    "What are the main trends discussed in this report?",
    "What recommendations are provided?",
    "How is generative AI changing marketing?",
    "What challenges are organizations facing?",
    "Summarize the report.",
]


def load_document(file_path: Path):
    """Extract, chunk, embed, and store a document before benchmarking queries."""

    file_bytes = file_path.read_bytes()

    print(f"Loading document: {file_path.name}")
    text = extract_text(file_path.name, file_bytes)

    chunks = split_text(text)

    print(f"Chunks generated: {len(chunks):,}")
    print("Loading embedding model...")

    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_documents(chunks)

    vector_store = InMemoryVectorStore()
    vector_store.add(chunks, embeddings)

    print("Document ready.\n")

    return embedding_service, vector_store


def benchmark_question(
    question: str,
    embedding_service: EmbeddingService,
    vector_store: InMemoryVectorStore,
):
    """Measure retrieval, prompt construction, and answer generation."""

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    results = retrieve_relevant_chunks(
        question,
        embedding_service,
        vector_store,
    )
    retrieval_time = time.perf_counter() - retrieval_start

    prompt_start = time.perf_counter()
    prompt = build_rag_prompt(question, results)
    prompt_time = time.perf_counter() - prompt_start

    total_time = time.perf_counter() - total_start

    return {
        "question": question,
        "retrieved_chunks": results,
        "retrieval_time": retrieval_time,
        "prompt_time": prompt_time,
        "total_time": total_time,
    }


def main():
    if len(sys.argv) != 2:
        print(
            'Usage: python src/benchmark_qa.py '
            '"/path/to/marketing_report.pdf"'
        )
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    embedding_service, vector_store = load_document(file_path)

    results = []

    for number, question in enumerate(QUESTIONS, start=1):
        print("=" * 70)
        print(f"Question {number}: {question}")
        print("=" * 70)

        result = benchmark_question(
            question,
            embedding_service,
            vector_store,
        )

        results.append(result)

        print("\nTop Retrieved Chunks")
        print("-" * 60)

        for i, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(f"\nChunk {i}")
            print(f"Score: {chunk.score:.4f}")
            print(chunk.text[:300])
            print("...")

        print("Timing")
        print("-" * 40)
        print(f"Retrieval:         {result['retrieval_time']:.4f} seconds")
        print(f"Prompt building:   {result['prompt_time']:.4f} seconds")
        print(f"LLM generation:    {result['generation_time']:.4f} seconds")
        print(f"Total response:    {result['total_time']:.4f} seconds")
        print()

    total_times = [result["total_time"] for result in results]

    print("\n" + "=" * 70)
    print("QUESTION-ANSWER BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Questions tested:       {len(total_times)}")
    print(f"Average response time:  {statistics.mean(total_times):.4f} seconds")
    print(f"Minimum response time:  {min(total_times):.4f} seconds")
    print(f"Maximum response time:  {max(total_times):.4f} seconds")

    if len(total_times) > 1:
        print(
            f"Standard deviation:     "
            f"{statistics.stdev(total_times):.4f} seconds"
        )


if __name__ == "__main__":
    main()