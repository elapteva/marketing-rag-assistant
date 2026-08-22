from pathlib import Path
import sys
import time

from chunking import split_text
from document_processor import DocumentProcessingError, extract_text
from embeddings import EmbeddingService
from vector_store import InMemoryVectorStore


def benchmark_document(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = file_path.name
    file_bytes = file_path.read_bytes()

    print(f"\nBenchmarking: {filename}")
    print(f"File size: {len(file_bytes) / 1024:.2f} KB")

    # Load the model before timing the document-processing pipeline.
    print("Loading embedding model...")
    embedding_service = EmbeddingService()
    print("Embedding model loaded.")

    total_start = time.perf_counter()

    extraction_start = time.perf_counter()
    text = extract_text(filename, file_bytes)
    extraction_time = time.perf_counter() - extraction_start

    if not text.strip():
        raise DocumentProcessingError("No readable text was found.")

    chunking_start = time.perf_counter()
    chunks = split_text(text)
    chunking_time = time.perf_counter() - chunking_start

    embedding_start = time.perf_counter()
    embeddings = embedding_service.embed_documents(chunks)
    embedding_time = time.perf_counter() - embedding_start

    storage_start = time.perf_counter()
    vector_store = InMemoryVectorStore()
    vector_store.add(chunks, embeddings)
    storage_time = time.perf_counter() - storage_start

    total_time = time.perf_counter() - total_start

    print("\nProcessing results")
    print("-" * 48)
    print(f"Characters extracted: {len(text):,}")
    print(f"Words extracted:      {len(text.split()):,}")
    print(f"Chunks generated:     {len(chunks):,}")
    print(f"Embedding shape:      {embeddings.shape}")

    print("\nTiming results")
    print("-" * 48)
    print(f"Text extraction:      {extraction_time:.4f} seconds")
    print(f"Chunking:             {chunking_time:.4f} seconds")
    print(f"Embedding generation: {embedding_time:.4f} seconds")
    print(f"Vector storage:       {storage_time:.4f} seconds")
    print(f"Total processing:     {total_time:.4f} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python src/benchmark_processing.py '
            '"/path/to/marketing_report.xlsx"'
        )
        raise SystemExit(1)

    try:
        benchmark_document(Path(sys.argv[1]))
    except Exception as exc:
        print(f"\nBenchmark failed: {exc}")
        raise SystemExit(1)