import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from chunking import split_text
from embeddings import EmbeddingService
from vector_store import InMemoryVectorStore
from retrieval import retrieve_relevant_chunks
from prompt_builder import build_rag_prompt



# Shared test data


TEST_TEXT = """
The summer social media campaign generated a 25 percent increase
in conversions. Social media advertising performed better than
display advertising during the campaign.

Email marketing produced strong customer engagement, while paid
search generated the highest click-through rate.

The company plans to increase its social media advertising budget
during the next marketing campaign.
"""

TEST_QUESTION = "Which advertising channel performed better?"



# Shared embedding model

# scope="module" means the SentenceTransformer model is loaded
# only once instead of being loaded again for every test.


@pytest.fixture(scope="module")
def embedding_service():
    return EmbeddingService()



# IT-01
# Chunking -> Embedding Generation


def test_chunking_to_embeddings(embedding_service):

    # Step 1: Divide the document into chunks
    chunks = split_text(
        TEST_TEXT,
        chunk_size=150,
        overlap=30
    )

    # Verify that chunking produced output
    assert len(chunks) > 0

    # Step 2: Convert the chunks into embeddings
    embeddings = embedding_service.embed_documents(chunks)

    # Every chunk should have one corresponding embedding
    assert len(embeddings) == len(chunks)

    # Embeddings should contain numerical dimensions
    assert embeddings.shape[1] > 0



# IT-02
# Embeddings -> Vector Store


def test_embeddings_to_vector_store(embedding_service):

    # Generate chunks
    chunks = split_text(
        TEST_TEXT,
        chunk_size=150,
        overlap=30
    )

    # Generate embeddings
    embeddings = embedding_service.embed_documents(chunks)

    # Create vector store
    vector_store = InMemoryVectorStore()

    # Add text and embeddings to vector store
    vector_store.add(
        chunks,
        embeddings
    )

    # Generate a query embedding
    query_embedding = embedding_service.embed_query(
        "social media campaign"
    )

    # Search the vector store
    results = vector_store.search(
        query_embedding,
        top_k=2
    )

    # Verify that the vector store returns results
    assert len(results) > 0



# IT-03
# Query Embedding -> Semantic Retrieval


def test_query_to_retrieval(embedding_service):

    # Prepare document
    chunks = split_text(
        TEST_TEXT,
        chunk_size=150,
        overlap=30
    )

    embeddings = embedding_service.embed_documents(chunks)

    # Store document embeddings
    vector_store = InMemoryVectorStore()

    vector_store.add(
        chunks,
        embeddings
    )

    # Perform semantic retrieval through retrieval.py
    results = retrieve_relevant_chunks(
        TEST_QUESTION,
        embedding_service,
        vector_store,
        top_k=2
    )

    # Retrieval should return at least one result
    assert len(results) > 0

    # Combine retrieved text so we can check its relevance
    retrieved_text = " ".join(
        result.text for result in results
    ).lower()

    # The most relevant information should discuss social media
    assert "social media" in retrieved_text



# IT-04
# Retrieval -> Prompt Builder


def test_retrieval_to_prompt_builder(embedding_service):

    # Prepare and index the document
    chunks = split_text(
        TEST_TEXT,
        chunk_size=150,
        overlap=30
    )

    embeddings = embedding_service.embed_documents(chunks)

    vector_store = InMemoryVectorStore()

    vector_store.add(
        chunks,
        embeddings
    )

    # Retrieve relevant chunks
    results = retrieve_relevant_chunks(
        TEST_QUESTION,
        embedding_service,
        vector_store,
        top_k=2
    )

    assert len(results) > 0


    # Send retrieved information to the prompt builder
    prompt = build_rag_prompt(
        TEST_QUESTION,
        results
    )

    # Verify that a prompt was successfully generated
    assert prompt is not None
    assert len(prompt) > 0

    # Verify that the user's question appears in the prompt
    assert TEST_QUESTION.lower() in prompt.lower()

    # Verify that retrieved document information appears
    assert "social media" in prompt.lower()