import os

import streamlit as st
from dotenv import load_dotenv

from auth import render_login
from chunking import split_text
from document_processor import DocumentProcessingError, extract_text
from embeddings import EmbeddingService
from llm import LLMConfigurationError, generate_answer
from prompt_builder import build_rag_prompt
from response_formatter import format_retrieval_summary
from retrieval import retrieve_relevant_chunks
from utils import validate_uploaded_file
from vector_store import InMemoryVectorStore


load_dotenv()

st.set_page_config(
    page_title="Marketing RAG Assistant",
    page_icon="📊",
    layout="wide"
)



# Authentication


if not render_login():
    st.stop()


# Embedding model


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_service():
    return EmbeddingService()



# Process multiple uploaded documents


def process_documents(files: list[tuple[str, bytes]]) -> None:

    all_texts = []
    all_chunks = []
    filenames = []

    for filename, file_bytes in files:

        # Extract document text
        text = extract_text(filename, file_bytes)

        if not text.strip():
            raise DocumentProcessingError(
                f"No readable text was found in {filename}."
            )

        # Split document into chunks
        chunks = split_text(text)

        if not chunks:
            raise DocumentProcessingError(
                f"No text chunks were generated for {filename}."
            )

        all_texts.append(text)
        all_chunks.extend(chunks)
        filenames.append(filename)

    if not all_chunks:
        raise DocumentProcessingError(
            "No readable content was found in the uploaded reports."
        )

    # Create embeddings for ALL document chunks
    embeddings = get_embedding_service().embed_documents(
        all_chunks
    )

    # Create one vector store containing all documents
    store = InMemoryVectorStore()

    store.add(
        all_chunks,
        embeddings
    )

    # Save processed information to Streamlit session state
    st.session_state.active_filenames = filenames

    st.session_state.document_text = "\n\n".join(
        all_texts
    )

    st.session_state.chunks = all_chunks
    st.session_state.vector_store = store

    # Remove previous question results
    st.session_state.pop(
        "last_results",
        None
    )

    st.session_state.pop(
        "last_answer",
        None
    )



# Application title


st.title(
    "AI-Powered Marketing Report Analysis "
    "and Decision Support System"
)

st.write(
    "Upload one or more marketing reports, inspect their "
    "content, and ask document-grounded questions."
)



# Tabs


upload_tab, analysis_tab, question_tab = st.tabs(
    [
        "1. Upload & Process",
        "2. Document Analysis",
        "3. Ask a Question"
    ]
)



# TAB 1
# Upload and Process Documents

with upload_tab:

    uploaded_files = st.file_uploader(
        "Choose one or more reports",
        type=[
            "pdf",
            "xlsx",
            "pptx"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        valid_files = []

        # Validate every uploaded file
        for uploaded_file in uploaded_files:

            data = uploaded_file.getvalue()

            valid, message = validate_uploaded_file(
                uploaded_file.name,
                len(data)
            )

            if not valid:

                st.error(
                    f"{uploaded_file.name}: {message}"
                )

            else:

                st.success(
                    f"{uploaded_file.name}: {message}"
                )

                valid_files.append(
                    (
                        uploaded_file.name,
                        data
                    )
                )

        # Process all valid files together
        if valid_files:

            if st.button(
                "Process and index reports",
                type="primary",
                use_container_width=True
            ):

                try:

                    with st.spinner(
                        "Processing and indexing reports..."
                    ):

                        process_documents(
                            valid_files
                        )

                    st.success(
                        f"{len(valid_files)} report(s) "
                        "processed successfully."
                    )

                except DocumentProcessingError as exc:

                    st.error(
                        str(exc)
                    )

                except Exception as exc:

                    st.exception(
                        exc
                    )

    # Display currently indexed reports
    active_filenames = st.session_state.get(
        "active_filenames",
        []
    )

    if active_filenames:

        st.info(
            "Active reports: **"
            + ", ".join(active_filenames)
            + "**"
        )



# TAB 2
# Document Analysis


with analysis_tab:

    text = st.session_state.get(
        "document_text"
    )

    chunks = st.session_state.get(
        "chunks",
        []
    )

    filenames = st.session_state.get(
        "active_filenames",
        []
    )

    if not text:

        st.warning(
            "Upload and process one or more reports first."
        )

    else:

        # Document statistics
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Reports",
            f"{len(filenames):,}"
        )

        c2.metric(
            "Characters",
            f"{len(text):,}"
        )

        c3.metric(
            "Words",
            f"{len(text.split()):,}"
        )

        c4.metric(
            "Chunks",
            f"{len(chunks):,}"
        )

        # Show indexed filenames
        st.markdown(
            "### Indexed Reports"
        )

        for filename in filenames:

            st.write(
                f"• {filename}"
            )

        # Combined extracted text preview
        st.text_area(
            "Combined extracted-text preview",
            text[:4000],
            height=280,
            disabled=True
        )

        # Preview first chunks
        with st.expander(
            "View first five chunks"
        ):

            for i, chunk in enumerate(
                chunks[:5],
                start=1
            ):

                st.markdown(
                    f"**Chunk {i}**"
                )

                st.write(
                    chunk
                )



# TAB 3
# Ask Questions


with question_tab:

    store = st.session_state.get(
        "vector_store"
    )

    filenames = st.session_state.get(
        "active_filenames",
        []
    )

    if store is None:

        st.warning(
            "Process one or more reports first."
        )

    else:

        if filenames:

            st.caption(
                f"Searching across "
                f"{len(filenames)} indexed report(s)."
            )

        question = st.text_input(
            "Question",
            placeholder=(
                "What are the main trends "
                "across these reports?"
            )
        )

        top_k = st.slider(
            "Supporting sections",
            1,
            5,
            3
        )

        if st.button(
            "Retrieve evidence and answer",
            type="primary",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "Enter a question."
                )

            else:

                # Retrieve relevant chunks across
                # ALL indexed documents
                results = retrieve_relevant_chunks(
                    question,
                    get_embedding_service(),
                    store,
                    top_k
                )

                st.session_state.last_results = (
                    results
                )

                st.session_state.last_answer = None

                # Generate AI response if OpenAI
                # configuration is available
                if os.getenv(
                    "OPENAI_API_KEY"
                ):

                    try:

                        prompt = build_rag_prompt(
                            question,
                            results
                        )

                        st.session_state.last_answer = (
                            generate_answer(
                                prompt
                            )
                        )

                    except LLMConfigurationError as exc:

                        st.info(
                            str(exc)
                        )


        # Display retrieval / response

        results = st.session_state.get(
            "last_results",
            []
        )

        answer = st.session_state.get(
            "last_answer"
        )

        if results:

            st.success(
                format_retrieval_summary(
                    results
                )
            )

            if answer:

                st.markdown(
                    "### AI-generated answer"
                )

                st.write(
                    answer
                )

            else:

                st.markdown(
                    "### Retrieval-only result"
                )

                st.write(
                    "OpenAI is not configured, so the "
                    "application is showing retrieved "
                    "evidence only."
                )

            st.markdown(
                "### Supporting report sections"
            )

            for i, result in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"Source {i} — "
                    f"similarity {result.score:.3f}"
                ):

                    st.write(
                        result.text
                    )