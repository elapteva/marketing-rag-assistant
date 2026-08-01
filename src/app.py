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
st.set_page_config(page_title="Marketing RAG Assistant", page_icon="📊", layout="wide")

if not render_login():
    st.stop()

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_service():
    return EmbeddingService()

def process_document(filename: str, file_bytes: bytes) -> None:
    text = extract_text(filename, file_bytes)
    if not text.strip():
        raise DocumentProcessingError("No readable text was found.")
    chunks = split_text(text)
    embeddings = get_embedding_service().embed_documents(chunks)
    store = InMemoryVectorStore()
    store.add(chunks, embeddings)
    st.session_state.active_filename = filename
    st.session_state.document_text = text
    st.session_state.chunks = chunks
    st.session_state.vector_store = store
    st.session_state.pop("last_results", None)
    st.session_state.pop("last_answer", None)

st.title("AI-Powered Marketing Report Analysis and Decision Support System")
st.write("Upload a marketing report, inspect its content, and ask document-grounded questions.")

upload_tab, analysis_tab, question_tab = st.tabs(
    ["1. Upload & Process", "2. Document Analysis", "3. Ask a Question"]
)

with upload_tab:
    uploaded = st.file_uploader("Choose a report", type=["pdf", "xlsx", "pptx"])
    if uploaded:
        data = uploaded.getvalue()
        valid, message = validate_uploaded_file(uploaded.name, len(data))
        if not valid:
            st.error(message)
        else:
            st.success(message)
            if st.button("Process and index report", type="primary", use_container_width=True):
                try:
                    with st.spinner("Processing report..."):
                        process_document(uploaded.name, data)
                    st.success("Report processed successfully.")
                except DocumentProcessingError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.exception(exc)
    if st.session_state.get("active_filename"):
        st.info(f"Active report: **{st.session_state.active_filename}**")

with analysis_tab:
    text = st.session_state.get("document_text")
    chunks = st.session_state.get("chunks", [])
    if not text:
        st.warning("Upload and process a report first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Characters", f"{len(text):,}")
        c2.metric("Words", f"{len(text.split()):,}")
        c3.metric("Chunks", f"{len(chunks):,}")
        st.text_area("Extracted-text preview", text[:4000], height=280, disabled=True)
        with st.expander("View first five chunks"):
            for i, chunk in enumerate(chunks[:5], start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(chunk)

with question_tab:
    store = st.session_state.get("vector_store")
    if store is None:
        st.warning("Process a report first.")
    else:
        question = st.text_input("Question", placeholder="Which campaign performed best?")
        top_k = st.slider("Supporting sections", 1, 5, 3)
        if st.button("Retrieve evidence and answer", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question.")
            else:
                results = retrieve_relevant_chunks(question, get_embedding_service(), store, top_k)
                st.session_state.last_results = results
                st.session_state.last_answer = None
                if os.getenv("OPENAI_API_KEY"):
                    try:
                        st.session_state.last_answer = generate_answer(build_rag_prompt(question, results))
                    except LLMConfigurationError as exc:
                        st.info(str(exc))
        results = st.session_state.get("last_results", [])
        answer = st.session_state.get("last_answer")
        if results:
            st.success(format_retrieval_summary(results))
            if answer:
                st.markdown("### AI-generated answer")
                st.write(answer)
            else:
                st.markdown("### Retrieval-only result")
                st.write("OpenAI is not configured, so the app is showing retrieved evidence.")
            st.markdown("### Supporting report sections")
            for i, result in enumerate(results, start=1):
                with st.expander(f"Source {i} — similarity {result.score:.3f}"):
                    st.write(result.text)



