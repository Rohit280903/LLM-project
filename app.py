import os
import streamlit as st
from dotenv import load_dotenv
import hashlib
# Load .env for local development
load_dotenv()

from ingestion.loader import load_document
from ingestion.preprocessing import clean_text_for_embedding  # ← fixed import
from ingestion.chunking import chunk_text

from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.retriever import Retriever                           # ← now used
from rag.generator import GroqGenerator
import mlflow

# Add this function OUTSIDE the button handler, near the top of app.py
# Cache answers based on question + document fingerprint to speed up repeated queries
@st.cache_data(show_spinner=False)
def get_cached_answer(question: str, rewritten_query: str, chunks_key: str):
    """
    Cache answers by question + document fingerprint.
    chunks_key is a hash of the current chunks so cache invalidates on new upload.
    """
    relevant_chunks = st.session_state.retriever.retrieve(rewritten_query, top_k=5)
    answer = st.session_state.generator.generate_answer(
        rewritten_query,
        relevant_chunks,
        chat_history=[]  # Don't cache with history — history changes every turn
    )
    return answer, relevant_chunks

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="NotebookLM-style AI", layout="wide")
st.title("📘 NotebookLM-style AI Assistant")
st.write("Upload documents and ask grounded questions with citations.")

# ------------------------------
# Session State
# ------------------------------
defaults = {
    "vector_store": None,
    "chunks": [],
    "embedder": None,
    "generator": None,
    "retriever": None,
    "chat_history": [],
    "suggested_questions": [],
    "processed_files": [],
    "selected_question": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Initialise heavy objects once
if st.session_state.embedder is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = TextEmbedder()

if st.session_state.generator is None:
    try:
        st.session_state.generator = GroqGenerator()
    except ValueError as e:
        st.error(str(e))
        st.stop()

# ------------------------------
# Sidebar: File Upload
# ------------------------------
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    all_chunks = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                raw_text = load_document(file_path)
            except Exception as e:
                st.sidebar.error(f"Failed to load {uploaded_file.name}: {e}")
                continue

            # ✅ FIX: use embedding-safe cleaning (preserves punctuation & case)
            cleaned_text = clean_text_for_embedding(raw_text)
            chunks = chunk_text(cleaned_text)

            labeled_chunks = [
                f"[Doc: {uploaded_file.name} | Chunk {i+1}] {chunk}"
                for i, chunk in enumerate(chunks)
            ]
            all_chunks.extend(labeled_chunks)

        if all_chunks:
            embeddings = st.session_state.embedder.embed_texts(all_chunks)

            # ✅ FIX: always create a fresh vector store on new upload
            vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
            vector_store.add_embeddings(embeddings, all_chunks)
            # After building the vector store, add:
            st.session_state.chunks_key = hashlib.md5(
                "".join(all_chunks[:5]).encode()
            ).hexdigest()

            mlflow.set_experiment("rag-project")

            with mlflow.start_run(run_name=f"upload_{uploaded_files[0].name}"):
                mlflow.log_param("chunk_size", 500)
                mlflow.log_param("chunk_overlap", 100)
                mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
                mlflow.log_param("groq_model", "llama-3.1-8b-instant")
                mlflow.log_param("top_k", 5)
                mlflow.log_param("num_files", len(uploaded_files))

                mlflow.log_metric("num_chunks", len(all_chunks))
                mlflow.log_metric(
                    "avg_chunk_length",
                    sum(len(c) for c in all_chunks) / len(all_chunks)
                )

                # Log file names
                mlflow.set_tag("files", ", ".join(f.name for f in uploaded_files))
            st.session_state.vector_store = vector_store
            st.session_state.chunks = all_chunks
            st.session_state.processed_files = [f.name for f in uploaded_files]

            # ✅ FIX: reuse session generator instead of creating a new one
            try:
                st.session_state.suggested_questions = (
                    st.session_state.generator.generate_questions(all_chunks)
                )
            except Exception as e:
                st.sidebar.warning(f"Could not generate suggestions: {e}")

            # Build Retriever
            st.session_state.retriever = Retriever(
                embedder=st.session_state.embedder,
                vector_store=vector_store
            )

    st.sidebar.success(f"✅ Processed {len(all_chunks)} chunks")

# ------------------------------
# Sidebar: Processed File List
# ------------------------------
if st.session_state.processed_files:
    st.sidebar.subheader("📄 Loaded Documents")
    for fname in st.session_state.processed_files:
        st.sidebar.markdown(f"- `{fname}`")

# ------------------------------
# Sidebar: Suggested Questions
# ------------------------------
if st.session_state.suggested_questions:
    st.sidebar.subheader("💡 Suggested Questions")
    for i, q in enumerate(st.session_state.suggested_questions):
        if st.sidebar.button(q, key=f"sq_{i}"):
            st.session_state.input_query = q


# ------------------------------
# Main Area
# ------------------------------
st.header("💬 Ask a Question")

question = st.text_input(
    "Enter your question:",
    value=st.session_state.selected_question
)

col1, col2 = st.columns(2)

# ------------------------------
# Ask Question (with streaming)
# ------------------------------
if col1.button("Get Answer"):
    st.session_state.selected_question = ""

    if not st.session_state.vector_store:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            rewritten_query = st.session_state.generator.rewrite_query(question)

            import hashlib
            cache_key = hashlib.md5(
                (question + rewritten_query + st.session_state.get("chunks_key", "default")).encode()
            ).hexdigest()

            # ------------------------------
            # ✅ CACHE CHECK
            # ------------------------------
            if "qa_cache" not in st.session_state:
                st.session_state.qa_cache = {}

            if cache_key in st.session_state.qa_cache:
                # ✅ Cached response (instant)
                full_answer, relevant_chunks = st.session_state.qa_cache[cache_key]

                st.subheader("📌 Answer (Cached ⚡)")
                st.write(full_answer)

                cached_flag = 1

            else:
                # ------------------------------
                # ❌ NOT CACHED → RUN FULL PIPELINE
                # ------------------------------
                relevant_chunks = st.session_state.retriever.retrieve(
                    rewritten_query, top_k=5
                )

                st.subheader("📌 Answer")

                full_answer = st.write_stream(
                    st.session_state.generator.generate_answer_stream(
                        rewritten_query,
                        relevant_chunks,
                        chat_history=st.session_state.chat_history
                    )
                )

                # Save in cache
                st.session_state.qa_cache[cache_key] = (full_answer, relevant_chunks)

                cached_flag = 0

            # ✅ MLflow logging for Q&A
            mlflow.end_run()  # prevent active run issue in Streamlit

            with mlflow.start_run(run_name="qa_query"):
                mlflow.log_param("query", question[:100])
                mlflow.log_param("rewritten_query", rewritten_query[:100])
                mlflow.log_metric("num_chunks_retrieved", len(relevant_chunks))
                mlflow.log_metric("answer_length", len(full_answer))

            # Save to history
            st.session_state.chat_history.append((question, full_answer))

            # ✅ UX: clean citations panel
            with st.expander("📄 Source Chunks Used"):
                for chunk in relevant_chunks:
                    # Parse "[Doc: file.pdf | Chunk 3] text..."
                    if chunk.startswith("[Doc:"):
                        header_end = chunk.find("]")
                        header = chunk[1:header_end]   # "Doc: file.pdf | Chunk 3"
                        body = chunk[header_end+1:].strip()
                        parts = header.split("|")
                        doc_name = parts[0].replace("Doc:", "").strip()
                        chunk_num = parts[1].strip() if len(parts) > 1 else ""
                        st.markdown(f"**{doc_name}** — {chunk_num}")
                        st.caption(body[:300] + ("..." if len(body) > 300 else ""))
                        st.divider()
                    else:
                        st.markdown(chunk[:300])

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ------------------------------
# Summarize Document
# ------------------------------
if col2.button("Summarize Document"):
    if not st.session_state.chunks:
        st.warning("Upload a document first.")
    else:
        try:
            with st.spinner("Generating summary..."):
                summary = st.session_state.generator.summarize_document(
                    st.session_state.chunks
                )
            st.subheader("📝 Document Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization failed: {e}")

# ------------------------------
# Chat History
# ------------------------------
if st.session_state.chat_history:
    st.divider()
    st.subheader("🧠 Conversation History")

    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)