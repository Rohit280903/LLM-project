import os
import streamlit as st

from ingestion.loader import load_document
from ingestion.preprocessing import clean_text
from ingestion.chunking import chunk_text

from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.generator import GroqGenerator


# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="NotebookLM-style AI", layout="wide")
st.title("📘 NotebookLM-style AI Assistant")
st.write("Upload documents and ask grounded questions with citations.")

# ------------------------------
# Session State
# ------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embedder" not in st.session_state:
    st.session_state.embedder = TextEmbedder()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# ------------------------------
# Sidebar: File Upload
# ------------------------------
st.sidebar.header("📂 Upload Document")

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

            raw_text = load_document(file_path)
            cleaned_text = clean_text(raw_text)
            chunks = chunk_text(cleaned_text)

            # add document metadata
            labeled_chunks = [
                f"[Doc: {uploaded_file.name} | Chunk {i+1}] {chunk}"
                for i, chunk in enumerate(chunks)
            ]

            all_chunks.extend(labeled_chunks)

        embeddings = st.session_state.embedder.embed_texts(all_chunks)

        vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        vector_store.add_embeddings(embeddings, all_chunks)

        st.session_state.vector_store = vector_store
        st.session_state.chunks = all_chunks

        # Generate suggested questions
        generator = GroqGenerator()
        st.session_state.suggested_questions = generator.generate_questions(all_chunks)

    st.sidebar.success(f"Processed {len(all_chunks)} chunks")

# ------------------------------
# Sidebar: Suggested Questions
# ------------------------------
if st.session_state.suggested_questions:
    st.sidebar.subheader("💡 Suggested Questions")
    for q in st.session_state.suggested_questions:
        if st.sidebar.button(q):
            st.session_state.selected_question = q

# ------------------------------
# Main Area
# ------------------------------
st.header("💬 Ask a Question")

question = st.text_input(
    "Enter your question:",
    value=st.session_state.get("selected_question", "")
)

col1, col2 = st.columns(2)

# ------------------------------
# Ask Question
# ------------------------------
if col1.button("Get Answer"):
    if not st.session_state.vector_store:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating grounded answer..."):
            generator = GroqGenerator()

            # Agentic step: rewrite query
            rewritten_query = generator.rewrite_query(question)

            query_embedding = st.session_state.embedder.embed_query(rewritten_query)

            relevant_chunks = st.session_state.vector_store.similarity_search(
                query_embedding, top_k=5
            )

            answer = generator.generate_answer(
                rewritten_query, relevant_chunks
            )

            st.session_state.chat_history.append((question, answer))

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Context"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")

# ------------------------------
# Summarize Document
# ------------------------------
if col2.button("Summarize Document"):
    if not st.session_state.chunks:
        st.warning("Upload a document first.")
    else:
        with st.spinner("Generating summary..."):
            generator = GroqGenerator()
            summary = generator.summarize_document(st.session_state.chunks)

        st.subheader("📝 Document Summary")
        st.write(summary)

# ------------------------------
# Chat History
# ------------------------------
if st.session_state.chat_history:
    st.divider()
    st.subheader("🧠 Conversation History")

    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")
