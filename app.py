import os
import streamlit as st

from ingestion.loader import load_document
from ingestion.preprocessing import clean_text
from ingestion.chunking import chunk_text
from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from rag.generator import GroqGenerator


# ------------------------------
# Streamlit App Config
# ------------------------------
st.set_page_config(page_title="LLM Project", layout="wide")
st.title("📘 NotebookLM-style AI Assistant")

st.write("Upload documents and ask questions based on their content.")

# ------------------------------
# Session State Initialization
# ------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "embedder" not in st.session_state:
    st.session_state.embedder = TextEmbedder()

# ------------------------------
# Sidebar: File Upload
# ------------------------------
st.sidebar.header("📂 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:
    # Save file locally
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("File uploaded successfully")

    # ------------------------------
    # Document Processing Pipeline
    # ------------------------------
    with st.spinner("Processing document..."):
        raw_text = load_document(file_path)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)

        embeddings = st.session_state.embedder.embed_texts(chunks)

        vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        vector_store.add_embeddings(embeddings, chunks)

        st.session_state.vector_store = vector_store

    st.sidebar.success(f"Document processed into {len(chunks)} chunks")

# ------------------------------
# Main: Question Answering
# ------------------------------
st.header("💬 Ask a Question")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not st.session_state.vector_store:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            query_embedding = st.session_state.embedder.embed_query(question)
            relevant_chunks = st.session_state.vector_store.similarity_search(
                query_embedding, top_k=5
            )

            generator = GroqGenerator()
            answer = generator.generate_answer(question, relevant_chunks)

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📄 Retrieved Context"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")
