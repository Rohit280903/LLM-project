import faiss
import numpy as np
import pickle
import os
from typing import List


class FAISSVectorStore:
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index.

        Args:
            embedding_dim (int): Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[str]):
        """
        Add embeddings and corresponding text chunks to FAISS.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks size mismatch")

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k most similar text chunks.
        """
        query_embedding = np.array([query_embedding])
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    def save(self, index_path: str, chunks_path: str):
        """
        Persist FAISS index and text chunks to disk.

        Args:
            index_path (str): Path to save the FAISS index (e.g. 'data/index.faiss')
            chunks_path (str): Path to save the chunks list (e.g. 'data/chunks.pkl')
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.text_chunks, f)

    @classmethod
    def load(cls, index_path: str, chunks_path: str, embedding_dim: int):
        """
        Load a previously saved FAISS index and chunks from disk.

        Args:
            index_path (str): Path to the saved FAISS index
            chunks_path (str): Path to the saved chunks list
            embedding_dim (int): Dimension of embeddings

        Returns:
            FAISSVectorStore instance
        """
        store = cls(embedding_dim=embedding_dim)
        store.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            store.text_chunks = pickle.load(f)
        return store

    @staticmethod
    def exists(index_path: str, chunks_path: str) -> bool:
        """Check if a saved index exists on disk."""
        return os.path.exists(index_path) and os.path.exists(chunks_path)