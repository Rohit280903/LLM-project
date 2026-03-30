import numpy as np
from embeddings.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore


class Retriever:
    """
    Handles embedding a query and retrieving
    the most relevant chunks from the vector store.
    """

    def __init__(self, embedder: TextEmbedder, vector_store: FAISSVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5):
        """
        Embed the query and return top_k relevant chunks.

        Args:
            query (str): User's question
            top_k (int): Number of chunks to retrieve

        Returns:
            List of relevant text chunks
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        return results