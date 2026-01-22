from sentence_transformers import SentenceTransformer
from typing import List


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        """
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[list]:
        """
        Generate embeddings for multiple text chunks
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> list:
        """
        Generate embedding for a user query
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        return embedding


# Example usage (for testing)
if __name__ == "__main__":
    embedder = TextEmbedder()

    sample_chunks = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with many layers"
    ]

    embeddings = embedder.embed_texts(sample_chunks)
    query_embedding = embedder.embed_query("what is deep learning")

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Query embedding length: {len(query_embedding)}")
