def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list:
    """
    Split text into overlapping chunks
    
    Args:
        text (str): Cleaned text
        chunk_size (int): Number of words per chunk
        overlap (int): Overlapping words between chunks
    
    Returns:
        List of text chunks
    """

    words = text.split()
    chunks = []

    start = 0
    total_words = len(words)

    while start < total_words:
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap

        if start < 0:
            start = 0

    return chunks


# Example usage (for testing)
if __name__ == "__main__":
    sample_text = " ".join([f"word{i}" for i in range(1, 2000)])
    chunks = chunk_text(sample_text)

    print(f"Total chunks: {len(chunks)}")
    print("First chunk:", chunks[0][:300])
