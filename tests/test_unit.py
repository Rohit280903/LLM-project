from ingestion.preprocessing import clean_text_for_embedding
from ingestion.chunking import chunk_text

def test_clean_text_preserves_case():
    result = clean_text_for_embedding("Hello World!")
    assert "Hello" in result  # case preserved

def test_chunking_overlap():
    words = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_text(words, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert "word40" in chunks[0]  # overlap works