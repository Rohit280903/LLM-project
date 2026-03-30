import re


def clean_text_for_embedding(text: str) -> str:
    """
    Lightly clean text while preserving semantic meaning.
    Use this BEFORE chunking and embedding.
    Keeps punctuation, case, and sentence structure intact.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove non-printable / control characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Collapse excessive whitespace and blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_text_for_display(text: str) -> str:
    """
    Aggressively normalize text for display/search purposes only.
    Do NOT use this before embedding — it destroys semantic quality.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Example usage (for testing)
if __name__ == "__main__":
    sample_text = """
    This is a SAMPLE text!! Visit https://example.com
    Extra    spaces, symbols ### and new lines.
    """
    print("For embedding:")
    print(clean_text_for_embedding(sample_text))
    print("\nFor display:")
    print(clean_text_for_display(sample_text))