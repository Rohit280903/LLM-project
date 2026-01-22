import re


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters (keep letters and numbers)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Example usage (for testing)
if __name__ == "__main__":
    sample_text = """
    This is a SAMPLE text!! Visit https://example.com
    Extra    spaces, symbols ### and new lines.
    """
    print(clean_text(sample_text))
