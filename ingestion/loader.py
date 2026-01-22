import os
try:
    # prefer the maintained package name 'pypdf'
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("Missing PDF parser: install 'pypdf' (recommended) or 'PyPDF2'. Run: pip install pypdf")


def load_txt(file_path: str) -> str:
    """Load text from a .txt file"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """Load text from a .pdf file"""
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_document(file_path: str) -> str:
    """
    Detect file type and extract text
    Supports: PDF, TXT
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    if file_path.endswith(".pdf"):
        return load_pdf(file_path)

    elif file_path.endswith(".txt"):
        return load_txt(file_path)

    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")


# Example usage (for testing)
if __name__ == "__main__":
    path = "C:\\Users\\Admin\\Desktop\\LLM project\\data\\uploads\\Practical Machine Learning.pdf"
    text = load_document(path)
    print(text[:1000])  # print first 1000 characters
