import os
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """
    Try normal text extraction first.
    If the page has no text (scanned), fall back to OCR.
    """
    reader = PdfReader(file_path)
    text = ""
    scanned_pages = []

    # Step 1: Try normal extraction
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text += page_text + "\n"
        else:
            scanned_pages.append(i)  # mark as needing OCR

    # Step 2: OCR the scanned pages
    if scanned_pages:
        if not OCR_AVAILABLE:
            raise RuntimeError(
                "This PDF contains scanned pages but OCR packages are not installed. "
                "Run: pip install pytesseract pdf2image Pillow"
            )

        images = convert_from_path(file_path, dpi=300)

        for i in scanned_pages:
            if i < len(images):
                ocr_text = pytesseract.image_to_string(images[i])
                text += ocr_text + "\n"

    return text


def load_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".txt"):
        return load_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")