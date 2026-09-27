import pymupdf


def extract_pdf_content(file_path: str):
    """
    Extract text and word-level coordinates from every page
    of a text-based PDF.
    """

    document = pymupdf.open(file_path)

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")
            words = page.get_text("words")

            page_data = {
                "page_number": page_number,
                "text": text,
                "words": []
            }

            for word in words:
                page_data["words"].append({
                    "text": word[4],
                    "x0": word[0],
                    "y0": word[1],
                    "x1": word[2],
                    "y1": word[3]
                })

            pages.append(page_data)

    finally:
        document.close()

    return {
        "total_pages": len(pages),
        "pages": pages
    }


def has_extractable_text(document_data):
    """
    Determine whether the PDF contains usable selectable text.
    """

    for page in document_data["pages"]:
        if page["text"].strip():
            return True

    return False