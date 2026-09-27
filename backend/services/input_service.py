from pathlib import Path

from services.pdf_service import (
    extract_pdf_content,
    has_extractable_text
)

from services.ocr_service import ocr_pdf


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png"
}


def process_document(file_path: str):
    """
    Process a PDF or image and return a unified document structure.

    Supported:
        PDF
        JPG
        JPEG
        PNG

    The output format is kept consistent so that Gemini,
    RAG, and document Q&A do not need to know the original
    file type.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # ---------------------------------------------------------
    # PDF
    # ---------------------------------------------------------

    if extension == ".pdf":

        extracted_data = extract_pdf_content(
            str(path)
        )

        if has_extractable_text(extracted_data):

            return {
                "extraction_method": "PyMuPDF",
                "document": extracted_data
            }

        # Scanned/image PDF
        extracted_data = ocr_pdf(
            str(path)
        )

        return {
            "extraction_method": "OCR",
            "document": extracted_data
        }

    # ---------------------------------------------------------
    # IMAGE
    # ---------------------------------------------------------

    return process_image(
        str(path)
    )


def process_image(file_path: str):
    """
    Process a single image document using PaddleOCR.

    The image is converted into the same page-based
    representation used by PDF processing.
    """

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="en",
        enable_mkldnn=False
    )

    result = ocr.predict(file_path)

    page_words = []
    page_text_parts = []

    for res in result:

        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("rec_boxes", [])

        for i, text in enumerate(texts):

            if not text or not text.strip():
                continue

            confidence = (
                float(scores[i])
                if i < len(scores)
                else 0.0
            )

            if i < len(boxes):

                box = boxes[i]

                x0 = float(box[0])
                y0 = float(box[1])
                x1 = float(box[2])
                y1 = float(box[3])

            else:

                x0 = y0 = x1 = y1 = 0.0

            page_words.append({
                "text": text.strip(),
                "confidence": confidence,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            })

            page_text_parts.append(
                text.strip()
            )

    return {
        "extraction_method": "OCR",
        "document": {
            "total_pages": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": " ".join(page_text_parts),
                    "words": page_words
                }
            ]
        }
    }