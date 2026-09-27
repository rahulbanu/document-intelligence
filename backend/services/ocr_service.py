from pathlib import Path
import pymupdf
from paddleocr import PaddleOCR


# Initialize PaddleOCR once when the service starts.
# This avoids loading the OCR model again for every page/request.
ocr = PaddleOCR(
    lang="en",
     enable_mkldnn=False
)


def ocr_pdf(file_path: str):
    """
    Perform OCR on every page of a scanned PDF.

    Returns:
        {
            "total_pages": int,
            "pages": [
                {
                    "page_number": int,
                    "text": str,
                    "words": [
                        {
                            "text": str,
                            "confidence": float,
                            "x0": float,
                            "y0": float,
                            "x1": float,
                            "y1": float
                        }
                    ]
                }
            ]
        }
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    document = pymupdf.open(file_path)

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            # Render PDF page as an image.
            # 2x scale gives OCR a better-quality image.
            matrix = pymupdf.Matrix(2, 2)
            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_path = pdf_path.parent / (
                f"{pdf_path.stem}_page_{page_number}.png"
            )

            pixmap.save(str(image_path))

            # Run PaddleOCR on the rendered page.
            result = ocr.predict(str(image_path))

            page_words = []
            page_text_parts = []

            # PaddleOCR 3.x returns structured prediction objects.
            for res in result:

                # Extract recognized text.
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

                    # OCR bounding box:
                    # [x_min, y_min, x_max, y_max]
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

                    page_text_parts.append(text.strip())

            pages.append({
                "page_number": page_number,
                "text": " ".join(page_text_parts),
                "words": page_words
            })

            # Remove temporary rendered image.
            try:
                image_path.unlink()
            except OSError:
                pass

    finally:
        document.close()

    return {
        "total_pages": len(pages),
        "pages": pages
    }