import base64
import json
import os

import pymupdf
from dotenv import load_dotenv
from google import genai


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# 1. NATIVE PDF ANNOTATION DETECTION
# ============================================================

def detect_native_annotations(file_path):
    """
    Detect real PDF annotation objects.

    Supports:
        - Highlight
        - Underline
        - Strikeout
        - Squiggly
        - Other native PDF annotations

    These are annotations created using a PDF editor,
    PDF viewer, etc.
    """

    document = pymupdf.open(file_path)

    pages = []

    try:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            annotations = []

            for annotation in page.annots():

                annotation_type = annotation.type

                annotation_name = (
                    annotation_type[1]
                    if annotation_type
                    else "Unknown"
                )

                # Get text associated with annotation
                try:
                    text = page.get_textbox(
                        annotation.rect
                    ).strip()
                except Exception:
                    text = ""

                annotations.append({
                    "type": annotation_name,
                    "text": text,
                    "source": "native_pdf_annotation"
                })

            pages.append({
                "page_number": page_number,
                "annotations": annotations
            })

    finally:

        document.close()

    return {
        "total_pages": len(pages),
        "pages": pages
    }


# ============================================================
# 2. GEMINI VISION ANALYSIS
# ============================================================

def analyze_page_with_gemini(
    page_image_bytes,
    page_number
):
    """
    Analyze one rendered document page using Gemini Vision.

    Gemini visually examines the actual page image and tries
    to identify human-created markings such as:

        - underline
        - highlight
        - pencil marks
        - pen marks
        - strike-through
        - circles
        - boxes

    Printed document elements such as normal text,
    table borders, logos and lines should NOT be treated
    as annotations.
    """

    encoded_image = base64.b64encode(
        page_image_bytes
    ).decode("utf-8")

    prompt = f"""
You are the visual annotation detection component of a
professional Document AI system.

You are analyzing PAGE {page_number} of a document.

Look carefully at the actual page image.

Your task is to identify HUMAN-MADE MARKINGS that were
added to the original document.

Possible markings include:

- Underline
- Highlight
- Pen mark
- Pencil mark
- Strike-through
- Circle around text
- Box around text
- Other obvious handwritten or manually added marks

IMPORTANT:

1. Look at the VISUAL appearance of the page.
2. Do not assume markings are red.
3. Markings can be ANY color.
4. Pencil/graphite marks may be gray or faint.
5. Pen marks may be blue, black, red, green, etc.
6. Highlighting may be transparent yellow, green, blue,
   pink, orange, etc.
7. Do NOT classify normal printed text as a marking.
8. Do NOT classify normal table borders as a marking.
9. Do NOT classify logos as markings.
10. Do NOT classify page borders as markings.
11. Do NOT classify normal printed underlines that are part
    of the document design as user annotations.
12. Do NOT invent annotations.
13. Only report markings that are visually present.
14. Extract the text that the marking applies to whenever
    it is readable.
15. If the exact text cannot be confidently determined,
    use an empty string.
16. Return ONLY valid JSON.
17. Do not use Markdown code fences.

Return exactly:

{{
    "annotations": [
        {{
            "type": "Underline",
            "text": ""
        }}
    ]
}}

Allowed types:

Underline
Highlight
Strikeout
Pencil
Pen
Circle
Box

If there are no human-created markings:

{{
    "annotations": []
}}
"""

    try:

        response = client.interactions.create(
            model=MODEL_NAME,
            input=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image",
                    "data": encoded_image,
                    "mime_type": "image/png"
                }
            ]
        )

        output_text = (
            response.output_text
            .strip()
        )

        # ----------------------------------------------------
        # Remove Markdown fences if Gemini adds them
        # ----------------------------------------------------

        if output_text.startswith(
            "```json"
        ):

            output_text = output_text[
                7:
            ]

        elif output_text.startswith(
            "```"
        ):

            output_text = output_text[
                3:
            ]

        if output_text.endswith(
            "```"
        ):

            output_text = output_text[
                :-3
            ]

        output_text = output_text.strip()

        result = json.loads(
            output_text
        )

        annotations = result.get(
            "annotations",
            []
        )

        valid_types = {
            "Underline",
            "Highlight",
            "Strikeout",
            "Pencil",
            "Pen",
            "Circle",
            "Box"
        }

        cleaned_annotations = []

        for annotation in annotations:

            annotation_type = str(
                annotation.get(
                    "type",
                    ""
                )
            ).strip()

            text = str(
                annotation.get(
                    "text",
                    ""
                )
            ).strip()

            if annotation_type not in valid_types:
                continue

            cleaned_annotations.append({
                "type": annotation_type,
                "text": text,
                "source": "gemini_vision"
            })

        return cleaned_annotations

    except json.JSONDecodeError:

        print(
            f"Gemini returned invalid JSON "
            f"for page {page_number}"
        )

        return []

    except Exception as e:

        print(
            f"Gemini Vision error on page "
            f"{page_number}: {e}"
        )

        return []


# ============================================================
# 3. VISUAL ANNOTATION DETECTION
# ============================================================

def detect_visual_annotations(
    file_path,
    scale=2.0
):
    """
    Render every PDF page as an image and send it to
    Gemini Vision.

    This is designed for scanned/image-based documents
    where markings are physically part of the page image.
    """

    document = pymupdf.open(
        file_path
    )

    pages = []

    try:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            print(
                f"Analyzing page "
                f"{page_number} with Gemini Vision..."
            )

            matrix = pymupdf.Matrix(
                scale,
                scale
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_bytes = pixmap.tobytes(
                "png"
            )

            annotations = (
                analyze_page_with_gemini(
                    image_bytes,
                    page_number
                )
            )

            pages.append({
                "page_number": page_number,
                "annotations": annotations
            })

    finally:

        document.close()

    return {
        "total_pages": len(pages),
        "pages": pages
    }


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

def remove_duplicate_annotations(
    annotations
):
    """
    Remove duplicate annotation entries.

    Gemini may occasionally return the same marking
    more than once.
    """

    unique = []

    seen = set()

    for annotation in annotations:

        annotation_type = annotation.get(
            "type",
            ""
        )

        text = annotation.get(
            "text",
            ""
        )

        key = (
            annotation_type.lower().strip(),
            text.lower().strip()
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(annotation)

    return unique


# ============================================================
# 5. COMBINED PDF ANNOTATION DETECTION
# ============================================================

def detect_pdf_annotations(
    file_path
):
    """
    Complete annotation detection pipeline.

    It combines:

        A. Native PDF annotation detection
        B. Gemini Vision visual detection

    This allows the system to detect both:

        Native annotations:
            PDF highlight
            PDF underline
            PDF strikeout

        Visual markings:
            Pen
            Pencil
            Colored underline
            Handwritten marks
            Visual highlight
            Circle
            Box
            etc.
    """

    print(
        "\n========================================"
    )

    print(
        "NATIVE PDF ANNOTATION DETECTION"
    )

    print(
        "========================================"
    )

    native_result = (
        detect_native_annotations(
            file_path
        )
    )

    print(
        "\n========================================"
    )

    print(
        "GEMINI VISION DETECTION"
    )

    print(
        "========================================"
    )

    visual_result = (
        detect_visual_annotations(
            file_path
        )
    )

    # --------------------------------------------------------
    # Merge page-by-page
    # --------------------------------------------------------

    pages = []

    total_pages = max(
        native_result.get(
            "total_pages",
            0
        ),
        visual_result.get(
            "total_pages",
            0
        )
    )

    for page_number in range(
        1,
        total_pages + 1
    ):

        native_annotations = []

        visual_annotations = []

        # Native
        for page in native_result[
            "pages"
        ]:

            if page[
                "page_number"
            ] == page_number:

                native_annotations = (
                    page.get(
                        "annotations",
                        []
                    )
                )

                break

        # Visual
        for page in visual_result[
            "pages"
        ]:

            if page[
                "page_number"
            ] == page_number:

                visual_annotations = (
                    page.get(
                        "annotations",
                        []
                    )
                )

                break

        combined = (
            native_annotations
            + visual_annotations
        )

        combined = (
            remove_duplicate_annotations(
                combined
            )
        )

        pages.append({
            "page_number": page_number,
            "annotations": combined
        })

    return {
        "total_pages": len(pages),
        "pages": pages
    }
