import uuid
from datetime import datetime, timezone


def build_document_store(
    filename,
    file_type,
    extraction_method,
    document_data,
    understanding,
    annotations=None
):
    """
    Build the structured document store.

    Stores:
    - document information
    - pages
    - extracted text
    - word-level coordinates
    - document understanding
    - detected annotations/markings

    Markings are initially empty during /upload.
    They are added later when the user selects
    "YES — FIND MARKINGS".
    """

    document_id = str(uuid.uuid4())

    pages = []

    # =====================================================
    # BUILD PAGE DATA
    # =====================================================

    for page in document_data.get("pages", []):

        page_number = page.get(
            "page_number"
        )

        page_id = (
            f"{document_id}-page-{page_number}"
        )

        words = []

        # -------------------------------------------------
        # WORDS + COORDINATES
        # -------------------------------------------------

        for word_index, word in enumerate(
            page.get("words", [])
        ):

            source_id = (
                f"{page_id}-word-{word_index}"
            )

            words.append({
                "source_id": source_id,

                "text": word.get(
                    "text",
                    ""
                ),

                "confidence": word.get(
                    "confidence"
                ),

                "bounding_box": {
                    "x0": word.get("x0"),
                    "y0": word.get("y0"),
                    "x1": word.get("x1"),
                    "y1": word.get("y1")
                }
            })

        # -------------------------------------------------
        # PAGE OBJECT
        # -------------------------------------------------

        pages.append({

            "page_id": page_id,

            "page_number": page_number,

            "text": page.get(
                "text",
                ""
            ),

            "words": words,

            # Initially empty.
            # Markings are added later only if requested.
            "annotations": []

        })

    # =====================================================
    # INITIAL ANNOTATION STRUCTURE
    # =====================================================

    annotation_store = {
        "total_pages": len(pages),
        "pages": []
    }

    # =====================================================
    # IF ANNOTATIONS WERE PROVIDED
    # =====================================================

    if annotations:

        annotation_store = annotations

        for annotation_page in annotations.get(
            "pages",
            []
        ):

            page_number = annotation_page.get(
                "page_number"
            )

            detected_annotations = (
                annotation_page.get(
                    "annotations",
                    []
                )
            )

            # Find matching page
            for page in pages:

                if page["page_number"] != page_number:
                    continue

                # -----------------------------------------
                # ADD ANNOTATIONS TO PAGE
                # -----------------------------------------

                for annotation in detected_annotations:

                    page["annotations"].append({

                        "annotation_id": str(
                            uuid.uuid4()
                        ),

                        "type": annotation.get(
                            "type",
                            ""
                        ),

                        "text": annotation.get(
                            "text",
                            ""
                        ),

                        "source": annotation.get(
                            "source",
                            "unknown"
                        ),

                        "bounding_box": annotation.get(
                            "bounding_box"
                        ),

                        "confidence": annotation.get(
                            "confidence"
                        )

                    })

                break

    # =====================================================
    # COMPLETE DOCUMENT STORE
    # =====================================================

    document_store = {

        "document_id": document_id,

        "filename": filename,

        "file_type": file_type,

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "extraction_method": extraction_method,

        # -------------------------------------------------
        # DOCUMENT METADATA
        # -------------------------------------------------

        "metadata": {

            "total_pages": len(pages)

        },

        # -------------------------------------------------
        # PAGE DATA
        # -------------------------------------------------

        "pages": pages,

        # -------------------------------------------------
        # GEMINI DOCUMENT UNDERSTANDING
        # -------------------------------------------------

        "understanding": understanding,

        # -------------------------------------------------
        # MARKINGS / ANNOTATIONS
        # -------------------------------------------------

        "annotations": annotation_store

    }

    return document_store