from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from services.input_service import process_document
from services.document_understanding_service import (
    understand_document
)
from services.annotation_service import (
    detect_pdf_annotations
)
from services.document_store_service import (
    build_document_store
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Document Intelligence",
    description=(
        "AI-powered general-purpose "
        "document extraction and understanding system"
    ),
    version="2.0.0"
)


# =========================================================
# DIRECTORIES
# =========================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# =========================================================
# DOCUMENT FILE MAPPING
#
# document_id -> uploaded file path
#
# This allows the marking endpoint to know exactly
# which document the user selected.
# =========================================================

DOCUMENT_FILE_MAP = {}


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Document Intelligence API is running!"
    }


# =========================================================
# UPLOAD + DOCUMENT UNDERSTANDING
#
# IMPORTANT:
# Marking detection is NOT performed here.
#
# Workflow:
#
# Upload
#   ↓
# Extraction
#   ↓
# Document Understanding
#   ↓
# Return result
#
# Gemini Vision is NOT called here.
# =========================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    supported_extensions = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png"
    }

    # -----------------------------------------------------
    # CHECK FILENAME
    # -----------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    # -----------------------------------------------------
    # CHECK FILE EXTENSION
    # -----------------------------------------------------

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in supported_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported formats: "
                "PDF, JPG, JPEG, PNG."
            )
        )

    # -----------------------------------------------------
    # SAVE FILE
    # -----------------------------------------------------

    file_path = UPLOAD_DIR / file.filename

    try:

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save uploaded file: "
                f"{str(e)}"
            )
        )

    # -----------------------------------------------------
    # DOCUMENT EXTRACTION
    # -----------------------------------------------------

    try:

        processed = process_document(
            str(file_path)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed: "
                f"{str(e)}"
            )
        )

    extraction_method = processed[
        "extraction_method"
    ]

    extracted_data = processed[
        "document"
    ]

    # -----------------------------------------------------
    # DOCUMENT UNDERSTANDING
    #
    # Gemini analyzes the extracted document content.
    #
    # NO MARKING DETECTION HERE.
    # -----------------------------------------------------

    try:

        understanding = understand_document(
            extracted_data
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document understanding failed: "
                f"{str(e)}"
            )
        )

    # -----------------------------------------------------
    # EMPTY ANNOTATION STRUCTURE
    #
    # Markings will be detected only if the user
    # later chooses YES.
    # -----------------------------------------------------

    annotations = {

        "total_pages":
            extracted_data.get(
                "total_pages",
                0
            ),

        "pages": []

    }

    # -----------------------------------------------------
    # BUILD DOCUMENT STORE
    # -----------------------------------------------------

    try:

        document_store = build_document_store(

            filename=file.filename,

            file_type=extension,

            extraction_method=extraction_method,

            document_data=extracted_data,

            understanding=understanding,

            annotations=annotations

        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document store creation failed: "
                f"{str(e)}"
            )
        )

    # -----------------------------------------------------
    # GET GENERATED DOCUMENT ID
    # -----------------------------------------------------

    document_id = document_store[
        "document_id"
    ]

    # -----------------------------------------------------
    # STORE DOCUMENT ID → FILE PATH
    #
    # Example:
    #
    # "8c4f..." :
    #     uploads/my_document.pdf
    # -----------------------------------------------------

    DOCUMENT_FILE_MAP[
        document_id
    ] = str(file_path)

    # -----------------------------------------------------
    # RETURN BASIC DOCUMENT ANALYSIS
    # -----------------------------------------------------

    return {

        "document_id":
            document_id,

        "filename":
            document_store[
                "filename"
            ],

        "file_type":
            document_store[
                "file_type"
            ],

        "extraction_method":
            document_store[
                "extraction_method"
            ],

        "total_pages":
            document_store[
                "metadata"
            ][
                "total_pages"
            ],

        "annotations":
            annotations,

        "document":
            document_store

    }


# =========================================================
# ANALYZE MARKINGS
#
# THIS ENDPOINT IS CALLED ONLY WHEN THE USER CLICKS:
#
# YES — FIND MARKINGS
#
# It does:
#
# Native PDF annotations
#        +
# Gemini Vision
#
# It does NOT run during normal upload.
# =========================================================

@app.post(
    "/analyze-markings/{document_id}"
)
async def analyze_markings(
    document_id: str
):

    # -----------------------------------------------------
    # CHECK DOCUMENT ID
    # -----------------------------------------------------

    if document_id not in DOCUMENT_FILE_MAP:

        raise HTTPException(
            status_code=404,
            detail=(
                "Document not found. "
                "Please upload the document again."
            )
        )

    # -----------------------------------------------------
    # GET EXACT FILE PATH
    # -----------------------------------------------------

    file_path = Path(
        DOCUMENT_FILE_MAP[
            document_id
        ]
    )

    # -----------------------------------------------------
    # CHECK FILE EXISTS
    # -----------------------------------------------------

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "The uploaded document "
                "could not be found."
            )
        )

    # -----------------------------------------------------
    # ONLY PDF MARKING ANALYSIS FOR NOW
    #
    # Your current annotation service works with PDFs.
    # -----------------------------------------------------

    extension = file_path.suffix.lower()

    if extension != ".pdf":

        return {

            "document_id":
                document_id,

            "filename":
                file_path.name,

            "markings_found":
                False,

            "message":
                (
                    "Visual marking analysis is "
                    "currently available for PDF "
                    "documents."
                ),

            "annotations": {

                "total_pages": 0,

                "pages": []

            }

        }

    # -----------------------------------------------------
    # RUN MARKING DETECTION
    #
    # This is the expensive part.
    # It happens ONLY after the user selects YES.
    # -----------------------------------------------------

    try:

        annotations = detect_pdf_annotations(
            str(file_path)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Marking analysis failed: "
                f"{str(e)}"
            )
        )

    # -----------------------------------------------------
    # CHECK WHETHER ANY MARKINGS WERE FOUND
    # -----------------------------------------------------

    markings_found = any(

        page.get(
            "annotations",
            []
        )

        for page in annotations.get(
            "pages",
            []
        )

    )

    # -----------------------------------------------------
    # RETURN MARKING RESULTS
    # -----------------------------------------------------

    return {

        "document_id":
            document_id,

        "filename":
            file_path.name,

        "markings_found":
            markings_found,

        "annotations":
            annotations

    }