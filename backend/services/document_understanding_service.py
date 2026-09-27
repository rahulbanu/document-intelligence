import json
import os

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
# JSON CLEANING
# ============================================================

def clean_json_response(output_text):
    """
    Clean Gemini's response before JSON parsing.

    Gemini may sometimes return JSON inside Markdown
    code fences even when asked not to.
    """

    output_text = output_text.strip()

    if output_text.startswith("```json"):
        output_text = output_text[7:]

    elif output_text.startswith("```"):
        output_text = output_text[3:]

    if output_text.endswith("```"):
        output_text = output_text[:-3]

    return output_text.strip()


# ============================================================
# DOCUMENT UNDERSTANDING
# ============================================================

def understand_document(document_data):
    """
    Perform general-purpose document understanding.

    This is NOT restricted to financial documents.

    Supported document examples:

        - Fee receipts
        - Certificates
        - Invoices
        - Bank statements
        - Research papers
        - Academic documents
        - Applications
        - Letters
        - Reports
        - Legal documents
        - Forms
        - Screenshots
        - Scanned documents
        - Other arbitrary documents

    Input:
        document_data from PyMuPDF or PaddleOCR

    Output:
        Structured document understanding.
    """

    # --------------------------------------------------------
    # Build page-aware document text
    # --------------------------------------------------------

    document_text_parts = []

    pages = document_data.get(
        "pages",
        []
    )

    for page in pages:

        page_number = page.get(
            "page_number"
        )

        page_text = page.get(
            "text",
            ""
        ).strip()

        if page_text:

            document_text_parts.append(
                f"""
--- PAGE {page_number} ---

{page_text}
"""
            )

    document_text = "\n".join(
        document_text_parts
    )

    # --------------------------------------------------------
    # No text
    # --------------------------------------------------------

    if not document_text.strip():

        return {
            "error": (
                "No readable text was found "
                "in the document."
            )
        }

    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are the Document Understanding engine of a
general-purpose Document AI system.

Your job is to deeply understand the document provided below.

The document may be ANY type.

Examples include:

- academic document
- fee receipt
- certificate
- invoice
- bank statement
- research paper
- application
- letter
- report
- legal document
- government document
- form
- financial document
- screenshot
- scanned document
- any other document

Do NOT assume the document is financial.

Do NOT use a fixed financial schema.

Understand the actual document.

============================================================
IMPORTANT RULES
============================================================

1. Do not invent information.

2. Extract only information supported by the document.

3. Preserve important wording from the document.

4. Use the page number where information appears.

5. If information is unavailable, use an empty string,
   empty array, or null as appropriate.

6. Do not hallucinate names, dates, amounts, entities,
   sections or table values.

7. Distinguish between printed information and your own
   interpretation.

8. Identify meaningful entities.

9. Identify dates.

10. Identify monetary amounts.

11. Identify important numbers when they have meaning.

12. Identify document sections and headings.

13. Identify key-value information.

14. Identify tables and preserve their structure.

15. Identify important facts and statements.

16. Create a concise overall summary.

17. Create page-level summaries.

18. Create source references that connect important
    information to its page.

19. If a table exists, preserve its headers and rows
    as accurately as possible.

20. Do not create tables when the document does not
    contain a table.

21. Do not treat every random number as an important number.

22. Do not treat ordinary words as entities unless they
    represent meaningful people, organizations, places,
    documents, subjects, identifiers, etc.

23. Return ONLY valid JSON.

24. Do not wrap the JSON in Markdown.

============================================================
OUTPUT FORMAT
============================================================

Return exactly this structure:

{{
    "document_type": "",
    "title": "",
    "language": "",

    "summary": "",

    "page_summaries": [],

    "entities": [],

    "key_information": [],

    "dates": [],

    "amounts": [],

    "important_numbers": [],

    "sections": [],

    "tables": [],

    "important_points": [],

    "source_references": []
}}

============================================================
FIELD FORMATS
============================================================

For "page_summaries":

{{
    "page": 1,
    "summary": ""
}}

For "entities":

{{
    "type": "",
    "value": "",
    "context": "",
    "page": 1
}}

Examples of entity types:

Person
Organization
Institution
Location
Course
Document
Certificate
Identifier
Account
Product
Subject
Other

Only use types that actually apply to the document.

For "key_information":

{{
    "label": "",
    "value": "",
    "page": 1
}}

Examples:

Student Name
Roll Number
Course
Institution
Scholarship
Address
Invoice Number
Account Number

Do not restrict yourself to these examples.

For "dates":

{{
    "value": "",
    "context": "",
    "page": 1
}}

For "amounts":

{{
    "value": "",
    "currency": "",
    "context": "",
    "page": 1
}}

For "important_numbers":

{{
    "value": "",
    "context": "",
    "page": 1
}}

Only include numbers that have meaningful context.

For "sections":

{{
    "heading": "",
    "description": "",
    "page": 1
}}

For "tables":

{{
    "title": "",
    "page": 1,
    "headers": [],
    "rows": []
}}

Each row must be an array containing values
corresponding to the headers.

For example:

{{
    "title": "Fee Details",
    "page": 1,
    "headers": [
        "Semester",
        "Fee"
    ],
    "rows": [
        ["1st Semester", "10000"],
        ["2nd Semester", "10200"]
    ]
}}

For "important_points":

{{
    "point": "",
    "page": 1
}}

For "source_references":

{{
    "page": 1,
    "text": "",
    "reason": ""
}}

The source reference should contain the important
document text supporting the extracted information.

============================================================
DOCUMENT
============================================================

{document_text}
"""

    # ========================================================
    # CALL GEMINI
    # ========================================================

    try:

        response = client.interactions.create(
            model=MODEL_NAME,
            input=prompt
        )

        output_text = (
            response.output_text.strip()
        )

        output_text = clean_json_response(
            output_text
        )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = json.loads(
            output_text
        )

        # ----------------------------------------------------
        # Ensure expected top-level fields exist
        # ----------------------------------------------------

        expected_fields = [
            "document_type",
            "title",
            "language",
            "summary",
            "page_summaries",
            "entities",
            "key_information",
            "dates",
            "amounts",
            "important_numbers",
            "sections",
            "tables",
            "important_points",
            "source_references"
        ]

        for field in expected_fields:

            if field not in result:

                if field in [
                    "page_summaries",
                    "entities",
                    "key_information",
                    "dates",
                    "amounts",
                    "important_numbers",
                    "sections",
                    "tables",
                    "important_points",
                    "source_references"
                ]:

                    result[field] = []

                else:

                    result[field] = ""

        return result

    # ========================================================
    # JSON ERROR
    # ========================================================

    except json.JSONDecodeError:

        return {
            "error": (
                "Gemini returned an invalid JSON response."
            ),
            "raw_response": output_text
        }

    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        return {
            "error": (
                "Document understanding failed: "
                f"{str(e)}"
            )
        }