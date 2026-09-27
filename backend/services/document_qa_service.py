import json
import os

from dotenv import load_dotenv
from google import genai

from services.rag_service import (
    build_rag_index,
    load_rag_index,
    retrieve_relevant_chunks,
    retrieve_annotations
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY was not found."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# ASK DOCUMENT QUESTION
# ============================================================

def ask_document_question(
    document_store,
    question
):
    """
    Full RAG-based document Q&A.

    Pipeline:

    Question
        ↓
    Local embedding
        ↓
    FAISS retrieval
        ↓
    Relevant chunks
        ↓
    Annotation retrieval
        ↓
    Gemini 3.8 Flash
        ↓
    Grounded answer
    """

    if not question or not question.strip():

        return {
            "answer": "Please enter a question.",
            "answer_type": "error",
            "sources": [],
            "annotations": []
        }

    question = question.strip()

    try:

        # ====================================================
        # STEP 1 — LOAD OR BUILD RAG INDEX
        # ====================================================

        document_id = document_store.get(
            "document_id"
        )

        rag_index = None

        if document_id:

            rag_index = load_rag_index(
                document_id
            )

        if rag_index is None:

            rag_index = build_rag_index(
                document_store,
                save_index=True
            )

        # ====================================================
        # STEP 2 — SEMANTIC RETRIEVAL
        # ====================================================

        retrieved_chunks = retrieve_relevant_chunks(
            rag_index,
            question,
            top_k=6
        )

        # ====================================================
        # STEP 3 — ANNOTATION RETRIEVAL
        # ====================================================

        annotations = retrieve_annotations(
            document_store,
            question
        )

        # ====================================================
        # STEP 4 — BUILD RAG CONTEXT
        # ====================================================

        context_parts = []

        for chunk in retrieved_chunks:

            context_parts.append(
                f"""
[PAGE {chunk.get("page")}]
[SOURCE TYPE: {chunk.get("source_type")}]

{chunk.get("text")}
"""
            )

        retrieved_context = "\n".join(
            context_parts
        )

        # ----------------------------------------------------
        # Add explicit annotation context
        # ----------------------------------------------------

        annotation_context = ""

        if annotations:

            annotation_lines = []

            for annotation in annotations:

                annotation_lines.append(
                    f"""
Page {annotation["page"]}
Type: {annotation["type"]}
Text: {annotation["text"]}
"""
                )

            annotation_context = (
                "\n================ ANNOTATIONS "
                "================\n"
                + "\n".join(
                    annotation_lines
                )
            )

        # ====================================================
        # STEP 5 — GEMINI
        # ====================================================

        prompt = f"""
You are the final answer engine of a professional
Document AI system.

Answer the user's question using ONLY the retrieved
document information below.

The information was retrieved using semantic search
from the uploaded document.

============================================================
STRICT RULES
============================================================

1. Do not invent information.

2. Do not use outside knowledge.

3. If the answer is not supported by the retrieved
   document context, say:

   "This information is not available in the document."

4. Use page numbers whenever possible.

5. If the user asks about highlighted text, use the
   annotation information.

6. If the user asks about underlined text, use the
   annotation information.

7. If the user asks about marked text, use the
   annotation information.

8. Follow the requested output format.

9. If the user asks for a table, produce a table.

10. If the user asks for a list, produce a list.

11. If the user asks for a summary, provide a concise
    summary.

12. Answer the actual question directly.

13. Do not mention the RAG system, FAISS, embeddings,
    retrieval or Gemini to the user.

14. Include page references in sources.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED DOCUMENT CONTEXT
============================================================

{retrieved_context}

{annotation_context}

============================================================
OUTPUT
============================================================

Return ONLY valid JSON:

{{
    "answer": "",
    "answer_type": "",
    "sources": [],
    "annotations": []
}}

answer_type can be:

direct_answer
summary
list
table
annotation_result
not_found

sources format:

[
    {{
        "page": 1,
        "text": "supporting text"
    }}
]

annotations format:

[
    {{
        "page": 1,
        "type": "Highlight",
        "text": "highlighted text"
    }}
]
"""

        response = client.interactions.create(
            model=MODEL_NAME,
            input=prompt
        )

        output_text = (
            response.output_text.strip()
        )

        # ====================================================
        # CLEAN JSON
        # ====================================================

        if output_text.startswith(
            "```json"
        ):

            output_text = output_text[7:]

        elif output_text.startswith(
            "```"
        ):

            output_text = output_text[3:]

        if output_text.endswith(
            "```"
        ):

            output_text = output_text[:-3]

        output_text = output_text.strip()

        # ====================================================
        # PARSE
        # ====================================================

        try:

            result = json.loads(
                output_text
            )

        except json.JSONDecodeError:

            # Gemini returned a normal answer
            # instead of JSON.
            return {
                "answer": output_text,
                "answer_type": "direct_answer",
                "sources": [
                    {
                        "page": chunk.get(
                            "page"
                        ),
                        "text": chunk.get(
                            "text"
                        )
                    }
                    for chunk in retrieved_chunks
                ],
                "annotations": annotations
            }

        # ====================================================
        # SAFETY DEFAULTS
        # ====================================================

        result.setdefault(
            "answer",
            ""
        )

        result.setdefault(
            "answer_type",
            "direct_answer"
        )

        result.setdefault(
            "sources",
            []
        )

        result.setdefault(
            "annotations",
            []
        )

        return result

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        return {
            "answer": (
                "Unable to answer the question."
            ),

            "answer_type": "error",

            "sources": [],

            "annotations": [],

            "error": str(e)
        }