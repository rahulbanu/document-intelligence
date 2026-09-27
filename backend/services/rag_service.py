import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# INDEX DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_indexes"
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(document_store):
    """
    Convert the document into semantic-ish chunks.

    Each chunk keeps:
    - text
    - page number
    - source type
    - annotation information when applicable
    """

    chunks = []

    pages = document_store.get(
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

        # ----------------------------------------------------
        # Split page into paragraphs
        # ----------------------------------------------------

        paragraphs = [
            p.strip()
            for p in page_text.split("\n")
            if p.strip()
        ]

        # ----------------------------------------------------
        # Create text chunks
        # ----------------------------------------------------

        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:

            words = paragraph.split()

            # Keep chunks reasonably small
            if (
                current_length + len(words)
                > 180
                and current_chunk
            ):

                chunks.append({
                    "chunk_id": len(chunks),
                    "page": page_number,
                    "text": " ".join(
                        current_chunk
                    ),
                    "source_type": "document"
                })

                current_chunk = []
                current_length = 0

            current_chunk.extend(words)
            current_length += len(words)

        # Remaining text
        if current_chunk:

            chunks.append({
                "chunk_id": len(chunks),
                "page": page_number,
                "text": " ".join(
                    current_chunk
                ),
                "source_type": "document"
            })

        # ----------------------------------------------------
        # Store annotations as searchable chunks
        # ----------------------------------------------------

        annotations = page.get(
            "annotations",
            []
        )

        for annotation in annotations:

            annotation_type = annotation.get(
                "type",
                ""
            )

            annotation_text = annotation.get(
                "text",
                ""
            ).strip()

            if not annotation_text:
                continue

            annotation_chunk = (
                f"{annotation_type}: "
                f"{annotation_text}"
            )

            chunks.append({
                "chunk_id": len(chunks),
                "page": page_number,
                "text": annotation_chunk,
                "source_type": "annotation",
                "annotation_type": annotation_type,
                "annotation_text": annotation_text
            })

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def build_rag_index(
    document_store,
    save_index=True
):
    """
    Create embeddings and a FAISS vector index
    for one document.
    """

    chunks = create_chunks(
        document_store
    )

    if not chunks:

        raise ValueError(
            "No document content available "
            "for RAG indexing."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Create local embeddings
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # --------------------------------------------------------
    # FAISS cosine similarity
    #
    # With normalized vectors:
    # Inner Product = Cosine Similarity
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    document_id = document_store.get(
        "document_id",
        "temporary_document"
    )

    if save_index:

        document_directory = (
            INDEX_DIR
            / document_id
        )

        document_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        index_path = (
            document_directory
            / "index.faiss"
        )

        chunks_path = (
            document_directory
            / "chunks.json"
        )

        faiss.write_index(
            index,
            str(index_path)
        )

        with open(
            chunks_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

    return {
        "index": index,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "dimension": dimension
    }


# ============================================================
# LOAD EXISTING INDEX
# ============================================================

def load_rag_index(document_id):

    document_directory = (
        INDEX_DIR
        / document_id
    )

    index_path = (
        document_directory
        / "index.faiss"
    )

    chunks_path = (
        document_directory
        / "chunks.json"
    )

    if not index_path.exists():
        return None

    if not chunks_path.exists():
        return None

    index = faiss.read_index(
        str(index_path)
    )

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return {
        "index": index,
        "chunks": chunks,
        "total_chunks": len(chunks)
    }


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    rag_index,
    question,
    top_k=6
):
    """
    Retrieve the most relevant document chunks
    using semantic similarity.
    """

    index = rag_index["index"]

    chunks = rag_index["chunks"]

    # --------------------------------------------------------
    # Embed question locally
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # --------------------------------------------------------
    # FAISS similarity search
    # --------------------------------------------------------

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        chunk = chunks[
            int(index_position)
        ].copy()

        chunk["similarity"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


# ============================================================
# ANNOTATION RETRIEVAL
# ============================================================

def retrieve_annotations(
    document_store,
    question
):
    """
    Retrieve annotations when the question is
    explicitly related to highlighting, underlining,
    marking, etc.
    """

    question_lower = question.lower()

    annotation_keywords = [
        "highlight",
        "highlighted",
        "underline",
        "underlined",
        "marked",
        "mark",
        "pencil",
        "pen",
        "circle",
        "circled",
        "box",
        "boxed",
        "strike",
        "strikethrough"
    ]

    asks_about_annotations = any(
        keyword in question_lower
        for keyword in annotation_keywords
    )

    if not asks_about_annotations:
        return []

    results = []

    for page in document_store.get(
        "pages",
        []
    ):

        page_number = page.get(
            "page_number"
        )

        for annotation in page.get(
            "annotations",
            []
        ):

            annotation_type = annotation.get(
                "type",
                ""
            )

            annotation_text = annotation.get(
                "text",
                ""
            )

            # ------------------------------------------------
            # If a specific annotation type is requested
            # ------------------------------------------------

            type_matches = True

            if (
                "highlight" in question_lower
                or "highlighted" in question_lower
            ):
                type_matches = (
                    annotation_type.lower()
                    == "highlight"
                )

            elif (
                "underline" in question_lower
                or "underlined" in question_lower
            ):
                type_matches = (
                    annotation_type.lower()
                    == "underline"
                )

            if type_matches:

                results.append({
                    "page": page_number,
                    "type": annotation_type,
                    "text": annotation_text,
                    "source": annotation.get(
                        "source",
                        ""
                    )
                })

    return results