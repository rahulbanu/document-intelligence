import json
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="DOCUMENT INTELLIGENCE",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .understanding-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }

    .marking-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">DOCUMENT INTELLIGENCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Extract • Understand • Analyze Documents'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "document_result" not in st.session_state:
    st.session_state.document_result = None

if "marking_result" not in st.session_state:
    st.session_state.marking_result = None

if "marking_requested" not in st.session_state:
    st.session_state.marking_requested = False


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=[
        "pdf",
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    if st.button(
        "🚀 ANALYZE DOCUMENT",
        use_container_width=True
    ):

        # Reset previous results
        st.session_state.document_result = None
        st.session_state.marking_result = None
        st.session_state.marking_requested = False

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner(
            "Extracting and understanding document..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=600
                )

                if response.status_code != 200:

                    st.error(
                        "Document processing failed:\n"
                        + response.text
                    )

                else:

                    st.session_state.document_result = (
                        response.json()
                    )

                    st.success(
                        "Document analyzed successfully."
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI backend. "
                    "Make sure FastAPI is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The document took too long to process."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )


# =========================================================
# DISPLAY DOCUMENT RESULT
# =========================================================

result = st.session_state.document_result


if result is not None:

    document = result.get(
        "document",
        {}
    )

    understanding = document.get(
        "understanding",
        {}
    )

    pages = document.get(
        "pages",
        []
    )

    # =====================================================
    # OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        'Document Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Document",
            result.get(
                "filename",
                "-"
            )
        )

    with col2:
        st.metric(
            "Pages",
            result.get(
                "total_pages",
                0
            )
        )

    with col3:
        st.metric(
            "Extraction",
            result.get(
                "extraction_method",
                "-"
            )
        )

    with col4:
        st.metric(
            "Entities",
            len(
                understanding.get(
                    "entities",
                    []
                )
            )
        )

    # =====================================================
    # SECTION 1 — EXTRACTED CONTENT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '1. Extracted Content'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Raw text extracted directly from the document. "
        "The original extracted text is not rewritten by the AI."
    )

    for page in pages:

        page_number = page.get(
            "page_number",
            "?"
        )

        page_text = page.get(
            "text",
            ""
        )

        with st.expander(
            f"Page {page_number}",
            expanded=False
        ):

            if page_text.strip():

                st.text_area(
                    "Extracted text",
                    value=page_text,
                    height=250,
                    key=f"raw_page_{page_number}"
                )

            else:

                st.info(
                    "No text was extracted from this page."
                )

    # =====================================================
    # SECTION 2 — DOCUMENT UNDERSTANDING
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '2. Document Understanding'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write("**Document Type**")

        st.write(
            understanding.get(
                "document_type",
                "Not identified"
            )
        )

    with col2:

        st.write("**Title**")

        st.write(
            understanding.get(
                "title",
                "Not identified"
            )
        )

    with col3:

        st.write("**Language**")

        st.write(
            understanding.get(
                "language",
                "Not identified"
            )
        )

    # -----------------------------------------------------
    # SHORT SUMMARY
    # -----------------------------------------------------

    st.write("### Summary")

    summary = understanding.get(
        "summary",
        ""
    )

    st.markdown(
        f"""
        <div class="understanding-box">
        {summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # PAGE SUMMARIES
    # -----------------------------------------------------

    page_summaries = understanding.get(
        "page_summaries",
        []
    )

    if page_summaries:

        st.write("### Page Summaries")

        for item in page_summaries:

            if isinstance(item, dict):

                page_number = item.get(
                    "page",
                    item.get(
                        "page_number",
                        "?"
                    )
                )

                page_summary = item.get(
                    "summary",
                    ""
                )

                st.write(
                    f"**Page {page_number}:** "
                    f"{page_summary}"
                )

            else:

                st.write(str(item))

    # -----------------------------------------------------
    # KEY INFORMATION
    # -----------------------------------------------------

    key_information = understanding.get(
        "key_information",
        []
    )

    if key_information:

        st.write("### Key Information")

        for item in key_information:

            if isinstance(item, dict):

                label = item.get(
                    "label",
                    ""
                )

                value = item.get(
                    "value",
                    ""
                )

                st.write(
                    f"**{label}:** {value}"
                )

            else:

                st.write(
                    f"• {item}"
                )

    # -----------------------------------------------------
    # ENTITIES
    # -----------------------------------------------------

    entities = understanding.get(
        "entities",
        []
    )

    if entities:

        st.write("### Entities")

        for entity in entities:

            if isinstance(entity, dict):

                entity_type = entity.get(
                    "type",
                    "Entity"
                )

                entity_value = entity.get(
                    "value",
                    entity.get(
                        "name",
                        ""
                    )
                )

                st.write(
                    f"**{entity_type}:** "
                    f"{entity_value}"
                )

            else:

                st.write(
                    f"• {entity}"
                )

    # -----------------------------------------------------
    # DATES
    # -----------------------------------------------------

    dates = understanding.get(
        "dates",
        []
    )

    if dates:

        st.write("### Dates")

        for date in dates:

            if isinstance(date, dict):

                st.write(
                    f"**{date.get('date', '')}**"
                )

            else:

                st.write(
                    f"• {date}"
                )

    # -----------------------------------------------------
    # AMOUNTS
    # -----------------------------------------------------

    amounts = understanding.get(
        "amounts",
        []
    )

    if amounts:

        st.write("### Amounts")

        for amount in amounts:

            if isinstance(amount, dict):

                label = amount.get(
                    "label",
                    "Amount"
                )

                value = amount.get(
                    "value",
                    ""
                )

                st.write(
                    f"**{label}:** {value}"
                )

            else:

                st.write(
                    f"• {amount}"
                )

    # -----------------------------------------------------
    # IMPORTANT NUMBERS
    # -----------------------------------------------------

    important_numbers = understanding.get(
        "important_numbers",
        []
    )

    if important_numbers:

        st.write("### Important Numbers")

        for number in important_numbers:

            st.write(
                f"• {number}"
            )

    # -----------------------------------------------------
    # SECTIONS
    # -----------------------------------------------------

    sections = understanding.get(
        "sections",
        []
    )

    if sections:

        st.write("### Sections")

        for section in sections:

            if isinstance(section, dict):

                st.write(
                    f"**{section.get('title', '')}**"
                )

            else:

                st.write(
                    f"• {section}"
                )

    # -----------------------------------------------------
    # TABLES
    # -----------------------------------------------------

    tables = understanding.get(
        "tables",
        []
    )

    if tables:

        st.write("### Tables")

        for index, table in enumerate(
            tables,
            start=1
        ):

            st.write(
                f"**Table {index}**"
            )

            if isinstance(table, dict):

                st.json(table)

            else:

                st.write(table)

    # -----------------------------------------------------
    # IMPORTANT POINTS
    # -----------------------------------------------------

    important_points = understanding.get(
        "important_points",
        []
    )

    if important_points:

        st.write("### Important Points")

        for point in important_points:

            st.write(
                f"• {point}"
            )

    # =====================================================
    # SECTION 3 — MARKING QUESTION
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '3. Document Markings'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Do you want the system to find "
        "highlighted, underlined, handwritten, "
        "pen, pencil, strike-through, circle, "
        "or boxed text?"
    )

    col_yes, col_no = st.columns(2)

    # -----------------------------------------------------
    # YES
    # -----------------------------------------------------

    with col_yes:

        if st.button(
            "🔎 YES — FIND MARKINGS",
            use_container_width=True
        ):

            st.session_state.marking_requested = True

            document_id = result.get(
                "document_id"
            )

            with st.spinner(
                "Analyzing document markings..."
            ):

                try:

                    marking_response = requests.post(
                        f"{API_URL}/analyze-markings/"
                        f"{document_id}",
                        timeout=600
                    )

                    if (
                        marking_response.status_code
                        != 200
                    ):

                        st.error(
                            "Marking analysis failed:\n"
                            + marking_response.text
                        )

                    else:

                        st.session_state.marking_result = (
                            marking_response.json()
                        )

                        st.success(
                            "Marking analysis completed."
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Cannot connect to FastAPI backend."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "Marking analysis took too long."
                    )

                except Exception as e:

                    st.error(
                        f"Unexpected error: {str(e)}"
                    )

    # -----------------------------------------------------
    # NO
    # -----------------------------------------------------

    with col_no:

        if st.button(
            "✓ NO — FINISH",
            use_container_width=True
        ):

            st.session_state.marking_requested = False
            st.session_state.marking_result = None

            st.success(
                "Document analysis completed."
            )


# =========================================================
# MARKING RESULTS
# =========================================================

marking_result = (
    st.session_state.marking_result
)


if marking_result is not None:

    st.markdown(
        "### Detected Markings"
    )

    markings = marking_result.get(
        "annotations",
        {}
    )

    pages_with_markings = markings.get(
        "pages",
        []
    )

    found_any = False

    for page in pages_with_markings:

        page_number = page.get(
            "page_number",
            "?"
        )

        annotations = page.get(
            "annotations",
            []
        )

        if not annotations:
            continue

        found_any = True

        st.write(
            f"#### Page {page_number}"
        )

        for annotation in annotations:

            annotation_type = annotation.get(
                "type",
                "Unknown"
            )

            annotation_text = annotation.get(
                "text",
                ""
            )

            source = annotation.get(
                "source",
                ""
            )

            st.markdown(
                f"""
                <div class="marking-box">
                <b>Type:</b> {annotation_type}<br>
                <b>Text:</b> {annotation_text or "Text not confidently identified"}<br>
                <b>Source:</b> {source}
                </div>
                """,
                unsafe_allow_html=True
            )

    if not found_any:

        st.info(
            "No highlighted, underlined, "
            "handwritten, or other supported "
            "markings were detected."
        )


# =========================================================
# DOWNLOAD SECTION
# =========================================================

if result is not None:

    st.markdown(
        '<div class="section-title">'
        'Download Results'
        '</div>',
        unsafe_allow_html=True
    )

    document = result.get(
        "document",
        {}
    )

    understanding = document.get(
        "understanding",
        {}
    )

    pages = document.get(
        "pages",
        []
    )

    # -----------------------------------------------------
    # COMPLETE JSON
    # -----------------------------------------------------

    complete_json = json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    )

    st.download_button(
        "⬇ Download Complete Analysis JSON",
        data=complete_json,
        file_name="document_analysis.json",
        mime="application/json",
        use_container_width=True
    )

    # -----------------------------------------------------
    # RAW TEXT
    # -----------------------------------------------------

    raw_text_parts = []

    for page in pages:

        raw_text_parts.append(
            f"===== PAGE "
            f"{page.get('page_number', '?')} =====\n"
        )

        raw_text_parts.append(
            page.get(
                "text",
                ""
            )
        )

        raw_text_parts.append("\n\n")

    raw_text = "".join(
        raw_text_parts
    )

    st.download_button(
        "⬇ Download Raw Extracted Text",
        data=raw_text,
        file_name="raw_extracted_text.txt",
        mime="text/plain",
        use_container_width=True
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary_text = (
        f"DOCUMENT TYPE: "
        f"{understanding.get('document_type', '')}\n\n"
        f"TITLE: "
        f"{understanding.get('title', '')}\n\n"
        f"LANGUAGE: "
        f"{understanding.get('language', '')}\n\n"
        f"SUMMARY:\n"
        f"{understanding.get('summary', '')}\n\n"
    )

    st.download_button(
        "⬇ Download Summary",
        data=summary_text,
        file_name="document_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

    # -----------------------------------------------------
    # MARKINGS JSON
    # -----------------------------------------------------

    if marking_result is not None:

        markings_json = json.dumps(
            marking_result,
            indent=4,
            ensure_ascii=False
        )

        st.download_button(
            "⬇ Download Markings",
            data=markings_json,
            file_name="document_markings.json",
            mime="application/json",
            use_container_width=True
        )