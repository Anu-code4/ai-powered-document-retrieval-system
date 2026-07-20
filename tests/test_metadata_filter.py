

from retrievers.metadata_filter import extract_metadata_filter


# ==========================================================
# No Filters
# ==========================================================

def test_no_filters():

    filters = extract_metadata_filter(
        "Explain FAISS."
    )

    assert filters == {}


# ==========================================================
# PDF Filter
# ==========================================================

def test_pdf_filter():

    filters = extract_metadata_filter(
        "Search only PDF documents."
    )

    assert filters["file_type"] == ".pdf"


# ==========================================================
# DOCX Filter
# ==========================================================

def test_docx_filter():

    filters = extract_metadata_filter(
        "Search only DOCX files."
    )

    assert filters["file_type"] == ".docx"


# ==========================================================
# PDF Filename
# ==========================================================

def test_pdf_filename():

    filters = extract_metadata_filter(
        "Search report.pdf"
    )

    assert filters["source"] == "report.pdf"


# ==========================================================
# DOCX Filename
# ==========================================================

def test_docx_filename():

    filters = extract_metadata_filter(
        "Use notes.docx"
    )

    assert filters["source"] == "notes.docx"


# ==========================================================
# File Type + Filename
# ==========================================================

def test_filetype_and_filename():

    filters = extract_metadata_filter(
        "Search report.pdf in PDF files"
    )

    assert filters["file_type"] == ".pdf"
    assert filters["source"] == "report.pdf"


# ==========================================================
# Case Insensitive
# ==========================================================

def test_case_insensitive():

    filters = extract_metadata_filter(
        "Open REPORT.PDF"
    )

    assert filters["source"].lower() == "report.pdf"
    assert filters["file_type"] == ".pdf"