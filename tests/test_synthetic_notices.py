"""tests/test_synthetic_notices.py - Comprehensive integrity and validation tests for synthetic notices.

Validates:
1. labels.json loads and strictly parses against `LabelsFile` and `NoticeLabel` in src/schemas.py.
2. Exactly 30 synthetic notices exist across 6 categories (5 per category).
3. All notices have `is_synthetic is True`.
4. All RTI notices have `is_citizen_request is True` and `doc_type == DocumentType.rti`.
5. At least 4 degraded notices have `is_degraded is True`.
6. At least 5 out-of-scope notices have `is_out_of_scope is True` and `doc_type == DocumentType.other`.
7. All notice files exist on disk with non-zero byte size.
8. PIL.Image can open and inspect all PNG notices without error.
9. PyMuPDF can open and inspect all PDF notices without error.
10. NoticeLabel conversion to NoticeExtraction works seamlessly for all entries.
"""

from pathlib import Path
from typing import List

import pymupdf
import pytest
from PIL import Image

from src.schemas import DocumentType, LabelsFile, NoticeExtraction, NoticeLabel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "test_notices"
LABELS_JSON_PATH = DATA_DIR / "labels.json"


@pytest.fixture(scope="module")
def labels_file() -> LabelsFile:
    """Fixture providing parsed and validated LabelsFile instance."""
    assert LABELS_JSON_PATH.exists(), f"labels.json not found at {LABELS_JSON_PATH}"
    with open(LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    assert len(raw_text) > 0, "labels.json is empty"
    return LabelsFile.model_validate_json(raw_text)


class TestLabelsFileIntegrity:
    """Tests for labels.json structure, count, and schema compliance."""

    def test_labels_file_exists_and_parses(self, labels_file: LabelsFile):
        """Verify labels.json parses without schema errors and contains notices."""
        assert labels_file.version == "1.0"
        assert len(labels_file) == 30, f"Expected exactly 30 notices, found {len(labels_file)}"

    def test_category_distribution(self, labels_file: LabelsFile):
        """Verify exactly 5 notices per category across all 6 supported types."""
        expected_categories = [
            DocumentType.electricity,
            DocumentType.challan,
            DocumentType.tax,
            DocumentType.scheme_rejection,
            DocumentType.rti,
            DocumentType.other,
        ]
        counts = {cat: 0 for cat in expected_categories}
        for notice in labels_file:
            counts[notice.doc_type] += 1

        for cat in expected_categories:
            assert (
                counts[cat] == 5
            ), f"Category '{cat.value}' expected 5 notices, found {counts[cat]}"

    def test_all_notices_marked_synthetic(self, labels_file: LabelsFile):
        """Verify every notice label has is_synthetic set to True."""
        for notice in labels_file:
            assert (
                notice.is_synthetic is True
            ), f"Notice {notice.filename} has is_synthetic={notice.is_synthetic}, expected True"

    def test_rti_citizen_request_flag(self, labels_file: LabelsFile):
        """Verify every RTI notice is citizen-written with is_citizen_request set to True."""
        rti_notices = [n for n in labels_file if n.doc_type == DocumentType.rti]
        assert len(rti_notices) == 5, f"Expected 5 RTI notices, found {len(rti_notices)}"

        for rti in rti_notices:
            assert (
                rti.is_citizen_request is True
            ), f"RTI notice {rti.filename} must have is_citizen_request=True"
            assert (
                len(rti.key_facts) >= 2
            ), f"RTI notice {rti.filename} must have at least 2 key facts/questions"
            # Verify citizen queries / questions are captured in key facts
            has_inquiry = any(
                any(keyword in fact.lower() for keyword in ["citizen", "request", "sought", "inquir", "question"])
                for fact in rti.key_facts
            )
            assert (
                has_inquiry
            ), f"RTI notice {rti.filename} key facts should describe citizen inquiries: {rti.key_facts}"

    def test_out_of_scope_flags(self, labels_file: LabelsFile):
        """Verify out-of-scope notices are marked with doc_type='other' and is_out_of_scope=True."""
        oos_notices = [n for n in labels_file if n.is_out_of_scope]
        assert len(oos_notices) >= 5, f"Expected at least 5 out-of-scope notices, found {len(oos_notices)}"

        for oos in oos_notices:
            assert oos.doc_type == DocumentType.other, f"Notice {oos.filename} should have doc_type='other'"

    def test_degraded_scan_count(self, labels_file: LabelsFile):
        """Verify at least 4 notices are flagged as degraded scans."""
        degraded = [n for n in labels_file if n.is_degraded]
        assert (
            len(degraded) >= 4
        ), f"Expected at least 4 degraded notices, found {len(degraded)}"

    def test_dates_and_amounts_integrity(self, labels_file: LabelsFile):
        """Verify date validity and non-negative amount constraints."""
        for notice in labels_file:
            if notice.amount_inr is not None:
                assert (
                    notice.amount_inr >= 0.0
                ), f"Notice {notice.filename} amount_inr must be non-negative: {notice.amount_inr}"

            if notice.issue_date is not None:
                # Validated by Pydantic regex and calendar validator
                assert len(notice.issue_date) == 10
                assert notice.issue_date.count("-") == 2

            if notice.deadline_date is not None:
                assert len(notice.deadline_date) == 10
                assert notice.deadline_date.count("-") == 2

    def test_to_extraction_conversion(self, labels_file: LabelsFile):
        """Verify each NoticeLabel converts cleanly to a NoticeExtraction."""
        for notice in labels_file:
            extraction = notice.to_extraction(confidence=0.95, is_mock=True)
            assert isinstance(extraction, NoticeExtraction)
            assert extraction.doc_type == notice.doc_type
            assert extraction.amount_inr == notice.amount_inr
            assert extraction.issue_date == notice.issue_date
            assert extraction.deadline_date == notice.deadline_date
            assert extraction.confidence == 0.95
            assert extraction.is_mock is True


class TestSyntheticFilesOnDisk:
    """Tests for physical notice files in data/test_notices/."""

    def test_all_thirty_files_exist_with_non_zero_size(self, labels_file: LabelsFile):
        """Verify all 30 files referenced in labels.json physically exist on disk and have non-zero size."""
        for notice in labels_file:
            file_path = DATA_DIR / notice.filename
            assert file_path.exists(), f"File {notice.filename} does not exist at {file_path}"
            file_size = file_path.stat().st_size
            assert file_size > 0, f"File {notice.filename} is empty (0 bytes)"

    def test_file_format_distribution(self, labels_file: LabelsFile):
        """Verify there is a balanced mix of PNG and PDF files."""
        png_files = [n for n in labels_file if n.filename.endswith(".png")]
        pdf_files = [n for n in labels_file if n.filename.endswith(".pdf")]

        assert len(png_files) >= 10, f"Expected at least 10 PNG files, found {len(png_files)}"
        assert len(pdf_files) >= 5, f"Expected at least 5 PDF files, found {len(pdf_files)}"
        assert len(png_files) + len(pdf_files) == 30

    def test_pil_can_open_all_png_files(self, labels_file: LabelsFile):
        """Verify PIL can open every PNG notice without corruption or error."""
        png_notices = [n for n in labels_file if n.filename.endswith(".png")]
        for notice in png_notices:
            file_path = DATA_DIR / notice.filename
            with Image.open(file_path) as img:
                assert img.format == "PNG", f"{notice.filename} is not PNG format"
                assert img.width > 200, f"{notice.filename} width too small: {img.width}"
                assert img.height > 200, f"{notice.filename} height too small: {img.height}"

    def test_pymupdf_can_open_all_pdf_files(self, labels_file: LabelsFile):
        """Verify PyMuPDF can open every PDF notice without error and has valid pages."""
        pdf_notices = [n for n in labels_file if n.filename.endswith(".pdf")]
        for notice in pdf_notices:
            file_path = DATA_DIR / notice.filename
            doc = pymupdf.open(str(file_path))
            try:
                assert len(doc) >= 1, f"PDF {notice.filename} has no pages"
                page = doc[0]
                assert page.rect.width > 0
                assert page.rect.height > 0
            finally:
                doc.close()
