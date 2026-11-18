"""tests/test_extract.py - Comprehensive unit and integration tests for F2 extraction pipeline.

Covers:
1. Mock fallback on synthetic notices across all 6 categories (electricity, challan, tax,
   scheme_rejection, rti, other) verifying all 10 F2 fields, correct doc_type, and is_mock=True.
2. Mock fallback on uncataloged / unknown files (returns valid ExtractionResult with is_mock=True).
3. 10MB file size limit rejection with graceful error response.
4. Live multimodal extraction simulation with single retry recovery on malformed JSON or validation error.
5. Live multimodal extraction simulation with manual-entry fallback on double failure (confidence=0.0).
6. Input formats: string path, Path object, raw bytes.
7. PDF and PNG intake verification.
8. Invariant: mock extractions always have is_mock=True on both result and extraction.
9. MIME type inference and helper functions.
10. All 30 synthetic notices verification in offline mock mode.
"""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.extract import (
    DATA_DIR,
    DEFAULT_MODEL,
    MAX_FILE_SIZE_BYTES,
    _clean_json_text,
    clear_labels_cache,
    extract_from_bytes,
    extract_from_file,
    extract_notice,
    get_cached_labels,
    infer_mime_type,
    render_pdf_first_page_as_png,
)
from src.schemas import DocumentType, ExtractionResult, NoticeExtraction


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure labels cache is fresh for each test run."""
    clear_labels_cache()
    yield
    clear_labels_cache()


class TestMockExtractionFallback:
    """Test suite for offline mock fallback using labels.json and synthetic notices."""

    @pytest.mark.parametrize(
        "filename,expected_doc_type",
        [
            ("notice_01_elec_disconnection.png", DocumentType.electricity),
            ("notice_02_elec_meter_tamper.pdf", DocumentType.electricity),
            ("notice_06_challan_speeding.png", DocumentType.challan),
            ("notice_07_challan_signal.pdf", DocumentType.challan),
            ("notice_11_tax_property_demand.png", DocumentType.tax),
            ("notice_14_tax_commercial_tax.pdf", DocumentType.tax),
            ("notice_16_scheme_ration_card.pdf", DocumentType.scheme_rejection),
            ("notice_17_scheme_pension.png", DocumentType.scheme_rejection),
            ("notice_21_rti_road_repair.png", DocumentType.rti),
            ("notice_22_rti_ration_delay.pdf", DocumentType.rti),
            ("notice_26_oos_grocery_receipt.png", DocumentType.other),
            ("notice_30_oos_lease_agreement.pdf", DocumentType.other),
        ],
    )
    def test_mock_fallback_on_synthetic_notices(
        self,
        filename: str,
        expected_doc_type: DocumentType,
    ):
        """Verify extract_notice correctly loads and extracts ground truth for synthetic notices."""
        file_path = DATA_DIR / filename
        assert file_path.exists(), f"Synthetic notice file not found: {file_path}"

        result = extract_notice(file_path, force_mock=True)

        # Result container assertions
        assert isinstance(result, ExtractionResult)
        assert result.is_mock is True
        assert result.error is None
        assert result.raw_response is not None

        # Extraction payload assertions
        extraction = result.extraction
        assert isinstance(extraction, NoticeExtraction)
        assert extraction.is_mock is True
        assert extraction.doc_type == expected_doc_type

        # Verify against ground truth from labels.json
        labels_file = get_cached_labels()
        assert labels_file is not None
        ground_truth = labels_file.get_by_filename(filename)
        assert ground_truth is not None

        assert extraction.issuer == ground_truth.issuer
        assert extraction.reference_number == ground_truth.reference_number
        assert extraction.amount_inr == ground_truth.amount_inr
        assert extraction.issue_date == ground_truth.issue_date
        assert extraction.deadline_date == ground_truth.deadline_date
        assert extraction.key_facts == ground_truth.key_facts
        assert extraction.language_of_document == ground_truth.language_of_document
        assert extraction.confidence == 1.0
        assert extraction.missing_or_unclear == []

        # Verify all 10 F2 fields exist and are well-formed
        assert isinstance(extraction.doc_type, DocumentType)
        assert extraction.issuer is None or isinstance(extraction.issuer, str)
        assert extraction.reference_number is None or isinstance(extraction.reference_number, str)
        assert extraction.amount_inr is None or (isinstance(extraction.amount_inr, float) and extraction.amount_inr >= 0.0)
        assert extraction.issue_date is None or (isinstance(extraction.issue_date, str) and len(extraction.issue_date) == 10)
        assert extraction.deadline_date is None or (isinstance(extraction.deadline_date, str) and len(extraction.deadline_date) == 10)
        assert isinstance(extraction.key_facts, list)
        assert isinstance(extraction.language_of_document, str)
        assert 0.0 <= extraction.confidence <= 1.0
        assert isinstance(extraction.missing_or_unclear, list)

    def test_mock_fallback_all_thirty_synthetic_notices(self):
        """Exhaustively verify that all 30 synthetic notices in data/test_notices/ extract cleanly."""
        labels_file = get_cached_labels()
        assert labels_file is not None
        assert len(labels_file) == 30

        for notice_label in labels_file:
            path = DATA_DIR / notice_label.filename
            assert path.exists()
            res = extract_notice(path, force_mock=True)
            assert res.is_mock is True
            assert res.extraction.is_mock is True
            assert res.extraction.doc_type == notice_label.doc_type
            assert res.error is None
            assert res.extraction.amount_inr == notice_label.amount_inr
            assert res.extraction.reference_number == notice_label.reference_number

    def test_mock_fallback_unknown_file(self):
        """Verify uncataloged files return a safe generic mock extraction with is_mock=True."""
        dummy_content = b"Content of uncataloged document for testing"
        result = extract_notice(
            file_input=dummy_content,
            filename="unknown_document.png",
            force_mock=True,
        )

        assert isinstance(result, ExtractionResult)
        assert result.is_mock is True
        assert result.extraction.is_mock is True
        assert result.error is None
        assert result.extraction.doc_type == DocumentType.other
        assert result.extraction.confidence == 0.5
        assert len(result.extraction.missing_or_unclear) > 0
        assert "not cataloged" in result.extraction.missing_or_unclear[0].lower()

    def test_mock_flag_invariance(self):
        """Verify is_mock is consistently True on both result and extraction across various calls."""
        # 1. From real synthetic notice with force_mock=True
        res1 = extract_notice("notice_01_elec_disconnection.png", force_mock=True)
        assert res1.is_mock is True
        assert res1.extraction.is_mock is True

        # 2. From raw bytes with synthetic notice filename
        bytes_data = (DATA_DIR / "notice_06_challan_speeding.png").read_bytes()
        res2 = extract_notice(bytes_data, filename="notice_06_challan_speeding.png", force_mock=True)
        assert res2.is_mock is True
        assert res2.extraction.is_mock is True

        # 3. From arbitrary uncataloged bytes
        res3 = extract_notice(b"arbitrary bytes", filename="random.pdf", force_mock=True)
        assert res3.is_mock is True
        assert res3.extraction.is_mock is True


class TestIntakeValidationAndLimits:
    """Test suite for file size enforcement, format handling, and input types."""

    def test_file_size_limit_rejection_bytes(self):
        """Verify files larger than 10MB are rejected gracefully with an error result."""
        oversized_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
        result = extract_notice(oversized_bytes, filename="oversized.pdf")

        assert isinstance(result, ExtractionResult)
        assert result.error is not None
        assert "10MB" in result.error
        assert result.is_mock is False
        assert result.extraction.confidence == 0.0
        assert result.extraction.doc_type == DocumentType.other

    def test_file_size_limit_boundary_acceptance(self):
        """Verify files equal to or below 10MB are accepted for processing."""
        # 100 bytes is well below 10MB
        small_bytes = b"%PDF-1.4 sample content"
        result = extract_notice(small_bytes, filename="small.pdf", force_mock=True)
        assert result.error is None

    def test_input_types(self):
        """Verify intake supports str path, Path object, and raw bytes."""
        notice_name = "notice_01_elec_disconnection.png"
        file_path = DATA_DIR / notice_name
        file_bytes = file_path.read_bytes()

        # 1. Path object
        res_path = extract_notice(file_path, force_mock=True)
        assert res_path.error is None
        assert res_path.extraction.doc_type == DocumentType.electricity

        # 2. String path
        res_str = extract_notice(str(file_path), force_mock=True)
        assert res_str.error is None
        assert res_str.extraction.doc_type == DocumentType.electricity

        # 3. Raw bytes with filename
        res_bytes = extract_notice(file_bytes, filename=notice_name, force_mock=True)
        assert res_bytes.error is None
        assert res_bytes.extraction.doc_type == DocumentType.electricity

    def test_pdf_and_png_intake(self):
        """Verify intake operates cleanly on both PNG and PDF notice files."""
        png_path = DATA_DIR / "notice_01_elec_disconnection.png"
        pdf_path = DATA_DIR / "notice_02_elec_meter_tamper.pdf"

        res_png = extract_notice(png_path, force_mock=True)
        assert res_png.error is None
        assert res_png.extraction.doc_type == DocumentType.electricity

        res_pdf = extract_notice(pdf_path, force_mock=True)
        assert res_pdf.error is None
        assert res_pdf.extraction.doc_type == DocumentType.electricity

    def test_nonexistent_file_path(self):
        """Verify passing a non-existent file path returns an error result."""
        result = extract_notice("non_existent_folder/imaginary_file.pdf")
        assert result.error is not None
        assert "File not found" in result.error
        assert result.extraction.confidence == 0.0

    def test_unsupported_input_type(self):
        """Verify passing an unsupported type returns a clear error."""
        result = extract_notice(file_input=12345)  # type: ignore
        assert result.error is not None
        assert "Unsupported file_input type" in result.error
        assert result.extraction.confidence == 0.0

    def test_infer_mime_type(self):
        """Verify MIME type inference from extensions and magic bytes."""
        # By filename
        assert infer_mime_type(filename="doc.png") == "image/png"
        assert infer_mime_type(filename="doc.PNG") == "image/png"
        assert infer_mime_type(filename="doc.jpg") == "image/jpeg"
        assert infer_mime_type(filename="doc.jpeg") == "image/jpeg"
        assert infer_mime_type(filename="doc.pdf") == "application/pdf"
        assert infer_mime_type(filename="doc.PDF") == "application/pdf"

        # By magic bytes
        assert infer_mime_type(file_bytes=b"%PDF-1.4 ...") == "application/pdf"
        assert infer_mime_type(file_bytes=b"\x89PNG\r\n\x1a\n...") == "image/png"
        assert infer_mime_type(file_bytes=b"\xff\xd8\xff\xe0...") == "image/jpeg"

        # Fallback default
        assert infer_mime_type(filename="unknown.xyz") == "application/octet-stream"

    def test_helpers_extract_from_file_and_bytes(self):
        """Verify extract_from_file and extract_from_bytes helper functions."""
        file_path = DATA_DIR / "notice_06_challan_speeding.png"

        # extract_from_file
        res1 = extract_from_file(file_path, force_mock=True)
        assert res1.error is None
        assert res1.extraction.doc_type == DocumentType.challan

        # extract_from_bytes
        file_bytes = file_path.read_bytes()
        res2 = extract_from_bytes(
            file_bytes=file_bytes,
            mime_type="image/png",
            filename="notice_06_challan_speeding.png",
            force_mock=True,
        )
        assert res2.error is None
        assert res2.extraction.doc_type == DocumentType.challan

    def test_render_pdf_first_page_as_png(self):
        """Verify PyMuPDF renders the first page of a PDF synthetic notice."""
        pdf_path = DATA_DIR / "notice_02_elec_meter_tamper.pdf"
        pdf_bytes = pdf_path.read_bytes()

        png_bytes = render_pdf_first_page_as_png(pdf_bytes)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_clean_json_text_utility(self):
        """Verify markdown fence stripper utility."""
        raw_markdown = "```json\n{\"doc_type\": \"tax\"}\n```"
        assert _clean_json_text(raw_markdown) == '{"doc_type": "tax"}'

        raw_fence_no_lang = "```\n{\"doc_type\": \"challan\"}\n```"
        assert _clean_json_text(raw_fence_no_lang) == '{"doc_type": "challan"}'

        raw_clean = '{"doc_type": "electricity"}'
        assert _clean_json_text(raw_clean) == '{"doc_type": "electricity"}'


class TestLiveMultimodalExtractionAndRetry:
    """Test suite for live multimodal extraction, single retry recovery, and manual-entry fallback."""

    @patch("src.extract.is_live_gemini_available", return_value=True)
    @patch("src.extract.get_gemini_client")
    @patch("src.extract.generate_content_with_retry")
    def test_live_extraction_success_first_attempt(
        self,
        mock_generate,
        mock_get_client,
        mock_avail,
    ):
        """Verify live extraction succeeds on the first attempt when LLM returns valid JSON."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        valid_response_json = """
        {
            "doc_type": "electricity",
            "issuer": "TSSPDCL",
            "reference_number": "TSSPDCL-2026-99",
            "amount_inr": 3500.0,
            "issue_date": "2026-11-10",
            "deadline_date": "2026-11-25",
            "key_facts": ["Disconnection warning for residential meter"],
            "language_of_document": "English",
            "confidence": 0.95,
            "missing_or_unclear": [],
            "is_mock": false
        }
        """
        mock_response = MagicMock()
        mock_response.text = valid_response_json
        mock_generate.return_value = mock_response

        dummy_bytes = b"dummy_image_data"
        result = extract_notice(dummy_bytes, mime_type="image/png", filename="test.png")

        assert isinstance(result, ExtractionResult)
        assert result.is_mock is False
        assert result.error is None
        assert result.extraction.is_mock is False
        assert result.extraction.doc_type == DocumentType.electricity
        assert result.extraction.issuer == "TSSPDCL"
        assert result.extraction.amount_inr == 3500.0
        assert result.extraction.confidence == 0.95
        assert mock_generate.call_count == 1

    @patch("src.extract.is_live_gemini_available", return_value=True)
    @patch("src.extract.get_gemini_client")
    @patch("src.extract.generate_content_with_retry")
    def test_single_retry_recovery_on_malformed_json(
        self,
        mock_generate,
        mock_get_client,
        mock_avail,
    ):
        """Verify that when attempt 1 produces malformed JSON, attempt 2 (retry) recovers successfully."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Attempt 1: Syntax error in JSON (missing closing brace)
        bad_response = MagicMock()
        bad_response.text = '{"doc_type": "challan", "amount_inr": 1000.0'

        # Attempt 2: Valid JSON
        good_response = MagicMock()
        good_response.text = """
        {
            "doc_type": "challan",
            "issuer": "Cyberabad Traffic Police",
            "reference_number": "CYB-2026-0012",
            "amount_inr": 1000.0,
            "issue_date": "2026-11-05",
            "deadline_date": "2026-11-20",
            "key_facts": ["Violation of signal jumping at Madhapur"],
            "language_of_document": "English",
            "confidence": 0.92,
            "missing_or_unclear": [],
            "is_mock": false
        }
        """
        mock_generate.side_effect = [bad_response, good_response]

        dummy_bytes = b"dummy_image_data"
        result = extract_notice(dummy_bytes, mime_type="image/png", filename="test.png")

        # Verify retry succeeded
        assert isinstance(result, ExtractionResult)
        assert result.error is None
        assert result.is_mock is False
        assert result.extraction.is_mock is False
        assert result.extraction.doc_type == DocumentType.challan
        assert result.extraction.amount_inr == 1000.0
        assert mock_generate.call_count == 2

    @patch("src.extract.is_live_gemini_available", return_value=True)
    @patch("src.extract.get_gemini_client")
    @patch("src.extract.generate_content_with_retry")
    def test_single_retry_recovery_on_validation_error(
        self,
        mock_generate,
        mock_get_client,
        mock_avail,
    ):
        """Verify that when attempt 1 produces invalid schema values (e.g. invalid date), retry recovers."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Attempt 1: Invalid calendar date (February 31)
        invalid_schema_response = MagicMock()
        invalid_schema_response.text = """
        {
            "doc_type": "tax",
            "issuer": "BBMP",
            "reference_number": "BBMP-TAX-01",
            "amount_inr": 5000.0,
            "issue_date": "2026-02-31",
            "deadline_date": "2026-03-15",
            "key_facts": ["Property tax notice"],
            "language_of_document": "English",
            "confidence": 0.88,
            "missing_or_unclear": [],
            "is_mock": false
        }
        """

        # Attempt 2: Fixed valid calendar date
        valid_schema_response = MagicMock()
        valid_schema_response.text = """
        {
            "doc_type": "tax",
            "issuer": "BBMP",
            "reference_number": "BBMP-TAX-01",
            "amount_inr": 5000.0,
            "issue_date": "2026-02-28",
            "deadline_date": "2026-03-15",
            "key_facts": ["Property tax notice"],
            "language_of_document": "English",
            "confidence": 0.88,
            "missing_or_unclear": [],
            "is_mock": false
        }
        """
        mock_generate.side_effect = [invalid_schema_response, valid_schema_response]

        dummy_bytes = b"dummy_image_data"
        result = extract_notice(dummy_bytes, mime_type="image/png", filename="test.png")

        assert isinstance(result, ExtractionResult)
        assert result.error is None
        assert result.is_mock is False
        assert result.extraction.doc_type == DocumentType.tax
        assert result.extraction.issue_date == "2026-02-28"
        assert mock_generate.call_count == 2

    @patch("src.extract.is_live_gemini_available", return_value=True)
    @patch("src.extract.get_gemini_client")
    @patch("src.extract.generate_content_with_retry")
    def test_manual_entry_fallback_on_double_failure(
        self,
        mock_generate,
        mock_get_client,
        mock_avail,
    ):
        """Verify that when both attempt 1 and retry attempt 2 fail, manual entry fallback is returned."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Both attempts fail with malformed JSON
        bad_response_1 = MagicMock()
        bad_response_1.text = "This is not JSON at all."
        bad_response_2 = MagicMock()
        bad_response_2.text = "Still not valid JSON: {"

        mock_generate.side_effect = [bad_response_1, bad_response_2]

        dummy_bytes = b"dummy_image_data"
        result = extract_notice(dummy_bytes, mime_type="image/png", filename="test.png")

        assert isinstance(result, ExtractionResult)
        assert result.is_mock is False
        assert result.error is not None
        assert "Extraction failed after 1 retry" in result.error

        # Manual entry fallback template assertions
        fallback = result.extraction
        assert fallback.confidence == 0.0
        assert fallback.doc_type == DocumentType.other
        assert fallback.issuer is None
        assert fallback.reference_number is None
        assert fallback.amount_inr is None
        assert fallback.missing_or_unclear == [
            "Extraction failed; manual entry fallback required"
        ]
        assert mock_generate.call_count == 2

    @patch("src.extract.is_live_gemini_available", return_value=True)
    @patch("src.extract.get_gemini_client")
    @patch("src.extract.generate_content_with_retry")
    def test_live_extraction_api_exception_fallback(
        self,
        mock_generate,
        mock_get_client,
        mock_avail,
    ):
        """Verify that if Gemini API generation raises an unhandled error, error_result is returned."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_generate.side_effect = RuntimeError("Fatal API communication failure")

        dummy_bytes = b"dummy_image_data"
        result = extract_notice(dummy_bytes, mime_type="image/png", filename="test.png")

        assert isinstance(result, ExtractionResult)
        assert result.error is not None
        assert "Live Gemini extraction failed" in result.error
        assert result.extraction.confidence == 0.0

    @patch("src.extract.is_live_gemini_available", return_value=True)
    @patch("src.extract.get_gemini_client", return_value=None)
    def test_live_available_but_client_none_falls_back_to_mock(
        self,
        mock_get_client,
        mock_avail,
    ):
        """Verify that if is_live_gemini_available is True but client instantiation fails, mock fallback runs."""
        res = extract_notice("notice_01_elec_disconnection.png")
        assert res.is_mock is True
        assert res.extraction.is_mock is True
        assert res.extraction.doc_type == DocumentType.electricity
