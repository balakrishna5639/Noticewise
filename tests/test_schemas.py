"""Comprehensive unit and boundary tests for NoticeWise schemas.

Verifies:
- DocumentType enum coverage and invalid rejection
- NoticeExtraction field validation, bounds, defaults, and date checks
- Serialization/deserialization roundtrip
- ExtractionResult container and mock synchronization
- NoticeLabel and LabelsFile ground truth handling
"""

import json
import pytest
from pydantic import ValidationError

from src.schemas import (
    DATE_REGEX_PATTERN,
    DocumentType,
    ExtractionResult,
    LabelsFile,
    NoticeExtraction,
    NoticeLabel,
    SUPPORTED_DOC_TYPES,
)


class TestDocumentTypeEnum:
    """Test suite for DocumentType enum definitions and constraints."""

    @pytest.mark.parametrize(
        "doc_type_val",
        [
            "electricity",
            "challan",
            "tax",
            "scheme_rejection",
            "rti",
            "other",
        ],
    )
    def test_all_six_document_types_valid(self, doc_type_val: str):
        """All 6 required document types must be recognized and equal their string value."""
        dt = DocumentType(doc_type_val)
        assert dt == doc_type_val
        assert dt.value == doc_type_val

    def test_uppercase_aliases(self):
        """Uppercase attribute aliases match corresponding enum values."""
        assert DocumentType.ELECTRICITY == DocumentType.electricity
        assert DocumentType.CHALLAN == DocumentType.challan
        assert DocumentType.TAX == DocumentType.tax
        assert DocumentType.SCHEME_REJECTION == DocumentType.scheme_rejection
        assert DocumentType.RTI == DocumentType.rti
        assert DocumentType.OTHER == DocumentType.other

    def test_supported_doc_types_tuple(self):
        """SUPPORTED_DOC_TYPES includes the 5 core notice types and excludes 'other'."""
        assert len(SUPPORTED_DOC_TYPES) == 5
        assert DocumentType.other not in SUPPORTED_DOC_TYPES
        for expected in ["electricity", "challan", "tax", "scheme_rejection", "rti"]:
            assert DocumentType(expected) in SUPPORTED_DOC_TYPES

    def test_invalid_document_type_rejected(self):
        """Invalid document type strings must raise a ValueError."""
        with pytest.raises(ValueError):
            DocumentType("invalid_type")

        with pytest.raises(ValueError):
            DocumentType("court_summons")


class TestNoticeExtractionValidation:
    """Test suite for NoticeExtraction Pydantic model validation and constraints."""

    @pytest.mark.parametrize(
        "doc_type",
        [
            DocumentType.electricity,
            DocumentType.challan,
            DocumentType.tax,
            DocumentType.scheme_rejection,
            DocumentType.rti,
            DocumentType.other,
        ],
    )
    def test_valid_instantiation_for_all_doc_types(self, doc_type: DocumentType):
        """Valid NoticeExtraction can be constructed for all 6 document types."""
        extraction = NoticeExtraction(
            doc_type=doc_type,
            issuer="Test Authority",
            reference_number="REF-12345",
            amount_inr=1500.0,
            issue_date="2026-11-01",
            deadline_date="2026-11-20",
            key_facts=["Disputed bill amount", "Pay before disconnection"],
            language_of_document="English",
            confidence=0.95,
            missing_or_unclear=[],
            is_mock=False,
        )
        assert extraction.doc_type == doc_type
        assert extraction.amount_inr == 1500.0
        assert extraction.confidence == 0.95

    def test_default_values(self):
        """Verify model defaults for optional fields, collections, and flags."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            confidence=0.8,
        )
        assert extraction.issuer is None
        assert extraction.reference_number is None
        assert extraction.amount_inr is None
        assert extraction.issue_date is None
        assert extraction.deadline_date is None
        assert extraction.key_facts == []
        assert extraction.missing_or_unclear == []
        assert extraction.language_of_document == "English"
        assert extraction.is_mock is False

    def test_rejection_of_invalid_doc_type(self):
        """Providing an invalid doc_type raises a ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            NoticeExtraction(
                doc_type="random_nonexistent_type",  # type: ignore
                confidence=0.9,
            )
        assert "doc_type" in str(exc_info.value)

    def test_rejection_of_negative_amount_inr(self):
        """amount_inr < 0.0 must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            NoticeExtraction(
                doc_type=DocumentType.tax,
                amount_inr=-1.0,
                confidence=0.9,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_amount_inr_boundary_zero(self):
        """amount_inr = 0.0 (zero amount / fee waiver) is strictly valid."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.rti,
            amount_inr=0.0,
            confidence=0.9,
        )
        assert extraction.amount_inr == 0.0

    @pytest.mark.parametrize(
        "invalid_confidence",
        [-0.01, -1.0, 1.01, 2.0, 100.0],
    )
    def test_rejection_of_out_of_bounds_confidence(self, invalid_confidence: float):
        """confidence < 0.0 or > 1.0 must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            NoticeExtraction(
                doc_type=DocumentType.challan,
                confidence=invalid_confidence,
            )
        assert "confidence" in str(exc_info.value)

    @pytest.mark.parametrize(
        "valid_confidence",
        [0.0, 0.001, 0.5, 0.85, 0.999, 1.0],
    )
    def test_valid_confidence_boundaries(self, valid_confidence: float):
        """confidence within [0.0, 1.0] must be accepted."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.challan,
            confidence=valid_confidence,
        )
        assert extraction.confidence == pytest.approx(valid_confidence)

    @pytest.mark.parametrize(
        "invalid_date",
        [
            "18-11-2026",       # DD-MM-YYYY format
            "2026/11/18",       # Slashes instead of dashes
            "Nov 18, 2026",     # Human string
            "2026-1-1",         # Unpadded month/day
            "2026-11",          # Missing day
            "not-a-date",       # Arbitrary string
            "2026-13-01",       # Invalid month (regex or calendar)
            "2026-02-31",       # Impossible calendar day
        ],
    )
    def test_rejection_of_non_iso_dates(self, invalid_date: str):
        """Non-ISO and invalid calendar dates for issue_date or deadline_date must be rejected."""
        with pytest.raises(ValidationError):
            NoticeExtraction(
                doc_type=DocumentType.electricity,
                issue_date=invalid_date,
                confidence=0.9,
            )

        with pytest.raises(ValidationError):
            NoticeExtraction(
                doc_type=DocumentType.electricity,
                deadline_date=invalid_date,
                confidence=0.9,
            )

    def test_valid_iso_dates_and_none(self):
        """Valid YYYY-MM-DD dates and None are accepted."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.scheme_rejection,
            issue_date="2026-01-15",
            deadline_date="2026-02-14",
            confidence=0.88,
        )
        assert extraction.issue_date == "2026-01-15"
        assert extraction.deadline_date == "2026-02-14"

        # None values are allowed
        extraction_none = NoticeExtraction(
            doc_type=DocumentType.scheme_rejection,
            issue_date=None,
            deadline_date=None,
            confidence=0.88,
        )
        assert extraction_none.issue_date is None
        assert extraction_none.deadline_date is None

    def test_empty_fallback_template(self):
        """NoticeExtraction.empty_fallback() creates a valid manual-entry template."""
        fallback = NoticeExtraction.empty_fallback(DocumentType.tax)
        assert fallback.doc_type == DocumentType.tax
        assert fallback.confidence == 0.0
        assert fallback.is_mock is False
        assert len(fallback.missing_or_unclear) > 0
        assert fallback.issuer is None
        assert fallback.amount_inr is None


class TestSerializationRoundtrip:
    """Test suite verifying serialization and deserialization fidelity."""

    def test_json_roundtrip(self):
        """model_dump_json() and model_validate_json() roundtrip preserves all fields."""
        original = NoticeExtraction(
            doc_type=DocumentType.challan,
            issuer="Telangana State Police",
            reference_number="TS99AB1234-CH-987",
            amount_inr=1000.0,
            issue_date="2026-10-15",
            deadline_date="2026-11-15",
            key_facts=[
                "Violation: Over-speeding on Outer Ring Road",
                "Challan payable online via e-challan portal",
            ],
            language_of_document="English",
            confidence=0.98,
            missing_or_unclear=[],
            is_mock=False,
        )

        json_str = original.model_dump_json()
        assert isinstance(json_str, str)

        parsed_data = json.loads(json_str)
        assert parsed_data["doc_type"] == "challan"
        assert parsed_data["amount_inr"] == 1000.0
        assert parsed_data["confidence"] == 0.98

        restored = NoticeExtraction.model_validate_json(json_str)
        assert restored == original
        assert restored.doc_type == DocumentType.challan
        assert restored.reference_number == original.reference_number
        assert restored.key_facts == original.key_facts


class TestExtractionResult:
    """Test suite for ExtractionResult container model and helpers."""

    def test_valid_extraction_result(self):
        """ExtractionResult properly wraps a NoticeExtraction payload."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.rti,
            issuer="Ministry of Rural Development",
            confidence=0.9,
        )
        res = ExtractionResult(
            extraction=extraction,
            is_mock=False,
            raw_response="{\"doc_type\": \"rti\"}",
            error=None,
        )
        assert res.extraction.doc_type == DocumentType.rti
        assert res.is_mock is False
        assert res.error is None

    def test_mock_result_helper_and_flag_sync(self):
        """ExtractionResult.mock_result sets is_mock=True on both container and extraction."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            confidence=0.85,
            is_mock=False,
        )
        result = ExtractionResult.mock_result(extraction, raw_response="[fixture]")
        assert result.is_mock is True
        assert result.extraction.is_mock is True
        assert result.raw_response == "[fixture]"

    def test_mock_flag_sync_from_extraction(self):
        """If extraction.is_mock is True, result.is_mock automatically synchronizes to True."""
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            confidence=0.85,
            is_mock=True,
        )
        result = ExtractionResult(extraction=extraction)
        assert result.is_mock is True
        assert result.extraction.is_mock is True

    def test_error_result_helper(self):
        """ExtractionResult.error_result creates a valid fallback container with error text."""
        result = ExtractionResult.error_result(
            error_msg="LLM JSON parsing failed after retry",
            doc_type=DocumentType.other,
            raw_response="Malformed output",
        )
        assert result.error == "LLM JSON parsing failed after retry"
        assert result.extraction.doc_type == DocumentType.other
        assert result.extraction.confidence == 0.0
        assert result.is_mock is False
        assert result.raw_response == "Malformed output"


class TestNoticeLabelAndLabelsFile:
    """Test suite for NoticeLabel and LabelsFile ground truth models."""

    def test_notice_label_instantiation(self):
        """NoticeLabel correctly instantiates with all ground truth attributes."""
        label = NoticeLabel(
            filename="notice_01_electricity.png",
            doc_type=DocumentType.electricity,
            issuer="TSSPDCL",
            reference_number="USC-10928374",
            amount_inr=3450.50,
            issue_date="2026-10-20",
            deadline_date="2026-11-10",
            key_facts=["Disconnection notice for non-payment", "Bill cycle Sept 2026"],
            language_of_document="English",
            is_synthetic=True,
            is_degraded=False,
            is_out_of_scope=False,
            is_citizen_request=False,
            description="Synthetic TSSPDCL disconnection notice",
        )
        assert label.filename == "notice_01_electricity.png"
        assert label.is_synthetic is True
        assert label.amount_inr == 3450.50

    def test_notice_label_to_extraction_conversion(self):
        """to_extraction() converts NoticeLabel to a valid NoticeExtraction instance."""
        label = NoticeLabel(
            filename="notice_02_challan.png",
            doc_type=DocumentType.challan,
            issuer="Cyberabad Traffic Police",
            reference_number="CYB-998877",
            amount_inr=1000.0,
            issue_date="2026-10-01",
            deadline_date="2026-11-01",
            key_facts=["No helmet violation"],
            is_synthetic=True,
        )
        extraction = label.to_extraction(confidence=1.0, is_mock=True)
        assert isinstance(extraction, NoticeExtraction)
        assert extraction.doc_type == DocumentType.challan
        assert extraction.issuer == "Cyberabad Traffic Police"
        assert extraction.amount_inr == 1000.0
        assert extraction.confidence == 1.0
        assert extraction.is_mock is True

    def test_labels_file_from_dict_and_list(self):
        """LabelsFile validates both list of labels and dict with 'notices' key."""
        raw_items = [
            {
                "filename": "n1.png",
                "doc_type": "electricity",
                "amount_inr": 500.0,
            },
            {
                "filename": "n2.pdf",
                "doc_type": "rti",
                "is_citizen_request": True,
            },
        ]

        # Validating from direct list
        lf1 = LabelsFile.model_validate(raw_items)
        assert len(lf1) == 2
        assert lf1[0].filename == "n1.png"
        assert lf1[1].is_citizen_request is True

        # Validating from dict structure
        lf2 = LabelsFile.model_validate({"notices": raw_items})
        assert len(lf2) == 2
        assert lf2.get_by_filename("n1.png") is not None
        assert lf2.get_by_filename("n1.png").amount_inr == 500.0
        assert lf2.get_by_filename("nonexistent.png") is None

    def test_labels_file_iteration(self):
        """LabelsFile supports standard Python iteration."""
        labels = [
            NoticeLabel(filename=f"notice_{i}.png", doc_type=DocumentType.tax)
            for i in range(3)
        ]
        lf = LabelsFile(notices=labels)
        filenames = [item.filename for item in lf]
        assert filenames == ["notice_0.png", "notice_1.png", "notice_2.png"]
