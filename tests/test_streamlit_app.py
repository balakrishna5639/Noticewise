"""Automated test suite for NoticeWise Streamlit Application Skeleton (Milestone 6 / R5).

Covers:
1. Module imports and symbol availability (`from app import streamlit_app`).
2. Dynamic sample notice picker loading from `data/test_notices/labels.json`.
3. Localization dictionary completeness across English, Telugu, and Hindi.
4. PRD §7 F5 Abstention & Safety Card logic triggers (`doc_type == "other"` or `confidence < 0.60`).
5. Editable form state updates with citizen modifications and Pydantic validation.
6. 10MB input file size limit enforcement.
7. Document preview rendering logic (PIL Image & PyMuPDF PDF first-page rendering).
"""

import io
from pathlib import Path
from typing import Dict

from PIL import Image
import pytest
from pydantic import ValidationError

from app import streamlit_app
from src.extract import MAX_FILE_SIZE_BYTES, render_pdf_first_page_as_png
from src.schemas import DocumentType, NoticeExtraction


# ---------------------------------------------------------------------------
# 1. Module Imports and Interface Symbols
# ---------------------------------------------------------------------------

class TestStreamlitAppImportAndSymbols:
    """Verify app/streamlit_app.py can be imported cleanly and exports all required functions."""

    def test_module_imports_without_error(self) -> None:
        """Module must import without syntax errors or missing dependencies."""
        assert streamlit_app is not None
        assert hasattr(streamlit_app, "main")

    def test_required_symbols_exported(self) -> None:
        """All functional helper routines must be defined and callable."""
        required_callables = [
            "get_translations",
            "load_sample_notices",
            "validate_file_size",
            "should_trigger_abstention",
            "save_edited_extraction",
            "render_preview",
            "render_abstention_card",
            "render_phase2_placeholders",
            "main",
        ]
        for name in required_callables:
            assert hasattr(streamlit_app, name), f"Missing callable: {name}"
            assert callable(getattr(streamlit_app, name)), f"Symbol {name} is not callable"

    def test_constants_defined(self) -> None:
        """Required configuration constants must match architectural specifications."""
        assert streamlit_app.MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024
        assert "electricity" in streamlit_app.DOC_TYPE_OPTIONS
        assert "challan" in streamlit_app.DOC_TYPE_OPTIONS
        assert "tax" in streamlit_app.DOC_TYPE_OPTIONS
        assert "scheme_rejection" in streamlit_app.DOC_TYPE_OPTIONS
        assert "rti" in streamlit_app.DOC_TYPE_OPTIONS
        assert "other" in streamlit_app.DOC_TYPE_OPTIONS
        assert len(streamlit_app.LANGUAGE_OPTIONS) == 3


# ---------------------------------------------------------------------------
# 2. Dynamic Sample Notice Picker
# ---------------------------------------------------------------------------

class TestSampleNoticePickerLoading:
    """Verify loading and parsing of synthetic notices from data/test_notices/labels.json."""

    def test_load_sample_notices_returns_all_thirty_samples(self) -> None:
        """Must load all 30 synthetic notices generated in Milestone 4."""
        samples = streamlit_app.load_sample_notices()
        assert len(samples) == 30, f"Expected 30 samples, got {len(samples)}"

    def test_sample_notices_contain_all_five_supported_types(self) -> None:
        """Must contain samples for all 5 core civic categories plus out-of-scope."""
        samples = streamlit_app.load_sample_notices()
        doc_types = {s["doc_type"] for s in samples}

        for expected in ["electricity", "challan", "tax", "scheme_rejection", "rti", "other"]:
            assert expected in doc_types, f"Missing document category in samples: {expected}"

    def test_sample_notices_files_exist_on_disk(self) -> None:
        """Every loaded sample notice must point to an actual file on disk."""
        samples = streamlit_app.load_sample_notices()
        for s in samples:
            assert s["exists"] is True, f"Sample file marked not found: {s['filename']}"
            assert Path(s["file_path"]).is_file(), f"File does not exist: {s['file_path']}"
            assert Path(s["file_path"]).stat().st_size > 0

    def test_sample_notices_display_name_formatting(self) -> None:
        """Display names must contain filename and uppercase document type."""
        samples = streamlit_app.load_sample_notices()
        first = samples[0]
        assert first["filename"] in first["display_name"]
        assert first["doc_type"].upper() in first["display_name"]

    def test_load_sample_notices_nonexistent_path_returns_empty_list(self, tmp_path: Path) -> None:
        """Nonexistent path should log warning and return empty list cleanly without crashing."""
        fake_path = tmp_path / "nonexistent_labels.json"
        samples = streamlit_app.load_sample_notices(labels_path=fake_path)
        assert samples == []


# ---------------------------------------------------------------------------
# 3. Multilingual Localization Dictionary
# ---------------------------------------------------------------------------

class TestLocalizationDictionary:
    """Verify complete localization dictionary across English, Telugu, and Hindi."""

    @pytest.mark.parametrize("lang_key", ["English", "Telugu", "Hindi"])
    def test_localization_dict_contains_all_three_languages(self, lang_key: str) -> None:
        """LOCALIZED_STRINGS must contain entries for English, Telugu, and Hindi."""
        assert lang_key in streamlit_app.LOCALIZED_STRINGS

    def test_localization_keys_consistency(self) -> None:
        """All three languages must define the exact same set of translation keys."""
        en_keys = set(streamlit_app.LOCALIZED_STRINGS["English"].keys())
        te_keys = set(streamlit_app.LOCALIZED_STRINGS["Telugu"].keys())
        hi_keys = set(streamlit_app.LOCALIZED_STRINGS["Hindi"].keys())

        assert en_keys == te_keys, f"Telugu keys differ from English: {en_keys ^ te_keys}"
        assert en_keys == hi_keys, f"Hindi keys differ from English: {en_keys ^ hi_keys}"

    def test_critical_ui_strings_present(self) -> None:
        """Critical strings required by dispatch and PRD must be present."""
        en = streamlit_app.LOCALIZED_STRINGS["English"]

        # High-visibility mock warning (verbatim check)
        assert en["mock_warning"] == (
            "⚠️ [MOCK EXTRACTION]: Running offline without GEMINI_API_KEY. "
            "Results are fixture-based (MOCK) and do not reflect model accuracy."
        )

        # F5 Abstention warning and disclaimer
        assert "⚠️ NoticeWise is unable to safely process this document. Please visit or call your local issuing office." in en["abstention_warning"]
        assert "NoticeWise does not provide legal advice. Please verify with the issuing office." in en["abstention_disclaimer"]

        # 10MB limit message
        assert en["file_too_large"] == "File exceeds 10 MB limit"

        # Phase 2 module titles
        assert "Plain Language Explanation (F3) — [Phase 2]" in en["p2_f3_title"]
        assert "Official Action Checklist & Citations (F4) — [Phase 2]" in en["p2_f4_title"]
        assert "Response Letter / RTI Draft (F6) — [Phase 2]" in en["p2_f6_title"]
        assert "Deadline Calendar Reminder (.ics) (F7) — [Phase 2]" in en["p2_f7_title"]

    @pytest.mark.parametrize(
        "query,expected_lang",
        [
            ("English", "English"),
            ("english", "English"),
            ("EN", "English"),
            ("Telugu (తెలుగు)", "Telugu"),
            ("telugu", "Telugu"),
            ("te", "Telugu"),
            ("Hindi (हिन्दी)", "Hindi"),
            ("hindi", "Hindi"),
            ("hi", "Hindi"),
            ("", "English"),
            (None, "English"),
            ("unknown_lang", "English"),
        ],
    )
    def test_get_translations_lookup(self, query: str, expected_lang: str) -> None:
        """get_translations must resolve language variants, shorthand codes, and default safely."""
        t = streamlit_app.get_translations(query)
        assert t == streamlit_app.LOCALIZED_STRINGS[expected_lang]


# ---------------------------------------------------------------------------
# 4. PRD §7 F5 Abstention & Safety Card Logic
# ---------------------------------------------------------------------------

class TestAbstentionAndSafetyLogic:
    """Verify F5 safety abstention condition: doc_type == 'other' or confidence < 0.60."""

    @pytest.mark.parametrize(
        "doc_type,confidence,expected_trigger",
        [
            # Out of scope documents always trigger regardless of confidence
            ("other", 1.0, True),
            ("other", 0.95, True),
            ("other", 0.60, True),
            ("other", 0.20, True),
            (DocumentType.other, 0.99, True),
            ("OTHER", 0.85, True),

            # Low confidence (< 0.60) on supported types triggers abstention
            ("electricity", 0.59, True),
            ("electricity", 0.5999, True),
            ("challan", 0.50, True),
            ("tax", 0.10, True),
            ("scheme_rejection", 0.0, True),
            ("rti", 0.45, True),
            (DocumentType.electricity, 0.55, True),

            # Sufficient confidence (>= 0.60) on supported types does NOT trigger abstention
            ("electricity", 0.60, False),
            ("electricity", 0.6001, False),
            ("challan", 0.75, False),
            ("tax", 0.85, False),
            ("scheme_rejection", 0.90, False),
            ("rti", 1.0, False),
            (DocumentType.tax, 0.80, False),
        ],
    )
    def test_should_trigger_abstention(
        self,
        doc_type: str,
        confidence: float,
        expected_trigger: bool,
    ) -> None:
        """Evaluate trigger condition across category and confidence boundaries."""
        result = streamlit_app.should_trigger_abstention(doc_type=doc_type, confidence=confidence)
        assert result == expected_trigger


# ---------------------------------------------------------------------------
# 5. File Size Limit Validation
# ---------------------------------------------------------------------------

class TestFileSizeLimitValidation:
    """Verify 10MB input file size limit enforcement."""

    def test_file_size_within_limit_passes(self) -> None:
        """Payloads under 10MB must be accepted without error."""
        small_payload = b"small notice content" * 100
        is_valid, err_msg = streamlit_app.validate_file_size(small_payload)
        assert is_valid is True
        assert err_msg is None

    def test_file_size_exact_ten_mb_boundary_passes(self) -> None:
        """Payload exactly equal to 10MB (10 * 1024 * 1024 bytes) must pass."""
        ten_mb_exact = b"\x00" * MAX_FILE_SIZE_BYTES
        is_valid, err_msg = streamlit_app.validate_file_size(ten_mb_exact)
        assert is_valid is True
        assert err_msg is None

    def test_file_size_exceeding_ten_mb_rejected(self) -> None:
        """Payload strictly exceeding 10MB must be rejected with informative error."""
        over_limit = b"\x00" * (MAX_FILE_SIZE_BYTES + 1)
        is_valid, err_msg = streamlit_app.validate_file_size(over_limit)
        assert is_valid is False
        assert err_msg is not None
        assert "File exceeds 10 MB limit" in err_msg


# ---------------------------------------------------------------------------
# 6. Editable Extracted-Fields Form State Updates
# ---------------------------------------------------------------------------

class TestEditableFormStateUpdates:
    """Verify citizen modifications update NoticeExtraction model and enforce Pydantic validation."""

    @pytest.fixture
    def initial_extraction(self) -> NoticeExtraction:
        """Sample baseline extraction."""
        return NoticeExtraction(
            doc_type=DocumentType.electricity,
            issuer="TSSPDCL",
            reference_number="REF-12345",
            amount_inr=4250.0,
            issue_date="2026-10-15",
            deadline_date="2026-10-30",
            key_facts=["Original fact 1", "Original fact 2"],
            language_of_document="English",
            confidence=0.95,
            missing_or_unclear=[],
            is_mock=True,
        )

    def test_save_edited_extraction_success(self, initial_extraction: NoticeExtraction) -> None:
        """Valid citizen modifications must update all fields and retain mock flag."""
        updated = streamlit_app.save_edited_extraction(
            current=initial_extraction,
            doc_type="tax",
            issuer="Greater Hyderabad Municipal Corporation (GHMC)",
            reference_number="PTIN-889012",
            amount_inr=7500.0,
            issue_date="2026-11-01",
            deadline_date="2026-11-20",
            key_facts="Citizen verified fact 1\nCitizen verified fact 2",
            language_of_document="Telugu",
            confidence=0.98,
            missing_or_unclear="Minor spelling in address",
        )

        assert updated.doc_type == DocumentType.tax
        assert updated.issuer == "Greater Hyderabad Municipal Corporation (GHMC)"
        assert updated.reference_number == "PTIN-889012"
        assert updated.amount_inr == 7500.0
        assert updated.issue_date == "2026-11-01"
        assert updated.deadline_date == "2026-11-20"
        assert updated.key_facts == ["Citizen verified fact 1", "Citizen verified fact 2"]
        assert updated.language_of_document == "Telugu"
        assert updated.confidence == 0.98
        assert updated.missing_or_unclear == ["Minor spelling in address"]
        assert updated.is_mock is True  # Preserved from initial

    def test_save_edited_extraction_rejection_of_invalid_date(
        self, initial_extraction: NoticeExtraction
    ) -> None:
        """Non-ISO or non-calendar dates must raise Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            streamlit_app.save_edited_extraction(
                current=initial_extraction,
                doc_type="electricity",
                issuer=None,
                reference_number=None,
                amount_inr=100.0,
                issue_date="15-10-2026",  # Invalid non-ISO format
                deadline_date="2026-10-30",
                key_facts=[],
                language_of_document="English",
                confidence=0.9,
                missing_or_unclear=[],
            )

    def test_save_edited_extraction_rejection_of_negative_amount(
        self, initial_extraction: NoticeExtraction
    ) -> None:
        """Negative monetary amounts must raise Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            streamlit_app.save_edited_extraction(
                current=initial_extraction,
                doc_type="electricity",
                issuer=None,
                reference_number=None,
                amount_inr=-500.0,  # Negative amount rejected
                issue_date="2026-10-15",
                deadline_date="2026-10-30",
                key_facts=[],
                language_of_document="English",
                confidence=0.9,
                missing_or_unclear=[],
            )

    def test_save_edited_extraction_rejection_of_out_of_bounds_confidence(
        self, initial_extraction: NoticeExtraction
    ) -> None:
        """Confidence score outside [0.0, 1.0] must raise Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            streamlit_app.save_edited_extraction(
                current=initial_extraction,
                doc_type="electricity",
                issuer=None,
                reference_number=None,
                amount_inr=100.0,
                issue_date="2026-10-15",
                deadline_date="2026-10-30",
                key_facts=[],
                language_of_document="English",
                confidence=1.25,  # > 1.0 rejected
                missing_or_unclear=[],
            )

    def test_save_edited_extraction_empty_strings_convert_to_none(
        self, initial_extraction: NoticeExtraction
    ) -> None:
        """Blank string inputs for optional fields should be normalized to None."""
        updated = streamlit_app.save_edited_extraction(
            current=initial_extraction,
            doc_type="scheme_rejection",
            issuer="   ",  # Whitespace only
            reference_number="",
            amount_inr=None,
            issue_date="  ",
            deadline_date="",
            key_facts="",
            language_of_document="English",
            confidence=0.70,
            missing_or_unclear="",
        )

        assert updated.issuer is None
        assert updated.reference_number is None
        assert updated.amount_inr is None
        assert updated.issue_date is None
        assert updated.deadline_date is None
        assert updated.key_facts == []
        assert updated.missing_or_unclear == []


# ---------------------------------------------------------------------------
# 7. Document Preview Functionality
# ---------------------------------------------------------------------------

class TestDocumentPreviewLogic:
    """Verify document preview handling for image and PDF files."""

    def test_preview_image_valid_png_bytes(self) -> None:
        """PIL can open generated in-memory PNG bytes."""
        img = Image.new("RGB", (50, 50), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        opened = Image.open(io.BytesIO(png_bytes))
        assert opened.size == (50, 50)

    def test_preview_pdf_first_page_rendering(self) -> None:
        """PyMuPDF successfully renders first page of synthetic PDF notice."""
        sample_pdf = Path("data/test_notices/notice_02_elec_meter_tamper.pdf")
        assert sample_pdf.is_file(), "notice_02_elec_meter_tamper.pdf must exist"

        pdf_bytes = sample_pdf.read_bytes()
        rendered_png = render_pdf_first_page_as_png(pdf_bytes)

        assert len(rendered_png) > 0
        img = Image.open(io.BytesIO(rendered_png))
        assert img.width > 0 and img.height > 0
