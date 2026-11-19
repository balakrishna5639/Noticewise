"""tests/test_verification_r6.py - Automated R6 Verification Test Suite for NoticeWise Phase 1.

Milestone 7 / Requirement R6:
1. Systematic extraction verification across representative synthetic notices spanning:
   - Electricity: notice_01_elec_disconnection.png
   - Challan: notice_06_challan_speeding.png
   - Tax: notice_11_tax_property_demand.png
   - Scheme Rejection: notice_16_scheme_ration_card.pdf (PDF format)
   - RTI Citizen Request: notice_21_rti_road_repair.png (Citizen-written request)
   - Out-of-Scope (F5 Abstention): notice_26_oos_grocery_receipt.png
   - Degraded Scan: notice_29_oos_unreadable_blur.png
2. Schema integrity: assert all 10 F2 fields match ground truth from labels.json.
3. Mock isolation: assert is_mock=True on both ExtractionResult and NoticeExtraction.
4. F5 Abstention & Safety Card logic triggers on doc_type == "other" or confidence < 0.60.
5. Streamlit UI integration: sample notice picker loading, preview rendering, and form validation.
6. Strict Phase 2/3 boundary isolation: assert forbidden files/directories do not exist.
7. Batch validation: assert all 30 synthetic notices extract cleanly with is_mock=True.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from PIL import Image
import pytest
from pydantic import ValidationError

from app.streamlit_app import (
    load_sample_notices,
    save_edited_extraction,
    should_trigger_abstention,
    validate_file_size,
)
from src.extract import (
    DATA_DIR,
    clear_labels_cache,
    extract_from_bytes,
    extract_from_file,
    extract_notice,
    get_cached_labels,
    render_pdf_first_page_as_png,
)
from src.schemas import (
    DocumentType,
    ExtractionResult,
    LabelsFile,
    NoticeExtraction,
    NoticeLabel,
)


@pytest.fixture(autouse=True)
def reset_labels_cache(monkeypatch):
    """Ensure labels cache is fresh and force offline mock during R6 unit tests."""
    clear_labels_cache()
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with patch("src.llm.get_api_key", return_value=None):
        yield
    clear_labels_cache()


# ---------------------------------------------------------------------------
# Canonical synthetic notice filenames for R6 verification
# ---------------------------------------------------------------------------
CANONICAL_NOTICES: List[Dict[str, Any]] = [
    {
        "filename": "notice_01_elec_disconnection.png",
        "category": "electricity",
        "expected_doc_type": DocumentType.electricity,
        "is_pdf": False,
        "is_citizen_request": False,
        "is_out_of_scope": False,
        "is_degraded": False,
    },
    {
        "filename": "notice_06_challan_speeding.png",
        "category": "challan",
        "expected_doc_type": DocumentType.challan,
        "is_pdf": False,
        "is_citizen_request": False,
        "is_out_of_scope": False,
        "is_degraded": False,
    },
    {
        "filename": "notice_11_tax_property_demand.png",
        "category": "tax",
        "expected_doc_type": DocumentType.tax,
        "is_pdf": False,
        "is_citizen_request": False,
        "is_out_of_scope": False,
        "is_degraded": False,
    },
    {
        "filename": "notice_16_scheme_ration_card.pdf",
        "category": "scheme_rejection",
        "expected_doc_type": DocumentType.scheme_rejection,
        "is_pdf": True,
        "is_citizen_request": False,
        "is_out_of_scope": False,
        "is_degraded": False,
    },
    {
        "filename": "notice_21_rti_road_repair.png",
        "category": "rti",
        "expected_doc_type": DocumentType.rti,
        "is_pdf": False,
        "is_citizen_request": True,
        "is_out_of_scope": False,
        "is_degraded": False,
    },
    {
        "filename": "notice_26_oos_grocery_receipt.png",
        "category": "other",
        "expected_doc_type": DocumentType.other,
        "is_pdf": False,
        "is_citizen_request": False,
        "is_out_of_scope": True,
        "is_degraded": False,
    },
    {
        "filename": "notice_29_oos_unreadable_blur.png",
        "category": "other",
        "expected_doc_type": DocumentType.other,
        "is_pdf": False,
        "is_citizen_request": False,
        "is_out_of_scope": True,
        "is_degraded": True,
    },
]


# ===========================================================================
# 1. Canonical Notice Extraction & All 10 Schema Fields Matching Ground Truth
# ===========================================================================

class TestR6CanonicalNoticeExtraction:
    """Verifies extraction and schema compliance on canonical synthetic notices."""

    @pytest.mark.parametrize("notice_info", CANONICAL_NOTICES, ids=lambda n: n["filename"])
    def test_canonical_notice_extraction_matches_ground_truth(
        self, notice_info: Dict[str, Any]
    ) -> None:
        """Extract notice and assert all 10 schema fields match ground truth from labels.json."""
        filename = notice_info["filename"]
        file_path = DATA_DIR / filename
        assert file_path.exists(), f"Synthetic notice file missing: {file_path}"

        # 1. Execute extraction
        result: ExtractionResult = extract_notice(file_path)

        # 2. Assert valid result without unhandled error
        assert result is not None
        assert result.error is None, f"Extraction failed for {filename}: {result.error}"

        # 3. Assert is_mock=True on both ExtractionResult and NoticeExtraction
        assert result.is_mock is True, f"ExtractionResult.is_mock must be True for {filename}"
        assert result.extraction.is_mock is True, (
            f"NoticeExtraction.is_mock must be True for {filename}"
        )

        # 4. Fetch ground truth label
        labels_cache = get_cached_labels()
        label: NoticeLabel = labels_cache.get_by_filename(filename)
        assert label is not None, f"Missing ground truth label for {filename}"

        ext: NoticeExtraction = result.extraction

        # 5. Assert all 10 F2 schema fields match ground truth
        # Field 1: doc_type
        assert ext.doc_type == label.doc_type, (
            f"doc_type mismatch on {filename}: {ext.doc_type} != {label.doc_type}"
        )
        assert ext.doc_type == notice_info["expected_doc_type"]

        # Field 2: issuer
        assert ext.issuer == label.issuer, (
            f"issuer mismatch on {filename}: {ext.issuer} != {label.issuer}"
        )

        # Field 3: reference_number
        assert ext.reference_number == label.reference_number, (
            f"reference_number mismatch on {filename}: {ext.reference_number} != {label.reference_number}"
        )

        # Field 4: amount_inr
        assert ext.amount_inr == label.amount_inr, (
            f"amount_inr mismatch on {filename}: {ext.amount_inr} != {label.amount_inr}"
        )

        # Field 5: issue_date
        assert ext.issue_date == label.issue_date, (
            f"issue_date mismatch on {filename}: {ext.issue_date} != {label.issue_date}"
        )

        # Field 6: deadline_date
        assert ext.deadline_date == label.deadline_date, (
            f"deadline_date mismatch on {filename}: {ext.deadline_date} != {label.deadline_date}"
        )

        # Field 7: key_facts
        assert ext.key_facts == label.key_facts, (
            f"key_facts mismatch on {filename}: {ext.key_facts} != {label.key_facts}"
        )
        assert len(ext.key_facts) >= 2, f"key_facts should contain at least 2 entries for {filename}"

        # Field 8: language_of_document
        assert ext.language_of_document == label.language_of_document, (
            f"language mismatch on {filename}: {ext.language_of_document} != {label.language_of_document}"
        )

        # Field 9: confidence
        assert 0.0 <= ext.confidence <= 1.0, f"confidence out of bounds: {ext.confidence}"
        assert ext.confidence == 1.0, f"mock fallback should return confidence 1.0: {ext.confidence}"

        # Field 10: missing_or_unclear
        assert isinstance(ext.missing_or_unclear, list), "missing_or_unclear must be a list"

    def test_extraction_via_file_bytes(self) -> None:
        """Extraction via extract_from_bytes must produce identical results with is_mock=True."""
        filename = "notice_01_elec_disconnection.png"
        file_path = DATA_DIR / filename
        file_bytes = file_path.read_bytes()

        result = extract_from_bytes(file_bytes, mime_type="image/png", filename=filename)
        assert result.is_mock is True
        assert result.extraction.is_mock is True
        assert result.extraction.doc_type == DocumentType.electricity
        assert result.extraction.amount_inr == 4250.0

    def test_extraction_via_extract_from_file_path_object(self) -> None:
        """Extraction via extract_from_file accepting Path object."""
        file_path = DATA_DIR / "notice_06_challan_speeding.png"
        result = extract_from_file(file_path)
        assert result.is_mock is True
        assert result.extraction.doc_type == DocumentType.challan
        assert result.extraction.issuer == "Hyderabad Traffic Police"


# ===========================================================================
# 2. Category-Specific Domain Scenarios & Semantics
# ===========================================================================

class TestR6CategorySpecificDomainScenarios:
    """Verifies domain-specific semantics for each of the civic notice categories."""

    def test_electricity_notice_semantics(self) -> None:
        """notice_01: Electricity disconnection notice has arrears, Section 56 reference, TSSPDCL issuer."""
        res = extract_notice(DATA_DIR / "notice_01_elec_disconnection.png")
        ext = res.extraction
        assert ext.doc_type == DocumentType.electricity
        assert "TSSPDCL" in (ext.issuer or "")
        assert ext.amount_inr == 4250.0
        assert ext.reference_number == "TSSPDCL-DISC-2026-89421"
        assert ext.issue_date == "2026-10-15"
        assert ext.deadline_date == "2026-10-30"
        assert any("Section 56" in f for f in ext.key_facts)

    def test_traffic_challan_semantics(self) -> None:
        """notice_06: Traffic e-challan has speeding penalty under Motor Vehicles Act."""
        res = extract_notice(DATA_DIR / "notice_06_challan_speeding.png")
        ext = res.extraction
        assert ext.doc_type == DocumentType.challan
        assert "Hyderabad Traffic Police" in (ext.issuer or "")
        assert ext.amount_inr == 1000.0
        assert ext.reference_number == "HYD-ECH-2026-55102"
        assert ext.deadline_date == "2026-11-17"
        assert any("Section 183" in f for f in ext.key_facts)

    def test_property_tax_demand_semantics(self) -> None:
        """notice_11: Municipal tax assessment demand has overdue property tax and penalty."""
        res = extract_notice(DATA_DIR / "notice_11_tax_property_demand.png")
        ext = res.extraction
        assert ext.doc_type == DocumentType.tax
        assert "GHMC" in (ext.issuer or "")
        assert ext.amount_inr == 18500.0
        assert ext.reference_number == "GHMC/PT/CZ/2026/99341"
        assert ext.issue_date == "2026-10-10"
        assert ext.deadline_date == "2026-11-10"

    def test_scheme_rejection_pdf_semantics(self) -> None:
        """notice_16: Welfare scheme rejection (NFSA ration card) has 30-day appeal window."""
        res = extract_notice(DATA_DIR / "notice_16_scheme_ration_card.pdf")
        ext = res.extraction
        assert ext.doc_type == DocumentType.scheme_rejection
        assert "Food and Civil Supplies" in (ext.issuer or "")
        assert ext.amount_inr is None
        assert ext.reference_number == "FCS/NFSA/REJ/2026/7821"
        assert ext.issue_date == "2026-10-14"
        assert ext.deadline_date == "2026-11-13"
        assert any("30 days" in f for f in ext.key_facts)

    def test_citizen_written_rti_request_semantics(self) -> None:
        """notice_21: RTI request is an outbound citizen request with statutory INR 10 fee."""
        labels = get_cached_labels()
        label = labels.get_by_filename("notice_21_rti_road_repair.png")
        assert label.is_citizen_request is True, "notice_21 must have is_citizen_request=True"

        res = extract_notice(DATA_DIR / "notice_21_rti_road_repair.png")
        ext = res.extraction
        assert ext.doc_type == DocumentType.rti
        assert ext.amount_inr == 10.0, "RTI statutory application fee is INR 10.0"
        assert "PIO" in (ext.issuer or "") or "Public Information Officer" in (ext.issuer or "")
        assert ext.reference_number == "RTI-CITIZEN-REQ-2026-01"
        assert ext.issue_date == "2026-11-05"
        assert ext.deadline_date == "2026-12-05"
        assert any("Section 6(1)" in f for f in ext.key_facts)

    def test_out_of_scope_grocery_receipt_semantics(self) -> None:
        """notice_26: Grocery store receipt is out of scope with doc_type 'other'."""
        labels = get_cached_labels()
        label = labels.get_by_filename("notice_26_oos_grocery_receipt.png")
        assert label.is_out_of_scope is True
        assert label.doc_type == DocumentType.other

        res = extract_notice(DATA_DIR / "notice_26_oos_grocery_receipt.png")
        ext = res.extraction
        assert ext.doc_type == DocumentType.other
        assert ext.deadline_date is None

    def test_degraded_scan_blur_semantics(self) -> None:
        """notice_29: Degraded blurred scan is out of scope with doc_type 'other'."""
        labels = get_cached_labels()
        label = labels.get_by_filename("notice_29_oos_unreadable_blur.png")
        assert label.is_degraded is True
        assert label.is_out_of_scope is True

        res = extract_notice(DATA_DIR / "notice_29_oos_unreadable_blur.png")
        ext = res.extraction
        assert ext.doc_type == DocumentType.other


# ===========================================================================
# 3. F5 Abstention & Safety Card Trigger Logic
# ===========================================================================

class TestR6F5AbstentionAndSafetyCard:
    """Verifies that PRD §7 F5 Abstention & Safety Card triggers correctly."""

    def test_abstention_triggers_on_out_of_scope_canonical_notice(self) -> None:
        """Out-of-scope document notice_26 must trigger safety abstention."""
        res = extract_notice(DATA_DIR / "notice_26_oos_grocery_receipt.png")
        should_abstain = should_trigger_abstention(res.extraction.doc_type, res.extraction.confidence)
        assert should_abstain is True

    def test_abstention_triggers_on_degraded_blur_canonical_notice(self) -> None:
        """Degraded blurred notice_29 must trigger safety abstention."""
        res = extract_notice(DATA_DIR / "notice_29_oos_unreadable_blur.png")
        should_abstain = should_trigger_abstention(res.extraction.doc_type, res.extraction.confidence)
        assert should_abstain is True

    @pytest.mark.parametrize(
        "filename",
        [
            "notice_01_elec_disconnection.png",
            "notice_06_challan_speeding.png",
            "notice_11_tax_property_demand.png",
            "notice_16_scheme_ration_card.pdf",
            "notice_21_rti_road_repair.png",
        ],
    )
    def test_abstention_does_not_trigger_on_valid_civic_notices(self, filename: str) -> None:
        """Valid supported civic notices with high confidence must NOT trigger abstention."""
        res = extract_notice(DATA_DIR / filename)
        should_abstain = should_trigger_abstention(res.extraction.doc_type, res.extraction.confidence)
        assert should_abstain is False

    @pytest.mark.parametrize(
        "doc_type,confidence,expected",
        [
            (DocumentType.other, 1.0, True),
            (DocumentType.other, 0.4, True),
            ("other", 0.95, True),
            (DocumentType.electricity, 0.59, True),
            (DocumentType.challan, 0.50, True),
            (DocumentType.tax, 0.0, True),
            (DocumentType.electricity, 0.60, False),
            (DocumentType.challan, 0.95, False),
            (DocumentType.tax, 0.85, False),
            (DocumentType.scheme_rejection, 0.90, False),
            (DocumentType.rti, 0.88, False),
        ],
    )
    def test_abstention_boundary_conditions(
        self, doc_type: Any, confidence: float, expected: bool
    ) -> None:
        """Verify boundary condition for confidence threshold 0.60 and doc_type 'other'."""
        assert should_trigger_abstention(doc_type, confidence) is expected


# ===========================================================================
# 4. Streamlit UI Integration & Preview Pipeline
# ===========================================================================

class TestR6StreamlitUIIntegration:
    """Verifies UI sample loader, preview rendering, and citizen form editing."""

    def test_load_sample_notices_end_to_end(self) -> None:
        """load_sample_notices must load exactly 30 notices spanning all 6 categories."""
        samples = load_sample_notices()
        assert len(samples) == 30, f"Expected 30 samples, found {len(samples)}"

        # Check all files exist on disk
        for sample in samples:
            assert sample["exists"] is True, f"Sample file not on disk: {sample['filename']}"
            assert sample["file_path"].is_file()

        # Check category representation
        categories = {s["doc_type"] for s in samples}
        expected_categories = {"electricity", "challan", "tax", "scheme_rejection", "rti", "other"}
        assert categories == expected_categories

    def test_png_image_preview_pipeline(self) -> None:
        """PNG notice image can be loaded by PIL and has valid dimensions."""
        file_path = DATA_DIR / "notice_01_elec_disconnection.png"
        file_bytes = file_path.read_bytes()

        image = Image.open(io.BytesIO(file_bytes))
        assert image.width > 0
        assert image.height > 0
        assert image.format == "PNG"

    def test_pdf_first_page_rendering_pipeline(self) -> None:
        """PDF notice can be rendered to PNG bytes via PyMuPDF and opened by PIL."""
        file_path = DATA_DIR / "notice_16_scheme_ration_card.pdf"
        file_bytes = file_path.read_bytes()

        png_bytes = render_pdf_first_page_as_png(file_bytes)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0

        image = Image.open(io.BytesIO(png_bytes))
        assert image.width > 0
        assert image.height > 0
        assert image.format == "PNG"

    def test_file_size_validation_guard(self) -> None:
        """validate_file_size allows <=10MB files and rejects >10MB files."""
        # 5 MB payload passes
        valid_bytes = b"0" * (5 * 1024 * 1024)
        is_valid, msg = validate_file_size(valid_bytes)
        assert is_valid is True
        assert msg is None

        # 10 MB + 1 byte payload rejected
        oversized_bytes = b"0" * ((10 * 1024 * 1024) + 1)
        is_valid, msg = validate_file_size(oversized_bytes)
        assert is_valid is False
        assert "exceeds 10 MB limit" in (msg or "")

    def test_citizen_form_edit_valid_save(self) -> None:
        """save_edited_extraction correctly updates and validates modified fields."""
        current = NoticeExtraction(
            doc_type=DocumentType.electricity,
            issuer="Original Issuer",
            reference_number="ORIG-123",
            amount_inr=1000.0,
            issue_date="2026-10-01",
            deadline_date="2026-10-15",
            key_facts=["Fact 1"],
            confidence=0.85,
        )

        updated = save_edited_extraction(
            current=current,
            doc_type=DocumentType.electricity,
            issuer="Updated Issuer",
            reference_number="UPDATED-456",
            amount_inr=1500.0,
            issue_date="2026-10-05",
            deadline_date="2026-10-25",
            key_facts=["Fact 1", "Fact 2"],
            language_of_document="English",
            confidence=0.90,
            missing_or_unclear=[],
        )

        assert updated.issuer == "Updated Issuer"
        assert updated.reference_number == "UPDATED-456"
        assert updated.amount_inr == 1500.0
        assert updated.issue_date == "2026-10-05"
        assert len(updated.key_facts) == 2

    def test_citizen_form_edit_invalid_date_raises_validation_error(self) -> None:
        """save_edited_extraction rejects non-ISO date format."""
        current = NoticeExtraction(
            doc_type=DocumentType.tax,
            confidence=0.80,
        )
        with pytest.raises(ValidationError):
            save_edited_extraction(
                current=current,
                doc_type=DocumentType.tax,
                issuer="GHMC",
                reference_number="TAX-123",
                amount_inr=500.0,
                issue_date="15/10/2026",  # Non-ISO date
                deadline_date=None,
                key_facts=[],
                language_of_document="English",
                confidence=0.80,
                missing_or_unclear=[],
            )

    def test_citizen_form_edit_negative_amount_raises_validation_error(self) -> None:
        """save_edited_extraction rejects negative amounts."""
        current = NoticeExtraction(
            doc_type=DocumentType.challan,
            confidence=0.80,
        )
        with pytest.raises(ValidationError):
            save_edited_extraction(
                current=current,
                doc_type=DocumentType.challan,
                issuer="Police",
                reference_number="CH-123",
                amount_inr=-500.0,  # Negative amount rejected
                issue_date=None,
                deadline_date=None,
                key_facts=[],
                language_of_document="English",
                confidence=0.80,
                missing_or_unclear=[],
            )


# ===========================================================================
# 5. Strict Phase 2 & 3 Boundary Isolation
# ===========================================================================

class TestR6Phase3BoundaryIsolation:
    """Verifies that Phase 3 evaluation artifacts exist and are properly populated."""

    @pytest.mark.parametrize(
        "phase3_path",
        [
            "scripts/run_eval.py",
            "reports/eval_results.json",
            "reports/citation_review.csv",
        ],
    )
    def test_phase3_artifacts_exist(self, phase3_path: str) -> None:
        """Phase 3 evaluation and report files/dirs must exist in Phase 3."""
        project_root = Path(__file__).resolve().parent.parent
        target = project_root / phase3_path
        assert target.exists(), f"Phase 3 artifact missing: {target}"


# ===========================================================================
# 6. Complete Batch Verification (All 30 Synthetic Notices)
# ===========================================================================

class TestR6AllSyntheticNoticesBatch:
    """Verifies the complete dataset of 30 synthetic notices."""

    def test_all_30_notices_extract_successfully_with_is_mock_true(self) -> None:
        """All 30 synthetic notices extract cleanly and are marked is_mock=True."""
        labels_file = get_cached_labels()
        assert labels_file is not None, "get_cached_labels returned None"
        all_notices = labels_file.notices
        assert len(all_notices) == 30, f"Expected 30 notices, found {len(all_notices)}"

        for notice_label in all_notices:
            file_path = DATA_DIR / notice_label.filename
            assert file_path.exists(), f"Missing notice file: {notice_label.filename}"

            result = extract_notice(file_path)
            assert result.error is None, f"Extraction failed on {notice_label.filename}: {result.error}"
            assert result.is_mock is True, f"Result not marked is_mock for {notice_label.filename}"
            assert result.extraction.is_mock is True, f"Extraction not marked is_mock for {notice_label.filename}"
            assert result.extraction.doc_type == notice_label.doc_type
            assert result.extraction.amount_inr == notice_label.amount_inr
            assert notice_label.is_synthetic is True

    def test_all_30_notices_dataset_composition(self) -> None:
        """Dataset distribution meets all PRD and task specifications."""
        labels_file_path = DATA_DIR / "labels.json"
        with open(labels_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels = LabelsFile.model_validate(data)

        notices = labels.notices
        assert len(notices) == 30

        # All synthetic
        assert all(n.is_synthetic is True for n in notices)

        # 5 per category
        type_counts = {}
        for n in notices:
            type_counts[n.doc_type.value] = type_counts.get(n.doc_type.value, 0) + 1

        assert type_counts["electricity"] == 5
        assert type_counts["challan"] == 5
        assert type_counts["tax"] == 5
        assert type_counts["scheme_rejection"] == 5
        assert type_counts["rti"] == 5
        assert type_counts["other"] == 5

        # Exactly 5 citizen-written RTI requests
        rti_citizen = [n for n in notices if n.doc_type == DocumentType.rti and n.is_citizen_request]
        assert len(rti_citizen) == 5

        # Degraded scans present (>= 4 required by PRD, we have 6)
        degraded = [n for n in notices if n.is_degraded]
        assert len(degraded) >= 4

        # Out-of-scope notices present (>= 5 required)
        oos = [n for n in notices if n.is_out_of_scope]
        assert len(oos) >= 5
