"""Comprehensive End-to-End Test Suite for NoticeWise 5 Sample Notices across 3 Languages."""

import io
import json
from pathlib import Path
import fitz
import pytest

from src.explain import generate_grounded_explanation
from src.guardrails import should_abstain
from src.letters import draft_letter_content, generate_letter_pdf, generate_letter_text
from src.reminders import generate_ics_from_extraction
from src.retrieve import get_top_retrieval_score, search_kb
from src.schemas import DocumentType, LetterDraft, NoticeExtraction, RetrievedChunk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHED_DIR = PROJECT_ROOT / "data" / "cached_results"
EVAL_RESULTS_PATH = PROJECT_ROOT / "reports" / "eval_results.json"
CITATION_REVIEW_PATH = PROJECT_ROOT / "reports" / "citation_review.csv"

SAMPLES = [
    "notice_01_elec_disconnection",
    "notice_21_rti_road_repair",
    "notice_16_scheme_ration_card",
    "notice_07_challan_signal",
    "notice_11_tax_property_demand",
]

LANGUAGES = ["English", "Telugu", "Hindi"]


def load_cached_notice(sample_stem: str) -> NoticeExtraction:
    cached_file = CACHED_DIR / f"{sample_stem}.json"
    assert cached_file.exists(), f"Cached live run not found for {sample_stem}"
    with open(cached_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return NoticeExtraction.model_validate(data["extraction"])


class TestEndToEndNoticeFlow:
    """Verifies all 5 tabs, letter drafting, PDF generation, and .ics for target notices."""

    @pytest.mark.parametrize("sample_stem", SAMPLES)
    @pytest.mark.parametrize("language", LANGUAGES)
    def test_sample_end_to_end_flow(self, sample_stem: str, language: str):
        extraction = load_cached_notice(sample_stem)

        # Tab 1: Intake & Abstention Logic
        chunks = search_kb(
            query=f"{extraction.doc_type.value} {' '.join(extraction.key_facts)}",
            doc_type=extraction.doc_type.value,
            top_k=3,
        )
        top_score = get_top_retrieval_score(chunks)
        abstain, reason = should_abstain(extraction=extraction, top_retrieval_score=top_score)

        if sample_stem == "notice_11_tax_property_demand":
            # Must abstain due to lack of verified municipal source in KB (coverage gap)
            assert abstain is True, "notice_11_tax_property_demand must trigger abstention"
        elif sample_stem in [
            "notice_01_elec_disconnection",
            "notice_21_rti_road_repair",
            "notice_16_scheme_ration_card",
        ]:
            assert abstain is False, f"{sample_stem} should be grounded by verified KB sources"

        # Tab 2: Plain Language Explanation & Calendar Reminder (.ics)
        exp_res = generate_grounded_explanation(
            extraction=extraction,
            retrieved_chunks=chunks,
            language=language,
            force_mock=True,
        )
        assert exp_res.explanation is not None
        assert exp_res.language == language

        ics_str = generate_ics_from_extraction(extraction)
        if extraction.deadline_date:
            assert ics_str is not None
            assert "BEGIN:VCALENDAR" in ics_str
            assert "END:VCALENDAR" in ics_str

        # Tab 3: Action Plan & Verified Citations
        assert len(exp_res.checklist) > 0

        # Tab 4: Letter Draft & Multilingual PDF
        draft_model = draft_letter_content(
            extraction=extraction,
            retrieved_chunks=chunks,
            language=language,
            sender_name="Srikanth Rao",
            sender_address="H.No. 1-2-3, Hyderabad",
            sender_phone="9876543210",
        )

        # CRITICAL REGRESSION CHECK: draft_model.body and draft_model.disclaimer must exist
        assert hasattr(draft_model, "body"), "LetterDraft missing 'body' attribute"
        assert isinstance(draft_model.body, str), "LetterDraft.body must be a string"
        assert len(draft_model.body) > 50, "LetterDraft.body must contain full draft text"

        assert hasattr(draft_model, "disclaimer"), "LetterDraft missing 'disclaimer' attribute"
        assert isinstance(draft_model.disclaimer, str), "LetterDraft.disclaimer must be a string"

        # Citizen edits synchronization test
        edited_text = draft_model.body + "\n[CITIZEN EDIT: Additional Grounds]"
        draft_model.body = edited_text
        assert draft_model.body == edited_text

        # Generate PDF and check valid PDF bytes
        pdf_bytes = generate_letter_pdf(draft_model)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 3000
        assert pdf_bytes.startswith(b"%PDF")

        # Verify Unicode text rendering in PDF via PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_text = doc[0].get_text()
        assert len(pdf_text) > 100, f"PDF text empty for {sample_stem} in {language}"
        assert "NoticeWise Citizen Administrative Draft" in pdf_text

        if language == "Telugu":
            # Assert Telugu characters exist in the rendered text
            assert any("\u0c00" <= c <= "\u0c7f" for c in pdf_text), f"Telugu glyphs missing in PDF for {sample_stem}"
        elif language == "Hindi":
            # Assert Devanagari characters exist in the rendered text
            assert any("\u0900" <= c <= "\u097f" for c in pdf_text), f"Hindi glyphs missing in PDF for {sample_stem}"

        # Tab 5: Evaluation Dashboard artifacts check
        assert EVAL_RESULTS_PATH.exists()
        assert CITATION_REVIEW_PATH.exists()


class TestLetterDraftBugRegression:
    """Explicit regression test targeting the crash around line 1021 in app/streamlit_app.py."""

    def test_letter_draft_has_body_attribute_and_accepts_assignment(self):
        extraction = load_cached_notice("notice_01_elec_disconnection")
        chunks = search_kb("electricity disconnection", doc_type="electricity", top_k=2)
        draft = draft_letter_content(
            extraction=extraction,
            retrieved_chunks=chunks,
            language="English",
            sender_name="Test User",
            sender_address="Test Address",
            sender_phone="1234567890",
        )

        # 1. Attribute access must succeed without AttributeError
        try:
            body_val = draft.body
        except AttributeError as err:
            pytest.fail(f"AttributeError raised on draft.body: {err}")

        assert body_val is not None
        assert isinstance(body_val, str)
        assert len(body_val) > 0

        # 2. Assignment must succeed (as done on line 1026 of streamlit_app.py)
        draft.body = "Updated Body Content by Citizen"
        assert draft.body == "Updated Body Content by Citizen"

        # 3. Model dump and re-validation must succeed
        dumped = draft.model_dump()
        revalidated = LetterDraft.model_validate(dumped)
        assert revalidated.body == "Updated Body Content by Citizen"
