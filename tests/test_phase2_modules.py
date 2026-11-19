"""Unit tests for Phase 2 modules: retrieve, explain, and letters."""

from pathlib import Path
import pytest

from src.explain import generate_grounded_explanation, verify_checklist_citations
from src.letters import draft_letter_content, generate_letter_pdf, generate_letter_text
from src.retrieve import get_top_retrieval_score, search_kb
from src.schemas import ChecklistStep, DocumentType, NoticeExtraction, RetrievedChunk


class TestRetrieval:
    def test_search_kb_returns_ranked_chunks(self):
        chunks = search_kb("disconnection notice bill arrears", doc_type="electricity", top_k=3)
        assert len(chunks) == 3
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
        # Verify scores are sorted descending
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)
        # Verify top chunk has valid metadata
        assert chunks[0].source_url.startswith("http")
        assert len(chunks[0].chunk_id) > 0

    def test_search_kb_category_boost(self):
        chunks = search_kb("violation compounding penalty", doc_type="challan", top_k=2)
        assert len(chunks) > 0
        top = chunks[0]
        assert "challan" in top.chunk_id.lower() or "challan" in top.title.lower()


class TestExplain:
    def test_verify_checklist_citations_drops_hallucinated_chunk_id(self):
        valid_chunk = RetrievedChunk(
            chunk_id="VALID_CHUNK_1",
            title="Electricity Rules",
            source_url="https://tserc.gov.in",
            content="Consumer must submit 15 days notice before disconnection under Section 56",
            score=0.5,
        )
        steps = [
            ChecklistStep(
                step_number=1,
                action="Submit objection within 15 days",
                details="Refer to Section 56 disconnection rules",
                source_title="",
                source_url="",
                chunk_id="VALID_CHUNK_1",
            ),
            ChecklistStep(
                step_number=2,
                action="Pay illegal fee",
                details="Invented rule with fake chunk",
                source_title="",
                source_url="",
                chunk_id="FAKE_CHUNK_NONEXISTENT",
            ),
        ]
        verified = verify_checklist_citations(steps, [valid_chunk])
        assert len(verified) == 1
        assert verified[0].chunk_id == "VALID_CHUNK_1"
        assert verified[0].source_title == "Electricity Rules"
        assert verified[0].source_url == "https://tserc.gov.in"
        assert verified[0].step_number == 1

    def test_offline_explanation_generation(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.tax,
            issuer="GHMC",
            reference_number="GHMC-TAX-2026",
            amount_inr=5400.0,
            deadline_date="2026-12-15",
            confidence=0.92,
        )
        chunks = search_kb("property tax assessment special notice revision petition", doc_type="tax", top_k=3)
        res = generate_grounded_explanation(extraction, chunks, language="English", force_mock=True)

        assert res.language == "English"
        assert res.explanation.amount_inr == 5400.0
        assert res.explanation.days_left is not None
        assert len(res.checklist) > 0
        assert res.is_mock is True


class TestLetters:
    def test_letter_draft_and_pdf_generation(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            issuer="TSSPDCL",
            reference_number="TSS-12345",
            amount_inr=4250.0,
            deadline_date="2026-11-30",
            confidence=0.95,
        )
        chunks = search_kb("electricity disconnection notice", doc_type="electricity", top_k=2)
        draft = draft_letter_content(
            extraction=extraction,
            retrieved_chunks=chunks,
            sender_name="Srikanth Rao",
            sender_address="H.No. 4-12, Madhapur, Hyderabad",
            sender_phone="9876543210",
        )
        assert draft.sender_name == "Srikanth Rao"
        assert draft.doc_type == DocumentType.electricity
        assert "TSS-12345" in draft.subject

        # Generate text
        text = generate_letter_text(draft)
        assert "Srikanth Rao" in text
        assert "TSS-12345" in text

        # Generate PDF
        pdf_bytes = generate_letter_pdf(draft)
        assert len(pdf_bytes) > 5000
        assert pdf_bytes.startswith(b"%PDF")
