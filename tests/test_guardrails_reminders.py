"""Unit tests for guardrails (F5) and deadline reminders (F7)."""

from datetime import date, timedelta
import pytest

from src.guardrails import evaluate_guardrails, MIN_CONFIDENCE_THRESHOLD, MIN_RETRIEVAL_THRESHOLD
from src.reminders import compute_days_left, create_reminder_payload, generate_ics_from_extraction
from src.schemas import DocumentType, NoticeExtraction


class TestGuardrails:
    def test_abstention_on_other_doc_type(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.other,
            confidence=0.95,
        )
        decision = evaluate_guardrails(extraction)
        assert decision.should_abstain is True
        assert any("out-of-scope" in r.lower() for r in decision.reasons)
        assert len(decision.documents_to_bring) > 0

    def test_abstention_on_low_confidence(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            confidence=0.45,
        )
        decision = evaluate_guardrails(extraction)
        assert decision.should_abstain is True
        assert any("confidence" in r.lower() for r in decision.reasons)

    def test_abstention_on_low_retrieval_score(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.challan,
            confidence=0.90,
        )
        decision = evaluate_guardrails(extraction, top_retrieval_score=0.08)
        assert decision.should_abstain is True
        assert any("relevance score" in r.lower() for r in decision.reasons)

    def test_no_abstention_on_valid_notice(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            confidence=0.92,
        )
        decision = evaluate_guardrails(extraction, top_retrieval_score=0.45)
        assert decision.should_abstain is False
        assert len(decision.reasons) == 0


class TestReminders:
    def test_compute_days_left(self):
        today = date.today()
        future = (today + timedelta(days=10)).strftime("%Y-%m-%d")
        past = (today - timedelta(days=5)).strftime("%Y-%m-%d")

        assert compute_days_left(future) == 10
        assert compute_days_left(past) == -5
        assert compute_days_left(None) is None
        assert compute_days_left("invalid-date") is None

    def test_generate_ics(self):
        future_date = (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            issuer="TSSPDCL",
            reference_number="TSS-12345",
            amount_inr=1500.0,
            deadline_date=future_date,
            confidence=0.95,
        )
        ics_text = generate_ics_from_extraction(extraction)
        assert ics_text is not None
        assert "BEGIN:VCALENDAR" in ics_text
        assert "END:VCALENDAR" in ics_text
        assert "TSSPDCL" in ics_text
        assert "TSS-12345" in ics_text
        assert "BEGIN:VALARM" in ics_text
        assert "TRIGGER:-P3D" in ics_text

    def test_generate_ics_without_deadline(self):
        extraction = NoticeExtraction(
            doc_type=DocumentType.electricity,
            confidence=0.95,
        )
        assert generate_ics_from_extraction(extraction) is None
