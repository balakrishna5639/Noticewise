"""Guardrails and safety checks for NoticeWise (F5).

Provides:
- Deterministic abstention rules for unsupported or ambiguous documents:
  - doc_type == 'other'
  - extraction confidence < 0.60
  - retrieval top score < 0.15 (when knowledge base chunks are evaluated)
- Safe guidance recommending local office visit and checklist of documents to bring.
- Standard mandatory disclaimer: not legal advice.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.schemas import DocumentType, NoticeExtraction

# Minimum extraction confidence threshold to proceed without abstaining
MIN_CONFIDENCE_THRESHOLD: float = 0.60

# Minimum retrieval similarity threshold for grounding
MIN_RETRIEVAL_THRESHOLD: float = 0.15

LEGAL_DISCLAIMER: str = (
    "NoticeWise provides informational assistance only and does not constitute legal advice. "
    "Administrative procedures and dispute guidelines vary by jurisdiction and are subject to change. "
    "Always verify official deadlines, fees, and requirements directly with the issuing department or local office."
)

OFFICE_GUIDANCE_BY_TYPE: Dict[str, Dict[str, Any]] = {
    "electricity": {
        "office": "Local Electricity Board / DISCOM Sub-Division Office (Assistant Engineer / Revenue Officer)",
        "documents": [
            "Original electricity notice / disconnection warning",
            "Latest 3 paid electricity bills and receipts",
            "Proof of premises ownership or rental lease agreement",
            "Meter number and Consumer Service Connection details",
            "Government photo ID (Aadhaar / PAN / Voter ID)",
        ],
    },
    "challan": {
        "office": "Local Traffic Police Station / Traffic Composite Branch / Online e-Challan Portal",
        "documents": [
            "Traffic e-challan printout with violation photo / notice number",
            "Vehicle Registration Certificate (RC)",
            "Valid Driving License (DL)",
            "Pollution Under Control (PUC) Certificate & Insurance policy",
            "Proof of vehicle location at time of violation if contesting wrongful capture",
        ],
    },
    "tax": {
        "office": "Municipal Corporation Citizen Service Centre / Ward Revenue Inspector Office",
        "documents": [
            "Property tax demand notice or special notice",
            "Previous financial year property tax receipt and assessment order",
            "Property Khata / Patta certificate and sale deed copy",
            "Built-up area measurements / occupancy certificate",
            "Government photo ID of registered owner",
        ],
    },
    "scheme_rejection": {
        "office": "Tahsildar / Municipal Ward Office / District Welfare Officer (or Citizen Service Portal)",
        "documents": [
            "Official rejection letter or ineligibility intimation printout",
            "Original welfare scheme application acknowledgment / reference slip",
            "Income and caste certificate (if scheme is means-tested)",
            "Ration card, bank passbook copy, and Aadhaar card",
            "Supporting affidavits or certificates clarifying the specific rejection reason",
        ],
    },
    "rti": {
        "office": "Public Information Officer (PIO) / Assistant PIO of the concerned Public Authority",
        "documents": [
            "Draft RTI application form with clear, specific queries",
            "RTI application fee (₹10 Court Fee Stamp, IPO, DD, or online receipt)",
            "Proof of Indian citizenship (Voter ID / Passport / Aadhaar)",
            "BPL certificate (if claiming exemption from application fee)",
        ],
    },
    "other": {
        "office": "The issuing authority named on your document or your local Municipal / Legal Aid Clinic",
        "documents": [
            "Original copy of the document or letter",
            "Valid government identification proof",
            "Any prior correspondence, receipts, or agreements relating to this issue",
        ],
    },
}


class AbstentionDecision:
    """Container for abstention verdict, rationale, and recommended next steps."""

    def __init__(
        self,
        should_abstain: bool,
        reasons: List[str],
        office_to_visit: str,
        documents_to_bring: List[str],
        disclaimer: str = LEGAL_DISCLAIMER,
    ):
        self.should_abstain = should_abstain
        self.reasons = reasons
        self.office_to_visit = office_to_visit
        self.documents_to_bring = documents_to_bring
        self.disclaimer = disclaimer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_abstain": self.should_abstain,
            "reasons": self.reasons,
            "office_to_visit": self.office_to_visit,
            "documents_to_bring": self.documents_to_bring,
            "disclaimer": self.disclaimer,
        }


def evaluate_guardrails(
    extraction: NoticeExtraction,
    top_retrieval_score: Optional[float] = None,
) -> AbstentionDecision:
    """Evaluate deterministic safety guardrails against extracted fields and retrieval scores.

    Triggers abstention if:
    1. doc_type is 'other' (out-of-scope document).
    2. confidence is below MIN_CONFIDENCE_THRESHOLD (0.60).
    3. top_retrieval_score is provided and below MIN_RETRIEVAL_THRESHOLD (0.15).

    Returns:
        AbstentionDecision instance with clear reasoning and office guidance.
    """
    reasons: List[str] = []

    # Check 1: Document classification scope
    doc_type_val = (
        extraction.doc_type.value
        if hasattr(extraction.doc_type, "value")
        else str(extraction.doc_type)
    )
    if doc_type_val == DocumentType.other.value:
        reasons.append(
            "This document is classified as out-of-scope. NoticeWise currently supports "
            "Electricity Disputes, Traffic e-Challans, Municipal Property Tax notices, "
            "Government Scheme Rejections, and Citizen RTI requests."
        )

    # Check 2: Extraction confidence
    if extraction.confidence < MIN_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Extraction confidence ({extraction.confidence:.2f}) is below our safety threshold "
            f"({MIN_CONFIDENCE_THRESHOLD:.2f}). Key details may be blurry, cropped, or ambiguous."
        )

    # Check 3: Retrieval relevance score
    if top_retrieval_score is not None and top_retrieval_score < MIN_RETRIEVAL_THRESHOLD:
        reasons.append(
            f"We could not locate authoritative official procedures matching this document "
            f"(top relevance score: {top_retrieval_score:.2f} < {MIN_RETRIEVAL_THRESHOLD:.2f})."
        )

    should_abstain = len(reasons) > 0

    # Determine recommended office guidance
    guidance = OFFICE_GUIDANCE_BY_TYPE.get(doc_type_val, OFFICE_GUIDANCE_BY_TYPE["other"])

    return AbstentionDecision(
        should_abstain=should_abstain,
        reasons=reasons,
        office_to_visit=guidance["office"],
        documents_to_bring=guidance["documents"],
        disclaimer=LEGAL_DISCLAIMER,
    )


def should_abstain(
    extraction: NoticeExtraction,
    top_retrieval_score: Optional[float] = None,
) -> Tuple[bool, str]:
    """Convenience helper returning (should_abstain, combined_reasons)."""
    decision = evaluate_guardrails(extraction, top_retrieval_score=top_retrieval_score)
    reason_str = "; ".join(decision.reasons) if decision.reasons else ""
    return decision.should_abstain, reason_str


def get_office_guidance(doc_type: Any) -> Dict[str, Any]:
    """Retrieve designated office and documents to bring for document type."""
    type_str = doc_type.value if hasattr(doc_type, "value") else str(doc_type).lower()
    return OFFICE_GUIDANCE_BY_TYPE.get(type_str, OFFICE_GUIDANCE_BY_TYPE["other"])
