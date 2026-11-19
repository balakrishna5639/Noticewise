"""Core Pydantic schemas and interface definitions for NoticeWise.

Defines the F2 structured extraction schema, document types, extraction results,
and ground truth label data models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentType(str, Enum):
    """Supported document types for NoticeWise intake and classification."""

    electricity = "electricity"
    challan = "challan"
    tax = "tax"
    scheme_rejection = "scheme_rejection"
    rti = "rti"
    other = "other"

    # Conventional uppercase aliases
    ELECTRICITY = "electricity"
    CHALLAN = "challan"
    TAX = "tax"
    SCHEME_REJECTION = "scheme_rejection"
    RTI = "rti"
    OTHER = "other"


# The 5 primary supported civic notice types (excluding 'other')
SUPPORTED_DOC_TYPES: Tuple[DocumentType, ...] = (
    DocumentType.electricity,
    DocumentType.challan,
    DocumentType.tax,
    DocumentType.scheme_rejection,
    DocumentType.rti,
)

DATE_REGEX_PATTERN: str = r"^\d{4}-\d{2}-\d{2}$"


class NoticeExtraction(BaseModel):
    """Pydantic model representing structured extraction for a civic notice (F2 schema).

    All 10 required fields from PRD Section 7 are represented, with strict bounds
    on monetary amounts, confidence scores, and ISO date formatting.
    """

    doc_type: DocumentType = Field(
        ...,
        description="Classified document type among supported categories or 'other'",
    )
    issuer: Optional[str] = Field(
        default=None,
        description="Name of the issuing department, board, or authority",
    )
    reference_number: Optional[str] = Field(
        default=None,
        description="Unique reference, notice number, challan ID, or consumer number",
    )
    amount_inr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Monetary amount involved in INR (must be non-negative)",
    )
    issue_date: Optional[str] = Field(
        default=None,
        pattern=DATE_REGEX_PATTERN,
        description="Notice issue date in ISO format (YYYY-MM-DD)",
    )
    deadline_date: Optional[str] = Field(
        default=None,
        pattern=DATE_REGEX_PATTERN,
        description="Response deadline or due date in ISO format (YYYY-MM-DD)",
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description="Key salient points extracted from the document",
    )
    language_of_document: str = Field(
        default="English",
        description="Primary detected language of the document (e.g. English, Telugu, Hindi)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model extraction confidence score between 0.0 and 1.0",
    )
    missing_or_unclear: List[str] = Field(
        default_factory=list,
        description="List of fields or facts that were obscured, missing, or unclear",
    )
    is_mock: bool = Field(
        default=False,
        description="Flag indicating whether this extraction was produced by mock/fixture fallback",
    )
    is_deadline_computed: bool = Field(
        default=False,
        description="Flag indicating whether deadline_date was calculated in Python from issue_date + N days",
    )

    @field_validator("issue_date", "deadline_date", mode="after")
    @classmethod
    def validate_calendar_date(cls, v: Optional[str]) -> Optional[str]:
        """Ensure date strings are valid calendar dates (YYYY-MM-DD)."""
        if v is None:
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as err:
            raise ValueError(
                f"Date string '{v}' is not a valid calendar date (expected YYYY-MM-DD): {err}"
            ) from err
        return v

    @classmethod
    def empty_fallback(
        cls, doc_type: DocumentType = DocumentType.other
    ) -> "NoticeExtraction":
        """Generate an empty extraction template for manual entry fallback."""
        return cls(
            doc_type=doc_type,
            issuer=None,
            reference_number=None,
            amount_inr=None,
            issue_date=None,
            deadline_date=None,
            key_facts=[],
            language_of_document="English",
            confidence=0.0,
            missing_or_unclear=["Extraction failed; manual entry fallback required"],
            is_mock=False,
        )


class ExtractionResult(BaseModel):
    """Result container wrapping the extraction with telemetry, raw response, and status flags."""

    extraction: NoticeExtraction = Field(
        ...,
        description="Structured extraction payload",
    )
    is_mock: bool = Field(
        default=False,
        description="Flag indicating whether mock/fixture fallback was used",
    )
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw LLM text response before JSON parsing",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if extraction encountered issues or fell back",
    )

    @model_validator(mode="after")
    def sync_mock_flag(self) -> "ExtractionResult":
        """Ensure consistency between result level is_mock and extraction level is_mock."""
        if self.is_mock and not self.extraction.is_mock:
            self.extraction.is_mock = True
        elif self.extraction.is_mock and not self.is_mock:
            self.is_mock = True
        return self

    @classmethod
    def mock_result(
        cls,
        extraction: NoticeExtraction,
        raw_response: Optional[str] = None,
    ) -> "ExtractionResult":
        """Convenience constructor for mock extractions."""
        extraction.is_mock = True
        return cls(
            extraction=extraction,
            is_mock=True,
            raw_response=raw_response,
            error=None,
        )

    @classmethod
    def error_result(
        cls,
        error_msg: str,
        doc_type: DocumentType = DocumentType.other,
        raw_response: Optional[str] = None,
    ) -> "ExtractionResult":
        """Convenience constructor for manual entry error fallbacks."""
        fallback = NoticeExtraction.empty_fallback(doc_type=doc_type)
        return cls(
            extraction=fallback,
            is_mock=False,
            raw_response=raw_response,
            error=error_msg,
        )


class NoticeLabel(BaseModel):
    """Ground truth label schema for synthetic and evaluation test notices.

    Stored in data/test_notices/labels.json.
    """

    filename: str = Field(
        ...,
        description="Name of the notice file (e.g. notice_01_electricity.png)",
    )
    doc_type: DocumentType = Field(
        ...,
        description="Ground truth document type",
    )
    issuer: Optional[str] = Field(
        default=None,
        description="Expected issuing entity",
    )
    reference_number: Optional[str] = Field(
        default=None,
        description="Expected reference or notice number",
    )
    amount_inr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Expected amount in INR",
    )
    issue_date: Optional[str] = Field(
        default=None,
        pattern=DATE_REGEX_PATTERN,
        description="Expected issue date (YYYY-MM-DD)",
    )
    deadline_date: Optional[str] = Field(
        default=None,
        pattern=DATE_REGEX_PATTERN,
        description="Expected deadline date (YYYY-MM-DD)",
    )
    key_facts: List[str] = Field(
        default_factory=list,
        description="Expected salient facts",
    )
    language_of_document: str = Field(
        default="English",
        description="Ground truth language",
    )
    is_synthetic: bool = Field(
        default=True,
        description="Flag indicating this is a synthetic test notice (must always be True)",
    )
    is_degraded: bool = Field(
        default=False,
        description="Flag indicating whether image has simulated degradation (blur/noise/rotation)",
    )
    is_out_of_scope: bool = Field(
        default=False,
        description="Flag indicating whether notice is out-of-scope for NoticeWise",
    )
    is_citizen_request: bool = Field(
        default=False,
        description="Flag indicating citizen-written request (specifically for RTI type)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief narrative summary of this test notice",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Evaluation or testing notes",
    )

    @field_validator("issue_date", "deadline_date", mode="after")
    @classmethod
    def validate_calendar_date(cls, v: Optional[str]) -> Optional[str]:
        """Ensure date strings are valid calendar dates (YYYY-MM-DD)."""
        if v is None:
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as err:
            raise ValueError(
                f"Date string '{v}' is not a valid calendar date (expected YYYY-MM-DD): {err}"
            ) from err
        return v

    def to_extraction(
        self, confidence: float = 1.0, is_mock: bool = False
    ) -> NoticeExtraction:
        """Convert ground truth label to a NoticeExtraction instance."""
        return NoticeExtraction(
            doc_type=self.doc_type,
            issuer=self.issuer,
            reference_number=self.reference_number,
            amount_inr=self.amount_inr,
            issue_date=self.issue_date,
            deadline_date=self.deadline_date,
            key_facts=list(self.key_facts),
            language_of_document=self.language_of_document,
            confidence=confidence,
            missing_or_unclear=[],
            is_mock=is_mock,
        )


class LabelsFile(BaseModel):
    """Container schema for data/test_notices/labels.json ground truth file."""

    version: str = Field(
        default="1.0",
        description="Schema version of labels file",
    )
    description: str = Field(
        default="NoticeWise Synthetic Ground Truth Labels",
        description="Dataset description",
    )
    notices: List[NoticeLabel] = Field(
        default_factory=list,
        description="List of labeled notices",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_flexible_format(cls, data: Any) -> Any:
        """Support both top-level list `[NoticeLabel, ...]` and dict `{'notices': [...]}`."""
        if isinstance(data, list):
            return {"notices": data}
        if isinstance(data, dict) and "notices" not in data:
            # Map of {filename: label_dict}
            items = []
            for k, v in data.items():
                if isinstance(v, dict):
                    if "filename" not in v:
                        v = {**v, "filename": k}
                    items.append(v)
            if items:
                return {"notices": items}
        return data

    def get_by_filename(self, filename: str) -> Optional[NoticeLabel]:
        """Find a ground truth label by filename."""
        for notice in self.notices:
            if notice.filename == filename:
                return notice
        return None

    def __len__(self) -> int:
        return len(self.notices)

    def __iter__(self):
        return iter(self.notices)

    def __getitem__(self, item):
        return self.notices[item]


# ============================================================================
# Phase 2 Schemas: Retrieval, Grounded Explanation, Letters & Reminders
# ============================================================================


class RetrievedChunk(BaseModel):
    """Knowledge base chunk with source metadata and similarity score."""

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier, e.g. 'KB_ELEC_01_CHUNK_2'",
    )
    title: str = Field(
        ...,
        description="Official document title",
    )
    source_url: str = Field(
        ...,
        description="Authoritative government or utility source URL",
    )
    date_accessed: Optional[str] = Field(
        default=None,
        description="Date the official document was accessed or retrieved",
    )
    content: str = Field(
        ...,
        description="Text content of the procedure or rule chunk",
    )
    score: float = Field(
        default=0.0,
        description="Retrieval similarity score (TF-IDF or cosine similarity)",
    )


class ChecklistStep(BaseModel):
    """Actionable procedural step with cited authority and required paperwork."""

    step_number: int = Field(
        ...,
        description="Sequential index of the procedural step (1-based)",
    )
    action: str = Field(
        ...,
        description="Clear, directive summary of the required action",
    )
    details: str = Field(
        ...,
        description="Detailed procedural guidance in plain language",
    )
    source_title: str = Field(
        ...,
        description="Title of the authoritative government rule or guideline",
    )
    source_url: str = Field(
        ...,
        description="Direct URL to the official source procedure",
    )
    chunk_id: str = Field(
        ...,
        description="Referenced chunk identifier supporting this step",
    )
    required_documents: List[str] = Field(
        default_factory=list,
        description="Documents citizen must bring or attach for this step",
    )


class PlainExplanation(BaseModel):
    """Citizen-friendly plain-language breakdown of the notice."""

    what_it_is: str = Field(
        ...,
        description="Plain-language explanation of what this notice or document is",
    )
    consequences_if_ignored: str = Field(
        ...,
        description="Clear explanation of penalties, disconnection, or legal risk if ignored",
    )
    amount_inr: Optional[float] = Field(
        default=None,
        description="Demanded or dispute amount in INR, if applicable",
    )
    deadline_date: Optional[str] = Field(
        default=None,
        description="Final payment or dispute deadline (YYYY-MM-DD)",
    )
    days_left: Optional[int] = Field(
        default=None,
        description="Days remaining until deadline (always computed in code, never by LLM)",
    )
    office_to_visit: Optional[str] = Field(
        default=None,
        description="Designated local office, portal, or authority to contact",
    )
    documents_to_bring: List[str] = Field(
        default_factory=list,
        description="Key documents the citizen must have prepared",
    )


class GroundedExplanation(BaseModel):
    """Combined structured response encompassing plain explanation, cited checklist, and metadata."""

    language: str = Field(
        default="English",
        description="Response language (English, Telugu, or Hindi)",
    )
    explanation: PlainExplanation = Field(
        ...,
        description="Plain-language notice explanation",
    )
    checklist: List[ChecklistStep] = Field(
        default_factory=list,
        description="Step-by-step procedural action plan with citations",
    )
    disclaimer: str = Field(
        default="This is an automated explanation, not legal advice. Please verify procedures with the issuing office.",
        description="Standard legal disclaimer",
    )
    top_retrieval_score: float = Field(
        default=0.0,
        description="Highest similarity score among retrieved chunks",
    )
    is_mock: bool = Field(
        default=False,
        description="Flag indicating if offline mock fallback was used",
    )


class LetterDraft(BaseModel):
    """Formal complaint, dispute, or RTI letter draft with filled fields and marked placeholders."""

    doc_type: DocumentType = Field(
        ...,
        description="Document type for template structure",
    )
    language: str = Field(
        default="English",
        description="Language of letter (English, Hindi, or Telugu)",
    )
    recipient_title: str = Field(
        ...,
        description="Title and designation of receiving officer",
    )
    recipient_address: str = Field(
        ...,
        description="Address of the competent local office or department",
    )
    subject: str = Field(
        ...,
        description="Formal subject line referencing notice number and dispute grounds",
    )
    reference_line: str = Field(
        ...,
        description="Reference notice number and date",
    )
    salutation: str = Field(
        default="Respected Sir / Madam,",
        description="Formal opening salutation",
    )
    body_paragraphs: List[str] = Field(
        default_factory=list,
        description="Introductory and procedural factual paragraphs",
    )
    grounds_paragraph: str = Field(
        ...,
        description="Specific legal and factual grounds drafted from authoritative procedures",
    )
    closing: str = Field(
        default="Yours faithfully,",
        description="Formal closing sign-off",
    )
    sender_name: str = Field(
        default="[YOUR NAME]",
        description="Citizen applicant name or placeholder",
    )
    sender_address: str = Field(
        default="[YOUR RESIDENTIAL ADDRESS]",
        description="Citizen applicant address or placeholder",
    )
    sender_phone: Optional[str] = Field(
        default=None,
        description="Citizen contact phone or placeholder",
    )
    unfilled_placeholders: List[str] = Field(
        default_factory=list,
        description="List of placeholder fields requiring user attention before sending",
    )


class ReminderInfo(BaseModel):
    """Calendar reminder payload for .ics generation."""

    title: str = Field(
        ...,
        description="Calendar event title",
    )
    description: str = Field(
        ...,
        description="Calendar event description including procedure checklist",
    )
    due_date: str = Field(
        ...,
        description="Notice deadline date (YYYY-MM-DD)",
    )
    reminder_date: str = Field(
        ...,
        description="Early warning reminder date (3 days before deadline)",
    )
    issuer: Optional[str] = Field(
        default=None,
        description="Issuing authority name",
    )
    reference_number: Optional[str] = Field(
        default=None,
        description="Notice reference number",
    )

