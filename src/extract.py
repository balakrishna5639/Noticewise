"""Multimodal extraction pipeline (F2) for NoticeWise.

Provides:
- Multimodal structured extraction for civic notices (images & PDFs) using Gemini 2.5 Flash.
- Strict Pydantic validation adhering to the NoticeExtraction F2 schema.
- Resilient single-retry on malformed JSON or validation errors with error feedback.
- Graceful manual-entry fallback (`confidence=0.0`) if retries fail.
- Offline mock/fixture fallback from `data/test_notices/labels.json` when `GEMINI_API_KEY`
  is unavailable or `force_mock=True`, explicitly marking outputs with `is_mock=True`.
- 10MB input file size limit enforcement.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from google.genai import types
from pydantic import ValidationError

from src.llm import (
    DEFAULT_MODEL,
    generate_content_with_retry,
    get_gemini_client,
    is_live_gemini_available,
)
from src.schemas import (
    DocumentType,
    ExtractionResult,
    LabelsFile,
    NoticeExtraction,
    NoticeLabel,
)

try:
    import pymupdf  # type: ignore
except ImportError:
    pymupdf = None

logger = logging.getLogger(__name__)

# Maximum allowed file size for intake (10 Megabytes)
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "test_notices"
DEFAULT_LABELS_PATH: Path = DATA_DIR / "labels.json"

_LABELS_CACHE: Optional[LabelsFile] = None

EXTRACTION_SYSTEM_PROMPT: str = """You are NoticeWise, an expert AI assistant analyzing Indian civic, administrative, and government documents.
Your task is to accurately extract structured facts from the provided document image or PDF into a strict JSON object.

### Document Type (`doc_type`) Categories:
You must classify the document into EXACTLY ONE of the following 6 categories:
1. "electricity": State electricity board / DISCOM notices, bills, meter tampering notices, provisional assessments, disconnection warnings, or tariff revisions (e.g., BESCOM, TSSPDCL, UPPCL, TANGEDCO).
2. "challan": Traffic e-challans, motor vehicle compounding notices, speed violation citations, or traffic police penalty demands.
3. "tax": Municipal property tax demand notices, local body tax assessments, tax revision notices, commercial taxes, or water/sewerage tax assessments (e.g., BBMP, GHMC, MCD, Municipal Corporations).
4. "scheme_rejection": Formal rejection letters, ineligibility intimations, or benefit cancellation notices for government welfare schemes (e.g., PM-Kisan, PMAY, NFSA ration card, pension schemes).
5. "rti": Citizen-written Right to Information (RTI) applications, queries, or clarification requests addressed to a Public Information Officer (PIO). NOTE: RTI notices are citizen-originated inquiries, NOT incoming penalty notices or government demands!
6. "other": Out-of-scope documents such as private lease/rental agreements, civil court notices, employment offer letters, commercial advertisements, private loan demands, or unidentifiable documents.

### Required JSON Schema:
Return ONLY a valid JSON object matching this schema:
{
  "doc_type": "electricity" | "challan" | "tax" | "scheme_rejection" | "rti" | "other",
  "issuer": string or null,
  "reference_number": string or null,
  "amount_inr": float or null,
  "issue_date": "YYYY-MM-DD" or null,
  "deadline_date": "YYYY-MM-DD" or null,
  "key_facts": [string, ...],
  "language_of_document": string,
  "confidence": float between 0.0 and 1.0,
  "missing_or_unclear": [string, ...],
  "is_mock": false
}

CRITICAL FORMATTING INSTRUCTIONS:
- Output MUST be valid JSON only. Do not wrap in backticks or markdown fences if possible.
- "amount_inr" must be a non-negative float (e.g. 1500.0) or null if no money is demanded. Never negative.
- "issue_date" and "deadline_date" must be valid calendar dates strictly in "YYYY-MM-DD" ISO format, or null.
- "confidence" must be a float between 0.0 and 1.0 representing your confidence in the extraction accuracy.
- "is_mock" must always be false for live extractions.
"""


def infer_mime_type(
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    default: str = "application/octet-stream",
) -> str:
    """Infer MIME type from filename extension or magic bytes header.

    Args:
        file_bytes: Optional raw bytes to inspect magic headers.
        filename: Optional filename to inspect extension.
        default: Fallback MIME type if detection fails.

    Returns:
        MIME type string (e.g. 'image/png', 'image/jpeg', 'application/pdf').
    """
    if filename:
        lower = filename.lower()
        if lower.endswith(".png"):
            return "image/png"
        if lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if lower.endswith(".pdf"):
            return "application/pdf"

    if file_bytes:
        if file_bytes.startswith(b"%PDF-"):
            return "application/pdf"
        if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if file_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"

    return default


def render_pdf_first_page_as_png(pdf_bytes: bytes) -> bytes:
    """Render the first page of a PDF document to PNG bytes using PyMuPDF.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Raw bytes of the rendered PNG image.

    Raises:
        ImportError: If pymupdf is not installed.
        ValueError: If PDF bytes are invalid or have no pages.
    """
    if pymupdf is None:
        raise ImportError("pymupdf is required to render PDF pages as images")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        raise ValueError("PDF document contains no pages")
    page = doc[0]
    pix = page.get_pixmap()
    return pix.tobytes("png")


def _prepare_multimodal_part(file_bytes: bytes, mime_type: str) -> types.Part:
    """Create a Google GenAI Part object from raw bytes and MIME type.

    Directly passes image or PDF bytes using GenAI inline data.
    """
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)


def get_cached_labels(labels_path: Optional[Path] = None) -> Optional[LabelsFile]:
    """Load and cache ground truth labels from labels.json.

    Args:
        labels_path: Optional custom path to labels.json.

    Returns:
        Validated LabelsFile instance, or None if file cannot be loaded.
    """
    global _LABELS_CACHE
    if _LABELS_CACHE is not None and labels_path is None:
        return _LABELS_CACHE

    target_path = labels_path or DEFAULT_LABELS_PATH
    if not target_path.exists():
        logger.warning("labels.json not found at %s", target_path)
        return None

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels_file = LabelsFile.model_validate(data)
        if labels_path is None:
            _LABELS_CACHE = labels_file
        return labels_file
    except Exception as exc:
        logger.error("Failed to load or parse labels.json at %s: %s", target_path, exc)
        return None


def clear_labels_cache() -> None:
    """Clear the in-memory labels cache."""
    global _LABELS_CACHE
    _LABELS_CACHE = None


def _clean_json_text(text: str) -> str:
    """Strip markdown code block fences and surrounding whitespace from LLM text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _extract_mock_fallback(
    filename: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
) -> ExtractionResult:
    """Produce a deterministic offline mock extraction labeled with is_mock=True.

    Checks data/test_notices/labels.json by filename. If matched, returns the
    ground truth label converted to NoticeExtraction. If uncataloged, returns a
    safe generic mock extraction.
    """
    resolved_name = Path(filename).name if filename else None

    if resolved_name:
        labels_file = get_cached_labels()
        if labels_file:
            label = labels_file.get_by_filename(resolved_name)
            if label:
                extraction = label.to_extraction(confidence=1.0, is_mock=True)
                return ExtractionResult.mock_result(
                    extraction=extraction,
                    raw_response=label.model_dump_json(),
                )

    # Safe generic mock extraction for uncataloged files or direct byte inputs
    generic_extraction = NoticeExtraction(
        doc_type=DocumentType.other,
        issuer=None,
        reference_number=None,
        amount_inr=None,
        issue_date=None,
        deadline_date=None,
        key_facts=[f"Offline mock extraction for {resolved_name or 'unnamed document'}"],
        language_of_document="English",
        confidence=0.5,
        missing_or_unclear=[
            "Document not cataloged in mock labels.json; generated generic mock fallback"
        ],
        is_mock=True,
    )
    return ExtractionResult.mock_result(
        extraction=generic_extraction,
        raw_response=f'{{"mock": "uncataloged", "filename": "{resolved_name}"}}',
    )


def _extract_live_gemini(
    file_bytes: bytes,
    mime_type: str,
    filename: Optional[str] = None,
) -> ExtractionResult:
    """Execute live multimodal extraction via Gemini 2.5 Flash with single retry and fallback.

    Performs:
    1. First attempt generation with system prompt and multimodal part.
    2. Pydantic validation via NoticeExtraction.model_validate_json.
    3. Exactly 1 retry if JSON parsing or validation fails, passing feedback.
    4. Manual-entry fallback (`ExtractionResult.error_result`) if retry also fails.
    """
    client = get_gemini_client()
    if client is None:
        logger.warning(
            "Gemini client unavailable despite live check; falling back to offline mock"
        )
        return _extract_mock_fallback(filename=filename, file_bytes=file_bytes)

    try:
        media_part = _prepare_multimodal_part(file_bytes=file_bytes, mime_type=mime_type)
    except Exception as exc:
        logger.error("Failed to prepare multimodal media part: %s", exc)
        return ExtractionResult.error_result(
            error_msg=f"Failed to prepare document payload for Gemini: {exc}",
            doc_type=DocumentType.other,
        )

    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )

    # Attempt 1: Initial extraction
    initial_contents = [EXTRACTION_SYSTEM_PROMPT, media_part]
    raw_response_text_1: Optional[str] = None
    validation_error_1: Optional[str] = None

    try:
        response_1 = generate_content_with_retry(
            client=client,
            model=DEFAULT_MODEL,
            contents=initial_contents,
            config=config,
        )
        raw_response_text_1 = (
            response_1 if isinstance(response_1, str) else getattr(response_1, "text", "")
        )
        if not raw_response_text_1:
            raise ValueError("Gemini returned an empty response")

        cleaned_1 = _clean_json_text(raw_response_text_1)
        extraction_1 = NoticeExtraction.model_validate_json(cleaned_1)
        extraction_1.is_mock = False
        return ExtractionResult(
            extraction=extraction_1,
            is_mock=False,
            raw_response=raw_response_text_1,
            error=None,
        )
    except (ValidationError, json.JSONDecodeError, ValueError) as val_err:
        validation_error_1 = str(val_err)
        logger.warning(
            "Attempt 1 extraction validation failed (%s). Triggering 1 retry with feedback...",
            validation_error_1,
        )
    except Exception as api_err:
        logger.error("Attempt 1 live generation failed with API error: %s", api_err)
        return ExtractionResult.error_result(
            error_msg=f"Live Gemini extraction failed: {api_err}",
            doc_type=DocumentType.other,
        )

    # Attempt 2: Exactly 1 retry with feedback
    retry_feedback = (
        f"The previous output failed validation with the following error:\n{validation_error_1}\n\n"
        f"Previous raw output was:\n{raw_response_text_1}\n\n"
        f"Please correct the errors and return ONLY a valid JSON object strictly adhering to NoticeExtraction schema. "
        f"Ensure all dates are valid calendar dates formatted as YYYY-MM-DD or null, "
        f"amount_inr is a non-negative float >= 0.0 or null, "
        f"doc_type is one of ['electricity', 'challan', 'tax', 'scheme_rejection', 'rti', 'other'], "
        f"and confidence is between 0.0 and 1.0."
    )
    retry_contents = [EXTRACTION_SYSTEM_PROMPT, media_part, retry_feedback]
    raw_response_text_2: Optional[str] = None

    try:
        response_2 = generate_content_with_retry(
            client=client,
            model=DEFAULT_MODEL,
            contents=retry_contents,
            config=config,
        )
        raw_response_text_2 = (
            response_2 if isinstance(response_2, str) else getattr(response_2, "text", "")
        )
        if not raw_response_text_2:
            raise ValueError("Gemini returned an empty response on retry")

        cleaned_2 = _clean_json_text(raw_response_text_2)
        extraction_2 = NoticeExtraction.model_validate_json(cleaned_2)
        extraction_2.is_mock = False
        return ExtractionResult(
            extraction=extraction_2,
            is_mock=False,
            raw_response=raw_response_text_2,
            error=None,
        )
    except Exception as retry_err:
        logger.error(
            "Attempt 2 retry also failed: %s (initial error: %s). Falling back to manual entry template.",
            retry_err,
            validation_error_1,
        )
        error_summary = (
            f"Extraction failed after 1 retry: {retry_err} (initial error: {validation_error_1})"
        )
        return ExtractionResult.error_result(
            error_msg=error_summary,
            doc_type=DocumentType.other,
            raw_response=raw_response_text_2 or raw_response_text_1,
        )


def extract_notice(
    file_input: Union[str, Path, bytes],
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
    force_mock: bool = False,
) -> ExtractionResult:
    """Primary multimodal intake and extraction entrypoint for NoticeWise (F2).

    Args:
        file_input: Document path (str or Path) or raw document bytes.
        mime_type: Optional MIME type (inferred if None).
        filename: Optional filename hint (resolved from path if None).
        force_mock: Force offline mock extraction even if GEMINI_API_KEY is configured.

    Returns:
        ExtractionResult containing NoticeExtraction, telemetry, and mock/error flags.
    """
    file_bytes: bytes
    resolved_filename: Optional[str] = None

    # Step 1: Intake & validate input type
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not path.is_file():
            # Check fallback in data/test_notices/ for convenience
            candidate = DATA_DIR / path.name
            if candidate.is_file():
                path = candidate
            else:
                return ExtractionResult.error_result(
                    error_msg=f"File not found: {file_input}",
                    doc_type=DocumentType.other,
                )

        try:
            file_bytes = path.read_bytes()
        except Exception as read_err:
            return ExtractionResult.error_result(
                error_msg=f"Failed to read file '{file_input}': {read_err}",
                doc_type=DocumentType.other,
            )

        resolved_filename = path.name if filename is None else Path(filename).name

    elif isinstance(file_input, bytes):
        file_bytes = file_input
        resolved_filename = Path(filename).name if filename else None

    else:
        return ExtractionResult.error_result(
            error_msg=f"Unsupported file_input type: {type(file_input).__name__}",
            doc_type=DocumentType.other,
        )

    # Step 2: Enforce 10MB file size limit
    file_size = len(file_bytes)
    if file_size > MAX_FILE_SIZE_BYTES:
        return ExtractionResult.error_result(
            error_msg=f"File size ({file_size} bytes) exceeds maximum 10MB limit ({MAX_FILE_SIZE_BYTES} bytes)",
            doc_type=DocumentType.other,
        )

    # Step 3: Infer MIME type if not provided
    resolved_mime_type = mime_type or infer_mime_type(
        file_bytes=file_bytes,
        filename=resolved_filename,
    )

    # Step 4: Dispatch to Mock Fallback or Live Gemini Multimodal Extraction
    if force_mock or not is_live_gemini_available():
        logger.info(
            "Executing offline mock fallback for notice '%s' (force_mock=%s, live_avail=%s)",
            resolved_filename,
            force_mock,
            is_live_gemini_available(),
        )
        return _extract_mock_fallback(
            filename=resolved_filename,
            file_bytes=file_bytes,
        )

    logger.info("Executing live multimodal Gemini extraction for notice '%s'", resolved_filename)
    return _extract_live_gemini(
        file_bytes=file_bytes,
        mime_type=resolved_mime_type,
        filename=resolved_filename,
    )


def extract_from_file(
    file_path: Union[str, Path],
    force_mock: bool = False,
) -> ExtractionResult:
    """Convenience helper to extract structured facts from a file on disk.

    Args:
        file_path: Path to document file (str or Path).
        force_mock: Force offline mock extraction.

    Returns:
        ExtractionResult container.
    """
    return extract_notice(file_input=file_path, force_mock=force_mock)


def extract_from_bytes(
    file_bytes: bytes,
    mime_type: str,
    filename: Optional[str] = None,
    force_mock: bool = False,
) -> ExtractionResult:
    """Convenience helper to extract structured facts from in-memory bytes.

    Args:
        file_bytes: Raw bytes of the document.
        mime_type: Document MIME type.
        filename: Optional filename hint.
        force_mock: Force offline mock extraction.

    Returns:
        ExtractionResult container.
    """
    return extract_notice(
        file_input=file_bytes,
        mime_type=mime_type,
        filename=filename,
        force_mock=force_mock,
    )
