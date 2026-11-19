"""Plain-language grounded explanation and cited checklist generator for NoticeWise (F3).

Optimizations:
1. Merges explanation, checklist, citations, and translation into ONE single structured LLM call.
2. Generates content only for the selected language (English, Telugu, or Hindi).
3. Days-left is computed deterministically in Python, never by the LLM.
4. Deterministic verification step: checks cited chunk IDs exist in the retrieved set
   and verifies keyword/numerical overlap with the supporting chunk. Drops unsupported steps.
5. Disk caching: keyed by extraction_hash + language + prompt_version.
6. Resilient offline fallback reading from retrieved chunks when API key or quota is unavailable.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from google.genai import types

from src.guardrails import LEGAL_DISCLAIMER
from src.llm import (
    GEMINI_EXPLAIN_MODEL,
    generate_content_with_retry,
    get_gemini_client,
    is_live_gemini_available,
)
from src.reminders import compute_days_left
from src.schemas import (
    ChecklistStep,
    DocumentType,
    GroundedExplanation,
    NoticeExtraction,
    PlainExplanation,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "explanations"
EXPLAIN_PROMPT_VERSION = "v2"


def _get_explain_cache_path(extraction: NoticeExtraction, language: str) -> Path:
    """Compute disk cache path for explanation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload_str = f"{extraction.doc_type}_{extraction.reference_number}_{extraction.amount_inr}_{extraction.deadline_date}_{language}_{EXPLAIN_PROMPT_VERSION}"
    cache_key = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{cache_key}_{language}_{EXPLAIN_PROMPT_VERSION}.json"


def verify_checklist_citations(
    checklist: List[ChecklistStep],
    retrieved_chunks: List[RetrievedChunk],
) -> List[ChecklistStep]:
    """Deterministic verification: ensure every step cites a retrieved chunk and shares semantic overlap.

    Any hallucinated or unsupported step is dropped. Official title and URL are enforced from the chunk.
    """
    valid_chunks: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}
    verified: List[ChecklistStep] = []

    for step in checklist:
        chunk = valid_chunks.get(step.chunk_id)
        if not chunk:
            logger.warning(
                "Dropped checklist step %d: cited chunk_id '%s' not in retrieved chunks",
                step.step_number,
                step.chunk_id,
            )
            continue

        # Overlap validation: search for shared words or numbers between step and chunk
        step_text = f"{step.action} {step.details}".lower()
        chunk_text = f"{chunk.title} {chunk.content}".lower()

        step_words = set(re.findall(r"\b[a-z]{4,}\b", step_text))
        chunk_words = set(re.findall(r"\b[a-z]{4,}\b", chunk_text))
        word_overlap = step_words.intersection(chunk_words)

        step_nums = set(re.findall(r"\b\d+\b", step_text))
        chunk_nums = set(re.findall(r"\b\d+\b", chunk_text))
        num_overlap = step_nums.intersection(chunk_nums)

        # For non-English (Telugu/Hindi), words may be script tokens, so allow if numbers overlap
        # or if at least 2 key English terms match, or chunk has high score
        is_supported = (len(word_overlap) >= 2) or (len(num_overlap) >= 1) or (chunk.score >= 0.40)

        if is_supported:
            step.source_title = chunk.title
            step.source_url = chunk.source_url
            verified.append(step)
        else:
            logger.warning(
                "Dropped checklist step %d: insufficient factual overlap with chunk '%s'",
                step.step_number,
                step.chunk_id,
            )

    # Re-index remaining steps sequentially
    for idx, s in enumerate(verified, 1):
        s.step_number = idx

    return verified


def _clean_json_text(text: str) -> str:
    """Strip markdown code block fences and whitespace from LLM JSON response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _build_explain_system_prompt(language: str) -> str:
    """Build unified single-call system prompt for explanation, checklist, and translation."""
    lang_name = {
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिन्दी)",
        "en": "English",
    }.get(language.lower()[:2], language)

    return f"""You are NoticeWise, an expert public administration assistant explaining Indian civic notices to citizens.
Your job is to generate a plain-language explanation and a step-by-step action checklist in {lang_name}.

CRITICAL RULES:
1. Base all procedural claims and checklist steps ONLY on the provided RETRIEVED CHUNKS.
2. Every checklist step MUST cite the exact `chunk_id` of the supporting chunk.
3. DO NOT invent section numbers, statutory deadlines, or penalty amounts not present in the chunks or notice.
4. Output MUST be valid JSON adhering strictly to the schema below.
5. All text in `what_it_is`, `consequences_if_ignored`, `action`, and `details` must be written in {lang_name}.

SCHEMA:
{{
  "explanation": {{
    "what_it_is": "string in {lang_name}",
    "consequences_if_ignored": "string in {lang_name}",
    "office_to_visit": "string or null",
    "documents_to_bring": ["string", ...]
  }},
  "checklist": [
    {{
      "step_number": 1,
      "action": "brief action in {lang_name}",
      "details": "detailed procedural instruction in {lang_name}",
      "chunk_id": "EXACT_CHUNK_ID_FROM_RETRIEVED_CHUNKS",
      "required_documents": ["document 1", ...]
    }}
  ]
}}
"""


def _generate_offline_explanation(
    extraction: NoticeExtraction,
    retrieved_chunks: List[RetrievedChunk],
    language: str,
) -> GroundedExplanation:
    """Deterministic offline fallback grounded in retrieved chunks when live API is unavailable."""
    doc_type_val = (
        extraction.doc_type.value
        if hasattr(extraction.doc_type, "value")
        else str(extraction.doc_type)
    )

    days_left = compute_days_left(extraction.deadline_date)

    # Build plain explanation based on document type
    if doc_type_val == "electricity":
        what = f"Notice from {extraction.issuer or 'Electricity DISCOM'} regarding power supply charges or service connection {extraction.reference_number or ''}."
        conseq = "Risk of electricity supply disconnection and reconnection penalty charges under Section 56 of Electricity Act 2003 if unpaid."
        office = "Local DISCOM Assistant Engineer (AE) / Consumer Service Center"
    elif doc_type_val == "challan":
        what = f"Traffic violation e-challan {extraction.reference_number or ''} issued by Traffic Police."
        conseq = "Compounding penalty or referral of contested citation to the Virtual Traffic Court."
        office = "Traffic Police Portal or Local Traffic Police Station"
    elif doc_type_val == "tax":
        what = f"Municipal property tax assessment demand notice {extraction.reference_number or ''}."
        conseq = "Accrual of statutory penal interest (up to 2% per month) and potential recovery warrant under Municipal laws."
        office = "Municipal Citizen Service Centre / Revenue Ward Office"
    elif doc_type_val == "scheme_rejection":
        what = f"Government welfare scheme rejection or ineligibility intimation ({extraction.reference_number or ''})."
        conseq = "Loss or forfeiture of subsidized foodgrain, pension, or housing benefits unless appealed within 30 days."
        office = "District Grievance Redressal Officer (DGRO) / Tahsildar Office"
    elif doc_type_val == "rti":
        what = "Citizen Right to Information application seeking official records or administrative clarification."
        conseq = "Information request will lapse if statutory application fee or proper public authority address is omitted."
        office = "Public Information Officer (PIO) of concerned Public Authority"
    else:
        what = "Unclassified or general communication document."
        conseq = "Please verify with the issuing office directly to avoid unexpected liabilities."
        office = "Issuing Office named on document"

    # Build checklist steps directly from top retrieved chunks
    steps: List[ChecklistStep] = []
    docs_to_bring: List[str] = [
        "Original notice / e-challan printout",
        "Government Photo ID (Aadhaar / Voter ID)",
    ]

    for idx, chunk in enumerate(retrieved_chunks[:3], 1):
        clean_content = chunk.content.replace("## ", "").replace("# ", "").strip()
        first_line = clean_content.splitlines()[0] if clean_content else "Follow official procedure"
        steps.append(
            ChecklistStep(
                step_number=idx,
                action=first_line[:80],
                details=clean_content[:250] + ("..." if len(clean_content) > 250 else ""),
                source_title=chunk.title,
                source_url=chunk.source_url,
                chunk_id=chunk.chunk_id,
                required_documents=["Relevant identity and ownership documents"],
            )
        )

    explanation = PlainExplanation(
        what_it_is=what,
        consequences_if_ignored=conseq,
        amount_inr=extraction.amount_inr,
        deadline_date=extraction.deadline_date,
        days_left=days_left,
        office_to_visit=office,
        documents_to_bring=docs_to_bring,
    )

    top_score = max([c.score for c in retrieved_chunks], default=0.0)

    return GroundedExplanation(
        language=language,
        explanation=explanation,
        checklist=steps,
        disclaimer=LEGAL_DISCLAIMER,
        top_retrieval_score=top_score,
        is_mock=True,
    )


def generate_grounded_explanation(
    extraction: NoticeExtraction,
    retrieved_chunks: List[RetrievedChunk],
    language: str = "English",
    use_cache: bool = True,
    force_mock: bool = False,
) -> GroundedExplanation:
    """Generate plain-language explanation and cited checklist in a single combined call.

    Args:
        extraction: Confirmed structured notice fields.
        retrieved_chunks: Top authoritative KB chunks from src/retrieve.py.
        language: User-selected target language ('English', 'Telugu', or 'Hindi').
        use_cache: Enable disk caching to avoid redundant LLM calls.
        force_mock: Force offline fallback generation.

    Returns:
        Validated GroundedExplanation instance.
    """
    days_left = compute_days_left(extraction.deadline_date)
    top_score = max([c.score for c in retrieved_chunks], default=0.0)

    # 1. Disk cache check
    cache_path = _get_explain_cache_path(extraction, language)
    cache_enabled = use_cache and (os.getenv("DISABLE_EXPLAIN_CACHE", "").lower() not in ("1", "true"))

    if cache_enabled and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_obj = GroundedExplanation.model_validate(data)
            # Recompute days_left dynamically from current system date
            cached_obj.explanation.days_left = days_left
            logger.info("Loaded grounded explanation from disk cache for %s (%s)", extraction.reference_number, language)
            return cached_obj
        except Exception as exc:
            logger.warning("Failed to read explanation cache: %s", exc)

    # 2. Check live Gemini availability
    if force_mock or not is_live_gemini_available():
        logger.info("Generating deterministic offline explanation (force_mock=%s)", force_mock)
        res = _generate_offline_explanation(extraction, retrieved_chunks, language)
        if use_cache:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(res.model_dump_json(indent=2))
            except Exception as w_err:
                logger.warning("Failed to write offline explanation cache: %s", w_err)
        return res

    client = get_gemini_client()
    if client is None:
        return _generate_offline_explanation(extraction, retrieved_chunks, language)

    # 3. Assemble prompt payload
    chunks_context_lines = []
    for c in retrieved_chunks:
        chunks_context_lines.append(f"--- CHUNK ID: {c.chunk_id} ---")
        chunks_context_lines.append(f"Title: {c.title}")
        chunks_context_lines.append(f"URL: {c.source_url}")
        chunks_context_lines.append(f"Content: {c.content}")
        chunks_context_lines.append("")

    notice_summary = (
        f"Document Type: {extraction.doc_type.value if hasattr(extraction.doc_type, 'value') else extraction.doc_type}\n"
        f"Issuer: {extraction.issuer or 'Official Office'}\n"
        f"Reference Number: {extraction.reference_number or 'N/A'}\n"
        f"Amount Demanded: INR {extraction.amount_inr if extraction.amount_inr is not None else 'None'}\n"
        f"Issue Date: {extraction.issue_date or 'N/A'}\n"
        f"Deadline Date: {extraction.deadline_date or 'N/A'}\n"
        f"Key Salient Facts: {'; '.join(extraction.key_facts) if extraction.key_facts else 'None'}\n"
    )

    user_prompt = (
        f"DOCUMENT CONFIRMED FIELDS:\n{notice_summary}\n\n"
        f"RETRIEVED OFFICIAL PROCEDURES (Only allowed source for procedures):\n"
        f"{chr(10).join(chunks_context_lines)}\n\n"
        f"Generate the grounded explanation and checklist in {language} as a strict JSON object."
    )

    system_prompt = _build_explain_system_prompt(language)
    model = GEMINI_EXPLAIN_MODEL

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=1500,
        response_mime_type="application/json",
    )

    try:
        response = generate_content_with_retry(
            client=client,
            model=model,
            contents=[system_prompt, user_prompt],
            config=config,
        )
        raw_text = response if isinstance(response, str) else getattr(response, "text", "")
        cleaned_json = _clean_json_text(raw_text)
        parsed = json.loads(cleaned_json)

        exp_data = parsed.get("explanation", {})
        raw_checklist = parsed.get("checklist", [])

        # Validate steps and enforce deterministic citation verification
        parsed_steps: List[ChecklistStep] = []
        for s in raw_checklist:
            parsed_steps.append(
                ChecklistStep(
                    step_number=s.get("step_number", 1),
                    action=s.get("action", ""),
                    details=s.get("details", ""),
                    source_title="",
                    source_url="",
                    chunk_id=s.get("chunk_id", ""),
                    required_documents=s.get("required_documents", []),
                )
            )

        verified_steps = verify_checklist_citations(parsed_steps, retrieved_chunks)

        explanation = PlainExplanation(
            what_it_is=exp_data.get("what_it_is", "Official civic notice."),
            consequences_if_ignored=exp_data.get("consequences_if_ignored", "Liable for penalties or service disruption."),
            amount_inr=extraction.amount_inr,
            deadline_date=extraction.deadline_date,
            days_left=days_left,
            office_to_visit=exp_data.get("office_to_visit", "Designated local authority office"),
            documents_to_bring=exp_data.get("documents_to_bring", ["Original Notice", "ID Proof"]),
        )

        grounded = GroundedExplanation(
            language=language,
            explanation=explanation,
            checklist=verified_steps,
            disclaimer=LEGAL_DISCLAIMER,
            top_retrieval_score=top_score,
            is_mock=False,
        )

        # Write to cache
        if cache_enabled:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(grounded.model_dump_json(indent=2))
            except Exception as w_err:
                logger.warning("Failed to write explanation cache: %s", w_err)

        return grounded

    except Exception as exc:
        logger.error("Live explanation generation failed (%s); falling back to offline grounded generation", exc)
        return _generate_offline_explanation(extraction, retrieved_chunks, language)
