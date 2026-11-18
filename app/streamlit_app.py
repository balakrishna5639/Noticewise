"""NoticeWise — Citizen Notice Simplification & Action Assistant.

Streamlit application providing:
- Multilingual interface (English, Telugu, Hindi).
- Document intake via file uploader (PNG, JPG, PDF, max 10MB) or synthetic sample notice picker.
- Live document preview (Image or PyMuPDF PDF first-page rendering).
- Extraction execution via src.extract with prominent MOCK banner if offline.
- F5 Abstention & Safety Card when doc_type == 'other' or confidence < 0.60.
- Editable extracted fields form for all 10 F2 schema fields with session state persistence.
- Phase 2 module placeholders (Plain language explanation, checklist, letter draft, calendar reminder).
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import streamlit as st

from src.extract import (
    MAX_FILE_SIZE_BYTES,
    extract_notice,
    infer_mime_type,
    render_pdf_first_page_as_png,
)
from src.schemas import (
    DocumentType,
    ExtractionResult,
    LabelsFile,
    NoticeExtraction,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "test_notices"
DEFAULT_LABELS_PATH: Path = DATA_DIR / "labels.json"

DOC_TYPE_OPTIONS: List[str] = [
    "electricity",
    "challan",
    "tax",
    "scheme_rejection",
    "rti",
    "other",
]

LANGUAGE_OPTIONS: List[str] = [
    "English",
    "Telugu (తెలుగు)",
    "Hindi (हिन्दी)",
]

LOCALIZED_STRINGS: Dict[str, Dict[str, str]] = {
    "English": {
        "app_title": "NoticeWise — Citizen Notice Simplification & Action Assistant",
        "app_subtitle": "Turn confusing government notices into clear, actionable paperwork.",
        "language_label": "Select Language / భాష / भाषा",
        "intake_header": "1. Document Intake",
        "intake_mode_label": "Choose Intake Method",
        "mode_upload": "Upload Document (PNG, JPG, PDF)",
        "mode_sample": "Sample Notice Picker (Synthetic)",
        "upload_label": "Upload Notice (PNG, JPG, or PDF — max 10MB)",
        "sample_label": "Choose a pre-loaded synthetic notice:",
        "analyze_button": "Analyze Notice",
        "preview_header": "2. Document Preview",
        "preview_caption": "Notice Document Preview",
        "mock_warning": "⚠️ [MOCK EXTRACTION]: Running offline without GEMINI_API_KEY. Results are fixture-based (MOCK) and do not reflect model accuracy.",
        "abstention_title": "NoticeWise Safety Warning: Local Office Visit Recommended",
        "abstention_warning": "⚠️ NoticeWise is unable to safely process this document. Please visit or call your local issuing office.",
        "abstention_bring_header": "What to bring with you to the issuing office:",
        "abstention_item1": "Original physical notice document",
        "abstention_item2": "Government Photo ID (Aadhaar / Voter ID / PAN Card)",
        "abstention_item3": "Previous payment receipts or relevant utility account records",
        "abstention_disclaimer": "NoticeWise does not provide legal advice. Please verify with the issuing office.",
        "form_header": "3. Extracted Notice Details (Review & Edit)",
        "doc_type": "Document Type",
        "issuer": "Issuing Authority / Department",
        "reference_number": "Reference / Notice Number",
        "amount_inr": "Amount Demanded (INR)",
        "issue_date": "Issue Date (YYYY-MM-DD)",
        "deadline_date": "Deadline / Due Date (YYYY-MM-DD)",
        "key_facts": "Key Facts (one item per line)",
        "language_of_document": "Language of Document",
        "confidence": "Extraction Confidence Score",
        "missing_or_unclear": "Missing or Unclear Details (one item per line)",
        "save_button": "Save / Confirm Details",
        "save_success": "✅ Notice details confirmed and saved successfully!",
        "file_too_large": "File exceeds 10 MB limit",
        "phase2_header": "4. Citizen Action Modules (Phase 2)",
        "p2_f3_title": "Plain Language Explanation (F3) — [Phase 2]",
        "p2_f3_desc": "Translates bureaucratic legal jargon into simple, citizen-friendly language with clear consequences of non-compliance.",
        "p2_f4_title": "Official Action Checklist & Citations (F4) — [Phase 2]",
        "p2_f4_desc": "Provides step-by-step verified dispute procedures citing official rules, portals, and fee schedules.",
        "p2_f6_title": "Response Letter / RTI Draft (F6) — [Phase 2]",
        "p2_f6_desc": "Drafts formal representation or RTI application letters addressed to the competent administrative authority.",
        "p2_f7_title": "Deadline Calendar Reminder (.ics) (F7) — [Phase 2]",
        "p2_f7_desc": "Generates downloadable iCalendar reminders with automated alerts 7, 3, and 1 days before statutory deadlines.",
        "no_file_selected": "Please upload a document or choose a sample notice to begin analysis.",
    },
    "Telugu": {
        "app_title": "నోటీస్‌వైజ్ — పౌర నోటీసుల సరళీకరణ మరియు సహాయకుడు",
        "app_subtitle": "గందరగోళ ప్రభుత్వ నోటీసులను స్పష్టమైన, ఆచరణాత్మక పత్రాలుగా మార్చండి.",
        "language_label": "భాషను ఎంచుకోండి / Select Language",
        "intake_header": "1. పత్ర సమర్పణ (ఇన్‌టేక్)",
        "intake_mode_label": "సమర్పణ విధానాన్ని ఎంచుకోండి",
        "mode_upload": "పత్రాన్ని అప్‌లోడ్ చేయండి (PNG, JPG, PDF)",
        "mode_sample": "నమూనా నోటీసు ఎంపిక (సింథటిక్)",
        "upload_label": "నోటీసును అప్‌లోడ్ చేయండి (PNG, JPG, లేదా PDF — గరిష్టం 10MB)",
        "sample_label": "ముందుగా లోడ్ చేసిన నమూనా నోటీసును ఎంచుకోండి:",
        "analyze_button": "నోటీసును విశ్లేషించండి",
        "preview_header": "2. పత్రం ప్రివ్యూ",
        "preview_caption": "నోటీసు పత్రం ప్రివ్యూ",
        "mock_warning": "⚠️ [MOCK EXTRACTION]: Running offline without GEMINI_API_KEY. Results are fixture-based (MOCK) and do not reflect model accuracy.",
        "abstention_title": "నోటీస్‌వైజ్ భద్రతా హెచ్చరిక: స్థానిక కార్యాలయ సందర్శన సిఫార్సు చేయబడింది",
        "abstention_warning": "⚠️ NoticeWise is unable to safely process this document. Please visit or call your local issuing office.",
        "abstention_bring_header": "మీ వెంట తీసుకెళ్లవలసిన పత్రాలు:",
        "abstention_item1": "అసలు నోటీసు పత్రం (Original notice)",
        "abstention_item2": "ప్రభుత్వ గుర్తింపు కార్డు (ఆధార్ / ఓటర్ ఐడి / పాన్ కార్డ్)",
        "abstention_item3": "గత చెల్లింపు రసీదులు లేదా ఖాతా వివరాలు",
        "abstention_disclaimer": "NoticeWise does not provide legal advice. Please verify with the issuing office.",
        "form_header": "3. సేకరించిన నోటీసు వివరాలు (సమీక్షించండి & సవరించండి)",
        "doc_type": "పత్రం రకం",
        "issuer": "జారీ చేసిన విభాగం / అధికారి",
        "reference_number": "రిఫరెన్స్ / నోటీసు నంబర్",
        "amount_inr": "మొత్తం (రూపాయలు)",
        "issue_date": "జారీ తేదీ (YYYY-MM-DD)",
        "deadline_date": "గడువు తేదీ (YYYY-MM-DD)",
        "key_facts": "ముఖ్య విషయాలు (లైనుకు ఒకటి)",
        "language_of_document": "పత్రం భాష",
        "confidence": "విశ్వసనీయత స్కోర్",
        "missing_or_unclear": "అస్పష్ట లేదా తప్పిపోయిన అంశాలు",
        "save_button": "వివరాలను సేవ్ / నిర్ధారించండి",
        "save_success": "✅ నోటీసు వివరాలు విజయవంతంగా ధృవీకరించబడ్డాయి మరియు భద్రపరచబడ్డాయి!",
        "file_too_large": "File exceeds 10 MB limit",
        "phase2_header": "4. పౌర చర్యల విభాగాలు (ఫేజ్ 2)",
        "p2_f3_title": "Plain Language Explanation (F3) — [Phase 2]",
        "p2_f3_desc": "చట్టపరమైన నిబంధనలను సులభమైన, సరళమైన భాషలో వివరిస్తుంది.",
        "p2_f4_title": "Official Action Checklist & Citations (F4) — [Phase 2]",
        "p2_f4_desc": "అధికారిక నిబంధనలు మరియు ఫీజు వివరాలతో కూడిన దశలవారీ ప్రక్రియ.",
        "p2_f6_title": "Response Letter / RTI Draft (F6) — [Phase 2]",
        "p2_f6_desc": "సంబంధిత విభాగానికి పంపవలసిన అధికారిక సమాధాన లేఖను సిద్ధం చేస్తుంది.",
        "p2_f7_title": "Deadline Calendar Reminder (.ics) (F7) — [Phase 2]",
        "p2_f7_desc": "ముఖ్యమైన గడువుల కోసం క్యాలెండర్ రిమైండర్‌లను డౌన్‌లోడ్ చేస్తుంది.",
        "no_file_selected": "విశ్లేషణ ప్రారంభించడానికి దయచేసి పత్రాన్ని అప్‌లోడ్ చేయండి లేదా నమూనా నోటీసును ఎంచుకోండి.",
    },
    "Hindi": {
        "app_title": "नोटिसवाइज़ — नागरिक नोटिस सरलीकरण एवं कार्य सहायक",
        "app_subtitle": "जटिल सरकारी नोटिसों को स्पष्ट, कार्रवाई योग्य दस्तावेजों में बदलें।",
        "language_label": "भाषा चुनें / Select Language",
        "intake_header": "1. दस्तावेज़ इनटेक",
        "intake_mode_label": "इनटेक विधि चुनें",
        "mode_upload": "दस्तावेज़ अपलोड करें (PNG, JPG, PDF)",
        "mode_sample": "नमूना नोटिस चयनकर्ता (सिंथेटिक)",
        "upload_label": "नोटिस अपलोड करें (PNG, JPG, या PDF — अधिकतम 10MB)",
        "sample_label": "पहले से लोड किया गया नमूना नोटिस चुनें:",
        "analyze_button": "नोटिस का विश्लेषण करें",
        "preview_header": "2. दस्तावेज़ पूर्वावलोकन",
        "preview_caption": "नोटिस दस्तावेज़ पूर्वावलोकन",
        "mock_warning": "⚠️ [MOCK EXTRACTION]: Running offline without GEMINI_API_KEY. Results are fixture-based (MOCK) and do not reflect model accuracy.",
        "abstention_title": "नोटिसवाइज़ सुरक्षा चेतावनी: स्थानीय कार्यालय जाने की अनुशंसा",
        "abstention_warning": "⚠️ NoticeWise is unable to safely process this document. Please visit or call your local issuing office.",
        "abstention_bring_header": "साथ में क्या ले जाएं:",
        "abstention_item1": "मूल नोटिस दस्तावेज़ (Original notice)",
        "abstention_item2": "सरकारी पहचान पत्र (आधार / वोटर आईडी / पैन कार्ड)",
        "abstention_item3": "पिछली भुगतान रसीदें या संबंधित खाता रिकॉर्ड",
        "abstention_disclaimer": "NoticeWise does not provide legal advice. Please verify with the issuing office.",
        "form_header": "3. निकाले गए नोटिस विवरण (समीक्षा एवं संपादन)",
        "doc_type": "दस्तावेज़ का प्रकार",
        "issuer": "जारीकर्ता प्राधिकरण / विभाग",
        "reference_number": "संदर्भ / नोटिस संख्या",
        "amount_inr": "मांग राशि (रुपये)",
        "issue_date": "जारी करने की तारीख (YYYY-MM-DD)",
        "deadline_date": "अंतिम तिथि / नियत तारीख (YYYY-MM-DD)",
        "key_facts": "प्रमुख तथ्य (प्रति पंक्ति एक)",
        "language_of_document": "दस्तावेज़ की भाषा",
        "confidence": "विश्वास स्कोर",
        "missing_or_unclear": "लापता या अस्पष्ट विवरण",
        "save_button": "विवरण सहेजें / पुष्टि करें",
        "save_success": "✅ नोटिस विवरण सफलतापूर्वक पुष्ट और सहेजे गए!",
        "file_too_large": "File exceeds 10 MB limit",
        "phase2_header": "4. नागरिक कार्रवाई मॉड्यूल (चरण 2)",
        "p2_f3_title": "Plain Language Explanation (F3) — [Phase 2]",
        "p2_f3_desc": "कानूनी जटिलताओं को सरल, नागरिक-अनुकूल भाषा में बदलता है।",
        "p2_f4_title": "Official Action Checklist & Citations (F4) — [Phase 2]",
        "p2_f4_desc": "आधिकारिक नियमों और शुल्क विवरणों के साथ चरण-दर-चरण प्रक्रिया।",
        "p2_f6_title": "Response Letter / RTI Draft (F6) — [Phase 2]",
        "p2_f6_desc": "सक्षम प्रशासनिक प्राधिकरण को संबोधित औपचारिक पत्र या आरटीआई तैयार करता है।",
        "p2_f7_title": "Deadline Calendar Reminder (.ics) (F7) — [Phase 2]",
        "p2_f7_desc": "समय सीमा के लिए डाउनलोड करने योग्य iCalendar अनुस्मारक उत्पन्न करता है।",
        "no_file_selected": "विश्लेषण शुरू करने के लिए कृपया एक दस्तावेज़ अपलोड करें या एक नमूना नोटिस चुनें।",
    },
}


def get_translations(lang: str) -> Dict[str, str]:
    """Retrieve UI localization dictionary for the specified language.

    Args:
        lang: Language string (e.g. 'English', 'Telugu (తెలుగు)', 'Hindi (हिन्दी)',
              or shorthand 'te', 'hi', 'en').

    Returns:
        Dictionary of localized UI labels.
    """
    cleaned = (lang or "").strip().lower()
    if "telugu" in cleaned or cleaned == "te":
        return LOCALIZED_STRINGS["Telugu"]
    if "hindi" in cleaned or cleaned == "hi":
        return LOCALIZED_STRINGS["Hindi"]
    return LOCALIZED_STRINGS["English"]


def load_sample_notices(
    labels_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Dynamically load synthetic sample notices from labels.json.

    Args:
        labels_path: Optional path to custom labels.json.

    Returns:
        List of dictionaries with notice details suitable for UI selection.
    """
    target_path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
    if not target_path.exists():
        logger.warning("labels.json does not exist at %s", target_path)
        return []

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels_file = LabelsFile.model_validate(data)
    except Exception as exc:
        logger.error("Failed to parse labels.json at %s: %s", target_path, exc)
        return []

    samples: List[Dict[str, Any]] = []
    base_dir = target_path.parent
    for notice in labels_file.notices:
        file_path = base_dir / notice.filename
        type_str = notice.doc_type.value if hasattr(notice.doc_type, "value") else str(notice.doc_type)
        desc = notice.description or notice.issuer or "Synthetic notice"
        display_name = f"{notice.filename} [{type_str.upper()}] — {desc}"
        samples.append(
            {
                "filename": notice.filename,
                "doc_type": type_str,
                "issuer": notice.issuer,
                "description": notice.description,
                "display_name": display_name,
                "file_path": file_path,
                "exists": file_path.is_file(),
            }
        )

    return samples


def validate_file_size(
    file_bytes: bytes,
    max_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[bool, Optional[str]]:
    """Validate whether file bytes are within the allowed size threshold.

    Args:
        file_bytes: Byte payload of the candidate document.
        max_bytes: Maximum allowed size in bytes (defaults to 10MB).

    Returns:
        Tuple of (is_valid, error_message).
    """
    size = len(file_bytes)
    if size > max_bytes:
        mb_size = size / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        return (
            False,
            f"File exceeds 10 MB limit ({mb_size:.2f} MB > {max_mb:.0f} MB)",
        )
    return True, None


def should_trigger_abstention(
    doc_type: Union[DocumentType, str],
    confidence: float,
) -> bool:
    """Evaluate PRD Section 7 F5 Abstention & Safety condition.

    Abstention triggers if the document type is classified as 'other' or
    model confidence is strictly below 0.60.

    Args:
        doc_type: Document classification category.
        confidence: Extraction confidence score in [0.0, 1.0].

    Returns:
        True if safety abstention is required, False otherwise.
    """
    type_str = doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type)
    type_str = type_str.lower().strip()
    return type_str == "other" or float(confidence) < 0.60


def save_edited_extraction(
    current: NoticeExtraction,
    doc_type: Union[DocumentType, str],
    issuer: Optional[str],
    reference_number: Optional[str],
    amount_inr: Optional[float],
    issue_date: Optional[str],
    deadline_date: Optional[str],
    key_facts: Union[List[str], str],
    language_of_document: str,
    confidence: float,
    missing_or_unclear: Union[List[str], str],
) -> NoticeExtraction:
    """Validate and update extraction state with citizen modifications.

    Args:
        current: Original NoticeExtraction instance.
        doc_type: Edited document type.
        issuer: Edited issuing authority.
        reference_number: Edited reference number.
        amount_inr: Edited monetary amount in INR.
        issue_date: Edited issue date (YYYY-MM-DD).
        deadline_date: Edited deadline date (YYYY-MM-DD).
        key_facts: Edited salient facts (list or newline-delimited string).
        language_of_document: Edited document language.
        confidence: Edited or retained confidence score.
        missing_or_unclear: Edited missing items (list or newline-delimited string).

    Returns:
        Updated and validated NoticeExtraction instance.

    Raises:
        pydantic.ValidationError: If field validation fails.
    """
    # Normalize list fields
    if isinstance(key_facts, str):
        parsed_facts = [line.strip() for line in key_facts.splitlines() if line.strip()]
    else:
        parsed_facts = list(key_facts)

    if isinstance(missing_or_unclear, str):
        parsed_missing = [line.strip() for line in missing_or_unclear.splitlines() if line.strip()]
    else:
        parsed_missing = list(missing_or_unclear)

    # Clean text fields
    clean_issuer = issuer.strip() if issuer and issuer.strip() else None
    clean_ref = reference_number.strip() if reference_number and reference_number.strip() else None
    clean_issue_date = issue_date.strip() if issue_date and issue_date.strip() else None
    clean_deadline = deadline_date.strip() if deadline_date and deadline_date.strip() else None

    # Clean amount
    clean_amount: Optional[float] = None
    if amount_inr is not None:
        clean_amount = float(amount_inr)

    # Document type
    target_type = DocumentType(doc_type) if isinstance(doc_type, str) else doc_type

    return NoticeExtraction(
        doc_type=target_type,
        issuer=clean_issuer,
        reference_number=clean_ref,
        amount_inr=clean_amount,
        issue_date=clean_issue_date,
        deadline_date=clean_deadline,
        key_facts=parsed_facts,
        language_of_document=str(language_of_document).strip() or "English",
        confidence=float(confidence),
        missing_or_unclear=parsed_missing,
        is_mock=current.is_mock,
    )


def render_preview(file_bytes: bytes, filename: str) -> None:
    """Render image preview or PDF first page in the Streamlit UI."""
    mime_type = infer_mime_type(file_bytes=file_bytes, filename=filename)

    if mime_type.startswith("image/"):
        try:
            image = Image.open(io.BytesIO(file_bytes))
            st.image(image, caption=f"{filename} (Image Preview)", use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render image preview: {exc}")
    elif mime_type == "application/pdf":
        try:
            png_bytes = render_pdf_first_page_as_png(file_bytes)
            image = Image.open(io.BytesIO(png_bytes))
            st.image(
                image,
                caption=f"{filename} (PDF Page 1 Preview)",
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"📄 PDF Document: **{filename}** ({len(file_bytes):,} bytes)")
            st.caption(f"PyMuPDF preview unavailable: {exc}")
    else:
        st.info(f"📁 Document: **{filename}** ({len(file_bytes):,} bytes, MIME: {mime_type})")


def render_abstention_card(t: Dict[str, str]) -> None:
    """Display the F5 Abstention & Safety Card callout."""
    st.error(f"### {t['abstention_title']}")
    st.warning(t["abstention_warning"])
    st.markdown(f"**{t['abstention_bring_header']}**")
    st.markdown(f"- **{t['abstention_item1']}**")
    st.markdown(f"- **{t['abstention_item2']}**")
    st.markdown(f"- **{t['abstention_item3']}**")
    st.info(f"ℹ️ *{t['abstention_disclaimer']}*")


def render_phase2_placeholders(t: Dict[str, str]) -> None:
    """Display disabled/scheduled Phase 2 action module tabs."""
    st.markdown(f"### {t['phase2_header']}")
    tab_f3, tab_f4, tab_f6, tab_f7 = st.tabs(
        [
            t["p2_f3_title"],
            t["p2_f4_title"],
            t["p2_f6_title"],
            t["p2_f7_title"],
        ]
    )

    with tab_f3:
        st.info(
            f"**{t['p2_f3_title']}**\n\n"
            f"{t['p2_f3_desc']}\n\n"
            f"📌 *Status: Scheduled for Phase 2 integration with RAG statutory knowledge base.*"
        )
    with tab_f4:
        st.info(
            f"**{t['p2_f4_title']}**\n\n"
            f"{t['p2_f4_desc']}\n\n"
            f"📌 *Status: Scheduled for Phase 2 retrieval against statutory dispute SOPs.*"
        )
    with tab_f6:
        st.info(
            f"**{t['p2_f6_title']}**\n\n"
            f"{t['p2_f6_desc']}\n\n"
            f"📌 *Status: Scheduled for Phase 2 formal legal representation and RTI drafting.*"
        )
    with tab_f7:
        st.info(
            f"**{t['p2_f7_title']}**\n\n"
            f"{t['p2_f7_desc']}\n\n"
            f"📌 *Status: Scheduled for Phase 2 iCalendar (.ics) generation with alert triggers.*"
        )


def main() -> None:
    """Primary Streamlit application entrypoint."""
    st.set_page_config(
        page_title="NoticeWise — Citizen Notice Simplification",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state keys
    if "language" not in st.session_state:
        st.session_state["language"] = "English"
    if "extraction_result" not in st.session_state:
        st.session_state["extraction_result"] = None
    if "confirmed_extraction" not in st.session_state:
        st.session_state["confirmed_extraction"] = None
    if "save_confirmed" not in st.session_state:
        st.session_state["save_confirmed"] = False

    # Sidebar: Language Selector & Information
    with st.sidebar:
        st.title("🏛️ NoticeWise")
        selected_lang = st.selectbox(
            "Language / భాష / भाषा",
            options=LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state["language"])
            if st.session_state["language"] in LANGUAGE_OPTIONS
            else 0,
            key="language_selector",
        )
        st.session_state["language"] = selected_lang
        t = get_translations(selected_lang)

        st.divider()
        st.markdown(
            "**NoticeWise Foundation (Phase 1)**\n\n"
            "Assisting Indian citizens in understanding and responding to civic and administrative notices."
        )

    # App Header
    st.title(t["app_title"])
    st.caption(t["app_subtitle"])
    st.divider()

    # Layout: Two columns for Intake/Preview and Analysis/Form
    col_left, col_right = st.columns([1, 1], gap="large")

    active_bytes: Optional[bytes] = None
    active_filename: Optional[str] = None

    with col_left:
        st.subheader(t["intake_header"])
        intake_mode = st.radio(
            t["intake_mode_label"],
            options=[t["mode_upload"], t["mode_sample"]],
            horizontal=True,
        )

        if intake_mode == t["mode_upload"]:
            uploaded = st.file_uploader(
                t["upload_label"],
                type=["png", "jpg", "jpeg", "pdf"],
                help="Supported formats: PNG, JPG, JPEG, PDF. Maximum size: 10MB.",
            )
            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                is_valid, err_msg = validate_file_size(file_bytes)
                if not is_valid:
                    st.error(t["file_too_large"])
                    st.error(f"Details: {err_msg}")
                else:
                    active_bytes = file_bytes
                    active_filename = uploaded.name

        else:
            sample_notices = load_sample_notices()
            if sample_notices:
                display_options = [s["display_name"] for s in sample_notices]
                selected_display = st.selectbox(
                    t["sample_label"],
                    options=display_options,
                    index=0,
                )
                selected_sample = next(
                    (s for s in sample_notices if s["display_name"] == selected_display),
                    sample_notices[0],
                )
                if selected_sample["exists"]:
                    active_bytes = Path(selected_sample["file_path"]).read_bytes()
                    active_filename = selected_sample["filename"]
                else:
                    st.warning(f"Sample file not found on disk: {selected_sample['filename']}")
            else:
                st.warning("No sample notices loaded from data/test_notices/labels.json.")

        # Document Preview Section
        if active_bytes and active_filename:
            st.divider()
            st.subheader(t["preview_header"])
            render_preview(active_bytes, active_filename)

            st.write("")
            analyze_clicked = st.button(
                t["analyze_button"],
                type="primary",
                use_container_width=True,
            )
            if analyze_clicked:
                with st.spinner("Extracting structured notice details..."):
                    result = extract_notice(
                        file_input=active_bytes,
                        filename=active_filename,
                    )
                    st.session_state["extraction_result"] = result
                    st.session_state["confirmed_extraction"] = result.extraction
                    st.session_state["save_confirmed"] = False
                    st.rerun()

    with col_right:
        extraction_result: Optional[ExtractionResult] = st.session_state.get("extraction_result")
        current_extraction: Optional[NoticeExtraction] = st.session_state.get(
            "confirmed_extraction"
        )

        if current_extraction is None:
            st.info(f"👈 {t['no_file_selected']}")
            render_phase2_placeholders(t)
            return

        # Prominent MOCK Warning Banner
        if current_extraction.is_mock or (extraction_result and extraction_result.is_mock):
            st.warning(t["mock_warning"])

        # PRD Section 7 F5 Abstention & Safety Card
        if should_trigger_abstention(current_extraction.doc_type, current_extraction.confidence):
            render_abstention_card(t)

        st.subheader(t["form_header"])

        # Editable Extracted-Fields Form (PRD Section 7 F2)
        with st.form(key="extraction_form"):
            curr_type_val = (
                current_extraction.doc_type.value
                if hasattr(current_extraction.doc_type, "value")
                else str(current_extraction.doc_type)
            )
            type_index = (
                DOC_TYPE_OPTIONS.index(curr_type_val)
                if curr_type_val in DOC_TYPE_OPTIONS
                else 0
            )

            f_col1, f_col2 = st.columns(2)
            with f_col1:
                edit_doc_type = st.selectbox(
                    t["doc_type"],
                    options=DOC_TYPE_OPTIONS,
                    index=type_index,
                )
                edit_issuer = st.text_input(
                    t["issuer"],
                    value=current_extraction.issuer or "",
                )
                edit_ref = st.text_input(
                    t["reference_number"],
                    value=current_extraction.reference_number or "",
                )
                edit_amount = st.number_input(
                    t["amount_inr"],
                    min_value=0.0,
                    value=float(current_extraction.amount_inr or 0.0),
                    step=50.0,
                )
                edit_lang = st.text_input(
                    t["language_of_document"],
                    value=current_extraction.language_of_document or "English",
                )

            with f_col2:
                edit_issue_date = st.text_input(
                    t["issue_date"],
                    value=current_extraction.issue_date or "",
                    help="Format: YYYY-MM-DD",
                )
                edit_deadline_date = st.text_input(
                    t["deadline_date"],
                    value=current_extraction.deadline_date or "",
                    help="Format: YYYY-MM-DD",
                )
                edit_confidence = st.slider(
                    f"{t['confidence']} (0.0 to 1.0)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_extraction.confidence),
                    step=0.01,
                )
                st.progress(float(edit_confidence))

            edit_key_facts = st.text_area(
                t["key_facts"],
                value="\n".join(current_extraction.key_facts),
                height=110,
                help="One fact per line",
            )
            edit_missing = st.text_area(
                t["missing_or_unclear"],
                value="\n".join(current_extraction.missing_or_unclear),
                height=80,
                help="One missing or unclear item per line",
            )

            save_submitted = st.form_submit_button(
                t["save_button"],
                type="primary",
                use_container_width=True,
            )

            if save_submitted:
                try:
                    updated = save_edited_extraction(
                        current=current_extraction,
                        doc_type=edit_doc_type,
                        issuer=edit_issuer,
                        reference_number=edit_ref,
                        amount_inr=edit_amount if edit_amount > 0.0 else None,
                        issue_date=edit_issue_date,
                        deadline_date=edit_deadline_date,
                        key_facts=edit_key_facts,
                        language_of_document=edit_lang,
                        confidence=edit_confidence,
                        missing_or_unclear=edit_missing,
                    )
                    st.session_state["confirmed_extraction"] = updated
                    st.session_state["save_confirmed"] = True
                    st.rerun()
                except Exception as exc:
                    st.error(f"Validation failed: {exc}")

        if st.session_state.get("save_confirmed", False):
            st.success(t["save_success"])

        # Display Current Confirmed Extraction JSON
        with st.expander("🔍 View Confirmed Schema JSON Payload", expanded=False):
            st.json(current_extraction.model_dump())

        st.divider()
        render_phase2_placeholders(t)


if __name__ == "__main__":
    main()
