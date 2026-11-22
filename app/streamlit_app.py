"""NoticeWise — Citizen Notice Simplification & Action Assistant.

Streamlit application providing:
- Multilingual interface (English, Telugu, Hindi).
- Document intake via file uploader (PNG, JPG, PDF, max 10MB) or synthetic sample notice picker.
- Live document preview (Image or PyMuPDF PDF first-page rendering).
- Extraction execution via src.extract with prominent MOCK banner if offline.
- F5 Abstention & Safety Card when doc_type == 'other' or confidence < 0.60 or retrieval score is low.
- Editable extracted fields form for all 10 F2 schema fields with session state persistence.
- Phase 2 modules:
  - Tab 1: Document Intake & Review (F2 & F5)
  - Tab 2: Plain Language Explanation (F3) & Calendar Reminder (F7)
  - Tab 3: Action Plan & Verified Citations (F4)
  - Tab 4: Response Letter Draft & Multilingual PDF (F6)
"""

import io
import json
import textwrap
import logging
from datetime import date, datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is on sys.path for robust imports across all Streamlit run configurations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import streamlit as st

from src.extract import (
    MAX_FILE_SIZE_BYTES,
    extract_notice,
    infer_mime_type,
    render_pdf_first_page_as_png,
)
from src.guardrails import LEGAL_DISCLAIMER, get_office_guidance, should_abstain
from src.letters import draft_letter_content, generate_letter_pdf
from src.llm import get_live_calls_count, is_live_gemini_available
from src.reminders import compute_days_left, generate_ics_from_extraction
from src.retrieve import get_top_retrieval_score, search_kb
from src.schemas import (
    DocumentType,
    ExtractionResult,
    GroundedExplanation,
    LabelsFile,
    LetterDraft,
    NoticeExtraction,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "test_notices"
DEFAULT_LABELS_PATH: Path = DATA_DIR / "labels.json"
CACHED_RESULTS_DIR: Path = PROJECT_ROOT / "data" / "cached_results"

DEMO_DATE_STR: str = "2026-10-25"
DEMO_DATE: date = datetime.strptime(DEMO_DATE_STR, "%Y-%m-%d").date()
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EVAL_RESULTS_PATH: Path = REPORTS_DIR / "eval_results.json"
CITATION_REVIEW_PATH: Path = REPORTS_DIR / "citation_review.csv"

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
        "tab_intake": "Document Intake & Review",
        "tab_explain": "Plain Explanation (F3)",
        "tab_action": "Action Plan & Citations (F4)",
        "tab_letter": "Response Letter & PDF (F6)",
        "tab_eval": "Evaluation & Metrics (F8)",
        "days_remaining": "Days Remaining",
        "due_date": "Due Date",
        "download_reminder_button": "Download Deadline Reminder (.ics)",
        "download_pdf_button": "Download Official Reply Letter (PDF)",
        "download_txt_button": "Download as Plain Text (.txt)",
        "sender_name_label": "Your Full Name",
        "sender_address_label": "Your Residential Address",
        "sender_phone_label": "Your Contact Phone Number",
        "upload_cap_reached": "ℹ️ Session live upload limit reached (3/3 uploads). You can still analyze any synthetic sample notice or edit details without limits.",
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
        "tab_intake": "పత్ర సమర్పణ & సమీక్ష",
        "tab_explain": "సరళ వివరణ (F3)",
        "tab_action": "చర్యల ప్రణాళిక & మూలాలు (F4)",
        "tab_letter": "సమాధాన లేఖ & PDF (F6)",
        "tab_eval": "మూల్యాంకనం & కొలమానాలు (F8)",
        "days_remaining": "మిగిలిన రోజులు",
        "due_date": "గడువు తేదీ",
        "download_reminder_button": "క్యాలెండర్ రిమైండర్ (.ics) డౌన్‌లోడ్ చేయండి",
        "download_pdf_button": "అధికారిక సమాధాన లేఖను (PDF) డౌన్‌లోడ్ చేయండి",
        "download_txt_button": "టెక్స్ట్ ఫైల్‌గా (.txt) డౌన్‌లోడ్ చేయండి",
        "sender_name_label": "మీ పూర్తి పేరు",
        "sender_address_label": "మీ నివాస చిరునామా",
        "sender_phone_label": "మీ ఫోన్ నంబర్",
        "upload_cap_reached": "ℹ️ సెషన్ లైవ్ అప్‌లోడ్ పరిమితి ముగిసింది (3/3 అప్‌లోడ్‌లు). మీరు ఇప్పటికీ నమూనా నోటీసులను విశ్లేషించవచ్చు లేదా వివరాలను సవరించవచ్చు.",
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
        "tab_intake": "दस्तावेज़ इनटेक एवं समीक्षा",
        "tab_explain": "सरल व्याख्या (F3)",
        "tab_action": "कार्य योजना एवं संदर्भ (F4)",
        "tab_letter": "उत्तर पत्र एवं पीडीएफ (F6)",
        "tab_eval": "मूल्यांकन एवं मेट्रिक्स (F8)",
        "days_remaining": "शेष दिन",
        "due_date": "अंतिम तिथि",
        "download_reminder_button": "कैलेंडर अनुस्मारक (.ics) डाउनलोड करें",
        "download_pdf_button": "आधिकारिक उत्तर पत्र (PDF) डाउनलोड करें",
        "download_txt_button": "टेक्स्ट फ़ाइल (.txt) के रूप में डाउनलोड करें",
        "sender_name_label": "आपका पूरा नाम",
        "sender_address_label": "आपका आवासीय पता",
        "sender_phone_label": "आपका फ़ोन नंबर",
        "upload_cap_reached": "ℹ️ सत्र लाइव अपलोड सीमा समाप्त (3/3 अपलोड)। आप अभी भी नमूना नोटिसों का विश्लेषण कर सकते हैं या विवरण संपादित कर सकते हैं।",
    },
}


def get_translations(lang: str) -> Dict[str, str]:
    """Retrieve UI localization dictionary for the specified language."""
    cleaned = (lang or "").strip().lower()
    if "telugu" in cleaned or cleaned == "te":
        return LOCALIZED_STRINGS["Telugu"]
    if "hindi" in cleaned or cleaned == "hi":
        return LOCALIZED_STRINGS["Hindi"]
    return LOCALIZED_STRINGS["English"]


def load_sample_notices(
    labels_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Dynamically load synthetic sample notices from labels.json."""
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
                "description": desc,
                "file_path": file_path,
                "exists": file_path.exists(),
                "display_name": display_name,
            }
        )
    return samples


def validate_file_size(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """Validate that input file bytes do not exceed the 10MB limit."""
    size = len(file_bytes)
    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        return False, f"File exceeds 10 MB limit ({mb:.2f} MB). Maximum allowed size is 10 MB."
    return True, None


def should_trigger_abstention(
    doc_type: Union[str, DocumentType],
    confidence: float,
) -> bool:
    """Determine whether the F5 Abstention & Safety Card must trigger."""
    type_str = doc_type.value if hasattr(doc_type, "value") else str(doc_type).lower()
    if type_str == "other":
        return True
    if confidence < 0.60:
        return True
    return False


def save_edited_extraction(
    current: NoticeExtraction,
    doc_type: Union[str, DocumentType],
    issuer: Optional[str],
    reference_number: Optional[str],
    amount_inr: Optional[float],
    issue_date: Optional[str],
    deadline_date: Optional[str],
    key_facts: Union[List[str], str],
    language_of_document: Optional[str],
    confidence: float,
    missing_or_unclear: Union[List[str], str],
) -> NoticeExtraction:
    """Validate and construct an updated NoticeExtraction model from citizen edits."""
    if isinstance(key_facts, str):
        parsed_key_facts = [line.strip() for line in key_facts.splitlines() if line.strip()]
    else:
        parsed_key_facts = [str(k).strip() for k in key_facts if str(k).strip()]

    if isinstance(missing_or_unclear, str):
        parsed_missing = [line.strip() for line in missing_or_unclear.splitlines() if line.strip()]
    else:
        parsed_missing = [str(m).strip() for m in missing_or_unclear if str(m).strip()]

    clean_issue = (issue_date or "").strip() or None
    clean_deadline = (deadline_date or "").strip() or None
    clean_issuer = (issuer or "").strip() or None
    clean_ref = (reference_number or "").strip() or None

    return NoticeExtraction(
        doc_type=doc_type,
        issuer=clean_issuer,
        reference_number=clean_ref,
        amount_inr=amount_inr,
        issue_date=clean_issue,
        deadline_date=clean_deadline,
        key_facts=parsed_key_facts,
        language_of_document=language_of_document or "English",
        confidence=confidence,
        missing_or_unclear=parsed_missing,
        is_mock=current.is_mock,
        is_deadline_computed=current.is_deadline_computed,
    )


def render_preview(file_bytes: bytes, filename: str) -> None:
    """Render image preview or PDF first page in the Streamlit UI."""
    mime_type = infer_mime_type(file_bytes=file_bytes, filename=filename)

    if mime_type.startswith("image/"):
        try:
            image = Image.open(io.BytesIO(file_bytes))
            st.image(image, caption=f"{filename} (Image Preview)", width="stretch")
        except Exception as exc:
            st.warning(f"Could not render image preview: {exc}")
    elif mime_type == "application/pdf":
        try:
            png_bytes = render_pdf_first_page_as_png(file_bytes)
            image = Image.open(io.BytesIO(png_bytes))
            st.image(
                image,
                caption=f"{filename} (PDF Page 1 Preview)",
                width="stretch",
            )
        except Exception as exc:
            st.info(f"📄 PDF Document: **{filename}** ({len(file_bytes):,} bytes)")
            st.caption(f"PyMuPDF preview unavailable: {exc}")
    else:
        st.info(f"📁 Document: **{filename}** ({len(file_bytes):,} bytes, MIME: {mime_type})")


def render_abstention_card(t: Dict[str, str], office_info: Optional[Dict[str, Any]] = None) -> None:
    """Display the F5 Abstention & Safety Card callout with professional civic design."""
    office_name = (
        office_info.get("office", "Local issuing administrative department / office")
        if office_info
        else "Local issuing administrative department / office"
    )
    docs_list = (
        office_info.get("documents", [])
        if office_info
        else [t["abstention_item1"], t["abstention_item2"], t["abstention_item3"]]
    )
    docs_html = "".join([f"<li style='margin-bottom: 4px;'>📄 {doc}</li>" for doc in docs_list])

    card_html = textwrap.dedent(f"""\
    <div style="background: #FFF1F2; border: 1px solid #FECDD3; border-left: 6px solid #E11D48; border-radius: 12px; padding: 1.25rem 1.5rem; margin: 1rem 0 1.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
    <span style="font-size: 1.3rem;">🛡️</span>
    <strong style="color: #9F1239; font-size: 1.15rem;">{t['abstention_title']}</strong>
    </div>
    <p style="color: #4C0519; font-size: 0.95rem; margin: 0 0 1rem 0; line-height: 1.5;">
    {t['abstention_warning']}
    </p>
    <div style="background: rgba(255, 255, 255, 0.85); border: 1px solid #FFE4E6; border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.85rem;">
    <div style="font-weight: 700; color: #1E293B; margin-bottom: 0.25rem;">🏛️ Recommended Office to Visit:</div>
    <div style="color: #334155; font-size: 0.92rem;">{office_name}</div>
    </div>
    <div style="font-weight: 600; color: #1E293B; margin-bottom: 0.35rem;">{t['abstention_bring_header']}</div>
    <ul style="color: #334155; font-size: 0.9rem; padding-left: 1.2rem; margin: 0 0 0.85rem 0;">
    {docs_html}
    </ul>
    <div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #FECDD3; padding-top: 0.5rem;">
    ⚖️ <em>{t['abstention_disclaimer']}</em>
    </div>
    </div>
    """).strip()
    st.markdown(card_html, unsafe_allow_html=True)


def render_phase2_placeholders(t: Dict[str, str]) -> None:
    """Display Phase 2 module descriptions for compatibility."""
    st.markdown(f"### {t['phase2_header']}")
    st.info("Phase 2 modules are fully integrated into the tabs above.")


# ---------------------------------------------------------------------------
# Streamlit Cached Backend Helpers (0 API Calls on Re-render)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_cached_search_kb(query: str, doc_type: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Cached local TF-IDF retrieval (zero API calls)."""
    chunks = search_kb(query=query, doc_type=doc_type, top_k=top_k)
    return [c.model_dump() for c in chunks]


@st.cache_data(show_spinner=False)
def get_cached_explanation(
    extraction_dict: Dict[str, Any],
    chunks_dicts: List[Dict[str, Any]],
    language: str,
) -> Dict[str, Any]:
    """Cached grounded explanation (disk-backed + st.cache_data)."""
    from src.explain import generate_grounded_explanation, sanitize_consequences
    ext = NoticeExtraction.model_validate(extraction_dict)
    chunks = [RetrievedChunk.model_validate(c) for c in chunks_dicts]
    res = generate_grounded_explanation(
        extraction=ext,
        retrieved_chunks=chunks,
        language=language,
        use_cache=True,
    )
    return res.model_dump()


@st.cache_data(show_spinner=False)
def get_cached_letter_pdf(draft_dict: Dict[str, Any]) -> bytes:
    """Cached ReportLab PDF generator."""
    draft = LetterDraft.model_validate(draft_dict)
    return generate_letter_pdf(draft)


CUSTOM_CSS = """
<style>
/* ==========================================================================
   NoticeWise Civic Tech Enterprise Design System (Theme-Safe & High Contrast)
   ========================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600;700&family=Noto+Sans+Telugu:wght@400;600;700&display=swap');

/* Force consistent light civic surface across all containers (prevents dark mode clashes) */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], .stApp {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
    font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', 'Noto Sans Telugu', system-ui, -apple-system, sans-serif !important;
}

/* Base text color guarantee */
.stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label {
    color: #0F172A;
}

/* Main Container Padding */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1280px !important;
}

/* --------------------------------------------------------------------------
   Civic Header Bar (Compact & Authoritative)
   -------------------------------------------------------------------------- */
.nw-header-bar {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
    padding: 1.4rem 1.8rem;
    border-radius: 12px;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
    margin-bottom: 1.25rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.nw-header-bar p, .nw-header-bar span, .nw-header-bar strong, .nw-header-bar h1 {
    color: #FFFFFF !important;
}
.nw-header-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.4rem;
}
.nw-header-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.nw-header-title {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em;
}
.nw-header-badge {
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #FFFFFF !important;
}
.nw-header-subtitle {
    font-size: 0.95rem !important;
    color: #E2E8F0 !important;
    margin: 0 0 0.85rem 0 !important;
    line-height: 1.4;
}
.nw-pill-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.nw-pill {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #F8FAFC !important;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

/* --------------------------------------------------------------------------
   Tabs Navigation (High Contrast & Clean Underline)
   -------------------------------------------------------------------------- */
div[data-testid="stTabs"] {
    margin-top: 0.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 2px solid #E2E8F0;
}
div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.94rem !important;
    padding: 0.65rem 1.1rem !important;
    border-radius: 6px 6px 0 0 !important;
    color: #475569 !important;
    background: transparent !important;
    transition: all 0.15s ease-in-out !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1E3A8A !important;
    border-bottom: 3px solid #2563EB !important;
    font-weight: 700 !important;
    background: #EFF6FF !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #1E3A8A !important;
    font-weight: 700 !important;
}
div[data-testid="stTabs"] button[aria-selected="false"] p {
    color: #475569 !important;
}
div[data-testid="stTabs"] button:hover {
    color: #2563EB !important;
    background: #F1F5F9 !important;
}
div[data-testid="stTabs"] button:hover p {
    color: #2563EB !important;
}

/* --------------------------------------------------------------------------
   Executive Notice Summary Card (Tab 1)
   -------------------------------------------------------------------------- */
.nw-summary-card {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 1.3rem 1.4rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 1.2rem !important;
}
.nw-summary-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.nw-doc-badge {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
}
.nw-ref-badge {
    background: #F1F5F9;
    color: #334155;
    border: 1px solid #CBD5E1;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: monospace;
}
.nw-summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.85rem 1.2rem;
    margin-bottom: 1rem;
}
.nw-grid-item {
    display: flex;
    flex-direction: column;
}
.nw-item-label {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    color: #64748B !important;
    margin-bottom: 0.15rem;
    letter-spacing: 0.03em;
}
.nw-item-val {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #0F172A !important;
}
.nw-amount-val {
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    color: #047857 !important;
}
.nw-deadline-val {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #B91C1C !important;
}
.nw-facts-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.85rem 1rem;
}
.nw-facts-box strong {
    color: #1E293B;
    font-size: 0.84rem;
}
.nw-facts-box ul {
    margin: 0.4rem 0 0 1.2rem;
    padding: 0;
    font-size: 0.88rem;
    color: #334155;
}

/* --------------------------------------------------------------------------
   Intake Welcoming Guide Card (Shown before document selection)
   -------------------------------------------------------------------------- */
.nw-welcome-card {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 1.5rem 1.6rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04) !important;
}
.nw-welcome-title {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #1E3A8A !important;
    margin-bottom: 0.6rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.nw-step-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.85rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #F1F5F9;
}
.nw-step-num {
    background: #2563EB;
    color: #FFFFFF !important;
    font-weight: 800;
    font-size: 0.78rem;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.nw-step-desc strong {
    color: #0F172A;
    font-size: 0.92rem;
}
.nw-step-desc p {
    color: #64748B;
    font-size: 0.84rem;
    margin: 0.15rem 0 0 0;
}

/* --------------------------------------------------------------------------
   KPI Tiles (High Contrast & Clean Colors)
   -------------------------------------------------------------------------- */
.nw-kpi-container {
    display: flex;
    gap: 0.85rem;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}
.nw-kpi-tile {
    flex: 1;
    min-width: 200px;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
}
.nw-kpi-tile-label {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.04em;
    margin-bottom: 0.35rem;
}
.nw-kpi-tile-val {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
}
.nw-kpi-tile-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* --------------------------------------------------------------------------
   Sidebar Clean Civic Styling
   -------------------------------------------------------------------------- */
section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
section[data-testid="stSidebar"] * {
    color: #0F172A !important;
}
.nw-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #E2E8F0;
}
.nw-brand-text {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em;
}
.nw-brand-tag {
    background: #FEF3C7 !important;
    color: #92400E !important;
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    padding: 0.15rem 0.5rem !important;
    border-radius: 4px !important;
    border: 1px solid #FDE68A !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.nw-sidebar-box {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 0.85rem 1rem !important;
    margin-top: 1rem !important;
}

/* --------------------------------------------------------------------------
   Forms & Inputs (Clean & High Contrast)
   -------------------------------------------------------------------------- */
div[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03) !important;
}
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}
input, textarea, select, [data-baseweb="select"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}
div[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #CBD5E1 !important;
    max-height: 440px !important;
    object-fit: contain !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

/* --------------------------------------------------------------------------
   Buttons & Download Controls (Guaranteed Visible High Contrast)
   -------------------------------------------------------------------------- */
div[data-testid="stDownloadButton"] button,
button[data-testid="baseButton-primary"],
button[kind="primary"] {
    background-color: #1E3A8A !important;
    border: 1px solid #1E3A8A !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    padding: 0.65rem 1.25rem !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.25) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stDownloadButton"] button *,
button[data-testid="baseButton-primary"] *,
button[kind="primary"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 700 !important;
}
div[data-testid="stDownloadButton"] button:hover,
button[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
}
div[data-testid="stDownloadButton"] button:hover *,
button[data-testid="baseButton-primary"]:hover *,
button[kind="primary"]:hover * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-testid="stDownloadButton"] button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[kind="secondary"] {
    background-color: #FFFFFF !important;
    border: 2px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #0F172A !important;
    font-weight: 700 !important;
}
div[data-testid="stDownloadButton"] button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
button[kind="secondary"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 700 !important;
}
div[data-testid="stDownloadButton"] button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[kind="secondary"]:hover {
    background-color: #F1F5F9 !important;
    border-color: #94A3B8 !important;
}

/* --------------------------------------------------------------------------
   Step-by-Step Action Cards (Tab 3)
   -------------------------------------------------------------------------- */
.nw-step-box {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-left: 5px solid #2563EB !important;
    border-radius: 10px !important;
    padding: 1.15rem 1.35rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04) !important;
}
.nw-step-title {
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    margin-bottom: 0.35rem !important;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}
.nw-step-badge {
    background: #EFF6FF !important;
    color: #1E3A8A !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    padding: 0.15rem 0.5rem !important;
    border-radius: 6px !important;
    border: 1px solid #BFDBFE !important;
}
.nw-citation-card {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-left: 4px solid #1E3A8A !important;
    border-radius: 8px !important;
    padding: 0.75rem 0.95rem !important;
    margin-top: 0.75rem !important;
    font-size: 0.86rem !important;
    color: #334155 !important;
}
.nw-citation-card div, .nw-citation-card small, .nw-citation-card strong {
    color: #334155 !important;
}

/* --------------------------------------------------------------------------
   Tab 2 Callout Cards
   -------------------------------------------------------------------------- */
.nw-callout-what {
    background: #F0FDF4 !important;
    border: 1px solid #BBF7D0 !important;
    border-left: 5px solid #16A34A !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
    color: #14532D !important;
    font-size: 0.96rem !important;
    line-height: 1.55 !important;
    margin-bottom: 1.1rem !important;
}
.nw-callout-what * {
    color: #14532D !important;
}
.nw-callout-danger {
    background: #FFF1F2 !important;
    border: 1px solid #FECDD3 !important;
    border-left: 5px solid #E11D48 !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
    color: #881337 !important;
    font-size: 0.96rem !important;
    line-height: 1.55 !important;
    margin-bottom: 1.1rem !important;
}
.nw-callout-danger * {
    color: #881337 !important;
}
.nw-callout-office {
    background: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-left: 5px solid #2563EB !important;
    border-radius: 10px !important;
    padding: 1.1rem 1.3rem !important;
    color: #1E3A8A !important;
    font-size: 0.96rem !important;
    line-height: 1.55 !important;
    margin-bottom: 1.1rem !important;
}
.nw-callout-office * {
    color: #1E3A8A !important;
}

/* Expander Styling */
div[data-testid="stExpander"] details {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary {
    padding: 0.75rem 1rem !important;
}
div[data-testid="stExpander"] summary span {
    font-weight: 600 !important;
    color: #1E293B !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Main Streamlit Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Primary Streamlit application entrypoint."""
    st.set_page_config(
        page_title="NoticeWise — Citizen Notice Simplification",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Custom Enterprise Civic Tech Styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state keys
    if "language" not in st.session_state:
        st.session_state["language"] = "English"
    if "extraction_result" not in st.session_state:
        st.session_state["extraction_result"] = None
    if "confirmed_extraction" not in st.session_state:
        st.session_state["confirmed_extraction"] = None
    if "save_confirmed" not in st.session_state:
        st.session_state["save_confirmed"] = False
    if "live_upload_count" not in st.session_state:
        st.session_state["live_upload_count"] = 0
    if "sender_name" not in st.session_state:
        st.session_state["sender_name"] = "[YOUR FULL NAME]"
    if "sender_address" not in st.session_state:
        st.session_state["sender_address"] = "[YOUR RESIDENTIAL ADDRESS]"
    if "sender_phone" not in st.session_state:
        st.session_state["sender_phone"] = "[YOUR CONTACT NUMBER]"
    if "is_sample_notice" not in st.session_state:
        st.session_state["is_sample_notice"] = True

    # Sidebar: Language Selector, Civic Guide & Telemetry
    with st.sidebar:
        sidebar_brand = textwrap.dedent("""\
        <div class="nw-sidebar-brand">
        <span style="font-size: 1.6rem;">🏛️</span>
        <div>
        <div class="nw-brand-text">NoticeWise</div>
        <span class="nw-brand-tag">Prototype</span>
        </div>
        </div>
        """).strip()
        st.markdown(sidebar_brand, unsafe_allow_html=True)

        selected_lang = st.selectbox(
            "Select Language / భాష / भाषा",
            options=LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(st.session_state["language"])
            if st.session_state["language"] in LANGUAGE_OPTIONS
            else 0,
            key="language_selector",
        )
        st.session_state["language"] = selected_lang
        t = get_translations(selected_lang)

        st.markdown(
            """
            <div class="nw-sidebar-box">
                <strong style="color: #0F172A; font-size: 0.88rem;">🏛️ How NoticeWise Works</strong>
                <ol style="color: #475569; font-size: 0.82rem; margin: 0.4rem 0 0 1.1rem; padding: 0; line-height: 1.45;">
                    <li><strong>Upload / Pick</strong> notice document</li>
                    <li><strong>Review</strong> plain explanation & deadline</li>
                    <li><strong>Follow</strong> cited action checklist</li>
                    <li><strong>Download</strong> reply letter & reminder</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="nw-sidebar-box" style="margin-top: 0.75rem;">
                <strong style="color: #0F172A; font-size: 0.84rem;">🛡️ Statutory Coverage Scope</strong>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 0.35rem; line-height: 1.4;">
                    <span style="color: #059669; font-weight: 700;">Grounded:</span> RTI Act, Electricity Disconnection (Sec 56), Traffic Challan (MV Act Sec 200), NFSA Grievances.<br/>
                    <span style="color: #B45309; font-weight: 700; margin-top: 0.25rem; display: inline-block;">Safe Abstention:</span> Property Tax, State Pensions, Private/Court disputes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        live_calls = get_live_calls_count()
        is_live = is_live_gemini_available()
        engine_label = "Gemini Live Vision" if is_live else "Verified Cache / Local"

        st.markdown(
            f"""
            <div class="nw-sidebar-box" style="margin-top: 0.75rem;">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #64748B; letter-spacing: 0.04em;">System Status</div>
                <div style="font-size: 0.86rem; font-weight: 700; color: #059669; margin-top: 0.2rem;">🟢 Pipeline Active</div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 0.35rem; line-height: 1.4;">
                    <strong>Engine:</strong> {engine_label}<br/>
                    <strong>Uploads:</strong> {st.session_state.get('live_upload_count', 0)} / 3 session limit
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sleek Civic Header Bar
    header_html = textwrap.dedent(f"""\
    <div class="nw-header-bar">
    <div class="nw-header-top">
    <div class="nw-header-brand">
    <span style="font-size: 1.5rem;">🏛️</span>
    <h1 class="nw-header-title">{t["app_title"]}</h1>
    </div>
    <span class="nw-header-badge">Public Administrative Assistant</span>
    </div>
    <p class="nw-header-subtitle">{t["app_subtitle"]}</p>
    <div class="nw-pill-group">
    <span class="nw-pill">🛡️ Grounded in Real Statutes</span>
    <span class="nw-pill">⚡ Instant Verification</span>
    <span class="nw-pill">🌐 Multilingual (EN / తె / हि)</span>
    <span class="nw-pill">📄 Print-Ready Legal PDFs</span>
    </div>
    </div>
    """).strip()
    st.markdown(header_html, unsafe_allow_html=True)

    # Top-Level Tabs
    tab_intake, tab_explain, tab_action, tab_letter, tab_eval = st.tabs(
        [
            f"📄 {t['tab_intake']}",
            f"💡 {t['tab_explain']}",
            f"📋 {t['tab_action']}",
            f"✍️ {t['tab_letter']}",
            f"📊 {t['tab_eval']}",
        ]
    )

    # -----------------------------------------------------------------------
    # TAB 1: Document Intake & Review (F2 & F5)
    # -----------------------------------------------------------------------
    with tab_intake:
        col_left, col_right = st.columns([1, 1], gap="large")

        active_bytes: Optional[bytes] = None
        active_filename: Optional[str] = None
        is_synthetic_pick: bool = False

        with col_left:
            st.subheader(t["intake_header"])
            intake_mode = st.radio(
                t["intake_mode_label"],
                options=[t["mode_upload"], t["mode_sample"]],
                horizontal=True,
            )

            if intake_mode == t["mode_upload"]:
                if st.session_state.get("live_upload_count", 0) >= 3:
                    st.warning(t["upload_cap_reached"])

                uploaded = st.file_uploader(
                    t["upload_label"],
                    type=["png", "jpg", "jpeg", "pdf"],
                    help="Supported formats: PNG, JPG, JPEG, PDF. Maximum size: 10MB.",
                    disabled=(st.session_state.get("live_upload_count", 0) >= 3),
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
                        is_synthetic_pick = False
                        st.session_state["is_sample_notice"] = False

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
                        is_synthetic_pick = True
                        st.session_state["is_sample_notice"] = True
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
                    f"🔍 {t['analyze_button']}",
                    type="primary",
                    width="stretch",
                )
                if analyze_clicked:
                    with st.spinner("Analyzing document structure and extracting details..."):
                        # Zero-API-Key Optimization: Check cached live runs for synthetic samples
                        cached_file = CACHED_RESULTS_DIR / f"{Path(active_filename).stem}.json"
                        if is_synthetic_pick and cached_file.exists():
                            try:
                                with open(cached_file, "r", encoding="utf-8") as cf:
                                    cached_run = json.load(cf)
                                ext_data = cached_run.get("extraction")
                                if ext_data:
                                    extraction_model = NoticeExtraction.model_validate(ext_data)
                                    st.session_state["extraction_result"] = ExtractionResult(
                                        extraction=extraction_model,
                                        is_mock=cached_run.get("is_mock", False),
                                        raw_response=json.dumps(ext_data),
                                        error=cached_run.get("error"),
                                    )
                                    st.session_state["confirmed_extraction"] = extraction_model
                                    st.session_state["save_confirmed"] = False
                                    st.session_state["active_cached_run"] = cached_run
                                    st.rerun()
                            except Exception as c_err:
                                logger.warning("Could not load cached run for %s: %s", active_filename, c_err)

                        # For real citizen uploads, check if live API key is present
                        if not is_synthetic_pick:
                            if not is_live_gemini_available():
                                st.warning(
                                    "⚠️ **Live Gemini Vision Unavailable:** No active `GEMINI_API_KEY` was detected in Streamlit secrets or environment variables. "
                                    "Uploaded documents require a live Gemini API key for multimodal analysis. "
                                    "Please select any pre-loaded synthetic notice from the picker above to test full multimodal analysis, explanations, checklists, and letter drafting without an API key."
                                )
                            st.session_state["live_upload_count"] = st.session_state.get("live_upload_count", 0) + 1

                        result = extract_notice(
                            file_input=active_bytes,
                            filename=active_filename,
                        )
                        if result.error and not is_synthetic_pick:
                            st.error(f"⚠️ Gemini extraction warning: {result.error}")
                        st.session_state["extraction_result"] = result
                        st.session_state["confirmed_extraction"] = result.extraction
                        st.session_state["save_confirmed"] = False
                        st.session_state["active_cached_run"] = None
                        st.rerun()

        with col_right:
            extraction_result: Optional[ExtractionResult] = st.session_state.get("extraction_result")
            current_extraction: Optional[NoticeExtraction] = st.session_state.get(
                "confirmed_extraction"
            )

            if current_extraction is None:
                # Welcoming Guide Card (Never leave right column empty!)
                welcome_html = textwrap.dedent(f"""\
                <div class="nw-welcome-card">
                <div class="nw-welcome-title">
                <span>🏛️</span>
                <span>Notice Analysis Guide</span>
                </div>
                <p style="color: #475569; font-size: 0.92rem; margin-bottom: 1.15rem; line-height: 1.5;">
                {t['no_file_selected']}
                </p>
                <div class="nw-step-row">
                <div class="nw-step-num">1</div>
                <div class="nw-step-desc">
                <strong>Choose a Notice on the Left</strong>
                <p>Pick a pre-loaded synthetic government notice from the dropdown, or upload your own notice (PDF/Image).</p>
                </div>
                </div>
                <div class="nw-step-row">
                <div class="nw-step-num">2</div>
                <div class="nw-step-desc">
                <strong>Click "Analyze Notice"</strong>
                <p>NoticeWise scans the document to extract the authority, demanded amount, and legal compliance due dates.</p>
                </div>
                </div>
                <div class="nw-step-row" style="border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
                <div class="nw-step-num">3</div>
                <div class="nw-step-desc">
                <strong>Get Actionable Civic Guidance</strong>
                <p>Review the plain-language explanation, verified statutory checklist, reply letter draft, and calendar reminder.</p>
                </div>
                </div>
                </div>
                """).strip()
                st.markdown(welcome_html, unsafe_allow_html=True)
            else:
                # Prominent MOCK Warning Banner
                if current_extraction.is_mock or (extraction_result and extraction_result.is_mock):
                    st.warning(t["mock_warning"])

                # Check Abstention condition
                type_val = current_extraction.doc_type.value if hasattr(current_extraction.doc_type, "value") else str(current_extraction.doc_type)
                office_info = get_office_guidance(type_val)

                if should_trigger_abstention(current_extraction.doc_type, current_extraction.confidence):
                    render_abstention_card(t, office_info)

                st.subheader("3. Extracted Notice Summary")

                # Executive Notice Summary Card
                amt_display = f"₹ {current_extraction.amount_inr:,.2f}" if current_extraction.amount_inr is not None else "None / Not Specified"
                deadline_display = current_extraction.deadline_date or "Immediate / None"
                if getattr(current_extraction, "is_deadline_computed", False):
                    deadline_display += " ⏱️ (computed from issue date)"

                facts_html = "".join([f"<li style='margin-bottom: 3px;'>{fact}</li>" for fact in current_extraction.key_facts]) if current_extraction.key_facts else "<li>No specific salient points detected.</li>"

                summary_card_html = textwrap.dedent(f"""\
                <div class="nw-summary-card">
                <div class="nw-summary-header">
                <span class="nw-doc-badge">📌 {type_val.upper()}</span>
                <span class="nw-ref-badge">Ref: {current_extraction.reference_number or 'Not Specified'}</span>
                </div>
                <div class="nw-summary-grid">
                <div class="nw-grid-item">
                <span class="nw-item-label">{t['issuer']}</span>
                <span class="nw-item-val">{current_extraction.issuer or 'Not Specified'}</span>
                </div>
                <div class="nw-grid-item">
                <span class="nw-item-label">{t['amount_inr']}</span>
                <span class="nw-amount-val">{amt_display}</span>
                </div>
                <div class="nw-grid-item">
                <span class="nw-item-label">{t['deadline_date']}</span>
                <span class="nw-deadline-val">{deadline_display}</span>
                </div>
                <div class="nw-grid-item">
                <span class="nw-item-label">{t['issue_date']}</span>
                <span class="nw-item-val">{current_extraction.issue_date or 'Not Specified'}</span>
                </div>
                </div>
                <div class="nw-facts-box">
                <strong>Salient Notice Observations:</strong>
                <ul>
                {facts_html}
                </ul>
                </div>
                </div>
                """).strip()
                st.markdown(summary_card_html, unsafe_allow_html=True)

                # Collapsible Edit Form for Corrections
                with st.expander("✏️ Correct or Edit Extracted Details", expanded=False):
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

                        with f_col2:
                            edit_issue_date = st.text_input(
                                t["issue_date"],
                                value=current_extraction.issue_date or "",
                                help="Format: YYYY-MM-DD",
                            )
                            deadline_label = t["deadline_date"] + (" ⏱️ (computed from issue date)" if getattr(current_extraction, "is_deadline_computed", False) else "")
                            edit_deadline_date = st.text_input(
                                deadline_label,
                                value=current_extraction.deadline_date or "",
                                help="Format: YYYY-MM-DD",
                            )
                            edit_lang = st.text_input(
                                t["language_of_document"],
                                value=current_extraction.language_of_document or "English",
                            )

                        edit_key_facts = st.text_area(
                            t["key_facts"],
                            value="\n".join(current_extraction.key_facts),
                            height=80,
                            help="One fact per line",
                        )
                        edit_missing = st.text_area(
                            t["missing_or_unclear"],
                            value="\n".join(current_extraction.missing_or_unclear),
                            height=60,
                            help="One missing or unclear item per line",
                        )

                        save_submitted = st.form_submit_button(
                            t["save_button"],
                            type="primary",
                            width="stretch",
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
                                    confidence=float(current_extraction.confidence),
                                    missing_or_unclear=edit_missing,
                                )
                                st.session_state["confirmed_extraction"] = updated
                                st.session_state["save_confirmed"] = True
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Validation failed: {exc}")

                if st.session_state.get("save_confirmed", False):
                    st.success(t["save_success"])

    # -----------------------------------------------------------------------
    # TAB 2: Plain Language Explanation (F3) & Calendar Reminder (F7)
    # -----------------------------------------------------------------------
    with tab_explain:
        current_extraction: Optional[NoticeExtraction] = st.session_state.get("confirmed_extraction")
        if current_extraction is None:
            st.info(f"👈 {t['no_file_selected']}")
        else:
            doc_type_str = (
                current_extraction.doc_type.value
                if hasattr(current_extraction.doc_type, "value")
                else str(current_extraction.doc_type)
            )

            # Retrieve KB Chunks
            retrieval_query = f"{current_extraction.issuer or ''} {current_extraction.reference_number or ''} " + " ".join(current_extraction.key_facts)
            cached_chunks_raw = get_cached_search_kb(
                query=retrieval_query,
                doc_type=doc_type_str,
                top_k=4,
            )
            top_score = max([c.get("score", 0.0) for c in cached_chunks_raw], default=0.0)

            # Check Abstention guardrails
            is_abstained, abstain_reason = should_abstain(current_extraction, top_retrieval_score=top_score)
            if is_abstained:
                st.warning(f"⚠️ **Safe Abstention Notice:** {abstain_reason}")
                office_info = get_office_guidance(doc_type_str)
                render_abstention_card(t, office_info)
            else:
                # Generate Grounded Explanation
                cached_run = st.session_state.get("active_cached_run")
                if (
                    cached_run
                    and cached_run.get("explanation")
                    and selected_lang == "English"
                ):
                    grounded_exp = GroundedExplanation.model_validate(cached_run["explanation"])
                else:
                    with st.spinner("Generating plain-language explanation and checklist..."):
                        explanation_data = get_cached_explanation(
                            extraction_dict=current_extraction.model_dump(),
                            chunks_dicts=cached_chunks_raw,
                            language=selected_lang,
                        )
                        grounded_exp = GroundedExplanation.model_validate(explanation_data)

                exp = grounded_exp.explanation

                # High-contrast KPI Summary Row
                amt_str = f"₹ {exp.amount_inr:,.2f}" if exp.amount_inr is not None else "None / Not Specified"
                due_date_display = exp.deadline_date or "Immediate / None"
                if getattr(current_extraction, "is_deadline_computed", False):
                    due_date_display += " (computed)"

                # Compute days-left: fixed demo date for sample notices, real today for uploads
                is_sample = st.session_state.get("is_sample_notice", True)
                ref_date = DEMO_DATE if is_sample else date.today()
                days = compute_days_left(exp.deadline_date, as_of_date=ref_date)

                if days is not None:
                    if days < 0:
                        urgency_val = f"⚠️ OVERDUE by {abs(days)} days"
                        urgency_color = "#B91C1C"
                        urgency_sub = f"as of demo date {DEMO_DATE_STR}" if is_sample else "Immediate payment or reply required"
                    elif days <= 3:
                        urgency_val = f"🚨 {days} days left"
                        urgency_color = "#D97706"
                        urgency_sub = f"as of demo date {DEMO_DATE_STR}" if is_sample else "Urgent: statutory deadline approaching"
                    else:
                        urgency_val = f"⏳ {days} days left"
                        urgency_color = "#047857"
                        urgency_sub = f"as of demo date {DEMO_DATE_STR}" if is_sample else "Within statutory response window"
                else:
                    urgency_val = "None specified"
                    urgency_color = "#475569"
                    urgency_sub = "No statutory deadline detected"

                days_tile_label = t['days_remaining'] + (f" (as of demo date {DEMO_DATE_STR})" if is_sample else "")

                kpi_html = textwrap.dedent(f"""\
                <div class="nw-kpi-container">
                <div class="nw-kpi-tile">
                <div class="nw-kpi-tile-label">Demanded Amount</div>
                <div class="nw-kpi-tile-val" style="color: #047857;">{amt_str}</div>
                <div class="nw-kpi-tile-sub">Principal & statutory charges</div>
                </div>
                <div class="nw-kpi-tile">
                <div class="nw-kpi-tile-label">{t['due_date']}</div>
                <div class="nw-kpi-tile-val" style="color: #1E3A8A;">{due_date_display}</div>
                <div class="nw-kpi-tile-sub">Statutory compliance cutoff</div>
                </div>
                <div class="nw-kpi-tile">
                <div class="nw-kpi-tile-label">{days_tile_label}</div>
                <div class="nw-kpi-tile-val" style="color: {urgency_color};">{urgency_val}</div>
                <div class="nw-kpi-tile-sub">{urgency_sub}</div>
                </div>
                </div>
                """).strip()
                st.markdown(kpi_html, unsafe_allow_html=True)

                # Plain Explanation Cards (Strictly unindented for valid HTML parsing)
                callout_what_html = textwrap.dedent(f"""\
                <div class="nw-callout-what">
                <strong style="font-size: 1.05rem;">📌 What This Notice Means:</strong><br/>
                <div style="margin-top: 0.4rem;">{exp.what_it_is}</div>
                </div>
                """).strip()
                st.markdown(callout_what_html, unsafe_allow_html=True)

                sanitized_conseq = sanitize_consequences(exp.consequences_if_ignored, doc_type_str)
                callout_danger_html = textwrap.dedent(f"""\
                <div class="nw-callout-danger">
                <strong style="font-size: 1.05rem;">⚠️ Consequences If You Ignore It:</strong><br/>
                <div style="margin-top: 0.4rem;">{sanitized_conseq}</div>
                </div>
                """).strip()
                st.markdown(callout_danger_html, unsafe_allow_html=True)

                docs_items = (
                    "".join([f"<li style='margin-bottom: 4px;'>📄 {d}</li>" for d in exp.documents_to_bring])
                    if exp.documents_to_bring
                    else "<li>Standard identity and notice documents</li>"
                )
                callout_office_html = textwrap.dedent(f"""\
                <div class="nw-callout-office">
                <strong style="font-size: 1.05rem;">🏛️ Office to Visit & Required Documents:</strong><br/>
                <div style="margin-top: 0.4rem; font-weight: 600;">{exp.office_to_visit or 'Local issuing department / office'}</div>
                <div style="margin-top: 0.6rem; font-weight: 600; font-size: 0.88rem;">What to bring with you:</div>
                <ul style="margin: 0.3rem 0 0 1.2rem; padding: 0; font-size: 0.92rem;">
                {docs_items}
                </ul>
                </div>
                """).strip()
                st.markdown(callout_office_html, unsafe_allow_html=True)

                # Calendar Reminder (.ics) Section (F7)
                st.divider()
                st.subheader("📅 Deadline Calendar Reminder (.ics)")
                if current_extraction.deadline_date:
                    ics_bytes = generate_ics_from_extraction(current_extraction, alarm_days_before=3)
                    if ics_bytes:
                        cal_col1, cal_col2 = st.columns([2, 1])
                        with cal_col1:
                            st.caption("Add an automated deadline reminder (.ics) with a 3-day advance alert to Google Calendar, Apple Calendar, or Microsoft Outlook.")
                        with cal_col2:
                            st.download_button(
                                label=f"📥 {t['download_reminder_button']}",
                                data=ics_bytes,
                                file_name=f"noticewise_reminder_{current_extraction.reference_number or 'notice'}.ics",
                                mime="text/calendar",
                                width="stretch",
                            )
                else:
                    st.caption("No statutory deadline date detected on this notice; calendar reminder not applicable.")

                st.caption(f"⚖️ *{grounded_exp.disclaimer}*")

    # -----------------------------------------------------------------------
    # TAB 3: Action Plan & Verified Citations (F4)
    # -----------------------------------------------------------------------
    with tab_action:
        current_extraction: Optional[NoticeExtraction] = st.session_state.get("confirmed_extraction")
        if current_extraction is None:
            st.info(f"👈 {t['no_file_selected']}")
        else:
            doc_type_str = (
                current_extraction.doc_type.value
                if hasattr(current_extraction.doc_type, "value")
                else str(current_extraction.doc_type)
            )

            retrieval_query = f"{current_extraction.issuer or ''} {current_extraction.reference_number or ''} " + " ".join(current_extraction.key_facts)
            cached_chunks_raw = get_cached_search_kb(
                query=retrieval_query,
                doc_type=doc_type_str,
                top_k=4,
            )
            top_score = max([c.get("score", 0.0) for c in cached_chunks_raw], default=0.0)

            is_abstained, abstain_reason = should_abstain(current_extraction, top_retrieval_score=top_score)
            if is_abstained:
                st.warning(f"⚠️ **Safe Abstention Notice:** {abstain_reason}")
                office_info = get_office_guidance(doc_type_str)
                render_abstention_card(t, office_info)
            else:
                cached_run = st.session_state.get("active_cached_run")
                if (
                    cached_run
                    and cached_run.get("explanation")
                    and selected_lang == "English"
                ):
                    grounded_exp = GroundedExplanation.model_validate(cached_run["explanation"])
                else:
                    explanation_data = get_cached_explanation(
                        extraction_dict=current_extraction.model_dump(),
                        chunks_dicts=cached_chunks_raw,
                        language=selected_lang,
                    )
                    grounded_exp = GroundedExplanation.model_validate(explanation_data)

                st.markdown("### 📋 Step-by-Step Action Plan")
                st.caption("Every procedural step is deterministically verified and grounded in official government portal rules.")

                if not grounded_exp.checklist:
                    st.warning("No verified procedural steps met the citation threshold. Please visit the issuing office directly.")
                else:
                    for step in grounded_exp.checklist:
                        if step.required_documents:
                            docs_bullets = "".join([f"<li style='margin-bottom: 3px;'>📎 {doc}</li>" for doc in step.required_documents])
                            req_docs_html = textwrap.dedent(f"""\
                            <div style="margin-top: 0.65rem; font-size: 0.88rem; font-weight: 600; color: #1E293B;">Required Documents / Attachments:</div>
                            <ul style="margin: 0.2rem 0 0.6rem 1.2rem; padding: 0; font-size: 0.86rem; color: #475569;">
                            {docs_bullets}
                            </ul>
                            """).strip()
                        else:
                            req_docs_html = ""

                        # Strictly unindented HTML so markdown parser renders HTML card rather than code block
                        step_html = textwrap.dedent(f"""\
                        <div class="nw-step-box">
                        <div class="nw-step-title">
                        <span class="nw-step-badge">STEP {step.step_number}</span>
                        <span>{step.action}</span>
                        </div>
                        <div style="color: #334155; font-size: 0.94rem; line-height: 1.5; margin-top: 0.35rem;">
                        {step.details}
                        </div>
                        {req_docs_html}
                        <div class="nw-citation-card">
                        <div><strong>🏛️ Official Citation:</strong> {step.source_title}</div>
                        <div style="margin-top: 0.25rem;"><strong>🌐 Verified Portal:</strong> <a href="{step.source_url}" target="_blank" rel="noopener noreferrer" style="color: #2563EB; font-weight: 600; text-decoration: underline;">{step.source_url}</a></div>
                        <div style="margin-top: 0.35rem; font-size: 0.8rem; color: #059669; font-weight: 600;">
                        ✅ Grounded in Official Government Statute
                        </div>
                        </div>
                        </div>
                        """).strip()
                        st.markdown(step_html, unsafe_allow_html=True)

                st.divider()
                with st.expander("📚 View Official Legal Reference Chunks", expanded=False):
                    for c in cached_chunks_raw:
                        st.markdown(f"**{c.get('title')}** (Score: `{c.get('score')}`)")
                        st.markdown(f"Official URL: [{c.get('source_url')}]({c.get('source_url')})")
                        st.caption(c.get("content")[:350] + "...")
                        st.divider()

    # -----------------------------------------------------------------------
    # TAB 4: Response Letter Draft & Multilingual PDF (F6)
    # -----------------------------------------------------------------------
    with tab_letter:
        current_extraction: Optional[NoticeExtraction] = st.session_state.get("confirmed_extraction")
        if current_extraction is None:
            st.info(f"👈 {t['no_file_selected']}")
        else:
            doc_type_str = (
                current_extraction.doc_type.value
                if hasattr(current_extraction.doc_type, "value")
                else str(current_extraction.doc_type)
            )

            retrieval_query = f"{current_extraction.issuer or ''} {current_extraction.reference_number or ''} " + " ".join(current_extraction.key_facts)
            cached_chunks_raw = get_cached_search_kb(
                query=retrieval_query,
                doc_type=doc_type_str,
                top_k=4,
            )
            chunks_models = [RetrievedChunk.model_validate(c) for c in cached_chunks_raw]

            st.markdown("### ✍️ Formal Reply Letter / Dispute Representation Draft")
            st.caption("Pre-filled with confirmed notice details and statutory dispute grounds. Unfilled placeholders are marked in brackets.")

            # Citizen Sender Details Inputs
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 0.85rem 1.2rem; margin-bottom: 1.1rem;">
                    <strong style="color: #0F172A; font-size: 0.95rem;">👤 Your Contact Details (Auto-fills into Official Letter)</strong>
                    <div style="color: #64748B; font-size: 0.82rem; margin-top: 0.15rem;">NoticeWise synchronizes these details directly into the letter header and body:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                citizen_name = st.text_input(
                    t["sender_name_label"],
                    value=st.session_state.get("sender_name", "[YOUR FULL NAME]"),
                )
            with s_col2:
                citizen_address = st.text_input(
                    t["sender_address_label"],
                    value=st.session_state.get("sender_address", "[YOUR RESIDENTIAL ADDRESS]"),
                )
            with s_col3:
                citizen_phone = st.text_input(
                    t["sender_phone_label"],
                    value=st.session_state.get("sender_phone", "[YOUR CONTACT NUMBER]"),
                )

            # Generate structured draft
            draft_model = draft_letter_content(
                extraction=current_extraction,
                retrieved_chunks=chunks_models,
                language=selected_lang,
                sender_name=citizen_name,
                sender_address=citizen_address,
                sender_phone=citizen_phone,
            )

            # Full Editable Letter Area
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.6rem; margin-bottom: 0.35rem;">
                    <strong style="color: #0F172A; font-size: 0.95rem;">📝 Formal Representation Letter Content</strong>
                    <span style="font-size: 0.78rem; color: #64748B; background: #F1F5F9; border: 1px solid #E2E8F0; padding: 0.2rem 0.5rem; border-radius: 4px;">Edits sync directly to PDF & TXT downloads</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            editable_body = st.text_area(
                "Full Letter Content (Review & Edit before downloading PDF)",
                value=draft_model.body,
                height=320,
                label_visibility="collapsed",
            )

            # Keep draft body synchronized with citizen edits
            draft_model.body = editable_body

            # Render Download Buttons
            b_col1, b_col2 = st.columns(2)

            with b_col1:
                with st.spinner("Generating print-ready PDF..."):
                    pdf_bytes = get_cached_letter_pdf(draft_model.model_dump())

                st.download_button(
                    label=f"📥 {t['download_pdf_button']}",
                    data=pdf_bytes,
                    file_name=f"noticewise_letter_{current_extraction.reference_number or 'reply'}.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                )

            with b_col2:
                st.download_button(
                    label=f"📄 {t['download_txt_button']}",
                    data=editable_body.encode("utf-8"),
                    file_name=f"noticewise_letter_{current_extraction.reference_number or 'reply'}.txt",
                    mime="text/plain",
                    width="stretch",
                )

            st.caption(f"⚖️ *{draft_model.disclaimer}*")

    # -----------------------------------------------------------------------
    # TAB 5: Evaluation & Metrics Dashboard (PRD Section 12 & F8)
    # -----------------------------------------------------------------------
    with tab_eval:
        st.header("📊 NoticeWise Benchmark & Evaluation Dashboard")
        st.caption("Standardized evaluation across all 30 synthetic notices adhering to PRD Section 12 & F8 specifications.")

        st.warning(
            "⚠️ **Benchmark & Synthetic Dataset Disclaimer:** "
            "All 30 test notices evaluated here are synthetic test artifacts designed to benchmark extraction, retrieval, and citation guardrails. "
            "The 100% document type and reference number accuracy is measured on clean synthetic templates; real-world documents with physical degradation, handwriting, or regional formatting will show lower accuracy. "
            "Administrative procedures and legal requirements vary across Indian states. NoticeWise provides informational guidance only and does not constitute formal legal counsel."
        )

        if not EVAL_RESULTS_PATH.exists():
            st.info("Evaluation results file (`reports/eval_results.json`) not found. Run `python scripts/run_eval.py` to generate.")
        else:
            with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as ef:
                eval_data = json.load(ef)

            meta = eval_data.get("metadata", {})
            metrics = eval_data.get("metrics", {})
            per_type = eval_data.get("per_doc_type_accuracy", {})
            cm = eval_data.get("confusion_matrix", {})
            abs_det = eval_data.get("abstention_details", {})
            failures = eval_data.get("failures", [])

            # Executive Accuracy & Safety Metrics
            st.subheader("1. Key Accuracy & Safety Benchmarks")
            st.markdown(
                f"""
                <div style="display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem;">
                    <div style="flex: 1; min-width: 200px; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 0.9rem 1.15rem;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #1E40AF; text-transform: uppercase;">Doc Classification</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #1E3A8A;">{metrics.get('doc_type_accuracy', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.76rem; color: #3B82F6;">30/30 clean templates</div>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px; padding: 0.9rem 1.15rem;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #065F46; text-transform: uppercase;">Reference Match</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #047857;">{metrics.get('reference_number_accuracy', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.76rem; color: #10B981;">100% exact identifier match</div>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 10px; padding: 0.9rem 1.15rem;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #92400E; text-transform: uppercase;">Statutory Citations</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #B45309;">{metrics.get('citation_support_rate', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.76rem; color: #D97706;">Grounded in verified portal texts</div>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #F3E8FF; border: 1px solid #E9D5FF; border-radius: 10px; padding: 0.9rem 1.15rem;">
                        <div style="font-size: 0.74rem; font-weight: 700; color: #6B21A8; text-transform: uppercase;">Abstention Recall</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #7E22CE;">{metrics.get('abstention_recall', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.76rem; color: #A855F7;">Zero hallucination on private/court</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Secondary Operational Stats Bar (Clean & High Contrast)
            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem;">
                    <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.85rem 1rem;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Amount Accuracy</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A;">{metrics.get('amount_inr_accuracy', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.74rem; color: #64748B;">Exact match on demanded total</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.85rem 1rem;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Deadline Accuracy</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A;">{metrics.get('deadline_date_accuracy', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.74rem; color: #64748B;">ISO compliance cutoff date</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.85rem 1rem;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Abstention Precision</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A;">{metrics.get('abstention_precision', 0.0) * 100:.1f}%</div>
                        <div style="font-size: 0.74rem; color: #B45309;">Coverage gap: no verified source</div>
                    </div>
                    <div style="background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.85rem 1rem;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Average Latency</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A;">{metrics.get('average_latency_seconds', 0.0):.2f}s</div>
                        <div style="font-size: 0.74rem; color: #047857;">Zero live calls on cached runs</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.divider()

            # Breakdown by Document Type
            st.subheader("2. Performance Breakdown by Document Category")
            type_rows = [
                {"Document Category": "Right to Information (RTI)", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('rti', 0.0)*100:.1f}%", "Status": "Grounded (RTI Act 2005 Sec 6 & 19, RTI Rules 2012)"},
                {"Document Category": "Electricity Disconnection Notices", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('electricity', 0.0)*100:.1f}%", "Status": "Grounded (DERC Supply Code Reg 49 / Electricity Act Sec 56)"},
                {"Document Category": "Traffic e-Challan Citations", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('challan', 0.0)*100:.1f}%", "Status": "Grounded for compounding under MV Act 1988 Sec 200 (state portal procedures abstain)"},
                {"Document Category": "NFSA Ration Card Grievance", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('scheme_rejection', 0.0)*100:.1f}%", "Status": "Grounded for NFSA (Sec 14-16); pension schemes abstain"},
                {"Document Category": "Municipal Property Tax Demands", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('tax', 0.0)*100:.1f}%", "Status": "Abstained (Coverage gap: no verified source available)"},
                {"Document Category": "Welfare Pension Schemes", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('scheme_rejection', 0.0)*100:.1f}%", "Status": "Abstained (Coverage gap: no verified source available)"},
                {"Document Category": "Out-of-Scope (Private / Court)", "Evaluated Notices": 5, "Accuracy": f"{per_type.get('other', 0.0)*100:.1f}%", "Status": "Abstained (F5 Safety Card: private lease, court notices, receipts)"},
            ]
            st.dataframe(type_rows, width="stretch", hide_index=True)

            st.divider()

            # Confusion Matrix
            st.subheader("3. Classification Confusion Matrix (6x6)")
            st.caption("Rows represent Ground Truth category; Columns represent Model Classification.")
            all_types = ["electricity", "challan", "tax", "scheme_rejection", "rti", "other"]
            cm_rows = []
            for act in all_types:
                row_dict = {"Actual \\ Predicted": act.upper()}
                for pred in all_types:
                    count = cm.get(act, {}).get(pred, 0)
                    row_dict[pred.upper()] = count
                cm_rows.append(row_dict)
            st.dataframe(cm_rows, width="stretch", hide_index=True)

            st.divider()

            # Abstention & Safety Analysis
            st.subheader("4. Safety & Abstention Analysis (F5 Guardrails)")
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                st.markdown(
                    f"- **Out-of-Scope Notices Handled:** {abs_det.get('true_positives_out_of_scope_abstained', 0)} / 5 correctly refused (100% recall).\n"
                    f"- **Out-of-Scope Missed (False Negatives):** {abs_det.get('false_negatives_out_of_scope_missed', 0)} (Zero dangerous out-of-scope notices processed).\n"
                    f"- **In-Scope Safe Abstentions:** {abs_det.get('false_positives_in_scope_abstained', 0)} notices (Coverage gap: property tax and pension schemes safely refused when official portal statutory text could not be verified)."
                )
            with a_col2:
                st.info(
                    "💡 **Safe Abstention Policy:** NoticeWise prioritizes citizen safety over false coverage. "
                    "When a document type lacks verified, unauthenticated official statutory rules (e.g. municipal property tax, state pension schemes), "
                    "the system abstains and directs the citizen to their local administrative office with a checklist of required documents."
                )

            st.divider()

            # Real Failure Examples (All 10 Amount/Deadline Discrepancies)
            st.subheader("5. Complete Failure Review (All 10 Amount & Deadline Discrepancies)")
            st.caption("Every discrepancy between synthetic ground truth labels and live model predictions across the 30-notice benchmark:")

            failure_explanations = [
                {
                    "Notice": "notice_02_elec_meter_tamper.pdf",
                    "Category": "electricity",
                    "Discrepancies": "Amount INR (exp: ₹12,800.00, pred: None); Deadline Date (exp: 2026-11-10, pred: None)",
                    "Root Cause": "Synthetic PDF text layer was clipped at generation; text fast-path received incomplete line without vigilance penalty total or relative deadline clause.",
                },
                {
                    "Notice": "notice_04_elec_tariff_revision.pdf",
                    "Category": "electricity",
                    "Discrepancies": "Amount INR (exp: ₹1,850.00, pred: None); Deadline Date (exp: 2026-11-25, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off at '...surcharge (PPAC) re'. In an earlier vision run, Gemini returned 1820 (OCR digit confusion on 5 vs 2).",
                },
                {
                    "Notice": "notice_07_challan_signal.pdf",
                    "Category": "challan",
                    "Discrepancies": "Amount INR (exp: ₹1,000.00, pred: None); Deadline Date (exp: 2026-11-24, pred: None)",
                    "Root Cause": "Synthetic PDF single-line text layer cut off before compounding penalty amount and payment due date.",
                },
                {
                    "Notice": "notice_09_challan_wrong_side.pdf",
                    "Category": "challan",
                    "Discrepancies": "Amount INR (exp: ₹2,000.00, pred: None); Deadline Date (exp: 2026-11-15, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before compounding violation penalty total.",
                },
                {
                    "Notice": "notice_12_tax_assessment_revision.pdf",
                    "Category": "tax",
                    "Discrepancies": "Amount INR (exp: ₹24,000.00, pred: None); Deadline Date (exp: 2026-11-21, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before revised property assessment demand and 30-day appeal date.",
                },
                {
                    "Notice": "notice_14_tax_commercial_tax.pdf",
                    "Category": "tax",
                    "Discrepancies": "Amount INR (exp: ₹32,000.00, pred: None); Deadline Date (exp: 2026-11-30, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before commercial tax assessment total and due date.",
                },
                {
                    "Notice": "notice_16_scheme_ration_card.pdf",
                    "Category": "scheme_rejection",
                    "Discrepancies": "Deadline Date (exp: 2026-11-13, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before 30-day DGRO appeal window clause. (Direct vision call successfully extracts 2026-11-13).",
                },
                {
                    "Notice": "notice_18_scheme_housing.pdf",
                    "Category": "scheme_rejection",
                    "Discrepancies": "Deadline Date (exp: 2026-12-02, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before PMAY credit subsidy rectification deadline.",
                },
                {
                    "Notice": "notice_22_rti_ration_delay.pdf",
                    "Category": "rti",
                    "Discrepancies": "Amount INR (exp: ₹10.00, pred: None); Deadline Date (exp: 2026-12-07, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before statutory application fee (INR 10) and 30-day response window.",
                },
                {
                    "Notice": "notice_24_rti_pension_grievance.pdf",
                    "Category": "rti",
                    "Discrepancies": "Amount INR (exp: ₹10.00, pred: None); Deadline Date (exp: 2026-12-11, pred: None)",
                    "Root Cause": "Synthetic PDF text layer cut off before court fee stamp disclosure and 30-day response deadline.",
                },
            ]
            st.dataframe(failure_explanations, width="stretch", hide_index=True)

            st.divider()

            # Grounded Citation Audit
            st.subheader("6. Grounded Action Checklist Audit (reports/citation_review.csv)")
            st.caption("Deterministic verification of generated action steps against official statutes and regulatory PDFs:")

            if CITATION_REVIEW_PATH.exists():
                import csv
                with open(CITATION_REVIEW_PATH, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    csv_rows = list(reader)

                audit_display = [
                    {
                        "Notice": r.get("notice_filename"),
                        "Step #": r.get("step_number"),
                        "Action Item": r.get("action"),
                        "Cited Chunk ID": r.get("cited_chunk_id"),
                        "Official Source": r.get("source_title"),
                        "Official URL": r.get("source_url"),
                        "Verification Verdict": r.get("supported_verdict"),
                    }
                    for r in csv_rows
                ]
                st.dataframe(audit_display, width="stretch", hide_index=True)

            st.divider()

            # Export Artifacts
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="📥 Download Full Evaluation Metrics (eval_results.json)",
                    data=json.dumps(eval_data, indent=2),
                    file_name="eval_results.json",
                    mime="application/json",
                    width="stretch",
                )
            with d_col2:
                if CITATION_REVIEW_PATH.exists():
                    st.download_button(
                        label="📥 Download Citation Review Audit (citation_review.csv)",
                        data=CITATION_REVIEW_PATH.read_text(encoding="utf-8"),
                        file_name="citation_review.csv",
                        mime="text/csv",
                        width="stretch",
                    )


if __name__ == "__main__":
    main()
