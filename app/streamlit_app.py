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


def render_abstention_card(t: Dict[str, str], office_info: Optional[Dict[str, Any]] = None) -> None:
    """Display the F5 Abstention & Safety Card callout."""
    st.error(f"### {t['abstention_title']}")
    st.warning(t["abstention_warning"])

    if office_info:
        st.markdown(f"🏛️ **Designated Office to Contact / Visit:**\n> {office_info.get('office', 'Local administrative office')}")
        st.markdown(f"**{t['abstention_bring_header']}**")
        for doc in office_info.get("documents", []):
            st.markdown(f"- 📄 {doc}")
    else:
        st.markdown(f"**{t['abstention_bring_header']}**")
        st.markdown(f"- **{t['abstention_item1']}**")
        st.markdown(f"- **{t['abstention_item2']}**")
        st.markdown(f"- **{t['abstention_item3']}**")

    st.info(f"ℹ️ *{t['abstention_disclaimer']}*")


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
    from src.explain import generate_grounded_explanation
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
            "**NoticeWise Civic Assistant (Phase 2)**\n\n"
            "Turning complex Indian government notices into verified explanations, cited checklists, reply letters, and deadline reminders."
        )

        st.divider()
        st.markdown("### 📊 Session & API Telemetry")
        live_calls = get_live_calls_count()
        is_live = is_live_gemini_available()
        st.metric("Live LLM Calls", f"{live_calls}")
        st.metric("Live Uploads Used", f"{st.session_state.get('live_upload_count', 0)} / 3")
        engine_label = "🟢 Gemini Live (3.8-Flash)" if is_live else "🟠 Offline Mock / Cache"
        st.caption(f"Engine: **{engine_label}**")

    # App Header
    st.title(t["app_title"])
    st.caption(t["app_subtitle"])
    st.divider()

    # Top-Level Tabs
    tab_intake, tab_explain, tab_action, tab_letter = st.tabs(
        [
            f"📄 {t['tab_intake']}",
            f"💡 {t['tab_explain']}",
            f"📋 {t['tab_action']}",
            f"✍️ {t['tab_letter']}",
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
                        if not is_synthetic_pick:
                            st.session_state["live_upload_count"] = st.session_state.get("live_upload_count", 0) + 1

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
            else:
                # Prominent MOCK Warning Banner
                if current_extraction.is_mock or (extraction_result and extraction_result.is_mock):
                    st.warning(t["mock_warning"])

                # Check Abstention condition
                type_val = current_extraction.doc_type.value if hasattr(current_extraction.doc_type, "value") else str(current_extraction.doc_type)
                office_info = get_office_guidance(type_val)

                if should_trigger_abstention(current_extraction.doc_type, current_extraction.confidence):
                    render_abstention_card(t, office_info)

                st.subheader(t["form_header"])

                # Editable Extracted-Fields Form (F2)
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
                        height=100,
                        help="One fact per line",
                    )
                    edit_missing = st.text_area(
                        t["missing_or_unclear"],
                        value="\n".join(current_extraction.missing_or_unclear),
                        height=70,
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

                with st.expander("🔍 View Confirmed Schema JSON Payload", expanded=False):
                    st.json(current_extraction.model_dump())

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
                with st.spinner("Generating plain-language explanation and checklist..."):
                    explanation_data = get_cached_explanation(
                        extraction_dict=current_extraction.model_dump(),
                        chunks_dicts=cached_chunks_raw,
                        language=selected_lang,
                    )
                    grounded_exp = GroundedExplanation.model_validate(explanation_data)

                exp = grounded_exp.explanation

                # High-level summary metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    amt_str = f"₹ {exp.amount_inr:,.2f}" if exp.amount_inr is not None else "None / Not Specified"
                    st.metric("Demanded Amount", amt_str)
                with m_col2:
                    st.metric(t["due_date"], exp.deadline_date or "Immediate / None")
                with m_col3:
                    days = exp.days_left
                    if days is not None:
                        if days < 0:
                            st.metric(t["days_remaining"], f"⚠️ OVERDUE by {abs(days)} days")
                        elif days <= 3:
                            st.metric(t["days_remaining"], f"🚨 {days} days (Urgent!)")
                        else:
                            st.metric(t["days_remaining"], f"⏳ {days} days left")
                    else:
                        st.metric(t["days_remaining"], "No deadline specified")

                st.divider()

                # Plain Explanation Cards
                st.markdown(f"### 📌 What This Notice Means")
                st.markdown(f"> **{exp.what_it_is}**")

                st.markdown(f"### ⚠️ Consequences If You Ignore It")
                st.error(exp.consequences_if_ignored)

                st.markdown(f"### 🏛️ Office to Visit / Contact")
                st.info(f"**Office:** {exp.office_to_visit or 'Local issuing department / office'}")

                if exp.documents_to_bring:
                    st.markdown("**What documents to bring:**")
                    for d in exp.documents_to_bring:
                        st.markdown(f"- 📄 {d}")

                # Calendar Reminder (.ics) Section (F7)
                st.divider()
                st.subheader("📅 Deadline Calendar Reminder (.ics)")
                if current_extraction.deadline_date:
                    ics_bytes = generate_ics_from_extraction(current_extraction, alarm_days_before=3)
                    if ics_bytes:
                        st.caption("Add this deadline to your Google Calendar, Apple Calendar, or Outlook with an automated 3-day advance alarm.")
                        st.download_button(
                            label=f"📥 {t['download_reminder_button']}",
                            data=ics_bytes,
                            file_name=f"noticewise_reminder_{current_extraction.reference_number or 'notice'}.ics",
                            mime="text/calendar",
                            use_container_width=True,
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
                explanation_data = get_cached_explanation(
                    extraction_dict=current_extraction.model_dump(),
                    chunks_dicts=cached_chunks_raw,
                    language=selected_lang,
                )
                grounded_exp = GroundedExplanation.model_validate(explanation_data)

                st.markdown("### 📋 Step-by-Step Action Plan")
                st.caption("Every step is deterministically verified and grounded in official government portal rules.")

                if not grounded_exp.checklist:
                    st.warning("No verified procedural steps met the citation threshold. Please visit the issuing office directly.")
                else:
                    for step in grounded_exp.checklist:
                        with st.container(border=True):
                            st.markdown(f"#### Step {step.step_number}: {step.action}")
                            st.markdown(step.details)

                            if step.required_documents:
                                st.markdown("**Required Documents / Attachments:**")
                                for doc in step.required_documents:
                                    st.markdown(f"- 📎 {doc}")

                            # Citation Box with verified live official URL
                            st.markdown(
                                f"""<div style="background-color: #F0F4F8; padding: 10px; border-radius: 6px; margin-top: 8px; border-left: 4px solid #1A365D;">
                                    <strong>🏛️ Official Citation:</strong> {step.source_title}<br/>
                                    <strong>🌐 Verified Portal:</strong> <a href="{step.source_url}" target="_blank">{step.source_url}</a><br/>
                                    <small><strong>Chunk ID:</strong> <code>{step.chunk_id}</code> | <strong>Verified Overlap:</strong> ✅ Grounded</small>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                st.divider()
                with st.expander("📚 View Retrieved Official Knowledge Base Sources (Top Chunks)", expanded=False):
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
            st.caption("Pre-filled with confirmed notice details and statutory dispute grounds. Unfilled citizen placeholders are marked in brackets.")

            # Citizen Sender Details Inputs
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
            editable_body = st.text_area(
                "Full Letter Content (Review & Edit before downloading PDF)",
                value=draft_model.body,
                height=320,
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
                    use_container_width=True,
                )

            with b_col2:
                st.download_button(
                    label=f"📄 {t['download_txt_button']}",
                    data=editable_body.encode("utf-8"),
                    file_name=f"noticewise_letter_{current_extraction.reference_number or 'reply'}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            st.caption(f"⚖️ *{draft_model.disclaimer}*")


if __name__ == "__main__":
    main()
