"""Official reply, complaint, and RTI letter generator for NoticeWise (F6).

Features:
- Structured templates per document type (electricity, challan, tax, scheme_rejection, rti).
- Clear marking of unfilled placeholders ([YOUR NAME], [YOUR ADDRESS], etc.).
- Grounds paragraph generated deterministically from retrieved KB chunks (or via single LLM call if requested).
- PDF generation via ReportLab using bundled TrueType Noto fonts (English, Hindi, Telugu).
- Returns both editable text and downloadable PDF bytes.
"""

import io
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.schemas import DocumentType, LetterDraft, NoticeExtraction, RetrievedChunk

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = PROJECT_ROOT / "fonts"

_FONTS_REGISTERED: bool = False


def register_noto_fonts() -> None:
    """Register bundled Noto Sans TrueType fonts in ReportLab for multilingual PDF rendering."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return

    try:
        regular_path = FONTS_DIR / "NotoSans-Regular.ttf"
        deva_path = FONTS_DIR / "NotoSansDevanagari-Regular.ttf"
        telugu_path = FONTS_DIR / "NotoSansTelugu-Regular.ttf"

        if regular_path.exists():
            pdfmetrics.registerFont(TTFont("NotoSans", str(regular_path)))
        if deva_path.exists():
            pdfmetrics.registerFont(TTFont("NotoSansDevanagari", str(deva_path)))
        if telugu_path.exists():
            pdfmetrics.registerFont(TTFont("NotoSansTelugu", str(telugu_path)))

        _FONTS_REGISTERED = True
        logger.info("Registered Noto fonts successfully")
    except Exception as exc:
        logger.warning("Could not register custom fonts, falling back to Helvetica: %s", exc)


def get_font_for_language(language: str) -> str:
    """Return the registered font name for the specified language."""
    register_noto_fonts()
    lang = language.lower()
    if "telugu" in lang or "te" == lang:
        return "NotoSansTelugu" if "NotoSansTelugu" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    elif "hindi" in lang or "hi" == lang:
        return "NotoSansDevanagari" if "NotoSansDevanagari" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    else:
        return "NotoSans" if "NotoSans" in pdfmetrics.getRegisteredFontNames() else "Helvetica"


def format_indic_markup(html_text: str) -> str:
    """Format mixed Indic (Hindi/Telugu) and Latin text with appropriate ReportLab font tags.

    Ensures that Devanagari and Telugu glyphs render via their dedicated Noto font while
    Latin characters, numbers, and common symbols render cleanly via NotoSans.
    """
    register_noto_fonts()
    tokens = re.split(r"(<[^>]+>)", html_text)
    out: List[str] = []
    for token in tokens:
        if not token:
            continue
        if token.startswith("<") and token.endswith(">"):
            out.append(token)
        else:
            pattern = re.compile(r"([\u0900-\u097F]+|[\u0C00-\u0C7F]+|[^\u0900-\u097F\u0C00-\u0C7F]+)")
            sub_tokens = pattern.findall(token)
            for st in sub_tokens:
                if re.search(r"[\u0900-\u097F]", st):
                    out.append(f"<font name='NotoSansDevanagari'>{st}</font>")
                elif re.search(r"[\u0C00-\u0C7F]", st):
                    out.append(f"<font name='NotoSansTelugu'>{st}</font>")
                else:
                    out.append(f"<font name='NotoSans'>{st}</font>")
    return "".join(out)


def draft_letter_content(
    extraction: NoticeExtraction,
    retrieved_chunks: List[RetrievedChunk],
    language: str = "English",
    sender_name: Optional[str] = None,
    sender_address: Optional[str] = None,
    sender_phone: Optional[str] = None,
    custom_grounds: Optional[str] = None,
) -> LetterDraft:
    """Generate a structured LetterDraft model with confirmed fields and marked placeholders."""
    name = (sender_name or "").strip() or "[YOUR FULL NAME]"
    address = (sender_address or "").strip() or "[YOUR RESIDENTIAL ADDRESS]"
    phone = (sender_phone or "").strip() or "[YOUR PHONE NUMBER]"

    unfilled: List[str] = []
    if name == "[YOUR FULL NAME]":
        unfilled.append("Applicant / Sender Full Name")
    if address == "[YOUR RESIDENTIAL ADDRESS]":
        unfilled.append("Sender Residential Address")
    if phone == "[YOUR PHONE NUMBER]":
        unfilled.append("Sender Contact Phone Number")

    doc_type_val = (
        extraction.doc_type.value
        if hasattr(extraction.doc_type, "value")
        else str(extraction.doc_type)
    )
    ref_num = extraction.reference_number or "[NOTICE REFERENCE NUMBER]"
    if ref_num == "[NOTICE REFERENCE NUMBER]":
        unfilled.append("Notice Reference / Challan Number")

    # Select authoritative grounding text from top chunk
    top_chunk_title = retrieved_chunks[0].title if retrieved_chunks else "Official Administrative Rules"
    top_chunk_url = retrieved_chunks[0].source_url if retrieved_chunks else "https://www.india.gov.in"

    amt_str = f"INR {extraction.amount_inr:,.2f}" if extraction.amount_inr is not None else "the demanded amount"
    lang_key = (language or "English").lower()

    if "telugu" in lang_key or "te" == lang_key:
        closing = "భవదీయుడు / భవదీయురాలు,"
        salutation = "గౌరవనీయులైన అధికారికి,"
        if doc_type_val == "electricity":
            recipient_title = "సహాయ ఇంజనీర్ (ఆపరేషన్స్) / డివిజనల్ ఇంజనీర్"
            recipient_address = f"{extraction.issuer or 'విద్యుత్ పంపిణీ సంస్థ (DISCOM)'}\nసబ్-డివిజన్ కార్యాలయం, తెలంగాణ."
            subject = f"విద్యుత్ నోటీసు / బకాయిలపై అభ్యంతర వినతి పత్రం (Ref: {ref_num})"
            ref_line = f"నోటీసు సంఖ్య: {ref_num}, తేదీ: {extraction.issue_date or '[తేదీ]'} సర్వీస్ నెం. [YOUR USC NUMBER]"
            body = [
                f"నేను పైన పేర్కొన్న చిరునామాలో గల విద్యుత్ కనెక్షన్ నెం. [YOUR USC NUMBER] వినియోగదారుడిని.",
                f"నాకు రూ. {amt_str} చెల్లించవలసిందిగా తేదీ {extraction.deadline_date or '[గడువు తేదీ]'} తో నోటీసు అందినది. ఈ బిల్లు అసెస్‌మెంట్ సరైనది కాదని నేను అభ్యంతరం తెలుపుతున్నాను.",
            ]
            grounds = custom_grounds or (
                f"విద్యుత్ చట్టం 2003 సెక్షన్ 56 నిబంధనలు మరియు {top_chunk_title} ({top_chunk_url}) ప్రకారం, "
                f"ఎలాంటి సర్వీస్ నిలిపివేత చర్యలకు ముందు వినియోగదారుడికి నోటీసు మరియు విచారణ పొందే హక్కు ఉంది. "
                f"మీటర్ రీడింగ్ మరియు లెడ్జర్ రికార్డులను పునఃపరిశీలించి సరైన బిల్లు జారీ చేయవలసిందిగా కోరుతున్నాను."
            )
        elif doc_type_val == "challan":
            recipient_title = "అసిస్టెంట్ కమిషనర్ ఆఫ్ పోలీస్ (ట్రాఫిక్) / ట్రాఫిక్ హెల్ప్‌డెస్క్"
            recipient_address = f"{extraction.issuer or 'ట్రాఫిక్ పోలీస్ కమీషనరేట్'}\nట్రాఫిక్ ఎన్‌ఫోర్స్‌మెంట్ విభాగం, తెలంగాణ."
            subject = f"ట్రాఫిక్ ఇ-చలానాపై అభ్యంతర వినతి పత్రం (చలానా సంఖ్య: {ref_num})"
            ref_line = f"ఇ-చలానా సంఖ్య: {ref_num}, తేదీ: {extraction.issue_date or '[తేదీ]'} వాహన సంఖ్య: [YOUR VEHICLE NUMBER]"
            body = [
                f"నేను మోటారు వాహన సంఖ్య [YOUR VEHICLE NUMBER] రిజిస్టర్డ్ యజమానిని.",
                f"నా వాహనంపై రూ. {amt_str} కాంపౌండింగ్ ఫీజుతో ఆటోమేటెడ్ ఇ-చలానా జారీ చేయబడింది.",
            ]
            grounds = custom_grounds or (
                f"ఆటోమేటెడ్ కెమెరా క్యాప్చర్ లోపం వల్ల ఈ చలానా తప్పుగా జారీ చేయబడింది. "
                f"{top_chunk_url} మార్గదర్శకాల ప్రకారం, సదరు సమయంలో నా వాహనం అక్కడ లేదు. "
                f"కెమెరా ఫుటేజ్ మరియు వాహన ఆర్సీని సరిచూసి ఈ చలానాను రద్దు చేయవలసిందిగా కోరుతున్నాను."
            )
        elif doc_type_val == "tax":
            recipient_title = "డిప్యూటీ కమిషనర్ / అసెస్‌మెంట్ అథారిటీ"
            recipient_address = f"{extraction.issuer or 'మున్సిపల్ కార్పొరేషన్'}\nరెవెన్యూ విభాగం, సిటిజన్ సర్వీస్ సెంటర్, హైదరాబాద్."
            subject = f"ఆస్తి పన్ను అసెస్‌మెంట్‌పై సెక్షన్ 282 రివిజన్ పిటిషన్ (నోటీసు: {ref_num})"
            ref_line = f"డిమాండ్ అసెస్‌మెంట్ సంఖ్య: {ref_num}, తేదీ: {extraction.issue_date or '[తేదీ]'}"
            body = [
                f"నేను పైన తెలిపిన చిరునామాలో గల ప్రాపర్టీ టాక్స్ ఐడెంటిఫికేషన్ నెం. (PTIN) [YOUR PTIN NUMBER] ఆస్తి యజమానిని.",
                f"సదరు ఆస్తిపై రూ. {amt_str} పన్ను విధిస్తూ అందిన స్పెషల్ నోటీసుపై నేను ఈ అభ్యంతరం తెలుపుతున్నాను.",
            ]
            grounds = custom_grounds or (
                f"మున్సిపల్ చట్టం సెక్షన్ 282 ప్రకారం 30 రోజుల పరిమితిలోపు ఈ రివిజన్ పిటిషన్ దాఖలు చేస్తున్నాను. "
                f"కొలతలు మరియు ఆస్తి వర్గీకరణ తప్పుగా ఉన్నందున, {top_chunk_title} ప్రకారం ప్రత్యక్ష విచారణ మరియు పునఃపరిశీలన చేపట్టాలని కోరుతున్నాను."
            )
        elif doc_type_val == "scheme_rejection":
            recipient_title = "జిల్లా గ్రీవెన్స్ రెడ్రెస్సల్ ఆఫీసర్ (DGRO) / ప్రాజెక్ట్ డైరెక్టర్"
            recipient_address = f"జిల్లా కలెక్టర్ కార్యాలయం / సంక్షేమ విభాగం\n{extraction.issuer or 'సంక్షేమ శాఖ'}, తెలంగాణ."
            subject = f"సంక్షేమ పథకం దరఖాస్తు తిరస్కరణపై అప్పీలు (Ref: {ref_num})"
            ref_line = f"దరఖాస్తు సంఖ్య: {ref_num}, తేదీ: {extraction.issue_date or '[తేదీ]'}"
            body = [
                f"నేను సంక్షేమ పథకానికి దరఖాస్తు చేసుకోగా, నా దరఖాస్తు తిరస్కరించబడినట్లు నోటీసు అందినది.",
                f"ఈ తిరస్కరణ నిర్ణయంపై నేను 30 రోజుల గడువులోగా ఈ అప్పీలు దాఖలు చేస్తున్నాను.",
            ]
            grounds = custom_grounds or (
                f"జాతీయ ఆహార భద్రతా చట్టం మరియు {top_chunk_title} ({top_chunk_url}) నిబంధనల ప్రకారం నా కుటుంబానికి పూర్తి అర్హత ఉంది. "
                f"తహసీల్దార్ ఆదాయ ధ్రువీకరణ పత్రం ఆధారంగా క్షేత్రస్థాయి విచారణ నిర్వహించి లబ్ధిని పునరుద్ధరించాలని కోరుతున్నాను."
            )
        elif doc_type_val == "rti":
            recipient_title = "పబ్లిక్ ఇన్ఫర్మేషన్ ఆఫీసర్ (PIO) / అసిస్టెంట్ PIO"
            recipient_address = f"{extraction.issuer or 'సంబంధిత ప్రభుత్వ కార్యాలయం'}\nప్రధాన కార్యాలయం / సచివాలయం."
            subject = "సమాచార హక్కు చట్టం 2005 సెక్షన్ 6(1) క్రింద సమాచారం కోరుతూ దరఖాస్తు"
            ref_line = "సెక్షన్ 6(1) క్రింద RTI దరఖాస్తు"
            body = [
                "నేను భారత పౌరుడిని. సమాచార హక్కు చట్టం 2005 నిబంధనల ప్రకారం క్రింది వివరాలను కోరుతున్నాను.",
                f"కోరిన వివరాలు:\n1. సర్టిఫైడ్ రికార్డుల కాపీలు: {'; '.join(extraction.key_facts) if extraction.key_facts else '[సమాచారం వివరాలు]'}.\n2. సంబంధిత ఫైళ్ల పరిశీలన అవకాశం కల్పించాలి.",
            ]
            grounds = custom_grounds or (
                f"సమాచార హక్కు నియమాలు 2012 ({top_chunk_url}) ప్రకారం దరఖాస్తు రుసుము రూ. 10 చెల్లించడమైనది. "
                f"సెక్షన్ 7(1) ప్రకారం 30 రోజుల గడువులోపు కోరిన సమాచారాన్ని సర్టిఫై చేసి అందించాలని కోరుతున్నాను."
            )
        else:
            recipient_title = "గౌరవనీయ అధికారి / అథారిటీ"
            recipient_address = f"{extraction.issuer or 'ప్రభుత్వ కార్యాలయం'}\nకార్యాలయ చిరునామా."
            subject = f"నోటీసు / పత్రంపై వివరణ కోరుతూ వినతి పత్రం (Ref: {ref_num})"
            ref_line = f"రిఫరెన్స్: {ref_num}"
            body = [f"నాకు అందిన పత్రం తేదీ {extraction.issue_date or '[తేదీ]'} ఆధారంగా ఈ వినతి పత్రం సమర్పిస్తున్నాను."]
            grounds = custom_grounds or "అధికారిక నిబంధనల ప్రకారం సదరు పత్రంపై పునఃపరిశీలన చేపట్టవలసిందిగా కోరుతున్నాను."

    elif "hindi" in lang_key or "hi" == lang_key:
        closing = "भवदीय,"
        salutation = "आदरणीय महोदय / महोदया,"
        if doc_type_val == "electricity":
            recipient_title = "सहायक अभियंता (संचालन) / प्रभागीय अभियंता"
            recipient_address = f"{extraction.issuer or 'विद्युत वितरण कंपनी (DISCOM)'}\nउप-प्रभाग कार्यालय, हैदराबाद, तेलंगाना।"
            subject = f"विवादित बिजली नोटिस / बिल के संबंध में आपत्ति अभ्यावेदन (Ref: {ref_num})"
            ref_line = f"नोटिस संख्या: {ref_num}, दिनांक: {extraction.issue_date or '[दिनांक]'} उपभोक्ता सेवा सं. [YOUR USC NUMBER]"
            body = [
                f"मैं उपरोक्त पते पर स्थित विद्युत सेवा कनेक्शन सं. [YOUR USC NUMBER] का नियमित उपभोक्ता हूँ।",
                f"मुझे {amt_str} की मांग करते हुए दिनांक {extraction.deadline_date or '[अंतिम तिथि]'} का नोटिस प्राप्त हुआ है। मैं इस बिल निर्धारण की शुद्धता पर आपत्ति दर्ज करता हूँ।",
            ]
            grounds = custom_grounds or (
                f"विद्युत अधिनियम 2003 की धारा 56 तथा {top_chunk_title} ({top_chunk_url}) के अनुसार, "
                f"उपभोक्ता को किसी भी विच्छेदन से पूर्व 15 दिनों का वैधानिक नोटिस और निष्पक्ष जांच का विधिक अधिकार प्राप्त है। "
                f"अतः मीटर रीडिंग व लेजर रिकॉर्ड का पुनःसत्यापन कर संशोधित बिल जारी किया जाए।"
            )
        elif doc_type_val == "challan":
            recipient_title = "सहायक पुलिस आयुक्त (यातायात) / ट्रैफिक हेल्पडेस्क"
            recipient_address = f"{extraction.issuer or 'ट्रैफिक पुलिस कमिश्नरेट'}\nयातायात प्रवर्तन शाखा, तेलंगाना।"
            subject = f"गलत स्वचालित ट्रैफ़िक ई-चालान के विरुद्ध अभ्यावेदन (चालान संख्या: {ref_num})"
            ref_line = f"ई-चालान संख्या: {ref_num}, दिनांक: {extraction.issue_date or '[दिनांक]'} वाहन सं. [YOUR VEHICLE NUMBER]"
            body = [
                f"मैं वाहन पंजीकरण संख्या [YOUR VEHICLE NUMBER] का पंजीकृत स्वामी हूँ।",
                f"स्वचालित कैमरा प्रणाली द्वारा {amt_str} की राशि का ई-चालान त्रुटिवश जारी कर दिया गया है।",
            ]
            grounds = custom_grounds or (
                f"उक्त चालान कैमरा नंबर प्लेट मास्किंग त्रुटि के कारण अनुचित रूप से उत्पन्न हुआ है। "
                f"{top_chunk_url} के निवारण दिशा-निर्देशों के अनुसार, कथित समय पर मेरा वाहन उस स्थान पर उपस्थित नहीं था। "
                f"कृपया फोटो साक्ष्य का मिलान कर इस चालान को निरस्त करने की कृपा करें।"
            )
        elif doc_type_val == "tax":
            recipient_title = "उपायुक्त / कर निर्धारण प्राधिकारी"
            recipient_address = f"{extraction.issuer or 'नगर निगम'}\nराजस्व अनुभाग, नागरिक सेवा केंद्र, हैदराबाद।"
            subject = f"संपत्ति कर निर्धारण के विरुद्ध धारा 282 पुनरीक्षण याचिका (नोटिस सं: {ref_num})"
            ref_line = f"मांग नोटिस सं: {ref_num}, दिनांक: {extraction.issue_date or '[दिनांक]'}"
            body = [
                f"मैं उपरोक्त पते पर स्थित संपत्ति संख्या (PTIN) [YOUR PTIN NUMBER] का स्वामी / अधिभोगी हूँ।",
                f"मुझे {amt_str} के संपत्ति कर निर्धारण का विशेष नोटिस प्राप्त हुआ है, जिस पर मुझे आपत्ति है।",
            ]
            grounds = custom_grounds or (
                f"नगर निगम अधिनियम की धारा 282 के अंतर्गत 30 दिनों की वैधानिक सीमा में यह पुनरीक्षण याचिका प्रस्तुत है। "
                f"प्लेंथ क्षेत्र व उपयोग श्रेणीकरण की त्रुटि सुधारने हेतु {top_chunk_title} के अनुरूप व्यक्तिगत सुनवाई का अवसर प्रदान करें।"
            )
        elif doc_type_val == "scheme_rejection":
            recipient_title = "जिला शिकायत निवारण अधिकारी (DGRO) / परियोजना निदेशक"
            recipient_address = f"जिला कलेक्टर कार्यालय / कल्याण अनुभाग\n{extraction.issuer or 'कल्याण विभाग'}, तेलंगाना।"
            subject = f"कल्याणकारी योजना आवेदन अस्वीकृति के विरुद्ध अपील (Ref: {ref_num})"
            ref_line = f"आवेदन संख्या: {ref_num}, दिनांक: {extraction.issue_date or '[दिनांक]'}"
            body = [
                f"मैंने संदर्भित योजना के अंतर्गत कल्याणकारी लाभों हेतु आवेदन किया था जिसे अनुचित रूप से निरस्त कर दिया गया है।",
                f"इस अस्वीकृति के विरुद्ध मैं विहित 30 दिनों की अवधि के भीतर यह अपील प्रस्तुत कर रहा हूँ।",
            ]
            grounds = custom_grounds or (
                f"राष्ट्रीय खाद्य सुरक्षा अधिनियम तथा {top_chunk_title} ({top_chunk_url}) के अंतर्गत मेरा परिवार सभी वैधानिक पात्रता मानदंडों को पूरा करता है। "
                f"तहसीलदार आय प्रमाण पत्र का संज्ञान लेते हुए स्थलीय पुनर्सत्यापन कर लाभ बहाल किया जाए।"
            )
        elif doc_type_val == "rti":
            recipient_title = "लोक सूचना अधिकारी (PIO) / सहायक PIO"
            recipient_address = f"{extraction.issuer or 'संबंधित लोक प्राधिकारी'}\nप्रधान कार्यालय / सचिवालय अनुभाग।"
            subject = "सूचना का अधिकार अधिनियम, 2005 की धारा 6(1) के अंतर्गत सूचना प्राप्ति हेतु आवेदन"
            ref_line = "धारा 6(1) के तहत आरटीआई आवेदन"
            body = [
                "मैं भारत का नागरिक हूँ तथा सूचना का अधिकार अधिनियम, 2005 के प्रावधानों के अंतर्गत निम्नलिखित सूचना प्राप्त करना चाहता हूँ।",
                f"वांछित सूचना का विवरण:\n1. प्रमाणित प्रतियों का विवरण: {'; '.join(extraction.key_facts) if extraction.key_facts else '[वांछित अभिलेख]'}.\n2. संबंधित विभागीय फाइलों का निरीक्षण।",
            ]
            grounds = custom_grounds or (
                f"आरटीआई नियम 2012 ({top_chunk_url}) के अनुसार निर्धारित आवेदन शुल्क 10 रुपये संलग्न है। "
                f"धारा 7(1) के अंतर्गत 30 दिनों की वैधानिक अवधि में प्रमाणित अभिलेख उपलब्ध कराने का कष्ट करें।"
            )
        else:
            recipient_title = "सक्षम प्राधिकारी / अधिकारी"
            recipient_address = f"{extraction.issuer or 'शासकीय कार्यालय'}\nमुख्य कार्यालय पता।"
            subject = f"नोटिस / शासकीय पत्र के संबंध में अभ्यावेदन (Ref: {ref_num})"
            ref_line = f"संदर्भ: {ref_num}"
            body = [f"दिनांक {extraction.issue_date or '[दिनांक]'} को प्राप्त पत्र के संदर्भ में यह अभ्यावेदन प्रस्तुत है।"]
            grounds = custom_grounds or "शासकीय नियमों के अनुरूप उक्त विषय की समीक्षा करने का कष्ट करें।"

    else:
        # English Default
        closing = "Yours faithfully,"
        salutation = "Respected Sir / Madam,"
        if doc_type_val == "electricity":
            recipient_title = "The Assistant Engineer (Operations) / Divisional Engineer"
            recipient_address = f"{extraction.issuer or 'Electricity Distribution Company (DISCOM)'}\nSub-Division Office, Hyderabad, Telangana."
            subject = f"Representation regarding Disputed Electricity Notice / Demand (Ref: {ref_num})"
            ref_line = f"Notice No: {ref_num}, dated {extraction.issue_date or '[ISSUE DATE]'} for Consumer Service No. [YOUR USC NUMBER]"
            body = [
                f"I am a consumer of electrical energy under Consumer Service Connection No. [YOUR USC NUMBER] situated at the address mentioned above.",
                f"I have received the subject notice demanding payment of {amt_str} by {extraction.deadline_date or '[DEADLINE DATE]'}. I dispute the correctness of this assessment / demand.",
            ]
            grounds = custom_grounds or (
                f"The demand is arbitrary and contrary to the applicable billing tariffs and consumer service regulations. "
                f"In accordance with {top_chunk_title} ({top_chunk_url}), a consumer is entitled to fair inspection and "
                f"provisional assessment review before any coercive disconnection under Section 56 of the Electricity Act 2003. "
                f"I request that meter reading and ledger records be verified and corrected billing issued."
            )

        elif doc_type_val == "challan":
            recipient_title = "The Assistant Commissioner of Police (Traffic) / Traffic Helpdesk"
            recipient_address = f"{extraction.issuer or 'Traffic Police Commissionerate'}\nTraffic Enforcement Branch, Telangana."
            subject = f"Representation challenging erroneous Traffic e-Challan (Challan No: {ref_num})"
            ref_line = f"e-Challan No: {ref_num}, dated {extraction.issue_date or '[ISSUE DATE]'} against Vehicle No. [YOUR VEHICLE NUMBER]"
            body = [
                f"I am the registered owner of motor vehicle bearing registration number [YOUR VEHICLE NUMBER].",
                f"An automated e-challan has been issued alleging traffic violation with a compounding penalty of {amt_str}.",
            ]
            grounds = custom_grounds or (
                f"The citation has been generated erroneously due to automated camera capture / number plate masking error. "
                f"As per the dispute redressal guidelines on {top_chunk_url}, I submit that my vehicle was not present at the alleged location "
                f"at the recorded time. I request immediate physical verification of the capture image against my vehicle RC and cancellation of this challan."
            )

        elif doc_type_val == "tax":
            recipient_title = "The Deputy Commissioner / Assessment Authority"
            recipient_address = f"{extraction.issuer or 'Municipal Corporation'}\nRevenue Section, Citizen Service Centre, Hyderabad."
            subject = f"Revision Petition under Section 282 against Property Tax Assessment (Notice No: {ref_num})"
            ref_line = f"Special Notice / Demand Assessment No: {ref_num}, dated {extraction.issue_date or '[ISSUE DATE]'}"
            body = [
                f"I am the owner / occupier of premises bearing Property Tax Identification Number (PTIN) [YOUR PTIN NUMBER] situated at the address above.",
                f"I have been served with the subject Special Notice assessing property tax at {amt_str}.",
            ]
            grounds = custom_grounds or (
                f"This Revision Petition is filed within the statutory 30-day limitation period under Section 282 of the Municipal Corporation Act. "
                f"The enhanced assessment is excessive due to incorrect plinth area measurement and wrongful classification of usage. "
                f"In accordance with {top_chunk_title}, I pray for personal hearing and re-inspection to reassess tax in accordance with law."
            )

        elif doc_type_val == "scheme_rejection":
            recipient_title = "The District Grievance Redressal Officer (DGRO) / Project Director"
            recipient_address = f"Office of the District Collector / Welfare Officer\n{extraction.issuer or 'Welfare Department'}, Telangana."
            subject = f"Appeal against wrongful rejection of Welfare Scheme Application (Ref: {ref_num})"
            ref_line = f"Application / Rejection Intimation No: {ref_num}, dated {extraction.issue_date or '[ISSUE DATE]'}"
            body = [
                f"I had submitted an application for welfare benefits under the scheme referenced above.",
                f"I received an intimation rejecting my application on grounds of ineligibility. I am filing this appeal within 30 days.",
            ]
            grounds = custom_grounds or (
                f"The rejection order was passed without verifying my actual socio-economic status. "
                f"Under {top_chunk_title} ({top_chunk_url}), my household fulfills all statutory poverty and eligibility criteria. "
                f"I attach valid Tahsildar income certificate and family Aadhaar copies, and pray for field re-verification and restoration of benefits."
            )

        elif doc_type_val == "rti":
            recipient_title = "The Public Information Officer (PIO) / Assistant PIO"
            recipient_address = f"{extraction.issuer or 'Concerned Public Authority'}\nHead Office / Secretarial Section."
            subject = "Application for obtaining Information under Section 6(1) of the Right to Information Act, 2005"
            ref_line = "RTI Application under Section 6(1)"
            body = [
                "I am a citizen of India and wish to seek information under the provisions of the Right to Information Act, 2005.",
                f"Particulars of Information Sought:\n1. Please provide certified copies of records relating to: {'; '.join(extraction.key_facts) if extraction.key_facts else '[DESCRIBE INFORMATION SOUGHT]'}.\n2. Please provide inspection of relevant departmental files.",
            ]
            grounds = custom_grounds or (
                f"The application fee of INR 10 is enclosed herewith via [IPO / DD / Court Fee Stamp No. ______] "
                f"in accordance with Rule 3 of the Right to Information Rules, 2012 ({top_chunk_url}). "
                f"I request that information be provided within the statutory period of 30 days as prescribed under Section 7(1)."
            )

        else:
            recipient_title = "The Competent Officer / Authority"
            recipient_address = f"{extraction.issuer or 'Official Office'}\nMain Office Address."
            subject = f"Representation regarding Notice / Document (Ref: {ref_num})"
            ref_line = f"Reference: {ref_num}"
            body = [
                f"I am writing in reference to the document referenced above received by me on {extraction.issue_date or '[DATE]'}.",
            ]
            grounds = custom_grounds or "I request clarification and review of the said document in accordance with official procedures."

    closing = "Yours faithfully,"

    draft = LetterDraft(
        doc_type=extraction.doc_type,
        language=language,
        recipient_title=recipient_title,
        recipient_address=recipient_address,
        subject=subject,
        reference_line=ref_line,
        salutation=salutation,
        body_paragraphs=body,
        grounds_paragraph=grounds,
        closing=closing,
        sender_name=name,
        sender_address=address,
        sender_phone=phone,
        unfilled_placeholders=unfilled,
    )
    draft.body = generate_letter_text(draft)
    return draft


def generate_letter_text(draft: LetterDraft) -> str:
    """Format LetterDraft into clean plain text for editing and copying."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append(f"FORMAL REPRESENTATION / DISPUTE LETTER ({draft.doc_type.value.upper()})")
    lines.append("=" * 60)
    lines.append("")
    lines.append("From:")
    lines.append(f"  {draft.sender_name}")
    lines.append(f"  {draft.sender_address}")
    if draft.sender_phone:
        lines.append(f"  Contact: {draft.sender_phone}")
    lines.append("")
    lines.append("To:")
    for addr_line in [draft.recipient_title] + draft.recipient_address.splitlines():
        lines.append(f"  {addr_line}")
    lines.append("")
    lines.append(f"Subject: {draft.subject}")
    lines.append(f"Ref: {draft.reference_line}")
    lines.append("")
    lines.append(draft.salutation)
    lines.append("")
    for p in draft.body_paragraphs:
        lines.append(p)
        lines.append("")
    lines.append("Grounds of Representation:")
    lines.append(draft.grounds_paragraph)
    lines.append("")
    lines.append(draft.closing)
    lines.append("")
    lines.append(f"({draft.sender_name})")
    lines.append("")
    if draft.unfilled_placeholders:
        lines.append("=" * 60)
        lines.append("NOTICEWISE ACTION CHECKLIST: UNFILLED PLACEHOLDERS")
        for u in draft.unfilled_placeholders:
            lines.append(f"  - [ ] {u}")
        lines.append("=" * 60)

    return "\n".join(lines)


def generate_letter_pdf(draft: LetterDraft) -> bytes:
    """Render LetterDraft into professional, print-ready PDF bytes using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    register_noto_fonts()
    font_name = "NotoSans" if "NotoSans" in pdfmetrics.getRegisteredFontNames() else "Helvetica"

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "LetterHeader",
        fontName=font_name,
        fontSize=13,
        leading=16,
        textColor=HexColor("#1A365D"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "LetterBody",
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=HexColor("#2D3748"),
        spaceAfter=8,
    )
    subject_style = ParagraphStyle(
        "LetterSubject",
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=HexColor("#1A202C"),
        spaceAfter=10,
    )
    bold_style = ParagraphStyle(
        "LetterBold",
        fontName=font_name,
        fontSize=10,
        leading=14,
        textColor=HexColor("#2B6CB0"),
        spaceAfter=6,
    )
    warning_style = ParagraphStyle(
        "LetterWarning",
        fontName=font_name,
        fontSize=9,
        leading=12,
        textColor=HexColor("#C53030"),
    )

    story = []

    # Title Banner
    story.append(Paragraph(format_indic_markup("<b>NoticeWise Citizen Administrative Draft</b>"), header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#3182CE"), spaceAfter=12))

    # Sender Info
    sender_html = f"<b>From:</b><br/>{draft.sender_name}<br/>{draft.sender_address.replace(chr(10), '<br/>')}<br/>Phone: {draft.sender_phone or 'N/A'}"
    story.append(Paragraph(format_indic_markup(sender_html), body_style))
    story.append(Spacer(1, 8))

    # Recipient Info
    recipient_html = f"<b>To:</b><br/>{draft.recipient_title}<br/>{draft.recipient_address.replace(chr(10), '<br/>')}"
    story.append(Paragraph(format_indic_markup(recipient_html), body_style))
    story.append(Spacer(1, 10))

    # Subject & Reference
    story.append(Paragraph(format_indic_markup(f"<b>Subject:</b> {draft.subject}"), subject_style))
    story.append(Paragraph(format_indic_markup(f"<b>Reference:</b> {draft.reference_line}"), body_style))
    story.append(Spacer(1, 8))

    # Salutation
    story.append(Paragraph(format_indic_markup(draft.salutation), body_style))

    # Body Paragraphs
    for p in draft.body_paragraphs:
        story.append(Paragraph(format_indic_markup(p.replace("\n", "<br/>")), body_style))

    # Grounds Paragraph
    story.append(Paragraph(format_indic_markup("<b>Grounds for Representation / Objection:</b>"), bold_style))
    story.append(Paragraph(format_indic_markup(draft.grounds_paragraph.replace("\n", "<br/>")), body_style))
    story.append(Spacer(1, 10))

    # Sign-off
    story.append(Paragraph(format_indic_markup(f"{draft.closing}<br/><br/><b>{draft.sender_name}</b>"), body_style))
    story.append(Spacer(1, 15))

    # Marked Placeholders Box if any
    if draft.unfilled_placeholders:
        unfilled_items = "".join([f"• {u}<br/>" for u in draft.unfilled_placeholders])
        box_data = [[
            Paragraph(
                format_indic_markup(
                    f"<b>Attention Required - Unfilled Placeholders:</b><br/>"
                    f"{unfilled_items}"
                    f"<i>Please write in your specific details before signing and submitting this letter.</i>"
                ),
                warning_style,
            )
        ]]
        table = Table(box_data, colWidths=["100%"])
        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FFF5F5")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor("#FEB2B2")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(table)

    doc.build(story)
    return buffer.getvalue()
