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

    # Template definitions
    if doc_type_val == "electricity":
        recipient_title = "The Assistant Engineer (Operations) / Divisional Engineer"
        recipient_address = f"{extraction.issuer or 'Electricity Distribution Company (DISCOM)'}\nSub-Division Office, Hyderabad, Telangana."
        subject = f"Representation regarding Disputed Electricity Notice / Demand (Ref: {ref_num})"
        ref_line = f"Notice No: {ref_num}, dated {extraction.issue_date or '[ISSUE DATE]'} for Consumer Service No. [YOUR USC NUMBER]"
        salutation = "Respected Sir / Madam,"
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
        salutation = "Respected Sir / Madam,"
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
        salutation = "Respected Sir / Madam,"
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
        salutation = "Respected Sir / Madam,"
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
        salutation = "Respected Sir / Madam,"
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
        salutation = "Respected Sir / Madam,"
        body = [
            f"I am writing in reference to the document referenced above received by me on {extraction.issue_date or '[DATE]'}.",
        ]
        grounds = custom_grounds or "I request clarification and review of the said document in accordance with official procedures."

    closing = "Yours faithfully,"

    return LetterDraft(
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

    font_name = get_font_for_language(draft.language)

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
    story.append(Paragraph(f"<b>NoticeWise Citizen Administrative Draft</b>", header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#3182CE"), spaceAfter=12))

    # Sender Info
    story.append(Paragraph(f"<b>From:</b><br/>{draft.sender_name}<br/>{draft.sender_address.replace(chr(10), '<br/>')}<br/>Phone: {draft.sender_phone or 'N/A'}", body_style))
    story.append(Spacer(1, 8))

    # Recipient Info
    recipient_html = f"<b>To:</b><br/>{draft.recipient_title}<br/>{draft.recipient_address.replace(chr(10), '<br/>')}"
    story.append(Paragraph(recipient_html, body_style))
    story.append(Spacer(1, 10))

    # Subject & Reference
    story.append(Paragraph(f"<b>Subject:</b> {draft.subject}", subject_style))
    story.append(Paragraph(f"<b>Reference:</b> {draft.reference_line}", body_style))
    story.append(Spacer(1, 8))

    # Salutation
    story.append(Paragraph(draft.salutation, body_style))

    # Body Paragraphs
    for p in draft.body_paragraphs:
        story.append(Paragraph(p.replace("\n", "<br/>"), body_style))

    # Grounds Paragraph
    story.append(Paragraph("<b>Grounds for Representation / Objection:</b>", bold_style))
    story.append(Paragraph(draft.grounds_paragraph.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 10))

    # Sign-off
    story.append(Paragraph(f"{draft.closing}<br/><br/><b>{draft.sender_name}</b>", body_style))
    story.append(Spacer(1, 15))

    # Marked Placeholders Box if any
    if draft.unfilled_placeholders:
        unfilled_items = "".join([f"• {u}<br/>" for u in draft.unfilled_placeholders])
        box_data = [[
            Paragraph(
                f"<b>Attention Required - Unfilled Placeholders:</b><br/>"
                f"{unfilled_items}"
                f"<i>Please write in your specific details before signing and submitting this letter.</i>",
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
