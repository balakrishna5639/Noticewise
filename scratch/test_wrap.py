import io
import re
import fitz
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("NotoSans", "fonts/NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSansDevanagari", "fonts/NotoSansDevanagari-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSansTelugu", "fonts/NotoSansTelugu-Regular.ttf"))

def wrap_text_for_fonts(text: str) -> str:
    # Preserve HTML line breaks
    parts = text.split("<br/>")
    wrapped_parts = []
    for part in parts:
        pattern = re.compile(r"([\u0900-\u097F]+|[\u0C00-\u0C7F]+|[^\u0900-\u097F\u0C00-\u0C7F]+)")
        tokens = pattern.findall(part)
        out = []
        for t in tokens:
            if re.search(r"[\u0900-\u097F]", t):
                out.append(f"<font name='NotoSansDevanagari'>{t}</font>")
            elif re.search(r"[\u0C00-\u0C7F]", t):
                out.append(f"<font name='NotoSansTelugu'>{t}</font>")
            else:
                out.append(f"<font name='NotoSans'>{t}</font>")
        wrapped_parts.append("".join(out))
    return "<br/>".join(wrapped_parts)

buf = io.BytesIO()
doc = SimpleDocTemplate(buf)
styles = getSampleStyleSheet()

t_hi = "प्रेषक (From): Srikanth Rao, Notice No: TSS-12345, राशि: INR 4,250.00. विषय: धारा 56 विद्युत विच्छेदन आपत्ति."
t_te = "పంపినవారు (From): Srikanth Rao, Notice No: TSS-12345, మొత్తం: INR 4,250.00. విషయం: సెక్షన్ 56 విద్యుత్ నోటీసుపై అభ్యంతరం."

p_hi = Paragraph(wrap_text_for_fonts(t_hi), styles["Normal"])
p_te = Paragraph(wrap_text_for_fonts(t_te), styles["Normal"])

doc.build([p_hi, Spacer(1, 15), p_te])
pdf_bytes = buf.getvalue()

fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
text = fitz_doc[0].get_text()
print("EXTRACTED:")
print(text)
