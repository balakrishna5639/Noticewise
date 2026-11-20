import io
import re
import fitz
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("NotoSans", "fonts/NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSansDevanagari", "fonts/NotoSansDevanagari-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoSansTelugu", "fonts/NotoSansTelugu-Regular.ttf"))

def format_indic_markup(html_text: str) -> str:
    tokens = re.split(r"(<[^>]+>)", html_text)
    out = []
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

raw_html_deva = "<b>Subject:</b> विद्युत अधिनियम 2003 - धारा 56 (Ref: TSS-12345)<br/><b>From:</b> Srikanth Rao"
raw_html_tel = "<b>Subject:</b> విద్యుత్ చట్టం 2003 - సెక్షన్ 56 (Ref: TSS-12345)<br/><b>From:</b> Srikanth Rao"

buf = io.BytesIO()
doc = SimpleDocTemplate(buf)
styles = getSampleStyleSheet()

p1 = Paragraph(format_indic_markup(raw_html_deva), styles["Normal"])
p2 = Paragraph(format_indic_markup(raw_html_tel), styles["Normal"])

doc.build([p1, p2])
fitz_doc = fitz.open(stream=buf.getvalue(), filetype="pdf")
print("EXTRACTED FROM REPORTLAB:")
print(fitz_doc[0].get_text())
