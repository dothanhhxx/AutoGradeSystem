"""python-docx helpers reproducing the VNICT 2026 template layout.

Template (VNICT 2026 paper, Google Doc): A4, 1-inch margins, Times New Roman; title 16 pt bold
centred; Heading 1 = 12 pt bold ("I. Introduction"); Heading 2 = 11 pt bold ("A. ..."); body
11 pt justified; figure captions centred italic "Fig. N. ..."; table captions centred 10 pt bold
upper-case "TABLE I. ..."; equations centred with "(n)"; references "[n] ...".
"""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _set_font(style_or_run, size=None, bold=None, italic=None):
    f = style_or_run.font
    f.name = FONT
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    f.color.rgb = RGBColor(0, 0, 0)
    el = style_or_run.element
    rpr = el.get_or_add_rPr() if hasattr(el, "get_or_add_rPr") else el.rPr
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), FONT)


class Paper:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Inches(11.69), Inches(8.27)
        for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
            setattr(sec, side, Inches(1))
        st = self.doc.styles
        _set_font(st["Normal"], 11)
        st["Normal"].paragraph_format.space_after = Pt(4)
        st["Normal"].paragraph_format.line_spacing = 1.08
        for name, size in (("Heading 1", 12), ("Heading 2", 11), ("Heading 3", 11)):
            _set_font(st[name], size, bold=True, italic=False)
            pf = st[name].paragraph_format
            pf.space_before, pf.space_after, pf.keep_with_next = Pt(10), Pt(4), True
        self.fig_n = 0
        self.tab_n = 0
        self.eq_n = 0
        self.h1_n = 0
        self.h2_n = 0

    # ---- text blocks ----
    def title(self, text):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_font(p.add_run(text), 16, bold=True)
        p.paragraph_format.space_after = Pt(12)

    def abstract(self, text, keywords):
        h = self.doc.add_heading("Abstract", level=1)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.para(text)
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _set_font(p.add_run("Keywords: " + keywords), bold=True)

    def h1(self, text, numbered=True, center=False):
        self.h2_n = 0
        if numbered:
            text = f"{ROMAN[self.h1_n]}. {text}"
            self.h1_n += 1
        h = self.doc.add_heading(text, level=1)
        if center:
            h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def h2(self, text):
        self.doc.add_heading(f"{chr(65 + self.h2_n)}. {text}", level=2)
        self.h2_n += 1

    def h3(self, text):
        p = self.doc.add_paragraph()
        _set_font(p.add_run(text), bold=True)
        p.paragraph_format.keep_with_next = True

    def para(self, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        """Supports **bold** and *italic* inline markup."""
        import re
        p = self.doc.add_paragraph()
        p.alignment = align
        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                _set_font(p.add_run(tok[2:-2]), bold=True)
            elif tok.startswith("*"):
                _set_font(p.add_run(tok[1:-1]), italic=True)
            else:
                _set_font(p.add_run(tok))
        return p

    def bullets(self, items):
        for it in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _set_font(p.add_run(it))

    def numbered(self, items):
        for it in items:
            p = self.doc.add_paragraph(style="List Number")
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _set_font(p.add_run(it))

    def equation(self, expr):
        self.eq_n += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_font(p.add_run(f"{expr},   ({self.eq_n})"), italic=False)
        return self.eq_n

    # ---- floats ----
    def figure(self, path, caption, width_in=6.2) -> int:
        self.fig_n += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(width_in))
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_font(c.add_run(f"Fig. {self.fig_n}. {caption}"), italic=True)
        return self.fig_n

    def table(self, caption, header, rows, col_widths=None, font_size=9) -> str:
        self.tab_n += 1
        num = ROMAN[self.tab_n - 1]
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.keep_with_next = True
        _set_font(c.add_run(f"TABLE {num}. {caption.upper()}"), 10, bold=True)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(header):
            cell = t.rows[0].cells[j]
            cell.text = ""
            _set_font(cell.paragraphs[0].add_run(str(h)), font_size, bold=True)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for i, r in enumerate(rows, start=1):
            for j, v in enumerate(r):
                cell = t.rows[i].cells[j]
                cell.text = ""
                bold = isinstance(v, str) and v.startswith("**") and v.endswith("**")
                txt = v[2:-2] if bold else str(v)
                _set_font(cell.paragraphs[0].add_run(txt), font_size, bold=bold)
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
        if col_widths:
            for j, w in enumerate(col_widths):
                for row in t.rows:
                    row.cells[j].width = Inches(w)
        # Rows never split across pages.
        for row in t.rows:
            trpr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trpr.append(cs)
        self.doc.add_paragraph()
        return num

    def references(self, refs):
        self.h1("Reference", numbered=False, center=True)
        for i, r in enumerate(refs, 1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _set_font(p.add_run(f"[{i}] {r}"), 10)

    def save(self, path):
        self.doc.save(str(path))
