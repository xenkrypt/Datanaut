"""Lay out the report as a PDF with fpdf2.

Fonts are DejaVu Sans from matplotlib's own package data (no font files are bundled with
Datanaut), so accents, degree signs and most punctuation render. A character the font does not
have is shown as "?" instead of an empty box. All text from the user's file has already been
cleaned and shortened by the model; it is also made printable here. The only date in the file is
the generation date carried by the report data, so the same input gives the same document.
"""

from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from typing import Any

from fpdf import FPDF
from fpdf.fonts import FontFace

from datanaut.report.charts import FIG_HEIGHT, FIG_WIDTH, render_chart
from datanaut.report.model import (
    Bullets,
    ChartBlock,
    Heading,
    KeyValues,
    Paragraph,
    ReportData,
    Section,
    Table,
)
from datanaut.report.text import font_dir, printable

MARGIN = 18.0
PAGE_W = 210.0
CONTENT_W = PAGE_W - 2 * MARGIN
INK = (30, 34, 44)
GREY = (110, 116, 128)
ACCENT = (36, 87, 197)
HEAD_FILL = (232, 238, 250)
BODY_PT, SMALL_PT, TABLE_PT = 10, 8.5, 8


class _Pdf(FPDF):
    def __init__(self, fingerprint: str, title: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self._prefix = fingerprint[:12]
        self._title = title
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=18)
        fonts = font_dir()
        self.add_font("DejaVu", "", os.path.join(fonts, "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", os.path.join(fonts, "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", os.path.join(fonts, "DejaVuSans-Oblique.ttf"))
        self.add_font("DejaVuMono", "", os.path.join(fonts, "DejaVuSansMono.ttf"))

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GREY)
        self.set_y(8)
        self.cell(w=0, h=4, text=printable(self._title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(220, 224, 232)
        self.line(MARGIN, 13, PAGE_W - MARGIN, 13)
        self.set_y(MARGIN)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-13)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GREY)
        self.cell(w=CONTENT_W / 2, h=5, text=f"Dataset fingerprint {self._prefix}")
        self.cell(w=CONTENT_W / 2, h=5, text=f"Page {self.page_no()}", align="R")


def _text(pdf: FPDF, size: float, *, bold: bool = False, italic: bool = False, colour=INK) -> None:
    style = ("B" if bold else "") + ("I" if italic else "")
    pdf.set_font("DejaVu", style, size)
    pdf.set_text_color(*colour)


def _paragraph(pdf: FPDF, block: Paragraph) -> None:
    if block.style == "normal":
        _text(pdf, BODY_PT)
    elif block.style == "estimate":
        _text(pdf, BODY_PT, italic=True, colour=(150, 80, 10))
    else:
        _text(pdf, SMALL_PT, italic=True, colour=GREY)
    pdf.multi_cell(w=0, h=5, text=printable(block.text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _bullets(pdf: FPDF, block: Bullets) -> None:
    _text(pdf, BODY_PT)
    for item in block.items:
        pdf.set_x(MARGIN + 3)
        pdf.multi_cell(
            w=CONTENT_W - 3, h=5, text="• " + printable(item), new_x="LMARGIN", new_y="NEXT"
        )
    pdf.ln(2)


def _widths(columns: list[str], rows: list[list[str]]) -> list[float]:
    weights = []
    for i, name in enumerate(columns):
        cells = sorted(len(r[i]) for r in rows) if rows else [0]
        typical = cells[int(0.8 * (len(cells) - 1))]
        weights.append(max(len(name) * 0.8, min(typical, 48), 6))
    total = sum(weights)
    widths = [max(14.0, CONTENT_W * w / total) for w in weights]
    scale = CONTENT_W / sum(widths)
    return [w * scale for w in widths]


def _table(pdf: FPDF, block: Table) -> None:
    if block.title:
        _text(pdf, 10, bold=True)
        pdf.multi_cell(w=0, h=6, text=printable(block.title), new_x="LMARGIN", new_y="NEXT")
    _text(pdf, TABLE_PT)
    heading = FontFace(emphasis="BOLD", fill_color=HEAD_FILL, color=INK)
    with pdf.table(
        col_widths=_widths(block.columns, block.rows),
        text_align="LEFT",
        line_height=4.2,
        padding=1,
        headings_style=heading,
        borders_layout="HORIZONTAL_LINES",
        first_row_as_headings=True,
    ) as table:
        table.row([printable(c) for c in block.columns])
        for row in block.rows:
            table.row([printable(c) for c in row])
    pdf.ln(3)


def _kv(pdf: FPDF, block: KeyValues) -> None:
    _text(pdf, BODY_PT)
    with pdf.table(
        col_widths=(CONTENT_W * 0.34, CONTENT_W * 0.66),
        text_align="LEFT",
        line_height=5,
        padding=0.8,
        borders_layout="NONE",
        first_row_as_headings=False,
    ) as table:
        for key, value in block.items:
            row = table.row()
            row.cell(printable(key), style=FontFace(emphasis="BOLD", color=GREY))
            row.cell(printable(value))
    pdf.ln(3)


def _chart(pdf: FPDF, png: bytes) -> None:
    height = CONTENT_W * FIG_HEIGHT / FIG_WIDTH
    if pdf.will_page_break(height + 4):
        pdf.add_page()
    pdf.image(io.BytesIO(png), x=MARGIN, w=CONTENT_W)
    pdf.ln(4)


def _heading(pdf: FPDF, text: str) -> None:
    if pdf.will_page_break(30):
        pdf.add_page()
    _text(pdf, 11.5, bold=True, colour=ACCENT)
    pdf.multi_cell(w=0, h=6, text=printable(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _section(pdf: _Pdf, section: Section, pngs: dict[str, bytes]) -> None:
    if pdf.will_page_break(55):
        pdf.add_page()
    pdf.start_section(printable(section.title), level=0)
    _text(pdf, 16, bold=True)
    pdf.multi_cell(w=0, h=9, text=printable(section.title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.5)
    pdf.line(MARGIN, pdf.get_y(), MARGIN + 30, pdf.get_y())
    pdf.ln(4)
    for block in section.blocks:
        if isinstance(block, Heading):
            _heading(pdf, block.text)
        elif isinstance(block, Paragraph):
            _paragraph(pdf, block)
        elif isinstance(block, Bullets):
            _bullets(pdf, block)
        elif isinstance(block, Table):
            _table(pdf, block)
        elif isinstance(block, KeyValues):
            _kv(pdf, block)
        elif isinstance(block, ChartBlock) and block.chart_id in pngs:
            _chart(pdf, pngs[block.chart_id])
    pdf.ln(4)


def _toc(pdf: FPDF, outline: list[Any]) -> None:
    _text(pdf, 11)
    for entry in outline:
        link = pdf.add_link(page=entry.page_number)
        pdf.cell(w=CONTENT_W - 20, h=7, text=printable(entry.name), link=link)
        pdf.cell(
            w=20,
            h=7,
            text=str(entry.page_number),
            align="R",
            link=link,
            new_x="LMARGIN",
            new_y="NEXT",
        )


def _cover(pdf: _Pdf, report: ReportData) -> None:
    c = report.cover
    pdf.add_page()
    pdf.ln(34)
    _text(pdf, 34, bold=True, colour=ACCENT)
    pdf.multi_cell(w=0, h=15, text=printable(c.product), new_x="LMARGIN", new_y="NEXT")
    _text(pdf, 16)
    pdf.multi_cell(w=0, h=9, text="Analysis report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _text(pdf, 10, italic=True, colour=GREY)
    pdf.multi_cell(w=0, h=5.5, text=printable(c.formal_title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(18)
    for label, value in (
        ("Dataset", c.dataset_name),
        ("Dataset version", c.version_id),
        ("Generated", c.generated),
    ):
        _text(pdf, 9, colour=GREY)
        pdf.cell(w=0, h=5, text=label, new_x="LMARGIN", new_y="NEXT")
        _text(pdf, 13)
        pdf.multi_cell(w=0, h=7, text=printable(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    _text(pdf, 9, colour=GREY)
    pdf.cell(w=0, h=5, text="Dataset fingerprint", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVuMono", "", 8.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(w=0, h=5, text=printable(c.fingerprint), new_x="LMARGIN", new_y="NEXT")


def _generated(text: str) -> datetime:
    try:
        return datetime.strptime(text.replace(" UTC", ""), "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError:
        return datetime(2000, 1, 1, tzinfo=UTC)


def build_pdf(report: ReportData) -> bytes:
    """The report as PDF bytes (nothing is written to disk)."""
    cover = report.cover
    pdf = _Pdf(cover.fingerprint, f"{cover.product} report: {cover.dataset_name}")
    pdf.set_title(printable(f"{cover.product} report: {cover.dataset_name}"))
    pdf.set_author(cover.product)
    pdf.set_creator(cover.product)
    pdf.set_creation_date(_generated(cover.generated))
    pdf.set_lang("en")
    _cover(pdf, report)

    pdf.add_page()
    _text(pdf, 18, bold=True)
    pdf.multi_cell(w=0, h=10, text="Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.insert_toc_placeholder(_toc, pages=1)

    pngs = {cid: render_chart(payload) for cid, payload in report.charts.items()}
    for section in report.sections:
        _section(pdf, section, pngs)
    return bytes(pdf.output())
