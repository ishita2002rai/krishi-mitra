"""Per-scan PDF report export (Increment 5), e.g. for a farmer to show an
agricultural extension officer."""

import unicodedata
from pathlib import Path

from fpdf import FPDF

from krishi_mitra.config import display_name

_UNICODE_REPLACEMENTS = {
    "‘": "'", "’": "'",  # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "–": "-", "—": "-", "‑": "-",  # en/em dash, non-breaking hyphen
    "…": "...",  # ellipsis
    "•": "-",  # bullet
    "≈": "~",  # approximately
    "⁻": "-",  # superscript minus (e.g. "L⁻¹")
    "→": "->", "←": "<-",  # arrows
}


def _normalize_spacing(text: str) -> str:
    """LLMs routinely use non-breaking/narrow/thin space variants (e.g. between a
    number and its unit) that aren't in Latin-1 and would otherwise render as literal
    "?" -- collapse any Unicode space-separator to a plain space, and drop zero-width/
    format characters (word joiners, BOM) entirely rather than mangling them."""
    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Zs":
            chars.append(" ")
        elif category == "Cf":
            continue
        else:
            chars.append(ch)
    return "".join(chars)


def _sanitize(text: str) -> str:
    """fpdf2's core fonts only support Latin-1/WinAnsi. LLM output routinely contains
    smart quotes, dashes, and superscripts outside that range, which otherwise crash
    PDF generation -- swap known cases for ASCII, then hard-fallback anything left."""
    text = _normalize_spacing(text)
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _section(pdf: FPDF, heading: str, body: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _sanitize(heading), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _sanitize(body), markdown=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def build_scan_report_pdf(scan: dict) -> bytes:
    """Render one scan record (see db.py's schema) as a single-page PDF report."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Krishi-Mitra Scan Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, _sanitize(f"Generated: {scan.get('created_at', '')}"), new_x="LMARGIN", new_y="NEXT")
    if scan.get("location"):
        pdf.cell(0, 7, _sanitize(f"Location: {scan['location']}"), new_x="LMARGIN", new_y="NEXT")
    if scan.get("plot_name"):
        pdf.cell(0, 7, _sanitize(f"Plot/field: {scan['plot_name']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    image_path = scan.get("image_path")
    if image_path and Path(image_path).exists():
        try:
            pdf.image(image_path, w=70)
            pdf.ln(4)
        except Exception:
            pass  # a broken/unsupported image file shouldn't block the rest of the report

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, _sanitize(f"Diagnosis: {display_name(scan['diagnosis'])}"), new_x="LMARGIN", new_y="NEXT")
    if scan.get("confidence") is not None:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Classifier confidence: {scan['confidence']:.1%}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if scan.get("reasoning"):
        _section(pdf, "Reasoning", scan["reasoning"])
    if scan.get("risk_assessment"):
        _section(pdf, "Weather-based risk assessment", scan["risk_assessment"])
    if scan.get("treatment_plan"):
        _section(pdf, "Treatment plan", scan["treatment_plan"])
    if scan.get("outcome_rating"):
        note = f" - {scan['outcome_note']}" if scan.get("outcome_note") else ""
        _section(pdf, "Farmer-reported outcome", f"{scan['outcome_rating']}/5{note}")

    return bytes(pdf.output())
