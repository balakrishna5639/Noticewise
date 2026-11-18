#!/usr/bin/env python3
"""scripts/make_synthetic_notices.py - Synthetic Notices and Labels Generator for NoticeWise.

Generates 30 realistic synthetic civic notices (images and PDFs) across 5 supported
categories (electricity, challan, tax, scheme rejection, RTI citizen requests),
plus out-of-scope non-civic documents, with degraded scan simulations and ground truth labels.

Ground truth labels are saved to `data/test_notices/labels.json` and validate against
`LabelsFile` in `src.schemas`.
"""

import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pymupdf  # PyMuPDF
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import DocumentType, LabelsFile, NoticeLabel

# Base output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_notices"

# Canvas dimensions (standard A4 aspect ratio in pixels for rendering)
PAGE_WIDTH = 1000
PAGE_HEIGHT = 1414


# ---------------------------------------------------------------------------
# Font Helpers
# ---------------------------------------------------------------------------
def _load_font(
    family: str = "arial", size: int = 16, bold: bool = False
) -> ImageFont.ImageFont:
    """Load a system TrueType font with graceful fallback to default PIL font."""
    font_candidates: List[str] = []
    if family == "arial":
        font_candidates = (
            ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf", "calibri.ttf"]
        )
    elif family == "times":
        font_candidates = (
            ["timesbd.ttf", "times.ttf"] if bold else ["times.ttf", "georgia.ttf"]
        )
    elif family == "courier":
        font_candidates = (
            ["courbd.ttf", "cour.ttf"] if bold else ["cour.ttf", "consola.ttf"]
        )
    elif family == "script":
        font_candidates = ["comic.ttf", "calibri.ttf", "arial.ttf"]
    else:
        font_candidates = ["arial.ttf"]

    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Visual Element Drawing Helpers
# ---------------------------------------------------------------------------
def _draw_border(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    margin: int = 30,
    double: bool = True,
    color: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    """Draw formal double or single outer margin border."""
    draw.rectangle([margin, margin, width - margin, height - margin], outline=color, width=2)
    if double:
        inner = margin + 5
        draw.rectangle([inner, inner, width - inner, height - inner], outline=color, width=1)


def _draw_official_emblem(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    radius: int = 36,
    state_name: str = "GOVT OF TELANGANA",
    color: Tuple[int, int, int] = (15, 45, 90),
) -> None:
    """Draw simulated circular governmental emblem/seal icon."""
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, outline=color, width=2)
    draw.ellipse(
        [x - radius + 4, y - radius + 4, x + radius - 4, y + radius - 4],
        outline=color,
        width=1,
    )
    # Inner emblem star / pillars
    draw.polygon(
        [
            (x, y - radius + 10),
            (x - 8, y + 6),
            (x + 8, y + 6),
        ],
        fill=color,
    )
    draw.rectangle([x - 12, y + 10, x + 12, y + 14], fill=color)
    draw.rectangle([x - 16, y + 16, x + 16, y + 20], fill=color)


def _draw_barcode(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int = 180,
    height: int = 35,
    seed: int = 42,
) -> None:
    """Draw a simulated 1D barcode strip."""
    rng = random.Random(seed)
    curr_x = x
    while curr_x < x + width:
        bar_w = rng.choice([1, 2, 3, 4])
        space_w = rng.choice([1, 2, 3])
        draw.rectangle([curr_x, y, curr_x + bar_w, y + height], fill=(20, 20, 20))
        curr_x += bar_w + space_w


def _draw_circular_stamp(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    center_y: int,
    radius: int = 48,
    top_text: str = "OFFICIAL SEAL",
    sub_text: str = "VERIFIED",
    color: Tuple[int, int, int] = (180, 30, 30),
) -> None:
    """Draw simulated official circular office rubber stamp."""
    bbox = [
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    ]
    draw.ellipse(bbox, outline=color, width=2)
    inner = [
        center_x - radius + 5,
        center_y - radius + 5,
        center_x + radius - 5,
        center_y + radius - 5,
    ]
    draw.ellipse(inner, outline=color, width=1)
    font_stamp = _load_font("arial", 11, bold=True)
    draw.text((center_x - 36, center_y - 14), top_text[:14], font=font_stamp, fill=color)
    draw.text((center_x - 26, center_y + 2), sub_text[:10], font=font_stamp, fill=color)


def _draw_banner(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    width: int,
    bg_color: Tuple[int, int, int] = (220, 40, 40),
    fg_color: Tuple[int, int, int] = (255, 255, 255),
    height: int = 42,
    font_size: int = 18,
) -> int:
    """Draw full-width colored header banner with centered title."""
    draw.rectangle([45, y, width - 45, y + height], fill=bg_color)
    font = _load_font("arial", font_size, bold=True)
    # Estimate text width
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = y + (height - th) // 2
    draw.text((tx, ty), text, font=font, fill=fg_color)
    return y + height + 15


def _draw_metadata_box(
    draw: ImageDraw.ImageDraw,
    left_items: List[Tuple[str, str]],
    right_items: List[Tuple[str, str]],
    y: int,
    width: int,
    height: Optional[int] = None,
) -> int:
    """Draw a clean 2-column key-value reference card."""
    row_count = max(len(left_items), len(right_items))
    calc_height = max(110, row_count * 28 + 24) if height is None else height
    box_rect = [50, y, width - 50, y + calc_height]
    draw.rectangle(box_rect, fill=(248, 250, 252), outline=(180, 190, 205), width=1)

    col1_x = 65
    col2_x = width // 2 + 10
    font_k = _load_font("arial", 13, bold=True)
    font_v = _load_font("arial", 13, bold=False)

    curr_y = y + 12
    for k, v in left_items:
        draw.text((col1_x, curr_y), f"{k}:", font=font_k, fill=(40, 50, 70))
        draw.text((col1_x + 150, curr_y), str(v), font=font_v, fill=(10, 15, 25))
        curr_y += 28

    curr_y = y + 12
    for k, v in right_items:
        draw.text((col2_x, curr_y), f"{k}:", font=font_k, fill=(40, 50, 70))
        draw.text((col2_x + 140, curr_y), str(v), font=font_v, fill=(10, 15, 25))
        curr_y += 28

    return y + calc_height + 15


def _draw_table(
    draw: ImageDraw.ImageDraw,
    headers: List[str],
    rows: List[List[str]],
    col_widths: List[int],
    y: int,
    x: int = 50,
) -> int:
    """Draw itemized financial / charge table."""
    font_th = _load_font("arial", 13, bold=True)
    font_td = _load_font("arial", 12, bold=False)
    table_w = sum(col_widths)
    header_h = 32

    # Draw header row
    draw.rectangle([x, y, x + table_w, y + header_h], fill=(230, 235, 245), outline=(160, 175, 195))
    curr_x = x
    for idx, (h, w) in enumerate(zip(headers, col_widths)):
        draw.text((curr_x + 8, y + 8), h, font=font_th, fill=(20, 30, 50))
        if idx > 0:
            draw.line([(curr_x, y), (curr_x, y + header_h)], fill=(160, 175, 195), width=1)
        curr_x += w

    curr_y = y + header_h
    for row_idx, row in enumerate(rows):
        is_last = row_idx == len(rows) - 1
        row_h = 28
        row_bg = (242, 245, 250) if is_last else (255, 255, 255)
        draw.rectangle([x, curr_y, x + table_w, curr_y + row_h], fill=row_bg, outline=(200, 210, 220))
        cx = x
        for idx, (cell, w) in enumerate(zip(row, col_widths)):
            fnt = font_th if is_last else font_td
            cell_col = (180, 20, 20) if is_last and idx == len(row) - 1 else (20, 20, 20)
            draw.text((cx + 8, curr_y + 6), str(cell), font=fnt, fill=cell_col)
            if idx > 0:
                draw.line([(cx, curr_y), (cx, curr_y + row_h)], fill=(200, 210, 220), width=1)
            cx += w
        curr_y += row_h

    return curr_y + 15


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Break long string into lines that fit within max_width pixels."""
    words = text.split()
    lines = []
    curr = []
    for w in words:
        test_line = " ".join(curr + [w])
        # Approximate width based on character length if draw is not available
        if len(test_line) * 7.5 > max_width and curr:
            lines.append(" ".join(curr))
            curr = [w]
        else:
            curr.append(w)
    if curr:
        lines.append(" ".join(curr))
    return lines


# ---------------------------------------------------------------------------
# Degradation Simulation (Scan tilt, noise, blur, aging)
# ---------------------------------------------------------------------------
def apply_scan_degradation(
    img: Image.Image,
    angle: float = 1.6,
    noise_sigma: float = 11.0,
    blur_radius: float = 0.9,
    contrast: float = 0.92,
) -> Image.Image:
    """Simulate realistic scanner degradation: rotation tilt, photocopy noise, and blur."""
    # 1. Subtle scan tilt
    rotated = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(248, 246, 240),
    )

    # 2. Add realistic photocopy / sensor noise
    arr = np.array(rotated, dtype=np.int16)
    noise = np.random.normal(loc=0, scale=noise_sigma, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)

    # 3. Scanner optics mild blur
    if blur_radius > 0.0:
        noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. Contrast adjustment
    enhancer = ImageEnhance.Contrast(noisy_img)
    return enhancer.enhance(contrast)


# ---------------------------------------------------------------------------
# Specific Notice Renderers
# ---------------------------------------------------------------------------
def _render_standard_official_header(
    draw: ImageDraw.ImageDraw,
    issuer_org: str,
    sub_title: str,
    address_line: str,
    emblem_text: str = "GOVT OF INDIA",
    emblem_color: Tuple[int, int, int] = (20, 50, 100),
) -> int:
    """Render authentic government department header block with emblem."""
    _draw_official_emblem(draw, 95, 80, radius=32, state_name=emblem_text, color=emblem_color)
    font_h1 = _load_font("arial", 20, bold=True)
    font_h2 = _load_font("arial", 14, bold=True)
    font_h3 = _load_font("arial", 12, bold=False)

    draw.text((150, 55), issuer_org, font=font_h1, fill=(20, 35, 65))
    draw.text((150, 85), sub_title, font=font_h2, fill=(60, 75, 100))
    draw.text((150, 110), address_line, font=font_h3, fill=(90, 100, 115))

    _draw_barcode(draw, PAGE_WIDTH - 240, 65, width=180, height=36)
    draw.line([(50, 140), (PAGE_WIDTH - 50, 140)], fill=(180, 190, 205), width=2)
    return 155


def render_electricity_notice(notice: NoticeLabel) -> Image.Image:
    """Render realistic electricity disconnection / penalty / arrears notice."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=35, double=True, color=(30, 45, 80))

    issuer_name = notice.issuer or "Electricity Supply Company"
    y = _render_standard_official_header(
        draw,
        issuer_org=issuer_name,
        sub_title="REVENUE & ENFORCEMENT CIRCLE - COMMERCIAL DIVISION",
        address_line="Power Distribution Bhavan, Central Operations Office",
        emblem_color=(25, 60, 120),
    )

    is_penalty = "penalty" in (notice.description or "").lower() or "tamper" in notice.filename
    banner_title = "STATUTORY NOTICE: PROVISIONAL PENALTY ASSESSMENT" if is_penalty else "FINAL NOTICE OF DISCONNECTION OF ELECTRICITY SUPPLY"
    banner_bg = (175, 30, 30) if not is_penalty else (190, 80, 20)
    y = _draw_banner(draw, banner_title, y, PAGE_WIDTH, bg_color=banner_bg)

    amt_str = f"INR {notice.amount_inr:,.2f}" if notice.amount_inr is not None else "N/A"
    left_items = [
        ("Notice Ref No", notice.reference_number or "N/A"),
        ("Date of Issue", notice.issue_date or "N/A"),
        ("Section Cites", "Section 56 / 126 of Electricity Act 2003"),
        ("Tariff Category", "LT-II Commercial / Residential Single Phase"),
    ]
    right_items = [
        ("Consumer / CA No", "114029419 / 60019284715"),
        ("Payment Due Date", notice.deadline_date or "N/A"),
        ("Total Demand Due", amt_str),
        ("Sanctioned Load", "5.0 kW / 240V"),
    ]
    y = _draw_metadata_box(draw, left_items, right_items, y, PAGE_WIDTH)

    # Narrative Body
    font_body_bold = _load_font("arial", 13, bold=True)
    font_body = _load_font("arial", 13, bold=False)
    draw.text((55, y), "SUBJECT: DEMAND FOR IMMEDIATE SETTLEMENT OF OUTSTANDING DUES / PENALTY", font=font_body_bold, fill=(20, 20, 20))
    y += 25

    intro = (
        f"Take notice that the consumer account registered under service number above has an accumulated "
        f"unpaid liability amounting to {amt_str}. Despite prior demand notices, the outstanding balance "
        f"remains unsettled in our records as of {notice.issue_date}."
    )
    for line in _wrap_text(intro, font_body, PAGE_WIDTH - 120):
        draw.text((55, y), line, font=font_body, fill=(35, 35, 35))
        y += 22
    y += 10

    # Key Facts / Salient Points
    draw.text((55, y), "Salient Findings & Audit Observations:", font=font_body_bold, fill=(20, 20, 20))
    y += 25
    for fact in notice.key_facts:
        bullet_text = f"•  {fact}"
        for line in _wrap_text(bullet_text, font_body, PAGE_WIDTH - 130):
            draw.text((65, y), line, font=font_body, fill=(45, 45, 45))
            y += 22
    y += 15

    # Itemized Table
    base_charge = (notice.amount_inr or 4000.0) * 0.75
    surcharge = (notice.amount_inr or 4000.0) * 0.20
    duty = (notice.amount_inr or 4000.0) * 0.05
    headers = ["Item #", "Description of Charge", "Assessment Period", "Amount (INR)"]
    rows = [
        ["1", "Energy Charges & Consumption Assessment", "Prior Cycle", f"{base_charge:,.2f}"],
        ["2", "Delayed Payment Surcharge / Vigilance Assessment", "Regulatory", f"{surcharge:,.2f}"],
        ["3", "Electricity Duty & Fuel Surcharge Adjustment (FSA)", "Statutory", f"{duty:,.2f}"],
        ["", "TOTAL AMOUNT PAYABLE BY DUE DATE", "", amt_str],
    ]
    y = _draw_table(draw, headers, rows, [80, 420, 200, 200], y, x=50)

    # Warning Box
    w_box = [50, y, PAGE_WIDTH - 50, y + 80]
    draw.rectangle(w_box, fill=(255, 245, 245), outline=(220, 50, 50), width=1)
    draw.text((65, y + 12), "WARNING / CONSEQUENCE OF NON-COMPLIANCE:", font=font_body_bold, fill=(180, 20, 20))
    w_msg = (
        f"If the full sum of {amt_str} is not credited on or before {notice.deadline_date}, your power supply "
        f"will be permanently disconnected under Section 56(1) of the Electricity Act without further intimation."
    )
    for line in _wrap_text(w_msg, font_body, PAGE_WIDTH - 140):
        draw.text((65, y + 36), line, font=font_body, fill=(150, 20, 20))
        y += 20
    y += 45

    # Footer Signatures and Rubber Stamp
    _draw_circular_stamp(draw, 220, y + 60, radius=48, top_text="TSSPDCL ACCOUNTS", sub_text="SEALED")
    draw.text((PAGE_WIDTH - 340, y + 40), "For & On Behalf of the Company,", font=font_body, fill=(40, 40, 40))
    draw.text((PAGE_WIDTH - 340, y + 65), "SUPERINTENDING ENGINEER (COMMERCIAL)", font=font_body_bold, fill=(20, 30, 60))
    draw.text((PAGE_WIDTH - 340, y + 85), "Authorised Assessing Officer", font=font_body, fill=(70, 70, 70))
    return img


def render_challan_notice(notice: NoticeLabel) -> Image.Image:
    """Render realistic traffic e-challan notice."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=35, double=True, color=(15, 30, 60))

    issuer_name = notice.issuer or "Traffic Police Department"
    y = _render_standard_official_header(
        draw,
        issuer_org=issuer_name.upper(),
        sub_title="AUTOMATED TRAFFIC ENFORCEMENT & E-CHALLAN SYSTEM",
        address_line="Central Traffic Command & Control Centre",
        emblem_color=(15, 35, 80),
    )

    y = _draw_banner(
        draw,
        "TRAFFIC E-CHALLAN / NOTICE OF MOTOR VEHICLE OFFENCE",
        y,
        PAGE_WIDTH,
        bg_color=(20, 50, 110),
    )

    amt_str = f"INR {notice.amount_inr:,.2f}" if notice.amount_inr is not None else "N/A"
    left_items = [
        ("Challan Number", notice.reference_number or "N/A"),
        ("Date of Offence", notice.issue_date or "N/A"),
        ("Statutory Act", "Motor Vehicles (Amendment) Act 2019"),
        ("Payment Portal", "echallan.parivahan.gov.in"),
    ]
    right_items = [
        ("Vehicle Registration", "TS09EA4821 / DL10CA3198"),
        ("Disposal Due Date", notice.deadline_date or "N/A"),
        ("Compounded Fine", amt_str),
        ("Enforcement Unit", "CCTV AI Sensor & Speed Camera"),
    ]
    y = _draw_metadata_box(draw, left_items, right_items, y, PAGE_WIDTH)

    font_b = _load_font("arial", 13, bold=True)
    font_r = _load_font("arial", 13, bold=False)

    draw.text((55, y), "SUMMARY OF RECORDED MOTOR VEHICLE INFRACTION:", font=font_b, fill=(20, 20, 20))
    y += 25

    intro = (
        f"The registered vehicle was photographed and recorded committing a motor vehicle violation under "
        f"the jurisdiction of {notice.issuer}. The digital evidence and automatic number plate recognition "
        f"(ANPR) timestamp confirms the offence on {notice.issue_date}."
    )
    for line in _wrap_text(intro, font_r, PAGE_WIDTH - 120):
        draw.text((55, y), line, font=font_r, fill=(30, 30, 30))
        y += 22
    y += 12

    # Violation Key Facts
    draw.text((55, y), "Offence Particulars & Evidentiary Log:", font=font_b, fill=(20, 20, 20))
    y += 25
    for fact in notice.key_facts:
        bullet_text = f"•  {fact}"
        for line in _wrap_text(bullet_text, font_r, PAGE_WIDTH - 130):
            draw.text((65, y), line, font=font_r, fill=(45, 45, 45))
            y += 22
    y += 15

    # Fine Structure Table
    headers = ["Item #", "Specific Offence Description", "MV Act Section", "Penalty (INR)"]
    rows = [
        ["1", "Exceeding Prescribed Speed Limit / Signal Violation", "Sec 183 / Sec 184", amt_str],
        ["", "TOTAL COMPOUNDING FINE PAYABLE", "", amt_str],
    ]
    y = _draw_table(draw, headers, rows, [80, 470, 180, 170], y, x=50)

    # Notice to Virtual Court
    w_box = [50, y, PAGE_WIDTH - 50, y + 85]
    draw.rectangle(w_box, fill=(255, 250, 240), outline=(210, 140, 20), width=1)
    draw.text((65, y + 12), "IMPORTANT INSTRUCTION - VIRTUAL TRAFFIC COURT:", font=font_b, fill=(180, 90, 10))
    v_msg = (
        f"You are required to settle this fine online by {notice.deadline_date}. Failure to settle within the "
        f"stipulated time will cause this challan to be automatically forwarded to the Virtual Court / Metropolitan "
        f"Magistrate for formal summons and registration of prosecution."
    )
    for line in _wrap_text(v_msg, font_r, PAGE_WIDTH - 140):
        draw.text((65, y + 36), line, font=font_r, fill=(120, 60, 10))
        y += 20
    y += 45

    _draw_circular_stamp(draw, 220, y + 60, radius=48, top_text="TRAFFIC POLICE", sub_text="E-CHALLAN", color=(20, 60, 130))
    draw.text((PAGE_WIDTH - 340, y + 40), "Digitally Authenticated by,", font=font_r, fill=(40, 40, 40))
    draw.text((PAGE_WIDTH - 340, y + 65), "ASSISTANT COMMISSIONER OF POLICE", font=font_b, fill=(20, 30, 60))
    draw.text((PAGE_WIDTH - 340, y + 85), "Traffic Enforcement Wing", font=font_r, fill=(70, 70, 70))
    return img


def render_tax_notice(notice: NoticeLabel) -> Image.Image:
    """Render realistic municipal / property tax demand notice."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=35, double=True, color=(30, 40, 40))

    issuer_name = notice.issuer or "Municipal Corporation"
    y = _render_standard_official_header(
        draw,
        issuer_org=issuer_name.upper(),
        sub_title="REVENUE DEPARTMENT - PROPERTY TAX & ASSESSMENT WING",
        address_line="Head Office, Municipal Corporation Complex",
        emblem_color=(20, 70, 50),
    )

    y = _draw_banner(
        draw,
        "DEMAND NOTICE OF PROPERTY TAX & MUNICIPAL ASSESSMENTS",
        y,
        PAGE_WIDTH,
        bg_color=(25, 85, 65),
    )

    amt_str = f"INR {notice.amount_inr:,.2f}" if notice.amount_inr is not None else "N/A"
    left_items = [
        ("Assessment / PTIN", notice.reference_number or "N/A"),
        ("Notice Issue Date", notice.issue_date or "N/A"),
        ("Governing Act", "Municipal Corporation Act / Section 220"),
        ("Zone / Circle", "Central Zone / Khairatabad Circle"),
    ]
    right_items = [
        ("Assessment Year", "2026 - 2027"),
        ("Payment Due Date", notice.deadline_date or "N/A"),
        ("Total Net Payable", amt_str),
        ("Property Use", "Mixed Residential & Commercial"),
    ]
    y = _draw_metadata_box(draw, left_items, right_items, y, PAGE_WIDTH)

    font_b = _load_font("arial", 13, bold=True)
    font_r = _load_font("arial", 13, bold=False)

    draw.text((55, y), "SUBJECT: SPECIAL NOTICE OF ASSESSMENT & DEMAND FOR TAX ARREARS", font=font_b, fill=(20, 20, 20))
    y += 25

    intro = (
        f"Please take notice that pursuant to the municipal assessment rolls for FY 2026-27, property tax "
        f"and statutory urban cess amounting to {amt_str} is due and payable for the property identified by "
        f"the assessment number above. All payments must be made on or before {notice.deadline_date}."
    )
    for line in _wrap_text(intro, font_r, PAGE_WIDTH - 120):
        draw.text((55, y), line, font=font_r, fill=(30, 30, 30))
        y += 22
    y += 12

    # Key Facts
    draw.text((55, y), "Assessment Basis & Statutory Findings:", font=font_b, fill=(20, 20, 20))
    y += 25
    for fact in notice.key_facts:
        bullet_text = f"•  {fact}"
        for line in _wrap_text(bullet_text, font_r, PAGE_WIDTH - 130):
            draw.text((65, y), line, font=font_r, fill=(45, 45, 45))
            y += 22
    y += 15

    # Table breakdown
    tax_base = (notice.amount_inr or 15000.0) * 0.70
    cess = (notice.amount_inr or 15000.0) * 0.15
    arrears = (notice.amount_inr or 15000.0) * 0.15
    headers = ["Component", "Description", "Rate / Basis", "Amount (INR)"]
    rows = [
        ["General Tax", "Annual Ratable Value (ARV) Tax", "Standard Slab", f"{tax_base:,.2f}"],
        ["Urban Dev Cess", "Statutory Education & Library Cess", "15% Surcharge", f"{cess:,.2f}"],
        ["Prior Arrears", "Unpaid Balance from Preceding Quarter", "Ledger Balance", f"{arrears:,.2f}"],
        ["", "TOTAL AMOUNT DEMANDED", "", amt_str],
    ]
    y = _draw_table(draw, headers, rows, [140, 410, 180, 170], y, x=50)

    # Distraint Warning
    w_box = [50, y, PAGE_WIDTH - 50, y + 80]
    draw.rectangle(w_box, fill=(253, 248, 248), outline=(190, 40, 40), width=1)
    draw.text((65, y + 12), "DISTRAINT & RECOVERY NOTICE UNDER MUNICIPAL ACT:", font=font_b, fill=(180, 20, 20))
    d_msg = (
        f"If the demanded sum of {amt_str} is not remitted prior to {notice.deadline_date}, warrant of distress "
        f"will be executed for attachment and public auction of goods or sealing of premises under Section 269."
    )
    for line in _wrap_text(d_msg, font_r, PAGE_WIDTH - 140):
        draw.text((65, y + 36), line, font=font_r, fill=(140, 20, 20))
        y += 20
    y += 45

    _draw_circular_stamp(draw, 220, y + 60, radius=48, top_text="REVENUE WING", sub_text="DEMANDED", color=(25, 90, 50))
    draw.text((PAGE_WIDTH - 340, y + 40), "By Order of Municipal Commissioner,", font=font_r, fill=(40, 40, 40))
    draw.text((PAGE_WIDTH - 340, y + 65), "DEPUTY COMMISSIONER (REVENUE)", font=font_b, fill=(20, 60, 40))
    draw.text((PAGE_WIDTH - 340, y + 85), "Assessment & Collection Authority", font=font_r, fill=(70, 70, 70))
    return img


def render_scheme_rejection_notice(notice: NoticeLabel) -> Image.Image:
    """Render realistic welfare scheme application rejection order."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=35, double=True, color=(60, 30, 30))

    issuer_name = notice.issuer or "Department of Social Welfare"
    y = _render_standard_official_header(
        draw,
        issuer_org=issuer_name.upper(),
        sub_title="DIRECTORATE OF SOCIAL ASSISTANCE & WELFARE SCHEMES",
        address_line="State Secretariat, Social Welfare Complex",
        emblem_color=(70, 25, 25),
    )

    y = _draw_banner(
        draw,
        "OFFICIAL ORDER: REJECTION OF SCHEME APPLICATION",
        y,
        PAGE_WIDTH,
        bg_color=(150, 40, 40),
    )

    left_items = [
        ("Application Order No", notice.reference_number or "N/A"),
        ("Date of Order", notice.issue_date or "N/A"),
        ("Scrutiny Authority", "District Scrutiny & Verification Committee"),
        ("Scheme Category", "National Food Security / Pension / PMAY"),
    ]
    right_items = [
        ("Applicant Name", "Beneficiary Applicant Records"),
        ("Appeal Deadline", notice.deadline_date or "N/A"),
        ("Application Status", "REJECTED / DISQUALIFIED"),
        ("Appellate Authority", "District Collector / Appellate Officer"),
    ]
    y = _draw_metadata_box(draw, left_items, right_items, y, PAGE_WIDTH)

    font_b = _load_font("arial", 13, bold=True)
    font_r = _load_font("arial", 13, bold=False)

    draw.text((55, y), "FORMAL COMMUNICATION OF DISQUALIFICATION & REASON FOR REJECTION:", font=font_b, fill=(20, 20, 20))
    y += 25

    intro = (
        f"This is to formally notify the applicant that the application registered under reference number "
        f"{notice.reference_number} has been thoroughly scrutinized and subsequently REJECTED by the competent "
        f"authority due to non-fulfillment of statutory eligibility criteria prescribed in the scheme guidelines."
    )
    for line in _wrap_text(intro, font_r, PAGE_WIDTH - 120):
        draw.text((55, y), line, font=font_r, fill=(30, 30, 30))
        y += 22
    y += 12

    # Key Facts
    draw.text((55, y), "Specific Grounds for Rejection & Field Verification Findings:", font=font_b, fill=(20, 20, 20))
    y += 25
    for fact in notice.key_facts:
        bullet_text = f"•  {fact}"
        for line in _wrap_text(bullet_text, font_r, PAGE_WIDTH - 130):
            draw.text((65, y), line, font=font_r, fill=(45, 45, 45))
            y += 22
    y += 15

    # Appellate Remedy Box
    a_box = [50, y, PAGE_WIDTH - 50, y + 105]
    draw.rectangle(a_box, fill=(245, 248, 255), outline=(50, 90, 160), width=1)
    draw.text((65, y + 12), "STATUTORY RIGHT TO APPEAL / SUBMIT RECTIFICATION:", font=font_b, fill=(20, 60, 140))
    appeal_msg = (
        f"Under Rule 14 of the Scheme Guidelines, if you are aggrieved by this decision, you have the right to file "
        f"an appeal before the First Appellate Authority on or before {notice.deadline_date} (within 30 days of this order). "
        f"You must enclose certified copies of income certificates, domicile records, or rectified affidavits with your petition."
    )
    for line in _wrap_text(appeal_msg, font_r, PAGE_WIDTH - 140):
        draw.text((65, y + 36), line, font=font_r, fill=(20, 50, 110))
        y += 20
    y += 45

    _draw_circular_stamp(draw, 220, y + 60, radius=48, top_text="SCHEME SCRUTINY", sub_text="REJECTED", color=(160, 30, 30))
    draw.text((PAGE_WIDTH - 340, y + 40), "By Order of Competent Authority,", font=font_r, fill=(40, 40, 40))
    draw.text((PAGE_WIDTH - 340, y + 65), "DISTRICT WELFARE OFFICER", font=font_b, fill=(70, 20, 20))
    draw.text((PAGE_WIDTH - 340, y + 85), "Directorate of Social Assistance", font=font_r, fill=(70, 70, 70))
    return img


def render_rti_citizen_request(notice: NoticeLabel) -> Image.Image:
    """Render authentic citizen-written RTI request letter seeking government information."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Simple, non-governmental single-line margin for citizen letter
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=40, double=False, color=(50, 50, 50))

    font_title = _load_font("times", 18, bold=True)
    font_b = _load_font("times", 14, bold=True)
    font_r = _load_font("times", 14, bold=False)

    y = 55
    # Citizen Header / Application Form Title
    draw.text((PAGE_WIDTH // 2 - 230, y), "FORM 'A': APPLICATION UNDER RIGHT TO INFORMATION ACT, 2005", font=font_title, fill=(15, 15, 15))
    y += 30
    draw.text((PAGE_WIDTH // 2 - 120, y), "[See Section 6(1) of the RTI Act, 2005]", font=font_r, fill=(70, 70, 70))
    y += 20
    draw.line([(55, y), (PAGE_WIDTH - 55, y)], fill=(80, 80, 80), width=1)
    y += 20

    # Addressed To:
    draw.text((65, y), "To,", font=font_b, fill=(20, 20, 20))
    y += 22
    target_pio = notice.issuer or "The Public Information Officer (PIO)"
    draw.text((65, y), f"The Central / State Public Information Officer (CPIO / SPIO),", font=font_b, fill=(20, 20, 20))
    y += 22
    draw.text((65, y), f"{target_pio},", font=font_r, fill=(30, 30, 30))
    y += 22
    draw.text((65, y), "Office of the Competent Public Authority, Division / Circle Office.", font=font_r, fill=(30, 30, 30))
    y += 30

    # From Citizen:
    draw.text((65, y), "1. Name of the Citizen / Applicant:", font=font_b, fill=(20, 20, 20))
    draw.text((360, y), "Sri K. Venkatesham / Citizen Applicant", font=font_r, fill=(10, 10, 10))
    y += 24
    draw.text((65, y), "2. Permanent Address of Applicant:", font=font_b, fill=(20, 20, 20))
    draw.text((360, y), "H.No. 4-21/A, Ward 102, Citizen Colony, Hyderabad - 500072", font=font_r, fill=(10, 10, 10))
    y += 24
    draw.text((65, y), "3. Citizen Tracking Reference:", font=font_b, fill=(20, 20, 20))
    draw.text((360, y), notice.reference_number or "RTI-CITIZEN-REQ-2026", font=font_r, fill=(10, 10, 10))
    y += 24
    draw.text((65, y), "4. Date of Application:", font=font_b, fill=(20, 20, 20))
    draw.text((360, y), notice.issue_date or "2026-11-05", font=font_r, fill=(10, 10, 10))
    y += 30

    # Subject
    draw.text((65, y), "SUBJECT:", font=font_b, fill=(15, 15, 15))
    subj_text = "Request for certified information and documents under Section 6(1) of the RTI Act 2005."
    draw.text((150, y), subj_text, font=font_b, fill=(15, 15, 15))
    y += 28

    # Narrative
    draw.text((65, y), "Sir / Madam,", font=font_r, fill=(20, 20, 20))
    y += 24
    p1 = (
        "I am a citizen of India. I hereby request you to provide certified, authentic copies of records "
        "and specific administrative information regarding the following public matter as detailed below:"
    )
    for line in _wrap_text(p1, font_r, PAGE_WIDTH - 140):
        draw.text((65, y), line, font=font_r, fill=(30, 30, 30))
        y += 22
    y += 12

    # Numbered Questions / Information Sought
    draw.text((65, y), "PARTICULARS OF INFORMATION SOUGHT (Key Citizen Queries):", font=font_b, fill=(20, 20, 20))
    y += 26
    for idx, fact in enumerate(notice.key_facts, start=1):
        q_line = f"Question {idx}:  {fact}"
        for line in _wrap_text(q_line, font_r, PAGE_WIDTH - 150):
            draw.text((80, y), line, font=font_r, fill=(25, 25, 25))
            y += 22
        y += 4
    y += 15

    # Statutory Fee & Section 7(1) Timeline Box
    fee_box = [60, y, PAGE_WIDTH - 60, y + 90]
    draw.rectangle(fee_box, fill=(250, 252, 250), outline=(70, 120, 70), width=1)
    draw.text((75, y + 12), "STATUTORY APPLICATION FEE & RESPONSE DEADLINE:", font=font_b, fill=(25, 90, 35))
    amt_str = f"INR {notice.amount_inr:.2f}" if notice.amount_inr is not None else "INR 10.00"
    fee_msg = (
        f"A prescribed application fee of {amt_str} has been paid via Indian Postal Order / Court Fee Stamp No. 948102. "
        f"As per Section 7(1) of the RTI Act 2005, you are requested to furnish the said information within 30 days "
        f"from receipt of this application (Statutory Response Deadline: {notice.deadline_date})."
    )
    for line in _wrap_text(fee_msg, font_r, PAGE_WIDTH - 160):
        draw.text((75, y + 36), line, font=font_r, fill=(35, 75, 45))
        y += 20
    y += 45

    # Citizen Signature
    draw.text((PAGE_WIDTH - 300, y + 30), "Yours faithfully,", font=font_r, fill=(30, 30, 30))
    draw.text((PAGE_WIDTH - 300, y + 60), "K. Venkatesham", font=font_b, fill=(20, 20, 20))
    draw.text((PAGE_WIDTH - 300, y + 80), "(Citizen Applicant)", font=font_r, fill=(60, 60, 60))
    draw.text((65, y + 80), "Enclosure: IPO Fee Receipt & ID Copy", font=font_r, fill=(70, 70, 70))
    return img


# ---------------------------------------------------------------------------
# Out of Scope Renderers (Grocery, Wedding, Note, Blur, Lease)
# ---------------------------------------------------------------------------
def render_grocery_receipt(notice: NoticeLabel) -> Image.Image:
    """Render realistic thermal cash receipt from a grocery supermarket."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (250, 250, 245))
    draw = ImageDraw.Draw(img)
    # Narrow thermal receipt centered on page
    rw = 550
    rx = (PAGE_WIDTH - rw) // 2
    draw.rectangle([rx, 60, rx + rw, PAGE_HEIGHT - 60], fill=(255, 255, 255), outline=(200, 200, 200), width=1)

    font_th = _load_font("courier", 16, bold=True)
    font_tr = _load_font("courier", 13, bold=False)

    y = 90
    draw.text((rx + 110, y), "*** FRESHDAILY SUPERMARKET ***", font=font_th, fill=(20, 20, 20))
    y += 24
    draw.text((rx + 120, y), "Fresh Foods Retail India Pvt Ltd", font=font_tr, fill=(60, 60, 60))
    y += 20
    draw.text((rx + 140, y), "GSTIN: 36AABCF9021K1ZM", font=font_tr, fill=(60, 60, 60))
    y += 24
    draw.text((rx + 30, y), "-" * 42, font=font_tr, fill=(100, 100, 100))
    y += 18
    draw.text((rx + 30, y), f"Receipt: {notice.reference_number}    Date: {notice.issue_date}", font=font_tr, fill=(40, 40, 40))
    y += 18
    draw.text((rx + 30, y), "Cashier: POS-Terminal 04      Time: 18:42:19", font=font_tr, fill=(40, 40, 40))
    y += 18
    draw.text((rx + 30, y), "-" * 42, font=font_tr, fill=(100, 100, 100))
    y += 22

    draw.text((rx + 30, y), "ITEM DESCRIPTION        QTY     PRICE      TOTAL", font=font_th, fill=(20, 20, 20))
    y += 20
    draw.text((rx + 30, y), "-" * 42, font=font_tr, fill=(100, 100, 100))
    y += 18

    items = [
        ("Organic Cow Milk 1L", "2", "68.00", "136.00"),
        ("Sona Masoori Rice 5kg", "1", "380.00", "380.00"),
        ("Sunflower Refined Oil 1L", "2", "145.00", "290.00"),
        ("Toor Dal Supreme 1kg", "2", "185.00", "370.00"),
        ("Farm Fresh Tomatoes 1kg", "1", "44.50", "44.50"),
        ("Fresh Brown Eggs (12pk)", "1", "110.00", "110.00"),
        ("Britannia Marie Biscuits", "2", "45.00", "90.00"),
    ]
    for name, qty, price, tot in items:
        line = f"{name[:22]:<22}  {qty:>3}  {price:>8}  {tot:>9}"
        draw.text((rx + 30, y), line, font=font_tr, fill=(30, 30, 30))
        y += 22
    y += 15

    draw.text((rx + 30, y), "=" * 42, font=font_tr, fill=(100, 100, 100))
    y += 20
    draw.text((rx + 30, y), f"SUB-TOTAL (7 Items):                  INR 1,420.50", font=font_th, fill=(20, 20, 20))
    y += 22
    draw.text((rx + 30, y), "CGST @ 2.5%:                            INR   18.25", font=font_tr, fill=(60, 60, 60))
    y += 20
    draw.text((rx + 30, y), "SGST @ 2.5%:                            INR   18.25", font=font_tr, fill=(60, 60, 60))
    y += 22
    draw.text((rx + 30, y), f"TOTAL AMOUNT PAID:                    INR {notice.amount_inr:,.2f}", font=font_th, fill=(10, 10, 10))
    y += 24
    draw.text((rx + 30, y), "Payment Mode: UPI / PhonePe (TXN ID: 9021481)", font=font_tr, fill=(40, 40, 40))
    y += 22
    draw.text((rx + 30, y), "-" * 42, font=font_tr, fill=(100, 100, 100))
    y += 30
    draw.text((rx + 110, y), "THANK YOU FOR SHOPPING WITH US!", font=font_th, fill=(40, 40, 40))
    y += 20
    draw.text((rx + 150, y), "Please Visit Again!", font=font_tr, fill=(70, 70, 70))
    y += 40
    _draw_barcode(draw, rx + 160, y, width=220, height=45)
    return img


def render_wedding_invitation(notice: NoticeLabel) -> Image.Image:
    """Render personal wedding reception invitation card."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 252, 248))
    draw = ImageDraw.Draw(img)
    # Elegant ornate double border in rich burgundy and gold
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=45, double=True, color=(140, 30, 45))

    font_title = _load_font("times", 24, bold=True)
    font_names = _load_font("script", 32, bold=True)
    font_body = _load_font("times", 15, bold=False)
    font_b = _load_font("times", 16, bold=True)

    y = 120
    draw.text((PAGE_WIDTH // 2 - 130, y), "|| SHREE GANESHAYA NAMAHA ||", font=font_b, fill=(160, 40, 40))
    y += 50
    draw.text((PAGE_WIDTH // 2 - 160, y), "TOGETHER WITH THEIR FAMILIES", font=font_body, fill=(100, 80, 80))
    y += 40
    draw.text((PAGE_WIDTH // 2 - 200, y), "CORDIALLY INVITE YOU TO CELEBRATE", font=font_body, fill=(100, 80, 80))
    y += 30
    draw.text((PAGE_WIDTH // 2 - 140, y), "THE WEDDING RECEPTION OF", font=font_body, fill=(100, 80, 80))
    y += 60

    draw.text((PAGE_WIDTH // 2 - 80, y), "Rahul", font=font_names, fill=(140, 25, 40))
    y += 45
    draw.text((PAGE_WIDTH // 2 - 15, y), "&", font=font_title, fill=(180, 130, 40))
    y += 45
    draw.text((PAGE_WIDTH // 2 - 90, y), "Sneha", font=font_names, fill=(140, 25, 40))
    y += 70

    draw.text((PAGE_WIDTH // 2 - 190, y), "ON SUNDAY, DECEMBER 14, 2026", font=font_title, fill=(140, 25, 40))
    y += 40
    draw.text((PAGE_WIDTH // 2 - 100, y), "AT 7:00 PM ONWARDS", font=font_b, fill=(80, 60, 60))
    y += 60

    draw.text((PAGE_WIDTH // 2 - 120, y), "VENUE: ROYAL PALACE GROUNDS", font=font_b, fill=(40, 40, 40))
    y += 28
    draw.text((PAGE_WIDTH // 2 - 140, y), "Gate No. 4, Bellary Road, Bengaluru", font=font_body, fill=(60, 60, 60))
    y += 70

    draw.text((PAGE_WIDTH // 2 - 160, y), "DINNER AND CELEBRATIONS TO FOLLOW", font=font_body, fill=(100, 80, 80))
    y += 50
    draw.text((PAGE_WIDTH // 2 - 100, y), "RSVP: The Sharma & Verma Family", font=font_body, fill=(120, 90, 90))
    return img


def render_personal_note(notice: NoticeLabel) -> Image.Image:
    """Render personal handwritten to-do list / diary note."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 250))
    draw = ImageDraw.Draw(img)

    # Notebook red margin line on left
    draw.line([(120, 0), (120, PAGE_HEIGHT)], fill=(230, 100, 100), width=2)
    # Blue ruled lines across notebook
    for ly in range(100, PAGE_HEIGHT - 80, 36):
        draw.line([(0, ly), (PAGE_WIDTH, ly)], fill=(210, 230, 245), width=1)

    font_note = _load_font("script", 18, bold=False)
    font_h = _load_font("script", 22, bold=True)

    draw.text((150, 70), "Personal To-Do List & Weekend Reminders:", font=font_h, fill=(20, 40, 100))
    y = 120
    draw.text((PAGE_WIDTH - 240, 70), "Date: 14/11/2026", font=font_note, fill=(50, 50, 110))

    tasks = [
        "1. Buy milk, eggs, toor dal & fresh vegetables from supermarket.",
        "2. Call dental clinic for routine root canal follow-up appointment.",
        "3. Drop laundry at dry cleaners before 4:00 PM.",
        "4. Pay apartment monthly maintenance charges online.",
        "5. Water balcony plants and replace soil in jade planter.",
        "6. Book train tickets for Diwali family trip to Nagpur.",
        "7. Finish reading Chapter 4 of machine learning systems book.",
        "8. Renew quarterly gym membership.",
    ]
    for t in tasks:
        draw.text((150, y), t, font=font_note, fill=(30, 50, 120))
        y += 72
    return img


def render_unreadable_blur(notice: NoticeLabel) -> Image.Image:
    """Render heavily blurred, corrupted, completely unreadable document image."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (240, 240, 235))
    draw = ImageDraw.Draw(img)

    # Draw irregular shapes, text fragments, and shadows
    font = _load_font("arial", 20, bold=True)
    draw.rectangle([60, 60, PAGE_WIDTH - 60, 180], fill=(180, 190, 200))
    draw.text((100, 100), "CONFIDENTIAL NOTICE AND DEMAND 2026", font=font, fill=(50, 50, 50))
    for i in range(250, PAGE_HEIGHT - 100, 45):
        draw.rectangle([80, i, 450, i + 25], fill=(140, 140, 150))
        draw.rectangle([500, i, PAGE_WIDTH - 80, i + 25], fill=(160, 160, 170))

    # Dark lighting vignette on top and bottom
    for row in range(0, 120):
        alpha = int((120 - row) * 1.5)
        draw.line([(0, row), (PAGE_WIDTH, row)], fill=(alpha, alpha, alpha), width=1)

    # Heavy Gaussian Blur to make all text unrecognizable blotches
    blurred = img.filter(ImageFilter.GaussianBlur(radius=15.0))
    return blurred


def render_lease_agreement(notice: NoticeLabel) -> Image.Image:
    """Render residential tenancy agreement snippet."""
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    _draw_border(draw, PAGE_WIDTH, PAGE_HEIGHT, margin=40, double=False, color=(30, 30, 30))

    font_t = _load_font("times", 18, bold=True)
    font_b = _load_font("times", 13, bold=True)
    font_r = _load_font("times", 13, bold=False)

    y = 60
    draw.text((PAGE_WIDTH // 2 - 180, y), "RESIDENTIAL TENANCY LEASE AGREEMENT", font=font_t, fill=(20, 20, 20))
    y += 35
    draw.text((PAGE_WIDTH // 2 - 140, y), "Agreement No: LEASE-2026-RES-77", font=font_b, fill=(70, 70, 70))
    y += 20
    draw.line([(55, y), (PAGE_WIDTH - 55, y)], fill=(80, 80, 80), width=1)
    y += 25

    intro = (
        "THIS AGREEMENT OF LEASE is made and executed on this 1st day of October, 2026 at Hyderabad, "
        "by and between Mr. Rajesh Sharma, residing at Banjara Hills (hereinafter called the 'LESSOR') "
        "and Ms. Priya Rao, software engineer (hereinafter called the 'LESSEE')."
    )
    for line in _wrap_text(intro, font_r, PAGE_WIDTH - 130):
        draw.text((65, y), line, font=font_r, fill=(25, 25, 25))
        y += 22
    y += 15

    clauses = [
        ("1. DEMISED PREMISES:", "Flat No. 302, Cyber Heights, Gachibowli, Hyderabad - 500032 comprising 2 BHK."),
        ("2. MONTHLY RENT:", "The Lessee shall pay a monthly rent of INR 22,000.00 (Twenty-Two Thousand Only) on or before the 5th day of every calendar month."),
        ("3. INTEREST-FREE SECURITY DEPOSIT:", "The Lessee has deposited an amount of INR 1,00,000.00 as refundable security deposit with the Lessor."),
        ("4. LEASE DURATION:", "The lease shall be valid for a continuous period of 11 months commencing from 2026-10-01."),
        ("5. TERMINATION NOTICE:", "Either party may terminate this agreement by providing one calendar month advance written notice."),
    ]
    for c_title, c_desc in clauses:
        draw.text((65, y), c_title, font=font_b, fill=(20, 20, 20))
        y += 20
        for line in _wrap_text(c_desc, font_r, PAGE_WIDTH - 150):
            draw.text((85, y), line, font=font_r, fill=(35, 35, 35))
            y += 20
        y += 8
    y += 40

    draw.text((85, y), "IN WITNESS WHEREOF the parties have set their hands on the day and year first written above:", font=font_r, fill=(40, 40, 40))
    y += 60

    draw.text((100, y), "Mr. Rajesh Sharma", font=font_b, fill=(20, 20, 20))
    draw.text((100, y + 20), "(LESSOR / LANDLORD)", font=font_r, fill=(60, 60, 60))

    draw.text((PAGE_WIDTH - 300, y), "Ms. Priya Rao", font=font_b, fill=(20, 20, 20))
    draw.text((PAGE_WIDTH - 300, y + 20), "(LESSEE / TENANT)", font=font_r, fill=(60, 60, 60))
    return img


# ---------------------------------------------------------------------------
# PDF Saving Helper (Using PyMuPDF)
# ---------------------------------------------------------------------------
def save_image_as_pdf(
    img: Image.Image,
    pdf_path: Path,
    searchable_text: str = "",
) -> None:
    """Save rendered Pillow image as a valid high-resolution PDF with PyMuPDF."""
    buf = io.BytesIO()
    # Save as high-quality JPEG for compact, standards-compliant PDF embedding
    img.save(buf, format="JPEG", quality=92)
    img_bytes = buf.getvalue()

    doc = pymupdf.open()
    # Standard A4 size in points (595 x 842 pt)
    page = doc.new_page(width=595, height=842)
    # Insert visual rendered document
    page.insert_image(page.rect, stream=img_bytes)

    # Insert searchable text layer
    if searchable_text:
        # Invisible text (render_mode=3) for searchability and text extraction
        page.insert_text(
            (30, 40),
            searchable_text[:2000],
            fontsize=7,
            render_mode=3,
        )
    doc.save(str(pdf_path), deflate=True)
    doc.close()


# ---------------------------------------------------------------------------
# Complete 30 Notice Ground Truth Catalog
# ---------------------------------------------------------------------------
SYNTHETIC_NOTICES_CATALOG: List[Dict[str, Any]] = [
    # ---------------- Category 1: electricity (5 notices) ----------------
    {
        "filename": "notice_01_elec_disconnection.png",
        "doc_type": DocumentType.electricity,
        "issuer": "Southern Power Distribution Company of Telangana Limited (TSSPDCL)",
        "reference_number": "TSSPDCL-DISC-2026-89421",
        "amount_inr": 4250.0,
        "issue_date": "2026-10-15",
        "deadline_date": "2026-10-30",
        "key_facts": [
            "Consumer Service Number: 114029419 under Banjara Hills Sub-Division",
            "Accumulated electricity arrears of INR 4,250.00 unpaid for September 2026 billing cycle",
            "Final disconnection notice issued under Section 56 of Electricity Act 2003",
            "Power supply will be disconnected without further notice on or after 2026-10-30",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "TSSPDCL statutory electricity disconnection demand notice for overdue residential billing arrears.",
        "format": "png",
    },
    {
        "filename": "notice_02_elec_meter_tamper.pdf",
        "doc_type": DocumentType.electricity,
        "issuer": "Bangalore Electricity Supply Company Limited (BESCOM)",
        "reference_number": "BESCOM/NZ/VIG/2026/4102",
        "amount_inr": 12800.0,
        "issue_date": "2026-10-20",
        "deadline_date": "2026-11-10",
        "key_facts": [
            "Account ID: 5930214890, Malleshwaram Division",
            "Vigilance inspection detected meter seal breakage and unmetered load",
            "Provisional assessment penalty of INR 12,800.00 levied under Section 126 of Electricity Act",
            "Objections may be filed before Assessing Officer before 2026-11-10",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "BESCOM provisional assessment penalty notice for meter inspection irregularities.",
        "format": "pdf",
    },
    {
        "filename": "notice_03_elec_arrears.png",
        "doc_type": DocumentType.electricity,
        "issuer": "Maharashtra State Electricity Distribution Company Limited (MSEDCL)",
        "reference_number": "MSEDCL-PUN-ARR-99214",
        "amount_inr": 6400.0,
        "issue_date": "2026-11-01",
        "deadline_date": "2026-11-20",
        "key_facts": [
            "Consumer Number: 170940381254, Kothrud Division, Pune",
            "Outstanding arrears amount of INR 6,400.00 including delayed payment surcharge",
            "Immediate payment required to prevent recovery proceedings and line cutoff by 2026-11-20",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "MSEDCL final demand notice for outstanding arrears and delayed payment surcharges.",
        "format": "png",
    },
    {
        "filename": "notice_04_elec_tariff_revision.pdf",
        "doc_type": DocumentType.electricity,
        "issuer": "Tata Power Delhi Distribution Limited (TPDDL)",
        "reference_number": "TPDDL/REV/ND/2026/7821",
        "amount_inr": 1850.0,
        "issue_date": "2026-11-05",
        "deadline_date": "2026-11-25",
        "key_facts": [
            "CA Number: 60019284715, Rohini Circle, New Delhi",
            "Power purchase cost adjustment surcharge (PPAC) retrospective debit of INR 1,850.00",
            "Payment due date fixed for 2026-11-25 before late surcharge addition",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "TPDDL supplementary bill and tariff revision assessment notice.",
        "format": "pdf",
    },
    {
        "filename": "notice_05_elec_degraded.png",
        "doc_type": DocumentType.electricity,
        "issuer": "Southern Power Distribution Company of Telangana Limited (TSSPDCL)",
        "reference_number": "TSSPDCL-HYD-55192-S",
        "amount_inr": 8950.0,
        "issue_date": "2026-10-28",
        "deadline_date": "2026-11-12",
        "key_facts": [
            "Consumer ID: 228019342, Charminar Division, Hyderabad",
            "Accumulated commercial tariff arrears and reconnection fee of INR 8,950.00",
            "Final notice to clear arrears by 2026-11-12 or face permanent line dismantlement",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": True,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Degraded scan of TSSPDCL commercial electricity arrears demand notice.",
        "format": "png",
    },
    # ---------------- Category 2: challan (5 notices) ----------------
    {
        "filename": "notice_06_challan_speeding.png",
        "doc_type": DocumentType.challan,
        "issuer": "Hyderabad Traffic Police",
        "reference_number": "HYD-ECH-2026-55102",
        "amount_inr": 1000.0,
        "issue_date": "2026-11-02",
        "deadline_date": "2026-11-17",
        "key_facts": [
            "Vehicle Registration: TS09EA4821",
            "Violation: Exceeding prescribed speed limit (88 km/h in 60 km/h zone) on Outer Ring Road Gachibowli",
            "Fine of INR 1,000.00 imposed under Section 183(1) of Motor Vehicles Act 1988",
            "Pay online at echallan.tspolice.gov.in before 2026-11-17",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Hyderabad Traffic Police camera-recorded over-speeding e-challan.",
        "format": "png",
    },
    {
        "filename": "notice_07_challan_signal.pdf",
        "doc_type": DocumentType.challan,
        "issuer": "Delhi Traffic Police",
        "reference_number": "DTP-ECH-2026-90214",
        "amount_inr": 1000.0,
        "issue_date": "2026-11-04",
        "deadline_date": "2026-11-24",
        "key_facts": [
            "Vehicle Number: DL10CA3198",
            "Violation: Jumping red light and crossing zebra crossing stop-line at ITO Intersection",
            "Penalty amount: INR 1,000.00 under Section 184 of Motor Vehicles Act",
            "Mandatory settlement deadline: 2026-11-24 prior to Virtual Court referral",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Delhi Traffic Police red light jumping and stop-line violation notice.",
        "format": "pdf",
    },
    {
        "filename": "notice_08_challan_helmet.png",
        "doc_type": DocumentType.challan,
        "issuer": "Bengaluru Traffic Police",
        "reference_number": "BTP-CH-BLR-2026-33910",
        "amount_inr": 500.0,
        "issue_date": "2026-11-06",
        "deadline_date": "2026-11-26",
        "key_facts": [
            "Vehicle: KA04EJ9123 (Two-Wheeler)",
            "Violation: Riding two-wheeler without ISI marked safety helmet at Indiranagar 100ft Road",
            "Fine of INR 500.00 under Section 129 r/w 177 of MV Act",
            "Payment due within 20 days by 2026-11-26 via Bangalore One or BTP portal",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Bengaluru Traffic Police helmet violation e-challan notice.",
        "format": "png",
    },
    {
        "filename": "notice_09_challan_wrong_side.pdf",
        "doc_type": DocumentType.challan,
        "issuer": "Mumbai Traffic Police",
        "reference_number": "MTP-E-2026-881234",
        "amount_inr": 2000.0,
        "issue_date": "2026-10-25",
        "deadline_date": "2026-11-15",
        "key_facts": [
            "Vehicle Registration: MH02DB6719",
            "Violation: Driving in opposite direction on one-way street (Wrong Side Driving) at Andheri East",
            "Compounded penalty of INR 2,000.00 under Section 184 MV Act",
            "Hearing / payment deadline before Virtual Court: 2026-11-15",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Mumbai Traffic Police dangerous wrong-side driving notice.",
        "format": "pdf",
    },
    {
        "filename": "notice_10_challan_degraded.png",
        "doc_type": DocumentType.challan,
        "issuer": "Hyderabad Traffic Police",
        "reference_number": "HYD-ECH-2026-77821",
        "amount_inr": 1200.0,
        "issue_date": "2026-10-18",
        "deadline_date": "2026-11-03",
        "key_facts": [
            "Vehicle: TS11AB5542",
            "Violations: Pillion riding without helmet and dangerous parking on footpath at Secunderabad Station",
            "Total compounded fine: INR 1,200.00",
            "Last date for payment: 2026-11-03 before Lok Adalat summons",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": True,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Degraded scan of Hyderabad Traffic Police multiple violations e-challan.",
        "format": "png",
    },
    # ---------------- Category 3: tax (5 notices) ----------------
    {
        "filename": "notice_11_tax_property_demand.png",
        "doc_type": DocumentType.tax,
        "issuer": "Greater Hyderabad Municipal Corporation (GHMC)",
        "reference_number": "GHMC/PT/CZ/2026/99341",
        "amount_inr": 18500.0,
        "issue_date": "2026-10-10",
        "deadline_date": "2026-11-10",
        "key_facts": [
            "Property Assessment PTIN: 1082049182, Central Zone, Khairatabad",
            "Property Tax Demand Notice for financial year 2026-27 including 5% advance rebate adjustment",
            "Total net tax payable: INR 18,500.00",
            "Due date for payment: 2026-11-10, after which 2% monthly penalty will accrue",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "GHMC property tax assessment demand notice for FY 2026-27.",
        "format": "png",
    },
    {
        "filename": "notice_12_tax_assessment_revision.pdf",
        "doc_type": DocumentType.tax,
        "issuer": "Bruhat Bengaluru Mahanagara Palike (BBMP)",
        "reference_number": "BBMP/REV/SZ/2026/14290",
        "amount_inr": 24000.0,
        "issue_date": "2026-10-22",
        "deadline_date": "2026-11-21",
        "key_facts": [
            "SAS Property Number: 084-W0152-39, South Zone, Jayanagar",
            "Revised property tax assessment due to undeclared built-up area detected during GIS drone survey",
            "Differential tax and penalty amount: INR 24,000.00",
            "Appeal or payment window expires on 2026-11-21 (30 days from notice)",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "BBMP re-assessment and differential tax notice based on GIS survey.",
        "format": "pdf",
    },
    {
        "filename": "notice_13_tax_municipal_arrears.png",
        "doc_type": DocumentType.tax,
        "issuer": "Brihanmumbai Municipal Corporation (BMC)",
        "reference_number": "BMC-KW-TAX-2026-0811",
        "amount_inr": 14200.0,
        "issue_date": "2026-11-03",
        "deadline_date": "2026-11-23",
        "key_facts": [
            "Property Account Number: KW04820199000, Andheri West Ward K-West",
            "Demand for municipal property tax and sewerage tax arrears of INR 14,200.00",
            "Notice issued under Section 202 of Mumbai Municipal Corporation Act 1888",
            "Payable on or before 2026-11-23 to prevent attachment of property",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "BMC property and sewerage tax arrears recovery notice.",
        "format": "png",
    },
    {
        "filename": "notice_14_tax_commercial_tax.pdf",
        "doc_type": DocumentType.tax,
        "issuer": "Municipal Corporation of Delhi (MCD)",
        "reference_number": "MCD/PROP/ND/2026/65123",
        "amount_inr": 32000.0,
        "issue_date": "2026-10-29",
        "deadline_date": "2026-11-30",
        "key_facts": [
            "UPIC: 5041928014, Karol Bagh Zone, New Delhi",
            "Annual commercial property tax assessment notice for FY 2026-27",
            "Total assessed demand: INR 32,000.00 including commercial use surcharge",
            "Last date for payment without interest is 2026-11-30",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "MCD commercial property tax demand notice with commercial surcharge.",
        "format": "pdf",
    },
    {
        "filename": "notice_15_tax_degraded.png",
        "doc_type": DocumentType.tax,
        "issuer": "Greater Hyderabad Municipal Corporation (GHMC)",
        "reference_number": "GHMC/WARR/WZ/2026/0412",
        "amount_inr": 21500.0,
        "issue_date": "2026-10-12",
        "deadline_date": "2026-11-05",
        "key_facts": [
            "PTIN: 1029481902, Serilingampally Circle 20",
            "Final Distraint Warrant Notice for non-payment of property tax dues",
            "Overdue balance: INR 21,500.00",
            "Mandatory settlement deadline: 2026-11-05 prior to sealing of commercial premises",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": True,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Degraded scan of GHMC final distraint warrant for property tax non-payment.",
        "format": "png",
    },
    # ---------------- Category 4: scheme_rejection (5 notices) ----------------
    {
        "filename": "notice_16_scheme_ration_card.pdf",
        "doc_type": DocumentType.scheme_rejection,
        "issuer": "Department of Consumer Affairs, Food and Civil Supplies, Govt. of Telangana",
        "reference_number": "FCS/NFSA/REJ/2026/7821",
        "amount_inr": None,
        "issue_date": "2026-10-14",
        "deadline_date": "2026-11-13",
        "key_facts": [
            "Application ID: FSC-2026-HYD-481920 submitted on 2026-08-10",
            "Rejection of National Food Security Act (NFSA) Priority Household Ration Card",
            "Ground for rejection: Annual household income exceeds rural/urban threshold of INR 1,50,000 based on tax records",
            "Citizen may file appeal before District Food and Civil Supplies Officer within 30 days by 2026-11-13",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Official rejection order for NFSA Priority Ration Card application due to income ceiling.",
        "format": "pdf",
    },
    {
        "filename": "notice_17_scheme_pension.png",
        "doc_type": DocumentType.scheme_rejection,
        "issuer": "Directorate of Social Security and Pensions, Govt. of Karnataka",
        "reference_number": "DSSP/OAP/2026/90124",
        "amount_inr": None,
        "issue_date": "2026-10-26",
        "deadline_date": "2026-11-25",
        "key_facts": [
            "Beneficiary Application No: SSP-BLR-2026-11490 for Sandhya Suraksha / Old Age Pension",
            "Application rejected after field verification",
            "Reason: Failure to furnish valid Karnataka domicile certificate of continuous residence (>10 years)",
            "Appellate authority: Assistant Commissioner (Revenue) within 30 days by 2026-11-25",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Rejection order for Old Age Social Security Pension due to missing domicile proof.",
        "format": "png",
    },
    {
        "filename": "notice_18_scheme_housing.pdf",
        "doc_type": DocumentType.scheme_rejection,
        "issuer": "Pradhan Mantri Awas Yojana (Urban), State Level Nodal Agency",
        "reference_number": "PMAY-U/REV/2026/34190",
        "amount_inr": None,
        "issue_date": "2026-11-02",
        "deadline_date": "2026-12-02",
        "key_facts": [
            "Beneficiary Registration: PMAY-BLR-URB-88319 under Credit Linked Subsidy Scheme (CLSS)",
            "Subsidy sanction rejected by State Screening Committee",
            "Ground for rejection: Existing ownership of pucca house recorded in municipal property index in family member's name",
            "Citizen can submit representation with clarification documents by 2026-12-02",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "PMAY-Urban interest subsidy scheme rejection order due to pre-existing property ownership.",
        "format": "pdf",
    },
    {
        "filename": "notice_19_scheme_scholarship.png",
        "doc_type": DocumentType.scheme_rejection,
        "issuer": "Social Welfare Department, Govt. of Maharashtra",
        "reference_number": "DSW/MAHA/PMS/2026/55210",
        "amount_inr": None,
        "issue_date": "2026-10-31",
        "deadline_date": "2026-11-20",
        "key_facts": [
            "Application ID: MAHA-PMS-2026-94812 under Post-Matric Scholarship Scheme for SC/ST/OBC Students",
            "Disqualification notice issued",
            "Reason for rejection: Discrepancy between Aadhaar-seeded bank account name and college enrollment records",
            "Rectification or grievance representation to be submitted online by 2026-11-20",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Post-Matric Scholarship rejection letter due to student Aadhaar-bank account mismatch.",
        "format": "png",
    },
    {
        "filename": "notice_20_scheme_degraded.png",
        "doc_type": DocumentType.scheme_rejection,
        "issuer": "Department of Women and Child Development",
        "reference_number": "WCD/PMMVY/REJ/2026/1944",
        "amount_inr": None,
        "issue_date": "2026-10-19",
        "deadline_date": "2026-11-18",
        "key_facts": [
            "Application ID: PMMVY-2026-TG-88120 for Pradhan Mantri Matru Vandana Yojana installment",
            "Application rejected at Project Child Development Officer (CDPO) level",
            "Reason: Mother and Child Protection (MCP) card registration submitted beyond statutory 150-day window",
            "Review petition allowed before District Collector by 2026-11-18",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": True,
        "is_out_of_scope": False,
        "is_citizen_request": False,
        "description": "Degraded scan of maternity financial benefit rejection order under PMMVY.",
        "format": "png",
    },
    # ---------------- Category 5: rti (5 citizen requests) ----------------
    {
        "filename": "notice_21_rti_road_repair.png",
        "doc_type": DocumentType.rti,
        "issuer": "Public Information Officer, Greater Hyderabad Municipal Corporation (GHMC)",
        "reference_number": "RTI-CITIZEN-REQ-2026-01",
        "amount_inr": 10.0,
        "issue_date": "2026-11-05",
        "deadline_date": "2026-12-05",
        "key_facts": [
            "Citizen-written formal RTI application under Section 6(1) of RTI Act 2005",
            "Applicant: K. Venkatesham, resident of Ward 102, Kukatpally, Hyderabad",
            "Information sought: Sanctioned budget, contractor name, tender copy, and approved completion timeline for main road resurfacing",
            "Applicant requests inspection of work measurements and reasons for 8-month delay",
            "Statutory fee of INR 10.00 attached via Indian Postal Order",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": True,
        "description": "Citizen-written RTI application seeking municipal road repair expenditure and contractor records.",
        "format": "png",
    },
    {
        "filename": "notice_22_rti_ration_delay.pdf",
        "doc_type": DocumentType.rti,
        "issuer": "Public Information Officer, Department of Food & Civil Supplies, Bengaluru Urban",
        "reference_number": "RTI-CITIZEN-REQ-2026-02",
        "amount_inr": 10.0,
        "issue_date": "2026-11-07",
        "deadline_date": "2026-12-07",
        "key_facts": [
            "Citizen RTI request filed by Smt. Lakshmi Devi regarding pending BPL Ration Card application (Ack No: BPL-2026-8819)",
            "Citizen requests daily progress report of application since submission in May 2026",
            "Citizen requests names and designations of dealing assistants who handled file",
            "Statutory 30-day response deadline under Section 7(1) expires on 2026-12-07",
            "Fee paid: INR 10.00 court fee stamp affixed",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": True,
        "description": "Citizen-written RTI application inquiring into 6-month delay of ration card processing.",
        "format": "pdf",
    },
    {
        "filename": "notice_23_rti_drainage_fund.png",
        "doc_type": DocumentType.rti,
        "issuer": "Public Information Officer / Executive Engineer, Delhi Jal Board",
        "reference_number": "RTI-CITIZEN-REQ-2026-03",
        "amount_inr": 10.0,
        "issue_date": "2026-11-09",
        "deadline_date": "2026-12-09",
        "key_facts": [
            "Citizen-written application seeking information regarding storm water drainage overhaul in Mayur Vihar Phase 1",
            "Citizen requests certified copies of contractor payments, quality audit reports, and penalty notices for overflow",
            "Citizen seeks details of official responsible for monsoon de-silting oversight",
            "Statutory fee of INR 10.00 paid via demand draft",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": True,
        "description": "Citizen-written RTI query regarding local storm drainage expenditure and desilting contracts.",
        "format": "png",
    },
    {
        "filename": "notice_24_rti_pension_grievance.pdf",
        "doc_type": DocumentType.rti,
        "issuer": "Public Information Officer, Social Welfare Commissionerate, Pune",
        "reference_number": "RTI-CITIZEN-REQ-2026-04",
        "amount_inr": 10.0,
        "issue_date": "2026-11-11",
        "deadline_date": "2026-12-11",
        "key_facts": [
            "Citizen query submitted by Ramesh Patil inquiring why disability pension monthly disbursal ceased from July 2026",
            "Citizen asks for official noting sheets, reasons for benefit suspension, and remedy procedure",
            "Citizen requests particulars of First Appellate Authority",
            "Fee of INR 10.00 paid via non-judicial court fee stamp",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": False,
        "is_citizen_request": True,
        "description": "Citizen-written RTI query seeking reasons and noting sheets for suspended disability pension.",
        "format": "pdf",
    },
    {
        "filename": "notice_25_rti_degraded.png",
        "doc_type": DocumentType.rti,
        "issuer": "Public Information Officer, Zilla Parishad, Rural Water Supply Division",
        "reference_number": "RTI-CITIZEN-REQ-2026-05",
        "amount_inr": 10.0,
        "issue_date": "2026-10-24",
        "deadline_date": "2026-11-24",
        "key_facts": [
            "Informal handwritten citizen RTI letter addressed to Gram Panchayat & Zilla Parishad PIO",
            "Citizen inquiries about drinking water pipeline grant and borewell fund allocation for Village Shivajinagar",
            "Citizen asks for certified ledger of borewell repairs executed in 2025-2026",
            "RTI application fee of INR 10.00 postal order attached",
            "Response sought within statutory 30 days by 2026-11-24",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": True,
        "is_out_of_scope": False,
        "is_citizen_request": True,
        "description": "Degraded scan of informal citizen RTI inquiry letter regarding village drinking water borewell funds.",
        "format": "png",
    },
    # ---------------- Category 6: other (5 out of scope) ----------------
    {
        "filename": "notice_26_oos_grocery_receipt.png",
        "doc_type": DocumentType.other,
        "issuer": "FreshDaily Supermarket Pvt. Ltd.",
        "reference_number": "FD-POS-89214",
        "amount_inr": 1420.50,
        "issue_date": "2026-11-12",
        "deadline_date": None,
        "key_facts": [
            "Point-of-sale printed thermal cash receipt from FreshDaily Supermarket",
            "Purchased items: Organic Milk 2L, Sona Masoori Rice 5kg, Sunflower Oil 1L, Toor Dal 1kg",
            "Total retail bill amount: INR 1,420.50 paid via UPI",
            "Commercial retail receipt with zero civic or governmental relevance",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": True,
        "is_citizen_request": False,
        "description": "Thermal cash register receipt from a retail grocery supermarket.",
        "format": "png",
    },
    {
        "filename": "notice_27_oos_wedding_invite.png",
        "doc_type": DocumentType.other,
        "issuer": None,
        "reference_number": None,
        "amount_inr": None,
        "issue_date": None,
        "deadline_date": None,
        "key_facts": [
            "Personal ceremonial wedding invitation card for the marriage of Rahul and Sneha",
            "Reception date: December 14, 2026 at Palace Grounds, Bengaluru",
            "Social family event invitation with no administrative or legal standing",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": True,
        "is_citizen_request": False,
        "description": "Private wedding reception invitation card.",
        "format": "png",
    },
    {
        "filename": "notice_28_oos_personal_note.png",
        "doc_type": DocumentType.other,
        "issuer": None,
        "reference_number": None,
        "amount_inr": None,
        "issue_date": "2026-11-14",
        "deadline_date": None,
        "key_facts": [
            "Informal personal handwritten to-do list and diary snippet",
            "Items: Buy groceries, call dental clinic for appointment, drop laundry, renew gym membership",
            "Private individual note outside the scope of NoticeWise",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": True,
        "is_citizen_request": False,
        "description": "Informal personal handwritten diary / to-do list page.",
        "format": "png",
    },
    {
        "filename": "notice_29_oos_unreadable_blur.png",
        "doc_type": DocumentType.other,
        "issuer": None,
        "reference_number": None,
        "amount_inr": None,
        "issue_date": None,
        "deadline_date": None,
        "key_facts": [
            "Severely out-of-focus, blurred, corrupted camera image of arbitrary paper",
            "Text is completely illegible and unextractable due to optical blur and low illumination",
            "Should trigger abstention guardrail with low extraction confidence",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": True,
        "is_out_of_scope": True,
        "is_citizen_request": False,
        "description": "Corrupted, out-of-focus, completely unreadable paper photo.",
        "format": "png",
    },
    {
        "filename": "notice_30_oos_lease_agreement.pdf",
        "doc_type": DocumentType.other,
        "issuer": None,
        "reference_number": "LEASE-2026-RES-77",
        "amount_inr": 22000.0,
        "issue_date": "2026-10-01",
        "deadline_date": None,
        "key_facts": [
            "Private residential tenancy lease agreement between landlord Mr. Rajesh Sharma and tenant Ms. Priya Rao",
            "Monthly rental amount: INR 22,000.00 with security deposit of INR 1,00,000.00",
            "Private contractual lease agreement, not a government notice or public authority communication",
        ],
        "language_of_document": "English",
        "is_synthetic": True,
        "is_degraded": False,
        "is_out_of_scope": True,
        "is_citizen_request": False,
        "description": "Private residential tenancy lease contract snippet.",
        "format": "pdf",
    },
]


# ---------------------------------------------------------------------------
# Generator Pipeline Function
# ---------------------------------------------------------------------------
def generate_dataset(output_dir: Optional[Path] = None) -> LabelsFile:
    """Generate all 30 synthetic notices and save ground truth labels.json."""
    target_dir = Path(output_dir or OUTPUT_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    labels_list: List[NoticeLabel] = []

    print(f"Generating 30 synthetic notices in: {target_dir}")

    for idx, item in enumerate(SYNTHETIC_NOTICES_CATALOG, start=1):
        filename = item["filename"]
        doc_format = item.get("format", "png")

        # Convert to Pydantic NoticeLabel for strict schema validation
        label_dict = {k: v for k, v in item.items() if k != "format"}
        notice_label = NoticeLabel.model_validate(label_dict)
        labels_list.append(notice_label)

        file_path = target_dir / filename

        # Render corresponding document image
        if notice_label.doc_type == DocumentType.electricity:
            raw_img = render_electricity_notice(notice_label)
        elif notice_label.doc_type == DocumentType.challan:
            raw_img = render_challan_notice(notice_label)
        elif notice_label.doc_type == DocumentType.tax:
            raw_img = render_tax_notice(notice_label)
        elif notice_label.doc_type == DocumentType.scheme_rejection:
            raw_img = render_scheme_rejection_notice(notice_label)
        elif notice_label.doc_type == DocumentType.rti:
            raw_img = render_rti_citizen_request(notice_label)
        else:  # other
            if "receipt" in filename:
                raw_img = render_grocery_receipt(notice_label)
            elif "wedding" in filename:
                raw_img = render_wedding_invitation(notice_label)
            elif "personal_note" in filename:
                raw_img = render_personal_note(notice_label)
            elif "blur" in filename:
                raw_img = render_unreadable_blur(notice_label)
            elif "lease" in filename:
                raw_img = render_lease_agreement(notice_label)
            else:
                raw_img = render_unreadable_blur(notice_label)

        # Apply degradation if requested (and not already heavily blurred)
        if notice_label.is_degraded and "blur" not in filename:
            final_img = apply_scan_degradation(
                raw_img,
                angle=random.uniform(-1.8, 1.8),
                noise_sigma=12.0,
                blur_radius=0.9,
                contrast=0.92,
            )
        else:
            final_img = raw_img

        # Save to disk based on format
        if doc_format == "pdf":
            # Build searchable text summary from key facts and metadata
            searchable_text = f"{notice_label.issuer or ''} {notice_label.reference_number or ''} " + " ".join(notice_label.key_facts)
            save_image_as_pdf(final_img, file_path, searchable_text=searchable_text)
        else:
            final_img.save(file_path, "PNG", optimize=True)

        size_kb = file_path.stat().st_size / 1024
        degraded_tag = "[DEGRADED] " if notice_label.is_degraded else ""
        citizen_tag = "[CITIZEN REQ] " if notice_label.is_citizen_request else ""
        print(f"[{idx:02d}/30] Saved {filename} ({size_kb:.1f} KB) - {notice_label.doc_type.value} {degraded_tag}{citizen_tag}")

    # Build and validate LabelsFile container
    labels_file = LabelsFile(
        version="1.0",
        description="NoticeWise Synthetic Ground Truth Labels (30 notices, 5 types + out-of-scope)",
        notices=labels_list,
    )

    labels_json_path = target_dir / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        f.write(labels_file.model_dump_json(indent=2))

    print(f"\nGround truth labels saved to: {labels_json_path}")
    print(f"Total notices labeled and verified: {len(labels_file.notices)}")
    return labels_file


if __name__ == "__main__":
    generate_dataset()
