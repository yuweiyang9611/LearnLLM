"""Build the complete Chinese LearnLLM study guide as a polished PDF.

Install the optional project dependencies with ``scripts/bootstrap.ps1
-WithDocs`` first.  The deterministic Markdown reader is tailored to this
repository and does not require network access while building the PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "pdf" / "LearnLLM-从零学习大语言模型.pdf"

GUIDE_SOURCE = PROJECT_ROOT / "LEARNING_GUIDE.md"
FOUNDATION_SOURCES = tuple(
    PROJECT_ROOT / "docs" / filename
    for filename in (
        "00_学习路线与环境.md",
        "01_数学与PyTorch基础.md",
        "02_Tokenizer与语言模型.md",
    )
)
MODEL_SOURCES = tuple(
    PROJECT_ROOT / "docs" / filename
    for filename in (
        "03_注意力与Transformer.md",
        "04_模型家族与训练生命周期.md",
    )
)
TRAINING_SOURCES = tuple(
    PROJECT_ROOT / "docs" / filename
    for filename in (
        "05_迷你GPT预训练与生成.md",
        "06_SFT_LoRA与对齐.md",
    )
)
APPLICATION_SOURCE = PROJECT_ROOT / "docs" / "07_评测_RAG与Agent.md"
PROJECT_SOURCE = PROJECT_ROOT / "docs" / "08_结课项目.md"
TEMPLATE_SOURCE = PROJECT_ROOT / "docs" / "实验记录模板.md"
REFERENCE_SOURCE = PROJECT_ROOT / "docs" / "PDF_章节对照与勘误.md"
PDF_MARKDOWN_SOURCES = (
    GUIDE_SOURCE,
    *FOUNDATION_SOURCES,
    *MODEL_SOURCES,
    *TRAINING_SOURCES,
    APPLICATION_SOURCE,
    PROJECT_SOURCE,
    TEMPLATE_SOURCE,
    REFERENCE_SOURCE,
)
PDF_BUILD_INPUTS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "requirements-docs.txt",
    *PDF_MARKDOWN_SOURCES,
)
SOURCE_DIGEST_KEY = "LearnLLM-source-sha256"

DEFAULT_FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
DEFAULT_FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
DEFAULT_FONT_CODE = Path(r"C:\Windows\Fonts\simsun.ttc")
DEFAULT_FONT_MATH = Path(r"C:\Windows\Fonts\ARIAL_UNICODE_MS.ttf")

PAGE_WIDTH, PAGE_HEIGHT = A4
LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT = landscape(A4)
LEFT_MARGIN = 18 * mm
RIGHT_MARGIN = 18 * mm
TOP_MARGIN = 22 * mm
BOTTOM_MARGIN = 18 * mm
BODY_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
LANDSCAPE_BODY_WIDTH = LANDSCAPE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

NAVY = HexColor("#12243A")
DEEP_NAVY = HexColor("#0A1728")
TEAL = HexColor("#0C7C86")
BLUE = HexColor("#2E5EAA")
INK = HexColor("#1E293B")
MUTED = HexColor("#64748B")
PALE = HexColor("#F4F7FA")
PALE_TEAL = HexColor("#EAF7F7")
BORDER = HexColor("#D7E1EA")
WARM = HexColor("#F8F5EF")


@dataclass(frozen=True)
class FontConfig:
    regular: Path
    bold: Path
    code: Path
    math: Path | None


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _ttfont(name: str, path: Path, *, collection_index: int = 0) -> TTFont:
    options: dict[str, int] = {"validate": 1}
    if path.suffix.lower() == ".ttc":
        options["subfontIndex"] = collection_index
    return TTFont(name, str(path), **options)


def register_fonts(config: FontConfig) -> None:
    required = {
        "regular Chinese font": config.regular,
        "bold Chinese font": config.bold,
        "code Chinese font": config.code,
    }
    missing = [f"  - {label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(
            "Required Chinese fonts were not found:\n"
            f"{details}\n\n"
            "Windows defaults are Microsoft YaHei (msyh.ttc/msyhbd.ttc) and "
            "SimSun (simsun.ttc). Override paths with --font-regular, "
            "--font-bold and --font-code, or with LEARNLLM_FONT_REGULAR, "
            "LEARNLLM_FONT_BOLD and LEARNLLM_FONT_CODE. TTF and TTC files are supported."
        )

    pdfmetrics.registerFont(_ttfont("MSYaHei", config.regular))
    pdfmetrics.registerFont(_ttfont("MSYaHei-Bold", config.bold))
    # The Windows SimSun collection uses index 1 for NSimSun. Custom TTF files
    # ignore this collection index.
    pdfmetrics.registerFont(_ttfont("NSimSun", config.code, collection_index=1))
    pdfmetrics.registerFontFamily(
        "MSYaHei",
        normal="MSYaHei",
        bold="MSYaHei-Bold",
        italic="MSYaHei",
        boldItalic="MSYaHei-Bold",
    )
    if config.math is not None and config.math.is_file():
        pdfmetrics.registerFont(_ttfont("ArialUnicode", config.math))


def build_styles() -> dict[str, ParagraphStyle]:
    common = dict(wordWrap="CJK")
    return {
        "CoverKicker": ParagraphStyle(
            "CoverKicker",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=10,
            leading=15,
            textColor=HexColor("#91E6E6"),
            alignment=TA_CENTER,
            spaceAfter=11 * mm,
        ),
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=31,
            leading=42,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=5 * mm,
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            **common,
            fontName="MSYaHei",
            fontSize=14,
            leading=23,
            textColor=HexColor("#DCE9F4"),
            alignment=TA_CENTER,
            spaceAfter=12 * mm,
        ),
        "CoverBody": ParagraphStyle(
            "CoverBody",
            **common,
            fontName="MSYaHei",
            fontSize=10,
            leading=18,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "CoverMeta": ParagraphStyle(
            "CoverMeta",
            **common,
            fontName="MSYaHei",
            fontSize=8.5,
            leading=14,
            textColor=HexColor("#B7C9D9"),
            alignment=TA_CENTER,
        ),
        "PartKicker": ParagraphStyle(
            "PartKicker",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=11,
            leading=17,
            textColor=HexColor("#94D8D8"),
            alignment=TA_CENTER,
            spaceAfter=8 * mm,
        ),
        "PartTitle": ParagraphStyle(
            "PartTitle",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=27,
            leading=38,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=7 * mm,
        ),
        "PartNote": ParagraphStyle(
            "PartNote",
            **common,
            fontName="MSYaHei",
            fontSize=10,
            leading=18,
            textColor=HexColor("#D1E2EE"),
            alignment=TA_CENTER,
        ),
        "ChapterTitle": ParagraphStyle(
            "ChapterTitle",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=21,
            leading=29,
            textColor=NAVY,
            spaceBefore=1 * mm,
            spaceAfter=6 * mm,
            keepWithNext=True,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=(0, 0, 4, 0),
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=14.5,
            leading=22,
            textColor=NAVY,
            spaceBefore=5 * mm,
            spaceAfter=2.8 * mm,
            keepWithNext=True,
        ),
        "Heading3": ParagraphStyle(
            "Heading3",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=11.5,
            leading=18,
            textColor=TEAL,
            spaceBefore=3.8 * mm,
            spaceAfter=1.8 * mm,
            keepWithNext=True,
        ),
        "Heading4": ParagraphStyle(
            "Heading4",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=10,
            leading=16,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=1.4 * mm,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            **common,
            fontName="MSYaHei",
            textColor=INK,
            fontSize=9.5,
            leading=16.2,
            spaceAfter=2.2 * mm,
            allowWidows=0,
            allowOrphans=0,
        ),
        "Lead": ParagraphStyle(
            "Lead",
            **common,
            fontName="MSYaHei",
            fontSize=10.4,
            leading=18,
            textColor=NAVY,
            spaceAfter=4 * mm,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            **common,
            fontName="MSYaHei",
            textColor=INK,
            fontSize=9.3,
            leading=15.4,
            leftIndent=5.5 * mm,
            firstLineIndent=-4.2 * mm,
            spaceAfter=1.2 * mm,
            allowWidows=0,
            allowOrphans=0,
        ),
        "Quote": ParagraphStyle(
            "Quote",
            **common,
            fontName="MSYaHei",
            fontSize=9,
            leading=15.5,
            leftIndent=4 * mm,
            rightIndent=2 * mm,
            textColor=HexColor("#37556A"),
            backColor=PALE_TEAL,
            borderColor=TEAL,
            borderWidth=0.7,
            borderPadding=(5, 8, 5, 9),
            spaceBefore=1.5 * mm,
            spaceAfter=3 * mm,
        ),
        "CodeLabel": ParagraphStyle(
            "CodeLabel",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=6.8,
            leading=9,
            textColor=TEAL,
            leftIndent=2 * mm,
            spaceBefore=1 * mm,
            spaceAfter=0.7 * mm,
            keepWithNext=True,
        ),
        "Code": ParagraphStyle(
            "Code",
            fontName="NSimSun",
            fontSize=7.7,
            leading=11.2,
            textColor=HexColor("#19324A"),
            backColor=PALE,
            borderColor=BORDER,
            borderWidth=0.6,
            borderPadding=(6, 8, 6, 8),
            leftIndent=1 * mm,
            rightIndent=1 * mm,
            spaceBefore=0,
            spaceAfter=3 * mm,
            splitLongWords=True,
        ),
        "Formula": ParagraphStyle(
            "Formula",
            **common,
            fontName="NSimSun",
            fontSize=9.2,
            leading=15.5,
            alignment=TA_CENTER,
            textColor=NAVY,
            backColor=HexColor("#F1F6FA"),
            borderColor=HexColor("#BFD4E3"),
            borderWidth=0.6,
            borderPadding=(6, 8, 6, 8),
            leftIndent=3 * mm,
            rightIndent=3 * mm,
            spaceBefore=1.2 * mm,
            spaceAfter=3 * mm,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            **common,
            fontName="MSYaHei",
            textColor=INK,
            fontSize=7.55,
            leading=11.2,
            spaceAfter=0,
        ),
        "TableHead": ParagraphStyle(
            "TableHead",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=7.7,
            leading=11.5,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "TOCTitle": ParagraphStyle(
            "TOCTitle",
            **common,
            fontName="MSYaHei-Bold",
            fontSize=24,
            leading=34,
            textColor=NAVY,
            spaceAfter=10 * mm,
        ),
        "Small": ParagraphStyle(
            "Small",
            **common,
            fontName="MSYaHei",
            fontSize=7.8,
            leading=12.5,
            textColor=MUTED,
            spaceAfter=2 * mm,
        ),
    }


@lru_cache(maxsize=1)
def source_digest() -> str:
    """Hash every source that affects the tracked learning-guide PDF."""

    digest = hashlib.sha256()
    for path in PDF_BUILD_INPUTS:
        relative_path = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def embedded_source_digest(path: Path) -> str | None:
    """Read the source digest from a generated PDF without rendering it."""

    from pypdf import PdfReader

    reader = PdfReader(path)
    metadata = reader.metadata
    keywords = str(metadata.get("/Keywords", "")) if metadata else ""
    pattern = rf"(?:^|;\s*){re.escape(SOURCE_DIGEST_KEY)}=([0-9a-f]{{64}})(?:;|$)"
    match = re.search(pattern, keywords)
    return match.group(1) if match else None


def check_source_digest(path: Path) -> bool:
    """Check that a tracked PDF was built from the current repository inputs."""

    expected = source_digest()
    if not path.is_file():
        print(f"PDF does not exist: {path}", file=sys.stderr)
        return False
    try:
        actual = embedded_source_digest(path)
    except Exception as exc:
        print(f"Could not read PDF source digest: {exc}", file=sys.stderr)
        return False
    if actual != expected:
        print("Tracked PDF is stale; rebuild it with this script.", file=sys.stderr)
        print(f"expected source SHA256: {expected}", file=sys.stderr)
        print(f"embedded source SHA256: {actual or '<missing>'}", file=sys.stderr)
        return False
    print(f"PDF source digest is current: {expected}")
    return True


def set_metadata(canvas) -> None:
    canvas.setTitle("从零学习 LLM：大语言模型原理、实验与项目实践")
    canvas.setAuthor("LearnLLM")
    canvas.setSubject("参考 Happy-LLM v1.0 知识主线的中文 LLM 自学教程")
    canvas.setKeywords(
        "LLM, Transformer, PyTorch, Tokenizer, LoRA, RAG; "
        f"{SOURCE_DIGEST_KEY}={source_digest()}"
    )


def draw_cover(canvas, doc) -> None:
    set_metadata(canvas)
    width, height = canvas._pagesize
    canvas.saveState()
    canvas.setFillColor(DEEP_NAVY)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColor(NAVY)
    canvas.circle(width * 0.88, height * 0.86, 72 * mm, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.setFillAlpha(0.35)
    canvas.circle(width * 0.13, height * 0.12, 50 * mm, stroke=0, fill=1)
    canvas.setFillAlpha(1)
    canvas.setStrokeColor(HexColor("#2B526A"))
    canvas.setLineWidth(0.6)
    for offset in range(7):
        y = height * 0.73 - offset * 8 * mm
        canvas.line(17 * mm, y, 35 * mm, y)
    canvas.restoreState()


def draw_part(canvas, doc) -> None:
    set_metadata(canvas)
    width, height = canvas._pagesize
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, height * 0.49, width, 3 * mm, stroke=0, fill=1)
    canvas.setStrokeColor(HexColor("#52758C"))
    canvas.setLineWidth(0.4)
    canvas.circle(width / 2, height / 2, 42 * mm, stroke=1, fill=0)
    canvas.circle(width / 2, height / 2, 53 * mm, stroke=1, fill=0)
    canvas.restoreState()


def draw_body(canvas, doc) -> None:
    set_metadata(canvas)
    width, height = canvas._pagesize
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.45)
    canvas.line(LEFT_MARGIN, height - 13 * mm, width - RIGHT_MARGIN, height - 13 * mm)
    canvas.line(LEFT_MARGIN, 12 * mm, width - RIGHT_MARGIN, 12 * mm)
    canvas.setFont("MSYaHei", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT_MARGIN, height - 9.4 * mm, "LearnLLM · 从零学习大语言模型")
    canvas.drawRightString(width - RIGHT_MARGIN, height - 9.4 * mm, "原理 · 实验 · 项目实践")
    canvas.drawCentredString(width / 2, 7.4 * mm, f"第 {canvas.getPageNumber()} 页")
    canvas.restoreState()


class GuideDocTemplate(BaseDocTemplate):
    """DocTemplate that creates outline bookmarks and a clickable TOC."""

    def beforeDocument(self) -> None:
        super().beforeDocument()
        self._heading_counter = 0

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        level_by_style = {
            "PartTitle": 0,
            "ChapterTitle": 0,
            "Heading2": 1,
        }
        level = level_by_style.get(flowable.style.name)
        if level is None:
            return
        self._heading_counter += 1
        title = flowable.getPlainText()
        key = f"heading-{self._heading_counter:04d}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=level >= 1)
        self.notify("TOCEntry", (level, title, self.page, key))


def create_document(output_path: Path) -> GuideDocTemplate:
    document = GuideDocTemplate(
        str(output_path),
        pagesize=A4,
        # Remove timestamps and random document IDs so repeated builds in the
        # same environment are stable. CI uses the embedded source digest
        # because runner font binaries can still differ between machines.
        invariant=1,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="从零学习 LLM",
        author="LearnLLM",
    )
    cover_frame = Frame(
        22 * mm,
        22 * mm,
        PAGE_WIDTH - 44 * mm,
        PAGE_HEIGHT - 44 * mm,
        id="cover-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    body_frame = Frame(
        LEFT_MARGIN,
        BOTTOM_MARGIN,
        BODY_WIDTH,
        PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
        id="body-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    part_frame = Frame(
        25 * mm,
        20 * mm,
        PAGE_WIDTH - 50 * mm,
        PAGE_HEIGHT - 40 * mm,
        id="part-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    landscape_frame = Frame(
        LEFT_MARGIN,
        BOTTOM_MARGIN,
        LANDSCAPE_BODY_WIDTH,
        LANDSCAPE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
        id="landscape-frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates(
        [
            PageTemplate("Cover", [cover_frame], onPage=draw_cover, pagesize=A4),
            PageTemplate("Normal", [body_frame], onPage=draw_body, pagesize=A4),
            PageTemplate("Part", [part_frame], onPage=draw_part, pagesize=A4),
            PageTemplate(
                "Landscape",
                [landscape_frame],
                onPage=draw_body,
                pagesize=landscape(A4),
            ),
        ]
    )
    return document


SMART_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})


def normalize_text(text: str) -> str:
    return text.translate(SMART_DASHES).replace("\x0c", " ").replace("\ufeff", "")


def latex_to_readable(source: str) -> str:
    text = normalize_text(source.strip())
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("$$", "")
    for environment in ("cases", "aligned", "split"):
        text = text.replace(rf"\begin{{{environment}}}", "")
        text = text.replace(rf"\end{{{environment}}}", "")
    text = text.replace("&", " ")
    text = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:mathrm|mathbf|mathcal|text)\{([^{}]+)\}", r"\1", text)
    text = text.replace("\\mathbb{R}", "R")
    for _ in range(4):
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)
    replacements = {
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\Delta": "Δ",
        r"\Theta": "Θ",
        r"\alpha": "α",
        r"\tau": "τ",
        r"\epsilon": "ε",
        r"\mu": "μ",
        r"\sigma": "σ",
        r"\gamma": "γ",
        r"\cdot": "·",
        r"\times": "×",
        r"\approx": "≈",
        r"\sim": "~",
        r"\le": "≤",
        r"\ge": "≥",
        r"\infty": "∞",
        r"\in": "∈",
        r"\mid": "|",
        r"\lVert": "||",
        r"\rVert": "||",
        r"\left": "",
        r"\right": "",
        r"\log": "log",
        r"\exp": "exp",
        r"\arg": "arg",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("\\,", " ").replace("\\!", "")
    text = text.replace("\\quad", "    ").replace("\\qquad", "        ")
    text = text.replace("\\\\", "\n")
    text = text.replace("{", "(").replace("}", ")")
    return text.strip()


def _soft_break(value: str) -> str:
    return re.sub(r"([\\/_\.=:])", lambda match: match.group(1) + "\u200b", value)


def inline_markup(source: str) -> str:
    """Convert the small inline Markdown subset used by this repository."""

    source = normalize_text(source)
    placeholders: dict[str, str] = {}

    def stash(fragment: str) -> str:
        key = f"@@INLINE{len(placeholders):04d}@@"
        placeholders[key] = fragment
        return key

    def code_replacer(match: re.Match[str]) -> str:
        code = _soft_break(html.escape(match.group(1), quote=False))
        return stash(
            f'<font name="NSimSun" color="#174D66" backColor="#EDF4F7"> {code} </font>'
        )

    def link_replacer(match: re.Match[str]) -> str:
        label = html.escape(match.group(1), quote=False)
        target = html.escape(match.group(2), quote=True)
        if re.match(r"https?://", match.group(2)):
            return stash(f'<link href="{target}" color="#2E5EAA"><u>{label}</u></link>')
        return stash(f'<font color="#2E5EAA"><u>{label}</u></font>')

    def formula_replacer(match: re.Match[str]) -> str:
        formula = html.escape(latex_to_readable(match.group(1)), quote=False)
        return stash(f'<font name="NSimSun" color="#173B5B">{formula}</font>')

    source = re.sub(r"`([^`]+)`", code_replacer, source)
    source = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_replacer, source)
    source = re.sub(r"\\\((.+?)\\\)", formula_replacer, source)
    escaped = html.escape(source, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", escaped)
    if "∇" in escaped and "ArialUnicode" in pdfmetrics.getRegisteredFontNames():
        escaped = escaped.replace("∇", '<font name="ArialUnicode">∇</font>')
    for key, fragment in placeholders.items():
        escaped = escaped.replace(key, fragment)
    return escaped


def visible_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WFA" else 1 for char in text)


def wrap_code_line(line: str, max_width: int) -> list[str]:
    if visible_width(line) <= max_width:
        return [line]
    indent = re.match(r"\s*", line).group(0)
    remaining = line
    wrapped: list[str] = []
    continuation = indent + "    "
    while visible_width(remaining) > max_width:
        width = 0
        cut = 0
        best_break = -1
        for index, char in enumerate(remaining):
            width += 2 if unicodedata.east_asian_width(char) in "WFA" else 1
            if char in " ,;)]}" or char in "/\\":
                best_break = index + 1
            if width > max_width:
                cut = best_break if best_break > max(8, index // 2) else index
                break
        if cut <= 0:
            break
        wrapped.append(remaining[:cut].rstrip())
        remaining = continuation + remaining[cut:].lstrip()
    wrapped.append(remaining)
    return wrapped


def wrap_code(code: str, max_width: int) -> str:
    lines: list[str] = []
    for line in normalize_text(code).splitlines() or [""]:
        lines.extend(wrap_code_line(line, max_width))
    return "\n".join(lines)


def split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row while preserving pipes inside inline code."""

    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    in_code = False
    escaped = False
    for char in stripped:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "`":
            in_code = not in_code
            current.append(char)
            continue
        if char == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def is_table_separator(line: str) -> bool:
    cells = split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def table_column_widths(rows: list[list[str]], available_width: float) -> list[float]:
    column_count = max(len(row) for row in rows)
    weights: list[float] = []
    for column in range(column_count):
        lengths = [
            min(max(visible_width(row[column]) if column < len(row) else 0, 5), 48)
            for row in rows
        ]
        weights.append(max(8.0, sum(lengths) / max(len(lengths), 1)))
    minimum = 0.09 * available_width
    raw_total = sum(weights)
    widths = [max(minimum, available_width * weight / raw_total) for weight in weights]
    scale = available_width / sum(widths)
    return [width * scale for width in widths]


def make_table(
    raw_rows: list[list[str]],
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> LongTable:
    column_count = max(len(row) for row in raw_rows)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in raw_rows]
    data: list[list[Paragraph]] = []
    for row_index, row in enumerate(normalized_rows):
        style = styles["TableHead"] if row_index == 0 else styles["TableCell"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])

    table = LongTable(
        data,
        colWidths=table_column_widths(normalized_rows, available_width),
        repeatRows=1,
        splitByRow=1,
        splitInRow=1,
        hAlign="LEFT",
    )
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands))
    table.spaceBefore = 1.5 * mm
    table.spaceAfter = 4 * mm
    return table


SPECIAL_START = re.compile(
    r"^(?:#{1,4}\s+|```|\s*[-*+]\s+|\s*\d+\.\s+|>\s?|\s*\|?|\$\$|\\\[|---\s*$)"
)


def parse_markdown(
    path: Path,
    styles: dict[str, ParagraphStyle],
    *,
    available_width: float = BODY_WIDTH,
) -> list[Flowable]:
    lines = path.read_text(encoding="utf-8").splitlines()
    story: list[Flowable] = []
    index = 0
    paragraph_count = 0

    while index < len(lines):
        line = normalize_text(lines[index])
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        fence = re.match(r"^```\s*([^\s`]*)", stripped)
        if fence:
            language = fence.group(1).strip() or "text"
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            story.append(Paragraph(f"CODE · {html.escape(language.upper())}", styles["CodeLabel"]))
            max_width = 84 if available_width < 600 else 126
            code = wrap_code("\n".join(code_lines), max_width)
            story.append(XPreformatted(html.escape(code), styles["Code"]))
            continue

        if stripped in {"$$", r"\["}:
            closing = "$$" if stripped == "$$" else r"\]"
            index += 1
            formula_lines: list[str] = []
            while index < len(lines) and lines[index].strip() != closing:
                formula_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            readable = latex_to_readable("\n".join(formula_lines))
            formula_html = "<br/>".join(
                html.escape(item, quote=False) for item in readable.splitlines()
            )
            story.append(Paragraph(formula_html, styles["Formula"]))
            continue

        inline_formula = re.match(r"^(?:\$\$|\\\[)(.+?)(?:\$\$|\\\])$", stripped)
        if inline_formula:
            readable = latex_to_readable(inline_formula.group(1))
            story.append(Paragraph(html.escape(readable), styles["Formula"]))
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            style_name = {1: "ChapterTitle", 2: "Heading2", 3: "Heading3", 4: "Heading4"}[level]
            story.append(Paragraph(inline_markup(heading.group(2)), styles[style_name]))
            if level == 1:
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=1.2,
                        color=TEAL,
                        spaceBefore=0,
                        spaceAfter=4 * mm,
                    )
                )
            index += 1
            continue

        if stripped == "---":
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=BORDER,
                    spaceBefore=2 * mm,
                    spaceAfter=3 * mm,
                )
            )
            index += 1
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and normalize_text(lines[index]).lstrip().startswith(">"):
                quote_lines.append(normalize_text(lines[index]).lstrip()[1:].lstrip())
                index += 1
            quote_html = "<br/>".join(inline_markup(item) for item in quote_lines)
            story.append(Paragraph(quote_html, styles["Quote"]))
            continue

        if index + 1 < len(lines) and "|" in line and is_table_separator(lines[index + 1]):
            rows = [split_markdown_row(line)]
            index += 2
            while index < len(lines):
                candidate = normalize_text(lines[index])
                if not candidate.strip() or "|" not in candidate:
                    break
                rows.append(split_markdown_row(candidate))
                index += 1
            story.append(make_table(rows, styles, available_width))
            continue

        bullet = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
        if bullet:
            depth = min(len(bullet.group(1).replace("\t", "    ")) // 2, 3)
            style = ParagraphStyle(
                f"Bullet-{depth}",
                parent=styles["Bullet"],
                leftIndent=styles["Bullet"].leftIndent + depth * 5 * mm,
            )
            story.append(Paragraph("• " + inline_markup(bullet.group(2)), style))
            index += 1
            continue

        numbered = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
        if numbered:
            depth = min(len(numbered.group(1).replace("\t", "    ")) // 2, 3)
            style = ParagraphStyle(
                f"Number-{depth}",
                parent=styles["Bullet"],
                leftIndent=styles["Bullet"].leftIndent + depth * 5 * mm,
            )
            story.append(
                Paragraph(f"{numbered.group(2)}. " + inline_markup(numbered.group(3)), style)
            )
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = normalize_text(lines[index])
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                break
            if re.match(r"^(#{1,4})\s+", candidate_stripped):
                break
            if candidate_stripped.startswith("```") or candidate_stripped in {"$$", r"\[", "---"}:
                break
            if candidate_stripped.startswith(">"):
                break
            if re.match(r"^\s*[-*+]\s+", candidate) or re.match(r"^\s*\d+\.\s+", candidate):
                break
            if index + 1 < len(lines) and "|" in candidate and is_table_separator(lines[index + 1]):
                break
            paragraph_lines.append(candidate_stripped)
            index += 1
        paragraph_count += 1
        style = styles["Lead"] if paragraph_count == 1 else styles["Body"]
        story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), style))

    return story


def add_part_page(
    story: list[Flowable],
    styles: dict[str, ParagraphStyle],
    kicker: str,
    title: str,
    note: str,
) -> None:
    story.extend(
        [
            NextPageTemplate("Part"),
            PageBreak(),
            Spacer(1, 59 * mm),
            Paragraph(inline_markup(kicker), styles["PartKicker"]),
            Paragraph(inline_markup(title), styles["PartTitle"]),
            Paragraph(inline_markup(note), styles["PartNote"]),
            NextPageTemplate("Normal"),
            PageBreak(),
        ]
    )


def append_markdown(
    story: list[Flowable],
    path: Path,
    styles: dict[str, ParagraphStyle],
    *,
    page_break: bool = False,
    available_width: float = BODY_WIDTH,
) -> None:
    if page_break:
        story.append(PageBreak())
    story.extend(parse_markdown(path, styles, available_width=available_width))


def make_cover(story: list[Flowable], styles: dict[str, ParagraphStyle]) -> None:
    story.extend(
        [
            Spacer(1, 23 * mm),
            Paragraph("LEARNLLM · SYSTEMATIC STUDY GUIDE", styles["CoverKicker"]),
            Paragraph("从零学习 LLM", styles["CoverTitle"]),
            Paragraph("大语言模型原理、实验与项目实践", styles["CoverSubtitle"]),
        ]
    )
    badge_data = [
        [
            Paragraph("原理优先", styles["CoverBody"]),
            Paragraph("CPU 可运行", styles["CoverBody"]),
            Paragraph("10 个实验", styles["CoverBody"]),
            Paragraph("自动测试", styles["CoverBody"]),
        ]
    ]
    badges = Table(badge_data, colWidths=[35 * mm] * 4, hAlign="CENTER")
    badges.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#183B52")),
                ("BOX", (0, 0), (-1, -1), 0.7, HexColor("#4C7A8F")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, HexColor("#4C7A8F")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(badges)
    story.append(Spacer(1, 22 * mm))
    cover_box = Table(
        [
            [
                Paragraph(
                    "从文本、token、注意力与 Transformer 出发，亲手训练 TinyGPT，"
                    "再理解 SFT、LoRA、评测、RAG 与 Agent。课程按 Windows + CPU 环境设计，"
                    "无需 GPU、API Key 或预训练模型下载。",
                    styles["CoverBody"],
                )
            ]
        ],
        colWidths=[142 * mm],
        hAlign="CENTER",
    )
    cover_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#132E43")),
                ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#2E6A75")),
                ("LEFTPADDING", (0, 0), (-1, -1), 15),
                ("RIGHTPADDING", (0, 0), (-1, -1), 15),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.append(cover_box)
    story.append(Spacer(1, 27 * mm))
    story.append(
        Paragraph(
            "参考《Happy-LLM v1.0》知识主线的原创教学改编<br/>"
            "LearnLLM v1.0 · 2026-07-15",
            styles["CoverMeta"],
        )
    )
    story.extend([NextPageTemplate("Normal"), PageBreak()])


def make_usage_page(story: list[Flowable], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph("如何使用本书", styles["ChapterTitle"]))
    story.append(
        Paragraph(
            "这不是一份只供阅读的摘要，而是一门需要运行代码、预测现象、记录失败并用测试验证理解的课程。"
            "建议先完整阅读导读，再按 00 到 09 的实验编号推进。每次只改变一个变量。",
            styles["Lead"],
        )
    )
    story.append(Paragraph("环境原则", styles["Heading2"]))
    for item in (
        "所有 Python 命令明确调用当前项目的 .venv，不向系统 Python 安装包。",
        "必做实验默认使用 CPU，不需要 API Key、GPU 或外部预训练模型。",
        "玩具模型用于验证原理与训练管线，不代表生产级 LLM 能力。",
    ):
        story.append(Paragraph("• " + inline_markup(item), styles["Bullet"]))
    story.append(Paragraph("第一条命令", styles["Heading2"]))
    command = ".\\.venv\\Scripts\\python.exe .\\experiments\\00_environment_check.py"
    story.append(Paragraph("CODE · POWERSHELL", styles["CodeLabel"]))
    story.append(XPreformatted(html.escape(command), styles["Code"]))
    story.append(Paragraph("快速验证", styles["Heading2"]))
    verify_command = ".\\.venv\\Scripts\\python.exe -m unittest discover -s .\\tests -v"
    story.append(Paragraph("CODE · POWERSHELL", styles["CodeLabel"]))
    story.append(XPreformatted(html.escape(verify_command), styles["Code"]))
    story.append(
        Paragraph(
            "改编说明：本书参考用户提供的《Happy-LLM v1.0》章节主线，"
            "讲解、代码、数据、测试和练习均为本项目重新设计。"
            "附录给出物理页码对照与勘误，避免盲目照抄示例。",
            styles["Quote"],
        )
    )
    story.append(PageBreak())


def make_toc(story: list[Flowable], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph("目录", styles["TOCTitle"]))
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.levelStyles = [
        ParagraphStyle(
            "TOC-Level-0",
            fontName="MSYaHei-Bold",
            fontSize=11,
            leading=17,
            textColor=NAVY,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=5,
        ),
        ParagraphStyle(
            "TOC-Level-1",
            fontName="MSYaHei",
            fontSize=9.2,
            leading=14.5,
            textColor=INK,
            leftIndent=12,
            firstLineIndent=-4,
            spaceBefore=2,
        ),
        ParagraphStyle(
            "TOC-Level-2",
            fontName="MSYaHei",
            fontSize=7.7,
            leading=12,
            textColor=MUTED,
            leftIndent=28,
            firstLineIndent=-5,
            spaceBefore=1,
        ),
    ]
    story.append(toc)


def build_story(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    story: list[Flowable] = []
    make_cover(story, styles)
    make_usage_page(story, styles)
    make_toc(story, styles)

    add_part_page(
        story,
        styles,
        "导读",
        "先建立完整心智模型",
        "从概率语言模型到 Transformer、训练、适配和应用系统的全景路线",
    )
    append_markdown(story, GUIDE_SOURCE, styles)

    add_part_page(
        story,
        styles,
        "第一部分",
        "实验基础",
        "隔离环境 · 数学与 PyTorch · Tokenizer 与语言模型",
    )
    for item_index, source in enumerate(FOUNDATION_SOURCES):
        append_markdown(
            story,
            source,
            styles,
            page_break=item_index > 0,
        )

    add_part_page(
        story,
        styles,
        "第二部分",
        "模型结构",
        "注意力与 Transformer · 模型家族与训练生命周期",
    )
    for item_index, source in enumerate(MODEL_SOURCES):
        append_markdown(
            story,
            source,
            styles,
            page_break=item_index > 0,
        )

    add_part_page(
        story,
        styles,
        "第三部分",
        "训练与适配",
        "TinyGPT 预训练与生成 · SFT、LoRA 与偏好对齐",
    )
    for item_index, source in enumerate(TRAINING_SOURCES):
        append_markdown(
            story,
            source,
            styles,
            page_break=item_index > 0,
        )

    add_part_page(
        story,
        styles,
        "第四部分",
        "应用系统",
        "评测、RAG 与 Agent：从会生成文本到可验证的系统",
    )
    append_markdown(story, APPLICATION_SOURCE, styles)

    add_part_page(
        story,
        styles,
        "第五部分",
        "独立实践",
        "用结课项目证明你能提出问题、控制变量、复现实验并解释边界",
    )
    append_markdown(story, PROJECT_SOURCE, styles)

    add_part_page(
        story,
        styles,
        "附录",
        "模板、参考与勘误",
        "实验记录模板 · 《Happy-LLM v1.0》物理页码对照与代码风险",
    )
    append_markdown(story, TEMPLATE_SOURCE, styles)
    story.extend([NextPageTemplate("Landscape"), PageBreak()])
    append_markdown(
        story,
        REFERENCE_SOURCE,
        styles,
        available_width=LANDSCAPE_BODY_WIDTH,
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "- 全书完 -<br/>源文件：LearnLLM 工作区 Markdown 讲义<br/>"
            "生成脚本：scripts/build_learning_guide_pdf.py",
            styles["Small"],
        )
    )
    return story


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="check that the output PDF embeds the digest of the current build inputs",
    )
    parser.add_argument(
        "--font-regular",
        type=Path,
        default=_env_path("LEARNLLM_FONT_REGULAR", DEFAULT_FONT_REGULAR),
        help="Chinese body font (env: LEARNLLM_FONT_REGULAR)",
    )
    parser.add_argument(
        "--font-bold",
        type=Path,
        default=_env_path("LEARNLLM_FONT_BOLD", DEFAULT_FONT_BOLD),
        help="Chinese bold font (env: LEARNLLM_FONT_BOLD)",
    )
    parser.add_argument(
        "--font-code",
        type=Path,
        default=_env_path("LEARNLLM_FONT_CODE", DEFAULT_FONT_CODE),
        help="Chinese code font (env: LEARNLLM_FONT_CODE)",
    )
    parser.add_argument(
        "--font-math",
        type=Path,
        default=_env_path("LEARNLLM_FONT_MATH", DEFAULT_FONT_MATH),
        help="Optional Unicode math fallback (env: LEARNLLM_FONT_MATH)",
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    if args.check_source:
        raise SystemExit(0 if check_source_digest(output_path) else 1)

    register_fonts(
        FontConfig(
            regular=args.font_regular.expanduser(),
            bold=args.font_bold.expanduser(),
            code=args.font_code.expanduser(),
            math=args.font_math.expanduser() if args.font_math else None,
        )
    )
    styles = build_styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = create_document(output_path)
    story = build_story(styles)
    document.multiBuild(story)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"PDF: {output_path}")
    print(f"SHA256: {digest}")
    print(f"Source SHA256: {source_digest()}")


if __name__ == "__main__":
    main()
