"""Perform deterministic structural checks on the generated learning-guide PDF."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "output" / "pdf" / "LearnLLM-从零学习大语言模型.pdf"
REQUIRED_TEXT = (
    "如何使用本书",
    "TinyGPT",
    "Transformer",
    "LoRA",
    "assistant-only",
    "RAG",
    "10_sft_tiny_gpt.py",
    "11_local_agent.py",
    "结课项目",
    "实验记录模板",
    "Happy-LLM",
)


def verify_pdf(path: Path, *, min_pages: int) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    facts: dict[str, object] = {"path": str(path)}

    if not path.is_file():
        return [f"PDF does not exist: {path}"], facts

    file_size = path.stat().st_size
    facts["bytes"] = file_size
    facts["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_size < 100_000:
        errors.append(f"PDF is unexpectedly small: {file_size} bytes")

    try:
        reader = PdfReader(path)
    except Exception as exc:  # pragma: no cover - exercised by corrupted files
        return [f"pypdf could not open the PDF: {exc}"], facts

    page_count = len(reader.pages)
    facts["pages"] = page_count
    if page_count < min_pages:
        errors.append(f"Expected at least {min_pages} pages, found {page_count}")

    metadata = reader.metadata
    title = str(metadata.title or "") if metadata else ""
    author = str(metadata.author or "") if metadata else ""
    facts["title"] = title
    facts["author"] = author
    if "LLM" not in title:
        errors.append(f"PDF title metadata does not contain 'LLM': {title!r}")
    if author != "LearnLLM":
        errors.append(f"Unexpected PDF author metadata: {author!r}")

    page_text: list[str] = []
    blank_pages: list[int] = []
    portrait_pages = 0
    landscape_pages = 0
    invalid_page_sizes: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
            errors.append(f"Could not extract page {page_number}: {exc}")
            text = ""
        page_text.append(text)
        if not text.strip():
            blank_pages.append(page_number)

        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        short_side, long_side = sorted((width, height))
        if short_side <= 0 or abs(long_side / short_side - 1.414) > 0.03:
            invalid_page_sizes.append(page_number)
        if width < height:
            portrait_pages += 1
        elif width > height:
            landscape_pages += 1

    facts["portrait_pages"] = portrait_pages
    facts["landscape_pages"] = landscape_pages
    if blank_pages:
        errors.append(f"Pages without extractable text: {blank_pages}")
    if invalid_page_sizes:
        errors.append(f"Pages are not close to A-series proportions: {invalid_page_sizes}")
    if portrait_pages == 0:
        errors.append("No portrait pages found")
    if landscape_pages == 0:
        errors.append("No landscape appendix pages found")

    all_text = "\n".join(page_text)
    missing_text = [item for item in REQUIRED_TEXT if item not in all_text]
    if missing_text:
        errors.append(f"Required guide text was not extracted: {missing_text}")

    try:
        outline_count = len(reader.outline)
    except Exception as exc:  # pragma: no cover - defensive for malformed outlines
        outline_count = 0
        errors.append(f"Could not read PDF outline: {exc}")
    facts["outline_entries"] = outline_count
    if outline_count < 10:
        errors.append(f"Expected at least 10 top-level outline entries, found {outline_count}")

    return errors, facts


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--min-pages", type=int, default=60)
    args = parser.parse_args()

    if args.min_pages < 1:
        parser.error("--min-pages must be at least 1")

    pdf_path = args.pdf.resolve()
    errors, facts = verify_pdf(pdf_path, min_pages=args.min_pages)
    if errors:
        print("PDF verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"PDF verification passed: {facts['path']}")
    print(
        f"pages={facts['pages']} portrait={facts['portrait_pages']} "
        f"landscape={facts['landscape_pages']} outline={facts['outline_entries']}"
    )
    print(f"sha256={facts['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
