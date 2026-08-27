"""Verify the generated GitHub Pages site without third-party dependencies."""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
SITE_PREFIX = "/LearnLLM/"
REQUIRED_SEARCH_TERMS = (
    "因果掩码",
    "交叉熵",
    "LoRA",
    "RAG",
    "zero_grad",
)


class DocumentParser(HTMLParser):
    """Collect local URLs and essential document metadata."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.html_language = ""
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.canonical_url = ""
        self.h1_count = 0
        self._inside_title = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "html":
            self.html_language = values.get("lang", "")
        elif tag == "title":
            self._inside_title = True
        elif tag == "h1":
            self.h1_count += 1
        elif tag == "link" and values.get("href"):
            self.links.append(values["href"])
            if "canonical" in values.get("rel", "").split():
                self.canonical_url = values["href"]
        elif tag == "a" and values.get("href"):
            self.links.append(values["href"])
        elif tag in {"img", "script", "source"} and values.get("src"):
            self.links.append(values["src"])
        elif tag == "meta":
            key = values.get("property") or values.get("name")
            if key:
                self.meta[key] = values.get("content", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


def resolve_local_target(site_root: Path, page: Path, raw_url: str) -> Path | None:
    """Resolve an HTML URL to its generated file, or return None for externals."""

    parsed = urlsplit(raw_url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None

    path = unquote(parsed.path)
    if path.startswith(SITE_PREFIX):
        candidate = site_root / path.removeprefix(SITE_PREFIX)
    elif path == SITE_PREFIX.rstrip("/"):
        candidate = site_root
    elif path.startswith("/"):
        return None
    else:
        candidate = page.parent / path

    if path.endswith("/") or candidate.is_dir():
        candidate /= "index.html"
    elif not candidate.suffix:
        candidate = candidate / "index.html"
    return candidate.resolve()


def verify_site(site_root: Path) -> list[str]:
    errors: list[str] = []
    required_files = (
        site_root / "index.html",
        site_root / "404.html",
        site_root / "robots.txt",
        site_root / "sitemap.xml",
        site_root / "search" / "search_index.json",
        site_root / "assets" / "images" / "learnllm-social.png",
    )
    for required in required_files:
        if not required.is_file():
            errors.append(f"missing required output: {required.relative_to(site_root)}")

    html_files = sorted(site_root.rglob("*.html"))
    if len(html_files) < 15:
        errors.append(f"expected at least 15 HTML pages, found {len(html_files)}")

    for page in html_files:
        text = page.read_text(encoding="utf-8")
        relative_page = page.relative_to(site_root)
        if EMAIL_PATTERN.search(text) or "mailto:" in text.lower():
            errors.append(f"email-like content found in {relative_page}")
        if "--8&lt;--" in text or "--8<--" in text:
            errors.append(f"unexpanded snippet marker found in {relative_page}")

        parser = DocumentParser()
        parser.feed(text)
        if not parser.html_language.lower().startswith("zh"):
            errors.append(f"missing Chinese document language in {relative_page}")
        if not parser.title:
            errors.append(f"missing page title in {relative_page}")
        if parser.h1_count != 1:
            errors.append(
                f"expected exactly one H1 in {relative_page}, found {parser.h1_count}"
            )
        if relative_page != Path("404.html") and not parser.canonical_url.startswith(
            "https://yuweiyang9611.github.io/LearnLLM/"
        ):
            errors.append(f"invalid canonical URL in {relative_page}")

        for meta_name in (
            "description",
            "robots",
            "og:title",
            "og:description",
            "og:url",
            "og:image",
            "og:image:alt",
            "twitter:card",
            "twitter:title",
            "twitter:description",
            "twitter:image",
            "twitter:image:alt",
        ):
            if not parser.meta.get(meta_name):
                errors.append(f"missing {meta_name} in {relative_page}")

        if relative_page == Path("404.html") and "noindex" not in parser.meta.get(
            "robots", ""
        ):
            errors.append("404.html must be marked noindex")

        for raw_url in parser.links:
            target = resolve_local_target(site_root, page, raw_url)
            if target is not None and not target.exists():
                errors.append(
                    f"broken local link in {relative_page}: {raw_url}"
                )

    inspectable_suffixes = {".css", ".html", ".js", ".json", ".txt", ".xml"}
    for file_path in site_root.rglob("*"):
        relative_file = file_path.relative_to(site_root)
        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in inspectable_suffixes
            or file_path.suffix.lower() == ".map"
            or (
                file_path.suffix.lower() == ".js"
                and relative_file.parts[:1] != ("javascripts",)
            )
        ):
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if EMAIL_PATTERN.search(text) or "mailto:" in text.lower():
            errors.append(f"email-like content found in {relative_file}")

    search_index = site_root / "search" / "search_index.json"
    if search_index.is_file():
        search_data = json.loads(search_index.read_text(encoding="utf-8"))
        search_text = json.dumps(search_data, ensure_ascii=False)
        for term in REQUIRED_SEARCH_TERMS:
            if term not in search_text:
                errors.append(f"search index is missing required term: {term}")

    return errors


def main() -> int:
    site_root = Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve()
    if not site_root.is_dir():
        print(f"ERROR: site directory does not exist: {site_root}")
        return 1

    errors = verify_site(site_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    page_count = len(list(site_root.rglob("*.html")))
    print(f"PASS: verified {page_count} HTML pages with no broken local links or emails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
