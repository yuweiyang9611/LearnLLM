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
# The shared notebook kit, copied unedited from the project index's kit/v1.
KIT_FILES = (
    Path("javascripts") / "theme.js",
    Path("stylesheets") / "notebook.css",
    Path("assets") / "images" / "favicon.svg",
)
# theme.js sets the colour scheme before the first paint, then the kit's styles, then the site's own.
KIT_HEAD_ORDER = ("javascripts/theme.js", "stylesheets/notebook.css", "stylesheets/extra.css")
CHAPTER_COUNT = 9
# Source read in place: scripts/bundle_source.py copies the course files into assets/source/ before the build,
# and a page shows them in the panel or links them on the 源码 page instead of sending the reader to GitHub.
REPOSITORY_URL = "https://github.com/yuweiyang9611/LearnLLM"
SOURCE_PAGE = "源码/"
MIN_BUNDLED_FILES = 30
REQUIRED_BUNDLED_FILES = (
    "experiments/00_environment_check.py",
    "experiments/01_math_foundations.py",
    "experiments/02_tokenization.py",
    "experiments/03_bigram_language_model.py",
    "experiments/04_attention.py",
    "experiments/05_transformer_block.py",
    "experiments/06_train_tiny_gpt.py",
    "experiments/07_generate.py",
    "experiments/08_lora.py",
    "experiments/09_tiny_rag.py",
    "experiments/10_sft_tiny_gpt.py",
    "experiments/11_local_agent.py",
    "examples/capstone_attention/run.py",
    "src/learn_llm/attention.py",
)
GITHUB_FILE_LINK = re.compile(re.escape(REPOSITORY_URL) + r"/(?P<kind>blob|tree)/[^/]+/(?P<path>.+)$")


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
        self._inside_head = False
        self.head_assets: list[str] = []
        self.running_heads = 0
        self.palette_controls = 0
        self.record_entries = 0
        self.stopped_marks = 0
        self.continue_links: list[str] = []
        self._inside_noscript = False
        # Source read in place: the panels' bundled paths, the links into the 源码 page, and every GitHub link
        # outside <noscript>, which is the one place a page may still name GitHub for a bundled file.
        self.source_panels: list[str] = []
        self.source_page_links: list[str] = []
        self.github_links: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {name: value or "" for name, value in attrs}
        classes = values.get("class", "").split()
        if tag == "noscript":
            self._inside_noscript = True
        if "llm-source" in classes and values.get("data-source"):
            self.source_panels.extend(values["data-source"].split())
        if tag == "a" and values.get("href"):
            href = values["href"]
            if href.startswith(REPOSITORY_URL) and not self._inside_noscript:
                self.github_links.append(href)
            if "?" in href and unquote(urlsplit(href).path).endswith(SOURCE_PAGE):
                self.source_page_links.append(href)
        if tag == "head":
            self._inside_head = True
        if self._inside_head and (
            (tag == "script" and values.get("src"))
            or (tag == "link" and "stylesheet" in values.get("rel", "").split())
        ):
            self.head_assets.append(values.get("src") or values.get("href", ""))
        if tag == "header" and "nb-head" in classes:
            self.running_heads += 1
        if values.get("data-md-component") == "palette":
            self.palette_controls += 1
        if "nb-entry" in classes:
            self.record_entries += 1
            if "aria-current" in values or values.get("data-record") == "done":
                self.stopped_marks += 1
        if "nb-stopped" in classes:
            self.stopped_marks += 1
        if tag == "a" and "llm-continue" in classes:
            self.continue_links.append(values.get("href", ""))
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
        elif tag == "head":
            self._inside_head = False
        elif tag == "noscript":
            self._inside_noscript = False

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


def notebook_errors(relative_page: Path, parser: DocumentParser) -> list[str]:
    """Check one page against the notebook design: language, kit order, one scheme control, no false record."""

    errors: list[str] = []
    if parser.html_language != "zh-CN":
        errors.append(f"expected lang=\"zh-CN\" in {relative_page}, found {parser.html_language!r}")

    positions = []
    for asset in KIT_HEAD_ORDER:
        found = [i for i, url in enumerate(parser.head_assets) if url.endswith(asset)]
        if not found:
            errors.append(f"{asset} is not loaded in the <head> of {relative_page}")
        positions.append(found[0] if found else -1)
    if -1 not in positions and positions != sorted(positions):
        errors.append(f"kit assets are out of order in {relative_page}: theme.js, notebook.css, extra.css")

    if parser.running_heads != 1:
        errors.append(f"expected one running head in {relative_page}, found {parser.running_heads}")
    if parser.palette_controls:
        errors.append(f"Material's palette toggle is back in {relative_page}; the kit's 白天/晚上 is the one control")
    # The site keeps no learning record, so nothing may claim one.
    if parser.stopped_marks:
        errors.append(f"{relative_page} shows a stopped or done mark, but the site keeps no record")

    if relative_page == Path("index.html"):
        if len(parser.continue_links) != 1:
            errors.append("the opening screen needs exactly one continue link")
        elif "00_" not in unquote(parser.continue_links[0]):
            errors.append("with no record, the continue link must offer chapter 00")
        if parser.record_entries != CHAPTER_COUNT:
            errors.append(
                f"expected {CHAPTER_COUNT} chapter rows on the opening screen, found {parser.record_entries}"
            )
    return errors


def load_source_bundle(site_root: Path) -> tuple[set[str], list[str]]:
    """Read assets/source/index.json and check that every file it lists is in the site."""

    errors: list[str] = []
    index_path = site_root / "assets" / "source" / "index.json"
    if not index_path.is_file():
        return set(), ["missing source bundle: assets/source/index.json (run scripts/bundle_source.py first)"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return set(), [f"invalid source bundle index: {error}"]
    if index.get("version") != 1 or not isinstance(index.get("files"), list):
        return set(), ["source bundle index must be version 1 with a files list"]
    if not str(index.get("blobBase", "")).startswith(f"{REPOSITORY_URL}/blob/"):
        errors.append("source bundle blobBase must point at the repository")
    bundled = set()
    for entry in index["files"]:
        path = str(entry.get("path", ""))
        bundled.add(path)
        if not (site_root / "assets" / "source" / (path + ".txt")).is_file():
            errors.append(f"bundled source file is missing: {path}")
        if not isinstance(entry.get("lines"), int) or not isinstance(entry.get("bytes"), int):
            errors.append(f"bundled source entry needs integer lines and bytes: {path}")
    if len(bundled) < MIN_BUNDLED_FILES:
        errors.append(f"source bundle must list at least {MIN_BUNDLED_FILES} files, found {len(bundled)}")
    for required in REQUIRED_BUNDLED_FILES:
        if required not in bundled:
            errors.append(f"source bundle is missing a file the chapters read: {required}")
    return bundled, errors


def source_errors(relative_page: Path, parser: DocumentParser, bundled: set[str]) -> list[str]:
    """Check one page against the source-in-place rule: no GitHub link for a bundled file or directory
    outside <noscript>, and every panel and 源码 link names a bundled path."""

    errors: list[str] = []
    for href in parser.github_links:
        match = GITHUB_FILE_LINK.match(href.split("#", 1)[0].split("?", 1)[0])
        if not match:
            continue
        path = unquote(match.group("path")).rstrip("/")
        if match.group("kind") == "blob" and path in bundled:
            errors.append(f"{relative_page} sends the reader to GitHub for a bundled file: {path}")
        elif match.group("kind") == "tree" and any(item.startswith(path + "/") for item in bundled):
            errors.append(f"{relative_page} sends the reader to GitHub for a bundled directory: {path}")
    for path in parser.source_panels:
        if path not in bundled:
            errors.append(f"{relative_page} embeds a source panel for a file outside the bundle: {path}")
    for href in parser.source_page_links:
        query = dict(part.split("=", 1) for part in urlsplit(href).query.split("&") if "=" in part)
        path = unquote(query.get("path", "")).strip("/")
        directory = unquote(query.get("dir", "")).strip("/")
        if path and path not in bundled:
            errors.append(f"{relative_page} links the 源码 page to a file outside the bundle: {path}")
        if directory and not any(item.startswith(directory + "/") for item in bundled):
            errors.append(f"{relative_page} links the 源码 page to a directory outside the bundle: {directory}")
        if not path and not directory:
            errors.append(f"{relative_page} links the 源码 page with neither path nor dir: {href}")
    return errors


def verify_site(site_root: Path) -> list[str]:
    errors: list[str] = []
    bundled, bundle_errors = load_source_bundle(site_root)
    errors.extend(bundle_errors)
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
    for kit_file in KIT_FILES:
        if not (site_root / kit_file).is_file():
            errors.append(f"missing notebook kit file: {kit_file.as_posix()}")

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

        errors.extend(notebook_errors(relative_page, parser))
        errors.extend(source_errors(relative_page, parser, bundled))

        if relative_page == Path(SOURCE_PAGE) / "index.html" and "noindex" not in parser.meta.get("robots", ""):
            errors.append("the 源码 page must be marked noindex: its content comes from the query string")

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
    bundled, _ = load_source_bundle(site_root)
    print(
        f"PASS: verified {page_count} HTML pages with no broken local links or emails, "
        f"and {len(bundled)} bundled source files read in place."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
