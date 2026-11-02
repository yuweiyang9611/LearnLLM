"""The site's notebook theme: the shared kit stays unedited and the page checks catch the old design."""
import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SHA-256 of kit/v1 in the project index repository, with LF line endings. A published kit version never
# changes, so a different hash means the site's copy was edited; change the kit upstream instead.
KIT_V1 = {
    "docs/stylesheets/notebook.css": "e5aea068eca83d10ad83a0fbbda3165fbbf0ecf575482066a8d8aba2d137635b",
    "docs/javascripts/theme.js": "2f26afc56d0152d02f363af644ac51b552a890e02d3e81542795aba6b8d5d5e0",
    "docs/assets/images/favicon.svg": "3ea07acd66b71bcef6462252c4e437ee6622f57898089bbb891410e57b20dc0e",
}

HEAD = (
    '<head><title>t</title>'
    '<link rel="stylesheet" href="assets/stylesheets/main.css">'
    '<script src="javascripts/theme.js"></script>'
    '<link rel="stylesheet" href="stylesheets/notebook.css">'
    '<link rel="stylesheet" href="stylesheets/extra.css">'
    "</head>"
)
CHAPTERS = "".join('<li class="nb-entry llm-row"></li>' for _ in range(9))
HOME_BODY = (
    '<body><header class="nb-head"></header>'
    '<a class="llm-continue" href="00_%E5%AD%A6%E4%B9%A0/">从 00 开始</a>'
    f"<ol>{CHAPTERS}</ol></body>"
)


def parse(html: str):
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import verify_site

    parser = verify_site.DocumentParser()
    parser.feed(html)
    return verify_site, parser


class KitCopyTests(unittest.TestCase):
    def test_kit_files_are_unedited_copies_of_v1(self):
        for relative, expected in KIT_V1.items():
            data = (PROJECT_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, relative)

    def test_material_keeps_one_scheme_control_and_the_kit_order(self):
        config = (PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        self.assertIn("font: false", config)
        self.assertNotIn("toggle:", config, "Material's palette toggle would be a second scheme control")
        self.assertLess(config.index("stylesheets/notebook.css"), config.index("stylesheets/extra.css"))


class NotebookPageCheckTests(unittest.TestCase):
    def test_opening_screen_passes(self):
        verify_site, parser = parse(f'<html lang="zh-CN">{HEAD}{HOME_BODY}</html>')
        self.assertEqual(verify_site.notebook_errors(Path("index.html"), parser), [])

    def test_old_material_page_is_caught(self):
        html = (
            '<html lang="zh"><head><link rel="stylesheet" href="stylesheets/extra.css"></head>'
            '<body><header class="md-header"><form data-md-component="palette"></form></header></body></html>'
        )
        verify_site, parser = parse(html)
        errors = " ".join(verify_site.notebook_errors(Path("a/index.html"), parser))
        self.assertIn('lang="zh-CN"', errors)
        self.assertIn("theme.js is not loaded", errors)
        self.assertIn("palette toggle", errors)
        self.assertIn("running head", errors)

    def test_stylesheet_before_theme_script_is_caught(self):
        head = HEAD.replace('<script src="javascripts/theme.js"></script>', "")
        head = head.replace("</head>", '<script src="javascripts/theme.js"></script></head>')
        verify_site, parser = parse(f'<html lang="zh-CN">{head}<body><header class="nb-head"></header></body></html>')
        self.assertIn("out of order", " ".join(verify_site.notebook_errors(Path("a/index.html"), parser)))

    def test_a_record_mark_without_a_record_is_caught(self):
        body = HOME_BODY.replace('<li class="nb-entry llm-row"></li>', '<li class="nb-entry" aria-current="step"></li>', 1)
        verify_site, parser = parse(f'<html lang="zh-CN">{HEAD}{body}</html>')
        self.assertIn("no record", " ".join(verify_site.notebook_errors(Path("index.html"), parser)))

    def test_opening_screen_must_offer_chapter_00(self):
        body = HOME_BODY.replace("00_%E5%AD%A6%E4%B9%A0/", "01_x/")
        verify_site, parser = parse(f'<html lang="zh-CN">{HEAD}{body}</html>')
        self.assertIn("chapter 00", " ".join(verify_site.notebook_errors(Path("index.html"), parser)))


if __name__ == "__main__":
    unittest.main()
