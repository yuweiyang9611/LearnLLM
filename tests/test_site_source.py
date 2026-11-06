"""Source read in place: the bundle carries the course files, and the site checks keep every page off GitHub for them."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bundle_source  # noqa: E402
import verify_site  # noqa: E402

HEAD = (
    '<head><title>t</title>'
    '<script src="../javascripts/theme.js"></script>'
    '<link rel="stylesheet" href="../stylesheets/notebook.css">'
    '<link rel="stylesheet" href="../stylesheets/extra.css">'
    "</head>"
)
BUNDLED = {"experiments/04_attention.py", "src/learn_llm/attention.py", "examples/capstone_attention/run.py"}
REPO = verify_site.REPOSITORY_URL


def source_errors(body: str) -> str:
    parser = verify_site.DocumentParser()
    parser.feed(f'<html lang="zh-CN">{HEAD}<body><header class="nb-head"></header>{body}</body></html>')
    return " ".join(verify_site.source_errors(Path("a/index.html"), parser, BUNDLED))


class BundleTests(unittest.TestCase):
    def test_bundle_copies_learning_files_and_writes_the_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src" / "pkg").mkdir(parents=True)
            (root / "src" / "pkg" / "a.py").write_text("x = 1\ny = 2", encoding="utf-8")
            (root / "src" / "pkg" / "__pycache__").mkdir()
            (root / "src" / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"\x00")
            (root / "src" / "pkg.egg-info").mkdir()
            (root / "src" / "pkg.egg-info" / "PKG-INFO").write_text("Name: pkg\n", encoding="utf-8")
            (root / "experiments").mkdir()
            (root / "experiments" / "00_check.py").write_bytes(b"print('ok')\r\n")
            (root / "experiments" / "big.py").write_text("#" * (bundle_source.SIZE_LIMIT + 1), encoding="utf-8")
            (root / "experiments" / "model.bin").write_bytes(b"\x00\x01")
            (root / "examples" / "Output").mkdir(parents=True)
            (root / "examples" / "Output" / "notes.md").write_text("# notes\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_a.py").write_text("pass\n", encoding="utf-8")

            output = root / "out"
            index = bundle_source.build_bundle(root, output, revision="a" * 40)

            self.assertEqual(index["version"], 1)
            self.assertEqual(index["revision"], "a" * 40)
            self.assertEqual(index["repository"], REPO)
            self.assertEqual(index["blobBase"], f"{REPO}/blob/{'a' * 40}/")
            # The exclusion is case-sensitive, so examples/Output stays: it is course content, not build output.
            self.assertEqual(
                [entry["path"] for entry in index["files"]],
                ["examples/Output/notes.md", "experiments/00_check.py", "src/pkg/a.py"],
            )
            by_path = {entry["path"]: entry for entry in index["files"]}
            self.assertEqual(by_path["src/pkg/a.py"]["lines"], 2)
            self.assertEqual(by_path["experiments/00_check.py"]["lines"], 1)
            self.assertEqual((output / "experiments" / "00_check.py.txt").read_bytes(), b"print('ok')\n")
            self.assertEqual(json.loads((output / "index.json").read_text(encoding="utf-8")), index)

    def test_repository_bundle_carries_every_file_the_chapters_read(self):
        bundled = {path.as_posix() for path in bundle_source.bundled_files(PROJECT_ROOT)}
        for required in verify_site.REQUIRED_BUNDLED_FILES:
            self.assertIn(required, bundled)
        self.assertGreaterEqual(len(bundled), verify_site.MIN_BUNDLED_FILES)
        self.assertFalse([path for path in bundled if ".egg-info" in path or "__pycache__" in path])

    def test_chapter_panels_and_source_links_name_bundled_files(self):
        bundled = {path.as_posix() for path in bundle_source.bundled_files(PROJECT_ROOT)}
        panels = 0
        for page in (PROJECT_ROOT / "docs").glob("*.md"):
            text = page.read_text(encoding="utf-8")
            for match in re.finditer(r'class="nb-code llm-source"[^>]*data-source="([^"]+)"', text):
                panels += 1
                for path in match.group(1).split():
                    self.assertIn(path, bundled, page.name)
            for match in re.finditer(r"\(源码\.md\?path=([^)#]+)", text):
                self.assertIn(match.group(1), bundled, page.name)
            for match in re.finditer(r"\(源码\.md\?dir=([^)#]+)", text):
                self.assertTrue(any(path.startswith(match.group(1) + "/") for path in bundled), page.name)
            # Outside <noscript>, a chapter never names GitHub for a file the site carries.
            prose = re.sub(r"<noscript>.*?</noscript>", "", text, flags=re.S)
            for match in re.finditer(re.escape(REPO) + r"/(blob|tree)/[^/]+/([^)\"#?\s]+)", prose):
                target = match.group(2).rstrip("/")
                self.assertNotIn(target, bundled, page.name)
                self.assertFalse(any(path.startswith(target + "/") for path in bundled), page.name)
        self.assertGreaterEqual(panels, 8, "chapters 00-03 and 05-08 embed the panel")

    def test_site_loads_the_panel_scripts_after_the_kit(self):
        config = (PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        order = [config.index(name) for name in ("javascripts/code-colors.js", "javascripts/source-panel.js", "javascripts/source-site.js")]
        self.assertEqual(order, sorted(order))
        self.assertIn("- 源码: 源码.md", config)
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
        self.assertLess(workflow.index("scripts/bundle_source.py"), workflow.index("mkdocs build --strict"))
        self.assertIn("robots: noindex", (PROJECT_ROOT / "docs" / "源码.md").read_text(encoding="utf-8"))


class SiteCheckTests(unittest.TestCase):
    def test_a_github_link_for_a_bundled_file_is_caught(self):
        errors = source_errors(f'<a href="{REPO}/blob/main/experiments/04_attention.py#L3">x</a>')
        self.assertIn("bundled file: experiments/04_attention.py", errors)

    def test_a_github_link_for_a_bundled_directory_is_caught(self):
        errors = source_errors(f'<a href="{REPO}/tree/main/examples/capstone_attention">x</a>')
        self.assertIn("bundled directory: examples/capstone_attention", errors)

    def test_github_links_outside_the_bundle_and_inside_noscript_pass(self):
        body = (
            f'<a href="{REPO}">GitHub</a>'
            f'<a href="{REPO}/blob/main/README.md">readme</a>'
            f'<a href="{REPO}/tree/main/data">data</a>'
            f'<noscript><a href="{REPO}/blob/main/experiments/04_attention.py">x</a></noscript>'
            f'<a href="{REPO}/releases">releases</a>'
        )
        self.assertEqual(source_errors(body), "")

    def test_a_panel_or_source_link_outside_the_bundle_is_caught(self):
        body = (
            '<div class="nb-code llm-source" data-source="experiments/04_attention.py experiments/99_missing.py"></div>'
            '<a href="../%E6%BA%90%E7%A0%81/?path=src/learn_llm/missing.py">a</a>'
            '<a href="../%E6%BA%90%E7%A0%81/?dir=labs">b</a>'
            '<a href="../%E6%BA%90%E7%A0%81/?path=experiments%2F04_attention.py#L1-L3">ok</a>'
            '<a href="../%E6%BA%90%E7%A0%81/?dir=examples%2Fcapstone_attention">ok</a>'
        )
        errors = source_errors(body)
        self.assertIn("outside the bundle: experiments/99_missing.py", errors)
        self.assertIn("file outside the bundle: src/learn_llm/missing.py", errors)
        self.assertIn("directory outside the bundle: labs", errors)
        self.assertNotIn("04_attention", errors)
        self.assertNotIn("capstone", errors)

    def test_bundle_index_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp)
            bundled, errors = verify_site.load_source_bundle(site)
            self.assertEqual(bundled, set())
            self.assertIn("missing source bundle", errors[0])
            source = site / "assets" / "source"
            source.mkdir(parents=True)
            files = [{"path": path, "bytes": 1, "lines": 1} for path in verify_site.REQUIRED_BUNDLED_FILES]
            files += [{"path": f"src/learn_llm/m{i}.py", "bytes": 1, "lines": 1} for i in range(verify_site.MIN_BUNDLED_FILES)]
            index = {"version": 1, "revision": "b" * 40, "repository": REPO, "blobBase": f"{REPO}/blob/main/", "files": files}
            (source / "index.json").write_text(json.dumps(index), encoding="utf-8")
            for entry in files[1:]:
                target = source / (entry["path"] + ".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x\n", encoding="utf-8")
            bundled, errors = verify_site.load_source_bundle(site)
            self.assertEqual(errors, ["bundled source file is missing: experiments/00_environment_check.py"])
            self.assertIn("src/learn_llm/m0.py", bundled)


if __name__ == "__main__":
    unittest.main()
