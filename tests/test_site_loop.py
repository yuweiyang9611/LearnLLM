"""The loop on every chapter: the band's static rules, the self-check bank, the chapter markers and the scripts
that run in the browser on Pyodide without torch."""
import json
import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import verify_site  # noqa: E402

DOCS = PROJECT_ROOT / "docs"
CHAPTER_PAGES = sorted(DOCS.glob("0[0-8]_*.md"))
RUNS_BY_CHAPTER = {
    "02": ["experiments/03_bigram_language_model.py"],
    "07": ["experiments/09_tiny_rag.py", "experiments/11_local_agent.py"],
}
HEAD = (
    '<head><title>t</title><meta name="llm-chapter" content="02">'
    '<script src="../javascripts/theme.js"></script>'
    '<link rel="stylesheet" href="../stylesheets/notebook.css">'
    '<link rel="stylesheet" href="../stylesheets/extra.css"></head>'
)
BAND = (
    '<ol class="nb-steps llm-steps" data-chapter="02">'
    '<li data-step="read" data-state="pending"><a href="#">读</a></li>'
    '<li data-step="code" data-state="pending"><a href="#llm-source">看代码</a></li>'
    '<li data-step="run" data-state="pending"><a href="#llm-source">跑</a></li>'
    '<li data-step="quiz" data-state="pending"><a href="#llm-quiz">自测</a><span class="nb-steps-count">3 题</span></li>'
    '<li data-step="note" data-state="pending"><a href="#llm-notepad">记一笔</a></li></ol>'
)
PANEL = (
    '<div class="nb-code llm-source" id="llm-source" data-run="experiments/03_bigram_language_model.py" '
    'data-source="experiments/02_tokenization.py experiments/03_bigram_language_model.py"></div>'
)
MARKERS = '<div class="llm-quiz" data-chapter="02"></div><div class="llm-notepad" data-chapter="02"></div>'


def loop_errors(body: str, head: str = HEAD) -> str:
    parser = verify_site.DocumentParser()
    parser.feed(f'<html lang="zh-CN">{head}<body><header class="nb-head"></header>{body}</body></html>')
    return " ".join(verify_site.loop_errors(Path("02_x/index.html"), parser))


def quiz_bank() -> dict:
    text = (DOCS / "javascripts" / "quiz-bank.js").read_text(encoding="utf-8")
    start = text.index("const BANK = ") + len("const BANK = ")
    end = text.index("\n  };\n", start) + len("\n  }")
    return json.loads(text[start:end])


class BandRuleTests(unittest.TestCase):
    def test_a_chapter_page_with_the_band_and_markers_passes(self):
        self.assertEqual(loop_errors(f"<h1>02</h1>{BAND}{PANEL}{MARKERS}"), "")

    def test_a_chapter_page_without_a_panel_or_a_runnable_script_has_dashed_steps(self):
        band = BAND.replace('data-step="code" data-state="pending"', 'data-step="code" data-state="none"')
        band = band.replace('data-step="run" data-state="pending"', 'data-step="run" data-state="none"')
        self.assertEqual(loop_errors(f"<h1>04</h1>{band}{MARKERS}"), "")
        self.assertIn("code step", loop_errors(f"<h1>04</h1>{BAND}{MARKERS}"))

    def test_the_band_must_follow_the_title_in_the_right_order(self):
        self.assertIn("follow the H1", loop_errors(f"<h1>02</h1><p>x</p>{BAND}{PANEL}{MARKERS}"))
        swapped = BAND.replace('data-step="read"', 'data-step="tmp"').replace('data-step="code"', 'data-step="read"').replace('data-step="tmp"', 'data-step="code"')
        self.assertIn("in that order", loop_errors(f"<h1>02</h1>{swapped}{PANEL}{MARKERS}"))
        self.assertIn("one loop band", loop_errors(f"<h1>02</h1>{BAND}{BAND}{PANEL}{MARKERS}"))

    def test_a_static_done_or_now_state_is_caught(self):
        for state in ("done", "now"):
            band = BAND.replace('data-step="read" data-state="pending"', f'data-step="read" data-state="{state}"')
            self.assertIn("read step", loop_errors(f"<h1>02</h1>{band}{PANEL}{MARKERS}"), state)

    def test_markers_and_runnable_files_are_checked(self):
        self.assertIn("self-check marker", loop_errors(f"<h1>02</h1>{BAND}{PANEL}"))
        self.assertIn("note marker", loop_errors(f'<h1>02</h1>{BAND}{PANEL}<div class="llm-quiz" data-chapter="02"></div>'))
        wrong = PANEL.replace("experiments/03_bigram_language_model.py", "experiments/02_tokenization.py", 1)
        self.assertIn("browser cannot run", loop_errors(f"<h1>02</h1>{BAND}{wrong}{MARKERS}"))
        other = PANEL.replace('data-run="experiments/03_bigram_language_model.py"', 'data-run="experiments/09_tiny_rag.py"')
        self.assertIn("panel does not show", loop_errors(f"<h1>02</h1>{BAND}{other}{MARKERS}"))

    def test_a_page_that_is_not_a_chapter_carries_no_band(self):
        head = HEAD.replace('<meta name="llm-chapter" content="02">', "")
        self.assertIn("not a chapter page", loop_errors(f"<h1>x</h1>{BAND}", head))
        self.assertEqual(loop_errors("<h1>x</h1><p>prose</p>", head), "")


class ChapterMarkupTests(unittest.TestCase):
    def test_every_chapter_carries_its_markers_and_the_panel_id(self):
        self.assertEqual(len(CHAPTER_PAGES), verify_site.CHAPTER_COUNT)
        for page in CHAPTER_PAGES:
            chapter = page.name[:2]
            text = page.read_text(encoding="utf-8")
            quiz = f'<div class="llm-quiz" id="llm-quiz" data-chapter="{chapter}"></div>'
            self.assertEqual(text.count(quiz), 1, page.name)
            last_heading = max(match.start() for match in re.finditer(r"^## ", text, flags=re.M))
            self.assertLess(text.index(quiz), last_heading, f"{page.name}: the self-check sits before the last section")
            self.assertTrue(text.rstrip().endswith(f'<div class="llm-notepad" id="llm-notepad" data-chapter="{chapter}"></div>'), page.name)
            panels = re.findall(r'<div class="nb-code llm-source"([^>]*)>', text)
            self.assertEqual(len(panels), 0 if chapter == "04" else 1, page.name)
            for attributes in panels:
                self.assertIn('id="llm-source"', attributes, page.name)
                run = re.search(r'data-run="([^"]*)"', attributes)
                self.assertEqual(run.group(1).split() if run else [], RUNS_BY_CHAPTER.get(chapter, []), page.name)

    def test_the_runnable_scripts_need_nothing_the_browser_lacks(self):
        runs = set()
        for paths in RUNS_BY_CHAPTER.values():
            runs.update(paths)
        self.assertEqual(runs, set(verify_site.BROWSER_RUNNABLE))
        run_site = (DOCS / "javascripts" / "run-site.js").read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r'^    "(experiments/[^"]+)": \[', run_site, flags=re.M)), runs)
        browser_common = (DOCS / "assets" / "browser" / "_common.py").read_text(encoding="utf-8")
        for name in ("PROJECT_ROOT", "SRC_DIR", "DATA_DIR", "CHECKPOINT_DIR", "def seed_everything"):
            self.assertIn(name, browser_common)
        self.assertFalse([line for line in browser_common.splitlines() if line.startswith(("import ", "from ")) and "torch" in line])
        for path in runs:
            source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
            imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
            self.assertFalse([line for line in imports if "torch" in line or "numpy" in line], path)
            for module in re.findall(r"^from learn_llm\.(\w+) import", source, flags=re.M):
                self.assertIn(module, {"rag", "agent"}, path)
                module_source = (PROJECT_ROOT / "src" / "learn_llm" / f"{module}.py").read_text(encoding="utf-8")
                self.assertNotIn("import torch", module_source, module)

    def test_site_loads_the_loop_scripts_in_order_with_a_pinned_runtime(self):
        config = (PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        names = ("source-site.js", "quiz-bank.js", "progress.js", "python-runner.js", "run-site.js", "quiz.js", "loop-site.js", "mathjax.js")
        order = [config.index(f"javascripts/{name}") for name in names]
        self.assertEqual(order, sorted(order))
        worker = (DOCS / "javascripts" / "python-worker.js").read_text(encoding="utf-8")
        version = re.search(r'const PYODIDE_VERSION = "(\d+\.\d+\.\d+)"', worker)
        self.assertIsNotNone(version, "the Pyodide version is pinned")
        self.assertIn(f"https://cdn.jsdelivr.net/pyodide/v${{PYODIDE_VERSION}}/full/", worker)
        runner = (DOCS / "javascripts" / "python-runner.js").read_text(encoding="utf-8")
        self.assertIn('"加载 Python 运行时"', runner)
        self.assertIn("execute: [20000", runner)
        progress = (DOCS / "javascripts" / "progress.js").read_text(encoding="utf-8")
        self.assertIn('const KEY = "learn-llm-progress-v1";', progress)


class QuizBankTests(unittest.TestCase):
    def test_three_questions_per_chapter_with_one_answer_each(self):
        bank = quiz_bank()
        self.assertEqual(sorted(bank), [f"{index:02d}" for index in range(verify_site.CHAPTER_COUNT)])
        ids = []
        answers = []
        for chapter, questions in bank.items():
            self.assertEqual(len(questions), 3, chapter)
            for question in questions:
                self.assertEqual(sorted(question), ["answer", "explanation", "id", "options", "question"])
                self.assertTrue(question["id"].startswith(f"{chapter}-"), question["id"])
                ids.append(question["id"])
                self.assertEqual(len(question["options"]), 4, question["id"])
                self.assertEqual(len(set(question["options"])), 4, question["id"])
                self.assertIn(question["answer"], range(4), question["id"])
                self.assertTrue(question["question"].strip() and question["explanation"].strip(), question["id"])
                answers.append(question["answer"])
        self.assertEqual(len(ids), 27)
        self.assertEqual(len(set(ids)), 27)
        self.assertEqual(set(answers), {0, 1, 2, 3}, "the right answer moves around")


if __name__ == "__main__":
    unittest.main()
