"""Copy the course's source files into the site, so a page reads them in place instead of on GitHub.

The bundle is written before `mkdocs build`: every file under src/, experiments/ and examples/ that a reader
can learn from becomes docs/assets/source/<path>.txt, and docs/assets/source/index.json lists them with the
repository revision they were copied at. The panel on a chapter and the 源码 page read from there, so the
code on a page never drifts from the repository. The bundle is generated, not committed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/yuweiyang9611/LearnLLM"
# The learning folders. Build output, caches and packaging metadata are never source a reader learns from.
SOURCE_ROOTS = ("src", "experiments", "examples")
SOURCE_EXTENSIONS = frozenset(
    {".py", ".md", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".txt", ".csv", ".ps1", ".sh"}
)
EXCLUDED_DIRECTORY = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache|\.venv|node_modules|build|dist|bin|obj|checkpoints|outputs|runs|[^/]*\.egg-info)/"
)
SIZE_LIMIT = 256 * 1024
BUNDLE_VERSION = 1


def bundled_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """The repository files the bundle carries, as paths relative to the repository root, sorted."""

    files: list[Path] = []
    for source_root in SOURCE_ROOTS:
        directory = root / source_root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            # Case-sensitive on purpose: an exclusion must never drop a course folder with a similar name.
            if EXCLUDED_DIRECTORY.search(relative.as_posix() + "/"):
                continue
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if path.stat().st_size > SIZE_LIMIT:
                continue
            files.append(relative)
    return sorted(files, key=lambda item: item.as_posix())


def repository_revision(root: Path = PROJECT_ROOT) -> str:
    """The commit the bundle is copied at: GITHUB_SHA on Actions, otherwise HEAD."""

    candidate = os.environ.get("GITHUB_SHA", "")
    if re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
        return candidate.lower()
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output.lower() if re.fullmatch(r"[0-9a-f]{40}", output) else "unknown"


def count_lines(text: str) -> int:
    lines = text.count("\n")
    if text and not text.endswith("\n"):
        lines += 1
    return lines


def build_bundle(root: Path = PROJECT_ROOT, output: Path | None = None, revision: str | None = None) -> dict:
    """Write the bundle under output (docs/assets/source by default) and return its index."""

    output = output or root / "docs" / "assets" / "source"
    if output.is_dir():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    entries = []
    for relative in bundled_files(root):
        text = (root / relative).read_bytes().decode("utf-8").replace("\r\n", "\n")
        target = output / (relative.as_posix() + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        entries.append(
            {"path": relative.as_posix(), "bytes": len(text.encode("utf-8")), "lines": count_lines(text)}
        )

    revision = revision or repository_revision(root)
    ref = revision if revision != "unknown" else "main"
    index = {
        "version": BUNDLE_VERSION,
        "revision": revision,
        "repository": REPOSITORY_URL,
        "blobBase": f"{REPOSITORY_URL}/blob/{ref}/",
        "files": entries,
    }
    (output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
    )
    return index


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    index = build_bundle(output=output)
    total = sum(entry["bytes"] for entry in index["files"])
    print(
        f"Bundled {len(index['files'])} files ({total / 1024:.0f} KB) at {index['revision'][:7]} "
        f"into {output or PROJECT_ROOT / 'docs' / 'assets' / 'source'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
