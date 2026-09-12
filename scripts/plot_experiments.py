"""Replot one or more run directories without loading model weights."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learn_llm.plotting import plot_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or manifest JSON files")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    directories = [p.parent if p.is_file() else p for p in args.runs]
    for path in plot_runs(directories, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
