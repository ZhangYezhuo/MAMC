#!/usr/bin/env python
"""Train the original MAMC model through the shared training entry point.

Uses the same command-line options as scripts/train.py, with --model legacy
selected by default. Run this file with --help for the supported options.
"""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    entrypoint = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    sys.argv = [str(entrypoint), "--model", "legacy", *sys.argv[1:]]
    runpy.run_path(str(entrypoint), run_name="__main__")
