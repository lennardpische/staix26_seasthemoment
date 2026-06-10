"""Runner for Expert Jasmine (Healthcare LightGBM).

Invoked in its own subprocess by moe_submission.ipynb. Jasmine's modules use
flat imports (`from config import ...`), so this dir goes on sys.path and the
process runs with cwd = repo root (where train/ and val/ live) for data detection.

Usage:  python experts/jasmine/run_expert.py
Writes: expert_jasmine.csv (via her get_output_path) to the working directory.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # enables flat `import config`, `import main`, ...

import main as jasmine_main  # noqa: E402


if __name__ == "__main__":
    jasmine_main.main()
