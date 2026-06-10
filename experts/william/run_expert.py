"""Runner for Expert William (Classical Statistics v12).

Invoked in its own subprocess by moe_submission.ipynb. William's pipeline is a
top-level script (faithful export of his notebook) that auto-detects the data
root and runs end-to-end. Running with cwd = repo root lets its `Path(".")`
fallback find train/ and val/.

Usage:  python experts/william/run_expert.py
Writes: expert_william.csv to the current working directory.
"""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    runpy.run_path(str(HERE / "pipeline_v12.py"), run_name="__main__")
