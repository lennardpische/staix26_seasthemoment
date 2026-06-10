"""Runner for Expert Eddy (XGBoost tree ensemble).

Invoked in its own subprocess by moe_submission.ipynb. Eddy's modules use
package imports (`from src.X import ...`), so this dir goes on sys.path to make
the `src` package importable.

Usage:  python experts/eddy/run_expert.py [DATA_ROOT] [OUT_CSV]
Writes: expert_eddy.csv (default) to the current working directory.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # enables `import src.*`

from src.predict import run  # noqa: E402


def _resolve_root():
    kaggle = Path("/kaggle/input/staix-challenge")
    if kaggle.exists():
        return kaggle
    if len(sys.argv) > 1 and sys.argv[1]:
        return Path(sys.argv[1])
    return Path.cwd()


if __name__ == "__main__":
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "expert_eddy.csv"
    run(_resolve_root(), out_csv)
