"""Runner for Expert Lenny (FT-Transformer).

Invoked in its own subprocess by moe_submission.ipynb so the `lenny.*` modules
never collide with the other experts' identically-named modules.

Usage:  python experts/lenny/run_expert.py [DATA_ROOT] [OUT_CSV]
Writes: expert_lenny.csv (default) to the current working directory.
"""
import sys
from pathlib import Path

EXPERTS_DIR = Path(__file__).resolve().parent.parent  # .../experts
sys.path.insert(0, str(EXPERTS_DIR))                   # enables `import lenny.*`

from lenny.predict import run  # noqa: E402  (path set above)


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "expert_lenny.csv"
    # Reuses checkpoints/ if present (resumes folds), TF-IDF/PCA fallback when
    # embeddings/ cache is absent. Exact same code as the transformer-lenny branch.
    run(data_root=data_root, output_path=out_csv)


if __name__ == "__main__":
    main()
