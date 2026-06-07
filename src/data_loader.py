"""Data loading utilities for STAI-X 2026 — transformer expert pipeline."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _find_data_root() -> Path:
    kaggle = Path("/kaggle/input/staix-challenge")
    if kaggle.exists():
        return kaggle
    local = Path(__file__).parent.parent
    if (local / "train" / "dose_sys_train.csv").exists():
        return local
    raise FileNotFoundError("Cannot locate data root. Expected 'train/dose_sys_train.csv'.")


def _compute_global_period_rank(train_cov: pd.DataFrame, val_cov: pd.DataFrame) -> dict[str, int]:
    """Rank all period_ids by cross-jurisdiction mean temp_avg_f (proxy for calendar season).

    Val periods are interleaved with train periods in temperature space, so the
    rank must be computed globally. Falls back to alphabetical order when
    temp_avg_f is entirely missing.
    """
    all_cov = pd.concat([train_cov, val_cov], ignore_index=True)
    period_temp = all_cov.groupby("period_id")["temp_avg_f"].mean()
    if period_temp.notna().sum() < 2:
        return {pid: i for i, pid in enumerate(sorted(all_cov["period_id"].unique()))}
    return period_temp.rank(method="first").astype(int).to_dict()


def load_train(data_root: Path | None = None) -> pd.DataFrame:
    """Return merged train dataframe (targets + covariates) with global period_rank."""
    root = Path(data_root) if data_root else _find_data_root()
    targets = pd.read_csv(root / "train" / "dose_sys_train.csv")
    train_cov = pd.read_csv(root / "train" / "covariates.csv")
    val_cov = pd.read_csv(root / "val" / "covariates.csv")
    rank_map = _compute_global_period_rank(train_cov, val_cov)
    df = targets.merge(train_cov, on=["period_id", "jurisdiction"], how="left")
    df["is_suppressed"] = df["rate_per_10000_ed_visits"].isna().astype(int)
    df["period_rank"] = df["period_id"].map(rank_map)
    return df


def load_val(data_root: Path | None = None) -> pd.DataFrame:
    """Return val covariates with global period_rank (target is hidden)."""
    root = Path(data_root) if data_root else _find_data_root()
    train_cov = pd.read_csv(root / "train" / "covariates.csv")
    val_cov = pd.read_csv(root / "val" / "covariates.csv")
    rank_map = _compute_global_period_rank(train_cov, val_cov)
    val_cov["period_rank"] = val_cov["period_id"].map(rank_map)
    return val_cov


def load_submission_template(data_root: Path | None = None) -> pd.DataFrame:
    root = Path(data_root) if data_root else _find_data_root()
    return pd.read_csv(root / "sample_submission.csv")


def load_image_flat(
    root: Path, split: str, jurisdiction: str, period_id: str, img_size: int = 32
) -> np.ndarray:
    """Load a MAT-density PNG, resize to img_size×img_size RGB, return flat float32 vector.

    Returns a zero vector when the image is missing or unreadable.
    """
    n = img_size * img_size * 3
    try:
        from PIL import Image
        path = root / split / "images" / "mat_density" / f"{jurisdiction}_{period_id}.png"
        if not path.exists():
            return np.zeros(n, dtype=np.float32)
        img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        return (np.array(img, dtype=np.float32) / 255.0).flatten()
    except Exception:
        return np.zeros(n, dtype=np.float32)


def load_images_for(
    root: Path, split: str, keys: pd.DataFrame, img_size: int = 32
) -> np.ndarray:
    """Load images for every row in keys (must have jurisdiction + period_id columns).

    Returned array shape: (len(keys), img_size * img_size * 3).
    Row order matches keys row order exactly.
    """
    return np.stack([
        load_image_flat(root, split, r.jurisdiction, r.period_id, img_size)
        for r in keys.itertuples(index=False)
    ], axis=0)
