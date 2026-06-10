"""Data loading utilities for STAI-X 2026 — transformer expert pipeline."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# COVID structural break — overdose rates accelerated sharply from this point
_COVID_START = pd.Timestamp("2020-03-31")


def _find_data_root() -> Path:
    kaggle = Path("/kaggle/input/staix-challenge")
    if kaggle.exists():
        return kaggle
    local = Path(__file__).parent.parent
    if (local / "train" / "dose_sys_train.csv").exists():
        return local
    raise FileNotFoundError("Cannot locate data root. Expected 'train/dose_sys_train.csv'.")


def load_period_date_map() -> pd.DataFrame | None:
    """Load period_id_map.json → DataFrame with period_id, year, month, quarter,
    month_sin, month_cos, months_since_covid, period_rank columns.

    Returns None if the file doesn't exist (falls back to temperature proxy).
    """
    path = Path(__file__).parent.parent / "period_id_map.json"
    if not path.exists():
        return None
    with open(path) as f:
        date_to_id = json.load(f)
    rows = []
    for date_str, pid in sorted(date_to_id.items()):
        dt = pd.Timestamp(date_str)
        rows.append({
            "period_id": pid,
            "year": dt.year,
            "month": dt.month,
            "quarter": dt.quarter,
            "month_sin": np.sin(2 * np.pi * dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dt.month / 12),
            "months_since_covid": max(0, (dt.year - _COVID_START.year) * 12
                                      + dt.month - _COVID_START.month),
        })
    df = pd.DataFrame(rows)
    df["period_rank"] = range(1, len(df) + 1)  # exact chronological rank
    return df


def _compute_global_period_rank(train_cov: pd.DataFrame, val_cov: pd.DataFrame) -> dict[str, int]:
    """Fallback: rank period_ids by cross-jurisdiction mean temp_avg_f.

    Only used when period_id_map.json is absent.
    """
    all_cov = pd.concat([train_cov, val_cov], ignore_index=True)
    period_temp = all_cov.groupby("period_id")["temp_avg_f"].mean()
    if period_temp.notna().sum() < 2:
        return {pid: i for i, pid in enumerate(sorted(all_cov["period_id"].unique()))}
    return period_temp.rank(method="first").astype(int).to_dict()


def _attach_date_features(df: pd.DataFrame, date_map: pd.DataFrame | None,
                           rank_map: dict | None) -> pd.DataFrame:
    """Attach period_rank + date features to any dataframe with a period_id column."""
    if date_map is not None:
        df = df.merge(date_map, on="period_id", how="left")
    else:
        df["period_rank"] = df["period_id"].map(rank_map)
    return df


def load_train(data_root: Path | None = None) -> pd.DataFrame:
    """Return merged train dataframe with targets, covariates, and date features."""
    root = Path(data_root) if data_root else _find_data_root()
    targets = pd.read_csv(root / "train" / "dose_sys_train.csv")
    train_cov = pd.read_csv(root / "train" / "covariates.csv")
    val_cov = pd.read_csv(root / "val" / "covariates.csv")

    date_map = load_period_date_map()
    rank_map = None if date_map is not None else _compute_global_period_rank(train_cov, val_cov)

    df = targets.merge(train_cov, on=["period_id", "jurisdiction"], how="left")
    df["is_suppressed"] = df["rate_per_10000_ed_visits"].isna().astype(int)
    df = _attach_date_features(df, date_map, rank_map)
    return df


def load_val(data_root: Path | None = None) -> pd.DataFrame:
    """Return val covariates with date features (target is hidden)."""
    root = Path(data_root) if data_root else _find_data_root()
    train_cov = pd.read_csv(root / "train" / "covariates.csv")
    val_cov = pd.read_csv(root / "val" / "covariates.csv")

    date_map = load_period_date_map()
    rank_map = None if date_map is not None else _compute_global_period_rank(train_cov, val_cov)

    return _attach_date_features(val_cov, date_map, rank_map)


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
