"""End-to-end transformer expert pipeline — outputs expert_transformer.csv."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import _find_data_root, load_train, load_val, load_submission_template
from .features import FeaturePipeline, SCORING_CATEGORIES
from .models import CATEGORIES, train_and_predict


def run(
    data_root: Path | str | None = None,
    output_path: Path | str | None = None,
    use_images: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Full transformer expert pipeline → expert_transformer.csv.

    Args:
        data_root:    Path to the dataset root (auto-detected when None).
        output_path:  Destination CSV (defaults to expert_transformer.csv at repo root).
        use_images:   Whether to load and encode MAT-density PNGs (set False to skip).
        verbose:      Print progress to stdout.
    """
    resolved_root = Path(data_root) if data_root else _find_data_root()
    output_path = Path(output_path) if output_path else Path("expert_transformer.csv")

    if verbose:
        print("Loading data...")
    train_long = load_train(resolved_root)
    val_cov = load_val(resolved_root)
    template = load_submission_template(resolved_root)
    if verbose:
        print(f"  Train: {train_long.shape}  Val: {val_cov.shape}  Template: {len(template)} rows")

    feat_root = resolved_root if use_images else None

    if verbose:
        print("Building features...")
    pipeline = FeaturePipeline()
    train_data, val_data = pipeline.fit_transform(train_long, val_cov, data_root=feat_root)
    if verbose:
        print(
            f"  Numeric: {pipeline.n_numeric}  "
            f"Text SVD: {pipeline.n_text_out}  "
            f"Image PCA: {pipeline.n_image_out}"
        )

    if verbose:
        print("Training (GroupKFold × 5)...")
    val_preds, _oof = train_and_predict(train_data, val_data, verbose=verbose)

    # Expand (N_val_rows × 3) predictions into long format keyed on template columns
    val_keys = val_data["keys"]
    rows = []
    for i, cat in enumerate(CATEGORIES):
        df = val_keys[["period_id", "jurisdiction"]].copy()
        df["overdose_category"] = cat
        df["rate_per_10000_ed_visits"] = np.clip(val_preds[:, i], 0, None)
        rows.append(df)
    preds_long = pd.concat(rows, ignore_index=True)

    # Merge onto template (authoritative for row_ids and row order)
    submission = template[["row_id", "period_id", "jurisdiction", "overdose_category"]].merge(
        preds_long, on=["period_id", "jurisdiction", "overdose_category"], how="left"
    )

    # Fallback: fill any unmatched rows with jurisdiction-category median from train
    fallback = (
        train_long.dropna(subset=["rate_per_10000_ed_visits"])
        .groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
        .median()
        .reset_index()
        .rename(columns={"rate_per_10000_ed_visits": "_fb"})
    )
    submission = submission.merge(fallback, on=["jurisdiction", "overdose_category"], how="left")
    nan_mask = submission["rate_per_10000_ed_visits"].isna()
    submission.loc[nan_mask, "rate_per_10000_ed_visits"] = submission.loc[nan_mask, "_fb"]
    submission.loc[submission["rate_per_10000_ed_visits"].isna(), "rate_per_10000_ed_visits"] = 0.0
    submission = submission[["row_id", "rate_per_10000_ed_visits"]]

    assert len(submission) == len(template), f"Row count: {len(submission)} vs {len(template)}"
    assert submission["rate_per_10000_ed_visits"].isna().sum() == 0, "NaN in expert predictions"

    submission.to_csv(output_path, index=False)
    if verbose:
        print(f"Expert predictions → {output_path}  ({len(submission)} rows)")
    return submission


if __name__ == "__main__":
    run()
