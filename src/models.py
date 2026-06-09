"""
Model training and prediction for STAI-X 2026 Competition
"""
import numpy as np
from lightgbm import LGBMRegressor
from config import LGBM_PARAMS, OVERDOSE_CATEGORIES


def train_models(train_panel, features):
    """
    Train category-specific models

    Args:
        train_panel: Training dataframe
        features: List of feature column names

    Returns:
        dict: Trained models keyed by category
    """
    models = {}

    for cat in OVERDOSE_CATEGORIES:
        cat_data = train_panel[train_panel["overdose_category"] == cat]

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(cat_data[features], cat_data["rate_per_10000_ed_visits"])

        models[cat] = model

    return models


def predict(val_panel, models, features):
    """
    Generate predictions

    Args:
        val_panel: Validation panel with features
        models: Trained models dict
        features: Feature names

    Returns:
        list: Predictions
    """
    preds = []

    for idx, row in val_panel.iterrows():
        cat = row["overdose_category"]
        feat_vector = row[features].values.reshape(1, -1)
        pred = models[cat].predict(feat_vector)[0]
        preds.append(pred)

    return preds


def apply_hierarchical_reconciliation(val_panel, pred_col="pred", alpha=0.6, residual=0.5):
    """
    Apply hierarchical reconciliation to ensure all_drugs >= all_opioids + all_stimulants

    Args:
        val_panel: Validation panel with predictions
        pred_col: Column name containing predictions
        alpha: Weight for top-down prediction (0-1)
        residual: Small residual for "other drugs"

    Returns:
        DataFrame with reconciled predictions
    """
    # Pivot to wide format
    pivoted = val_panel.pivot(
        index=["period_id", "jurisdiction"],
        columns="overdose_category",
        values=pred_col
    ).reset_index()

    # Bottom-up estimate
    all_drugs_bottom_up = pivoted["all_opioids"] + pivoted["all_stimulants"] + residual

    # Blend top-down and bottom-up
    pivoted["all_drugs_reconciled"] = (
        alpha * pivoted["all_drugs"] +
        (1 - alpha) * all_drugs_bottom_up
    )

    # Enforce hierarchy via proportional scaling
    for idx, row in pivoted.iterrows():
        total_subs = row["all_opioids"] + row["all_stimulants"]
        if total_subs > row["all_drugs_reconciled"]:
            scale_factor = row["all_drugs_reconciled"] / (total_subs + 1e-6)
            pivoted.at[idx, "all_opioids"] *= scale_factor
            pivoted.at[idx, "all_stimulants"] *= scale_factor

    pivoted["all_drugs"] = pivoted["all_drugs_reconciled"]

    # Melt back to long format
    melted = pivoted.drop(columns=["all_drugs_reconciled"]).melt(
        id_vars=["period_id", "jurisdiction"],
        value_name="final_rate",
        var_name="overdose_category"
    )

    return melted
