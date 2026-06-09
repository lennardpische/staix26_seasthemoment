"""
Out-of-fold validation for STAI-X 2026 Competition
"""
import numpy as np
from lightgbm import LGBMRegressor
from config import N_SPLITS, LGBM_PARAMS, OVERDOSE_CATEGORIES


def run_oof_validation(train_panel, features):
    """
    Run time-series cross-validation

    Args:
        train_panel: Training panel with features
        features: Feature names

    Returns:
        dict: OOF MAEs per category per fold
    """
    unique_periods = sorted(train_panel["period_id"].unique())

    if len(unique_periods) <= 5:
        print("Warning: Not enough periods for OOF validation")
        return None

    oof_maes = {cat: [] for cat in OVERDOSE_CATEGORIES}

    print("\n" + "="*80)
    print("OUT-OF-FOLD VALIDATION")
    print("="*80)

    for fold in range(N_SPLITS):
        split_point = len(unique_periods) * (fold + 1) // (N_SPLITS + 1)
        train_periods = unique_periods[:split_point]
        test_periods = unique_periods[split_point:split_point + len(unique_periods) // (N_SPLITS + 1)]

        if not test_periods:
            continue

        local_train = train_panel[train_panel["period_id"].isin(train_periods)]
        local_test = train_panel[train_panel["period_id"].isin(test_periods)].copy()

        print(f"\nFold {fold + 1}: Train={len(train_periods)} periods, Test={len(test_periods)} periods")

        # Train models for this fold
        for cat in OVERDOSE_CATEGORIES:
            cat_tr = local_train[local_train["overdose_category"] == cat]
            cat_te = local_test[local_test["overdose_category"] == cat]

            if not cat_te.empty:
                model = LGBMRegressor(**LGBM_PARAMS)
                model.fit(cat_tr[features], cat_tr["rate_per_10000_ed_visits"])
                preds = model.predict(cat_te[features])
                local_test.loc[local_test["overdose_category"] == cat, "pred"] = preds

        # Calculate MAEs
        print(f"  {'Category':<20} {'MAE':<12}")
        print(f"  {'-'*35}")
        for cat in OVERDOSE_CATEGORIES:
            cat_mask = local_test["overdose_category"] == cat
            if cat_mask.sum() > 0:
                true_vals = local_test[cat_mask]["rate_per_10000_ed_visits"]
                mae = np.mean(np.abs(true_vals - local_test[cat_mask]["pred"]))
                oof_maes[cat].append(mae)
                print(f"  {cat:<20} {mae:<12.4f}")

    return oof_maes
