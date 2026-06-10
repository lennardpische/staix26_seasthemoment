"""
Main execution script for STAI-X 2026 Expert Jasmine
Lag-based forecasting with advanced time-series features
"""
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import pandas as pd
import numpy as np

from config import EXPERT_NAME, OVERDOSE_CATEGORIES, RECONCILIATION_ALPHA, RESIDUAL_DRUGS
from data_loader import get_data_paths, load_train_data, load_validation_data, get_output_path
from features import run_feature_engineering, get_feature_names
from models import train_models, predict, apply_hierarchical_reconciliation
from validation import run_oof_validation
from utils import validate_submission, print_oof_results


def main():
    """Main execution pipeline"""

    print("="*80)
    print("STAI-X 2026: Expert Jasmine")
    print("="*80)

    # ===== GET DATA PATHS (auto-detect Kaggle vs local) =====
    paths = get_data_paths()
    print(f"Using data root: {paths['root']}")

    # ===== LOAD DATA =====
    train_target, train_cov = load_train_data()
    train_target = train_target.dropna(subset=["rate_per_10000_ed_visits"])

    val_cov, sample_sub = load_validation_data()

    print(f"\nAfter cleaning:")
    print(f"  Training samples: {len(train_target)}")
    print(f"  Validation samples: {len(sample_sub)}")

    # ===== JURISDICTION STATS =====
    print("\nCalculating jurisdiction embeddings...")
    jurisdiction_stats = train_target.groupby(['jurisdiction', 'overdose_category'])[
        'rate_per_10000_ed_visits'
    ].agg([
        ('jur_mean', 'mean'),
        ('jur_std', 'std'),
        ('jur_median', 'median'),
        ('jur_min', 'min'),
        ('jur_max', 'max'),
        ('jur_q25', lambda x: x.quantile(0.25)),
        ('jur_q75', lambda x: x.quantile(0.75))
    ]).reset_index()

    high_risk_jur = train_target.groupby('jurisdiction')[
        'rate_per_10000_ed_visits'
    ].mean().nlargest(15).index.tolist()

    # ===== FEATURE ENGINEERING =====
    print("\n" + "="*80)
    print("Building features with target lags...")
    print("="*80)
    train_feats = run_feature_engineering(
        train_cov, str(paths['train_img_dir']), target_df=train_target, build_target_lags=True
    )

    # ===== BUILD TRAINING PANEL =====
    if "rate_per_10000_ed_visits" not in train_feats.columns:
        train_panel = pd.merge(
            train_target, train_feats, on=["period_id", "jurisdiction"], how="left"
        )
    else:
        train_panel = train_feats.copy()

    train_panel = pd.merge(
        train_panel, jurisdiction_stats,
        on=["jurisdiction", "overdose_category"], how="left"
    )
    train_panel['is_high_risk_jurisdiction'] = \
        train_panel['jurisdiction'].isin(high_risk_jur).astype(float)

    # ===== DEFINE FEATURES =====
    base_features = get_feature_names(include_target_lags=False)
    target_lag_features = [col for col in train_panel.columns if col.startswith('target_')]
    features = base_features + target_lag_features

    print(f"\n{'='*80}")
    print(f"FEATURE SUMMARY")
    print(f"{'='*80}")
    print(f"Total features: {len(features)}")
    print(f"  - Base: {len(base_features)}")
    print(f"  - Target lags: {len(target_lag_features)}")

    # Impute
    train_panel[features] = train_panel[features].fillna(train_panel[features].median())

    # ===== OOF VALIDATION =====
    oof_maes = run_oof_validation(train_panel, features)

    if oof_maes:
        scores, block_avg = print_oof_results(oof_maes, OVERDOSE_CATEGORIES)
    else:
        print("\nSkipping OOF validation (not enough periods)")
        scores, block_avg = None, None

    # ===== TRAIN FINAL MODELS =====
    print("\n\nTraining final models on full data...")
    models = train_models(train_panel, features)

    # ===== PREPARE VALIDATION DATA =====
    print("\nPreparing validation data...")

    val_feats = run_feature_engineering(val_cov, str(paths['val_img_dir']), build_target_lags=False)

    val_panel = pd.merge(
        sample_sub.drop(columns=["rate_per_10000_ed_visits"]),
        val_feats,
        on=["period_id", "jurisdiction"],
        how="left"
    )
    val_panel = pd.merge(
        val_panel, jurisdiction_stats,
        on=["jurisdiction", "overdose_category"], how="left"
    )
    val_panel['is_high_risk_jurisdiction'] = \
        val_panel['jurisdiction'].isin(high_risk_jur).astype(float)

    # Fill target lags with last known values
    last_known = train_panel.groupby(['jurisdiction', 'overdose_category'])[
        'rate_per_10000_ed_visits'
    ].last().reset_index()
    last_known.columns = ['jurisdiction', 'overdose_category', 'last_known_value']

    val_panel = pd.merge(
        val_panel, last_known, on=['jurisdiction', 'overdose_category'], how='left'
    )

    for lag_col in target_lag_features:
        if lag_col not in val_panel.columns:
            val_panel[lag_col] = val_panel['last_known_value']
        else:
            val_panel[lag_col] = val_panel[lag_col].fillna(val_panel['last_known_value'])

    val_panel[features] = val_panel[features].fillna(0.0)

    # ===== GENERATE PREDICTIONS =====
    print("Generating predictions...")
    preds = predict(val_panel, models, features)
    val_panel["pred"] = preds

    # ===== HIERARCHICAL RECONCILIATION =====
    print("Applying hierarchical reconciliation...")
    melted = apply_hierarchical_reconciliation(
        val_panel,
        pred_col="pred",
        alpha=RECONCILIATION_ALPHA,
        residual=RESIDUAL_DRUGS
    )

    val_final = pd.merge(
        val_panel[["row_id", "period_id", "jurisdiction", "overdose_category"]],
        melted,
        on=["period_id", "jurisdiction", "overdose_category"],
        how="left"
    )

    val_final["rate_per_10000_ed_visits"] = np.maximum(0.0, val_final["final_rate"])

    # ===== OUTPUT =====
    output_file = get_output_path(f"{EXPERT_NAME}.csv")
    submission = val_final[["row_id", "rate_per_10000_ed_visits"]].sort_values("row_id")

    # Validate
    try:
        validate_submission(submission)
        print("\n✓ Submission validation passed")
    except AssertionError as e:
        print(f"\n✗ Submission validation FAILED: {e}")
        return

    submission.to_csv(output_file, index=False)

    # ===== SAVE METRICS =====
    if scores is not None:
        import json
        from datetime import datetime
        from pathlib import Path

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "oof_validation": {
                "per_category": {cat: float(score) for cat, score in scores.items()},
                "block_averaged_mae": float(block_avg)
            },
            "model_config": {
                "total_features": len(features),
                "base_features": len(base_features),
                "target_lag_features": len(target_lag_features)
            },
            "submission_stats": {
                "total_predictions": len(submission),
                "mean_rate": float(submission["rate_per_10000_ed_visits"].mean()),
                "std_rate": float(submission["rate_per_10000_ed_visits"].std()),
                "min_rate": float(submission["rate_per_10000_ed_visits"].min()),
                "max_rate": float(submission["rate_per_10000_ed_visits"].max()),
                "median_rate": float(submission["rate_per_10000_ed_visits"].median())
            }
        }

        metrics_file = Path(__file__).parent / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved: {metrics_file}")

    print(f"\n{'='*80}")
    print(f"✅ Expert Jasmine CSV saved: {output_file}")
    print(f"{'='*80}")
    print(f"\nPrediction statistics:")
    print(submission["rate_per_10000_ed_visits"].describe())


if __name__ == "__main__":
    main()
