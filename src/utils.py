"""
Utility functions for STAI-X 2026 Competition
"""
import numpy as np
import pandas as pd
from PIL import Image
from config import REGIONS, HIGH_RISK_KEYWORDS


def get_region(jurisdiction):
    """Map jurisdiction to region"""
    for region, states in REGIONS.items():
        if jurisdiction in states:
            return region
    return 'other'


def parse_mat_density_map(img_dir, jurisdiction, period_id):
    """
    Extract features from MAT density heatmap image

    Returns dict with 6 features:
    - mat_density_mean
    - mat_density_variance
    - mat_treatment_desert_ratio
    - mat_density_max
    - mat_density_min
    - mat_density_range
    """
    import os
    img_name = f"{jurisdiction}_{period_id}.png"
    img_path = os.path.join(img_dir, img_name)

    # Default values if image doesn't exist
    if not os.path.exists(img_path):
        return {
            'mat_density_mean': 0.20,
            'mat_density_variance': 0.05,
            'mat_treatment_desert_ratio': 0.40,
            'mat_density_max': 0.30,
            'mat_density_min': 0.10,
            'mat_density_range': 0.20
        }

    with Image.open(img_path).convert('L') as img:
        pixels = np.array(img, dtype=np.float32) / 255.0

    return {
        'mat_density_mean': float(np.mean(pixels)),
        'mat_density_variance': float(np.var(pixels)),
        'mat_treatment_desert_ratio': float(np.mean(pixels < 0.15)),
        'mat_density_max': float(np.max(pixels)),
        'mat_density_min': float(np.min(pixels)),
        'mat_density_range': float(np.max(pixels) - np.min(pixels))
    }


def calculate_hal_score(text):
    """
    Health Alert Level (HAL) score based on keyword counts

    Args:
        text: State DOH release text

    Returns:
        float: Count of high-risk keywords
    """
    if pd.isna(text) or text == "":
        return 0.0
    return float(sum(text.lower().count(word) for word in HIGH_RISK_KEYWORDS))


def calculate_lag_confidence(val_panel, train_panel):
    """
    Calculate confidence score for each validation row

    Confidence is based on:
    1. Lag availability (how many target lags are non-missing)
    2. History depth (how many periods of history exist)
    3. Consistency (low variance = high confidence)

    Args:
        val_panel: Validation dataframe with target lag features
        train_panel: Training dataframe

    Returns:
        np.array: Confidence score per row (0-1 scale)
    """
    confidence_scores = []

    for idx, row in val_panel.iterrows():
        jurisdiction = row['jurisdiction']
        category = row['overdose_category']

        # Check how many lags we actually have data for
        lag_cols = [c for c in val_panel.columns if c.startswith('target_lag_')]
        available_lags = sum([1 for col in lag_cols if not pd.isna(row.get(col, np.nan))])

        # Get history length for this jurisdiction-category
        hist = train_panel[
            (train_panel['jurisdiction'] == jurisdiction) &
            (train_panel['overdose_category'] == category)
        ]

        if len(hist) == 0:
            confidence_scores.append(0.0)
            continue

        history_length = len(hist)

        # Calculate confidence components
        lag_availability = available_lags / len(lag_cols) if len(lag_cols) > 0 else 0.0
        history_depth = min(history_length / 10.0, 1.0)  # Saturate at 10 periods

        # Check consistency (lower variance in recent history = higher confidence)
        recent_vals = hist['rate_per_10000_ed_visits'].tail(6).values
        if len(recent_vals) > 1:
            cv = np.std(recent_vals) / (np.mean(recent_vals) + 1e-6)
            consistency = 1.0 / (1.0 + cv)  # High variance = low confidence
        else:
            consistency = 0.5

        # Combined confidence (weighted average)
        confidence = 0.4 * lag_availability + 0.3 * history_depth + 0.3 * consistency
        confidence_scores.append(confidence)

    return np.array(confidence_scores)


def validate_submission(submission_df, expected_rows=918):
    """
    Validate submission file meets competition requirements

    Args:
        submission_df: DataFrame with row_id and rate_per_10000_ed_visits
        expected_rows: Expected number of rows (default 918)

    Returns:
        bool: True if valid

    Raises:
        AssertionError: If validation fails
    """
    assert len(submission_df) == expected_rows, \
        f"Expected {expected_rows} rows, got {len(submission_df)}"

    assert list(submission_df.columns) == ["row_id", "rate_per_10000_ed_visits"], \
        f"Incorrect columns: {list(submission_df.columns)}"

    assert submission_df["rate_per_10000_ed_visits"].isna().sum() == 0, \
        "Found NaN values"

    assert (submission_df["rate_per_10000_ed_visits"] >= 0).all(), \
        "Found negative values"

    return True


def print_oof_results(oof_maes, category_names):
    """
    Pretty print OOF validation results

    Args:
        oof_maes: Dict of category -> list of MAEs per fold
        category_names: List of category names

    Returns:
        tuple: (scores_dict, block_avg_mae)
    """
    print("\n" + "="*80)
    print("OUT-OF-FOLD VALIDATION RESULTS")
    print("="*80)

    # Calculate scores
    scores = {cat: np.mean(oof_maes[cat]) for cat in category_names}
    block_avg = np.mean(list(scores.values()))

    # Print results
    print(f"\n{'Category':<20} {'MAE':<12}")
    print("-" * 35)
    for cat in category_names:
        print(f"{cat:<20} {scores[cat]:<12.4f}")
    print("-" * 35)
    print(f"{'Block Average':<20} {block_avg:<12.4f}")
    print("="*80)

    return scores, block_avg
