"""
Feature engineering for STAI-X 2026 Competition
"""
import numpy as np
import pandas as pd
from config import REGION_ENCODING, GTRENDS_COLS
from utils import get_region, parse_mat_density_map, calculate_hal_score


def build_base_features(df):
    """
    Build base covariate features (no target lags)

    Args:
        df: DataFrame with covariates

    Returns:
        DataFrame with base features added
    """
    # Regional features
    df['region'] = df['jurisdiction'].apply(get_region)
    df['region_encoded'] = df['region'].map(REGION_ENCODING)

    # Economic features
    df["economic_stress_index"] = df["unemployment_rate"] * df["labor_force"]
    df["labor_force_participation_proxy"] = df.groupby("jurisdiction")["labor_force"].transform(
        lambda x: x / (x.max() + 1e-6)
    )
    df["unemployment_momentum"] = df.groupby("jurisdiction")["unemployment_rate"].diff().fillna(0.0)
    df["unemployment_acceleration"] = df.groupby("jurisdiction")["unemployment_momentum"].diff().fillna(0.0)

    # Environmental features
    df["extreme_weather_indicator"] = (
        (df["precip_in"] > df["precip_in"].median()) & (df["temp_avg_f"] < 32.0)
    ).astype(float)
    df["seasonal_temperature_deviation"] = df.groupby("jurisdiction")["temp_avg_f"].transform(
        lambda x: x - x.median()
    ).fillna(0.0)

    return df


def build_gtrends_features(df):
    """
    Build Google Trends features with lags and rolling stats

    Args:
        df: DataFrame with gtrends columns

    Returns:
        DataFrame with gtrends features added
    """
    df["supply_volatility_index"] = df[["gtrends_fentanyl", "gtrends_methamphetamine"]].mean(axis=1)
    df["harm_reduction_ratio"] = df["gtrends_naloxone"] / (df["gtrends_overdose"] + 1e-6)

    for col in GTRENDS_COLS:
        # Momentum (first derivative)
        df[f"search_momentum_{col}"] = df.groupby("jurisdiction")[col].diff().fillna(0.0)

        # Lags
        df[f"{col}_lag1"] = df.groupby("jurisdiction")[col].shift(1).fillna(df[col].median())
        df[f"{col}_lag2"] = df.groupby("jurisdiction")[col].shift(2).fillna(df[col].median())

        # Rolling statistics
        df[f"{col}_rolling_mean_3"] = df.groupby("jurisdiction")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df[f"{col}_rolling_std_3"] = df.groupby("jurisdiction")[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        ).fillna(0.0)

    return df


def build_text_features(df):
    """
    Build text-based features from state DOH releases

    Args:
        df: DataFrame with state_doh_release column

    Returns:
        DataFrame with text features added
    """
    df["health_alert_level_score"] = df["state_doh_release"].apply(calculate_hal_score)
    df["release_issued_flag"] = df["state_doh_release"].apply(
        lambda x: 1.0 if pd.notna(x) and str(x).strip() != "" else 0.0
    )
    df["release_word_count"] = df["state_doh_release"].apply(
        lambda x: float(len(str(x).split())) if pd.notna(x) else 0.0
    )

    return df


def build_image_features(df, img_dir):
    """
    Build features from MAT density images

    Args:
        df: DataFrame with jurisdiction and period_id
        img_dir: Path to image directory

    Returns:
        DataFrame with image features added
    """
    print(f"Parsing images from {img_dir}...")
    img_features = df.apply(
        lambda row: parse_mat_density_map(img_dir, row["jurisdiction"], row["period_id"]),
        axis=1
    )
    img_df = pd.DataFrame(img_features.tolist())
    df = pd.concat([df, img_df], axis=1)

    return df


def build_interaction_features(df):
    """
    Build feature interactions

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with interaction features added
    """
    df["stress_x_harm_reduction"] = df["economic_stress_index"] * df["harm_reduction_ratio"]
    df["weather_x_supply"] = df["extreme_weather_indicator"] * df["supply_volatility_index"]
    df["mat_accessibility"] = df["mat_density_mean"] / (df["mat_treatment_desert_ratio"] + 1e-6)
    df["mat_x_economic"] = df["mat_density_mean"] * df["economic_stress_index"]
    df["overdose_x_naloxone"] = df["gtrends_overdose"] * df["gtrends_naloxone"]
    df["fentanyl_x_mat"] = df["gtrends_fentanyl"] * df["mat_density_mean"]

    return df


def build_target_lag_features(df, target_df):
    """
    Build target lag features (lag model)

    Args:
        df: DataFrame with covariates
        target_df: DataFrame with rate_per_10000_ed_visits

    Returns:
        DataFrame with target lag features added
    """
    print("Building TARGET lag features...")

    df_with_target = pd.merge(
        df,
        target_df[["period_id", "jurisdiction", "overdose_category", "rate_per_10000_ed_visits"]],
        on=["period_id", "jurisdiction"],
        how="left"
    )

    grouped = df_with_target.groupby(['jurisdiction', 'overdose_category'])['rate_per_10000_ed_visits']

    # Simple lags
    for lag in range(1, 7):
        df_with_target[f"target_lag_{lag}"] = grouped.shift(lag)

    # EWMA
    for alpha in [0.3, 0.5, 0.7]:
        df_with_target[f'target_ewma_{alpha}'] = grouped.transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )

    # Rolling statistics
    for window in [3, 6]:
        df_with_target[f'target_rolling_mean_{window}'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_with_target[f'target_rolling_std_{window}'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0.0)
        df_with_target[f'target_rolling_max_{window}'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        df_with_target[f'target_rolling_min_{window}'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )

    # Trend features
    df_with_target['target_trend'] = grouped.diff()
    df_with_target['target_acceleration'] = df_with_target.groupby(
        ['jurisdiction', 'overdose_category']
    )['target_trend'].diff()
    df_with_target['is_increasing'] = (df_with_target['target_trend'] > 0).astype(float)

    # Volatility features
    df_with_target['target_volatility_3'] = grouped.transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )
    df_with_target['target_cv_3'] = df_with_target['target_volatility_3'] / (
        df_with_target['target_rolling_mean_3'] + 1e-6
    )

    # Ratio to historical mean
    df_with_target['target_ratio_to_mean'] = grouped.transform(
        lambda x: x / (x.mean() + 1e-6)
    )

    return df_with_target


def run_feature_engineering(cov_df, img_dir, target_df=None, build_target_lags=True):
    """
    Main feature engineering pipeline

    Args:
        cov_df: Covariates dataframe
        img_dir: Path to image directory
        target_df: Target dataframe (optional, for target lags)
        build_target_lags: Whether to build target lag features

    Returns:
        DataFrame with all features
    """
    df = cov_df.copy()
    df = df.sort_values(by=["jurisdiction", "period_id"]).reset_index(drop=True)

    # Build feature streams
    df = build_base_features(df)
    df = build_gtrends_features(df)
    df = build_text_features(df)
    df = build_image_features(df, img_dir)
    df = build_interaction_features(df)

    # Optionally build target lag features
    if build_target_lags and target_df is not None:
        df = build_target_lag_features(df, target_df)

    # Impute missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def get_feature_names(include_target_lags=False):
    """
    Get list of feature column names

    Args:
        include_target_lags: Whether to include target lag features

    Returns:
        list: Feature column names
    """
    base_features = [
        "region_encoded",
        "economic_stress_index", "labor_force_participation_proxy",
        "unemployment_momentum", "unemployment_acceleration",
        "extreme_weather_indicator", "seasonal_temperature_deviation",
        "supply_volatility_index", "harm_reduction_ratio",
        "health_alert_level_score", "release_issued_flag", "release_word_count",
        "mat_density_mean", "mat_density_variance", "mat_treatment_desert_ratio",
        "mat_density_max", "mat_density_min", "mat_density_range",
        "stress_x_harm_reduction", "weather_x_supply", "mat_accessibility",
        "mat_x_economic", "overdose_x_naloxone", "fentanyl_x_mat",
        "jur_mean", "jur_std", "jur_median", "jur_min", "jur_max", "jur_q25", "jur_q75",
        "is_high_risk_jurisdiction"
    ]

    gtrends_features = []
    for col in GTRENDS_COLS:
        gtrends_features.extend([
            f"search_momentum_{col}",
            f"{col}_lag1", f"{col}_lag2",
            f"{col}_rolling_mean_3", f"{col}_rolling_std_3"
        ])

    features = base_features + gtrends_features

    if include_target_lags:
        # These will be dynamically detected from dataframe columns
        # that start with 'target_'
        pass

    return features
