"""Functions for feature engineering"""

import pandas as pd
import numpy as np
import re
from scipy.ndimage import binary_erosion
from skimage.morphology import local_maxima
from skimage.measure import label


def create_tabular_features(df):
    """
    Function to construct features from tabular data.
    """
    # Define mapping from state to region
    state_to_region = {
        "AL": "East South Central",
        "AK": "Pacific",
        "AZ": "Mountain",
        "AR": "West South Central",
        "CA": "Pacific",
        "CO": "Mountain",
        "CT": "New England",
        "DE": "South Atlantic",
        "FL": "South Atlantic",
        "GA": "South Atlantic",
        "HI": "Pacific",
        "ID": "Mountain",
        "IL": "East North Central",
        "IN": "East North Central",
        "IA": "West North Central",
        "KS": "West North Central",
        "KY": "East South Central",
        "LA": "West South Central",
        "ME": "New England",
        "MD": "South Atlantic",
        "MA": "New England",
        "MI": "East North Central",
        "MN": "West North Central",
        "MS": "East South Central",
        "MO": "West North Central",
        "MT": "Mountain",
        "NE": "West North Central",
        "NV": "Mountain",
        "NH": "New England",
        "NJ": "Middle Atlantic",
        "NM": "Mountain",
        "NY": "Middle Atlantic",
        "NC": "South Atlantic",
        "ND": "West North Central",
        "OH": "East North Central",
        "OK": "West South Central",
        "OR": "Pacific",
        "PA": "Middle Atlantic",
        "RI": "New England",
        "SC": "South Atlantic",
        "SD": "West North Central",
        "TN": "East South Central",
        "TX": "West South Central",
        "UT": "Mountain",
        "VT": "New England",
        "VA": "South Atlantic",
        "WA": "Pacific",
        "WV": "South Atlantic",
        "WI": "East North Central",
        "WY": "Mountain",
    }

    # Mapping from period_id to date
    id_to_date = {
        "uTjgI1Sv": "2019-01-31",
        "wf016pk5": "2019-02-28",
        "BkTW58Ff": "2019-03-31",
        "shDD7wDP": "2019-04-30",
        "aZFXT65l": "2019-05-31",
        "fizSTkFs": "2019-06-30",
        "wQLd1SNL": "2019-07-31",
        "kmxcVN2e": "2019-08-31",
        "x24Jbzaz": "2019-09-30",
        "FuLb1kk4": "2019-10-31",
        "1Tl9271R": "2019-11-30",
        "Fj7ebbrB": "2019-12-31",
        "h9Re4kM3": "2020-01-31",
        "gqVDbZc7": "2020-02-29",
        "UKjQnuej": "2020-03-31",
        "TedmliP4": "2020-04-30",
        "PJu8Wb2C": "2020-05-31",
        "mZcpe0Ud": "2020-06-30",
        "NOtvYKB9": "2020-07-31",
        "YBlSTfgc": "2020-08-31",
        "rX3aMRGn": "2020-09-30",
        "44RA6kMl": "2020-10-31",
        "88NtGYTF": "2020-11-30",
        "tVb8fHGc": "2020-12-31",
        "aHT3VIho": "2021-01-31",
        "a239r7U4": "2021-02-28",
        "kTaI18at": "2021-03-31",
        "FZxIVFvr": "2021-04-30",
        "27DK2m8F": "2021-05-31",
        "y5ysDDpd": "2021-06-30",
        "NWU8bRHI": "2021-07-31",
        "3JILuYCd": "2021-08-31",
        "56aULHvm": "2021-09-30",
        "KDy1VIvO": "2021-10-31",
        "j1dZWmlF": "2021-11-30",
        "OugqP9RF": "2021-12-31",
        "4VCAqmuO": "2022-01-31",
        "nICRHvl9": "2022-02-28",
        "omhpgEVm": "2022-03-31",
        "WB9kCj4E": "2022-04-30",
        "iIi2mgES": "2022-05-31",
        "CDQGTxV0": "2022-06-30",
        "ePA08XXo": "2022-07-31",
        "MZ0ENeKD": "2022-08-31",
        "4MVfmuye": "2022-09-30",
        "N81HwK1a": "2022-10-31",
        "QpWgWZqu": "2022-11-30",
        "68B5zQl0": "2022-12-31",
        "BhtGJhRU": "2023-01-31",
        "9Dp3l3qq": "2023-02-28",
        "LALpfR23": "2023-03-31",
        "wa7tAVQg": "2023-04-30",
        "eVeAG5UX": "2023-05-31",
        "0Un18Xny": "2023-06-30",
        "9FQthr9A": "2023-07-31",
        "xtjIUpyk": "2023-08-31",
        "yIHgtqjY": "2023-09-30",
        "DpR0556d": "2023-10-31",
        "lSdEh765": "2023-11-30",
        "7cCeqHbf": "2023-12-31",
        "OqDkgaDk": "2024-01-31",
        "k4mmkR0U": "2024-02-29",
        "63zxcdKZ": "2024-03-31",
        "S2Qn2n8u": "2024-04-30",
        "i9aSkhZb": "2024-05-31",
        "UJgFAh3i": "2024-06-30",
        "OIpwoBOI": "2024-07-31",
        "lfTz14iT": "2024-08-31",
        "3CdEQbdr": "2024-09-30",
        "Kk6iVNym": "2024-10-31",
        "tLoy7Zpr": "2024-11-30",
        "S1xSdqr5": "2024-12-31",
        "Hy8SBtar": "2025-01-31",
        "rle4IZEn": "2025-02-28",
        "5Lptd03a": "2025-03-31",
        "jtUOZLP4": "2025-04-30",
        "dp3VfN8B": "2025-05-31",
        "dsZhPyK4": "2025-06-30",
        "aL5zkp6g": "2025-07-31",
        "yFh3wzPe": "2025-08-31",
        "lXSJn8AD": "2025-09-30",
        "kbpS9xmS": "2025-10-31",
        "DmKNJoJt": "2025-11-30",
        "if5b8Sut": "2025-12-31",
        "WNFmh9iQ": "2026-01-31",
        "myO2m6ax": "2026-02-28",
        "fN5pFXQU": "2026-03-31",
        "Ja22UVH5": "2026-04-30",
        "JuylH0n8": "2026-05-31",
        "E8YQntpd": "2026-06-30",
    }
    
    # Store date
    df["date"] = pd.to_datetime(df["period_id"].map(id_to_date))
                     
    # Store trends
    trend_cols = ["gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone", "gtrends_opioid", "gtrends_methamphetamine"]
    trend_arr = df[trend_cols].to_numpy()

    # Construct features
    df["weather_extremity"] = df["temp_avg_f"] * df["precip_in"]
    df["region"] = df["jurisdiction"].map(state_to_region)
    df["region"] = df["region"].astype("category")
    df["gtrends_total"] = np.nansum(trend_arr, axis = 1)
    df["gtrends_max"] = np.nanmax(trend_arr, axis = 1)
    df["gtrends_std"] = np.nanstd(trend_arr, axis = 1)

    return df


def make_nonoverlap_pattern(terms):
    """
    Compiled regex for non-overlapping term counts

    Longest-first sorting prevents shorter terms from being counted inside
    longer terms, ex: "nitazene" inside "isotonitazene" or
    "protonitazene" inside "n-pyrrolidino-protonitazene".
    """
    terms = sorted(
        set(t.lower().strip() for t in terms if t and t.strip()),
        key=len,
        reverse=True
    )

    escaped_terms = [re.escape(t) for t in terms]

    pattern = (
        r"(?<![A-Za-z0-9])(?:"
        + "|".join(escaped_terms)
        + r")(?![A-Za-z0-9])"
    )

    return re.compile(pattern, flags = re.IGNORECASE)


def create_text_features(df):
    """
    Function to construct new features from text column.
    """
    # Store text column
    text = df["state_doh_release"].fillna("").astype(str)

    # Compute simple text features
    df["text_presence"] = (~df["state_doh_release"].isna()).astype(int)
    df["text_presence"] = df["text_presence"].astype("category")
    df["num_char"] = text.apply(lambda text: len(text.strip().replace(" ", "")))
    df["num_numeric"] = text.apply(lambda text: sum([char.isnumeric() for char in text]))

    # Statistic-related terms
    stat_terms = [
        "percent", 
        "%", 
        "$"
    ]

    # Opioid-related terms
    opioid_terms = [
        "opioid",
        "n-pyrrolidino-protonitazene",
        "isotonitazene",
        "buprenorphine",
        "hydrocodone",
        "oxycodone",
        "methadone",
        "fentanyl",
        "nitazene",
    ]

    # Stimulant terms
    stimulant_terms = [
        "stimulant",
        "methamphetamine",
        "adderall",
        "cocaine",
    ]

    # Other drug terms
    other_drug_terms = [
        "benzodiazepine",
        "naltrexone",
        "naloxone",
        "xylazine",
        "narcan",
        "xanax",
    ]

    # Combine into general drug term list
    any_drug_terms = (opioid_terms + stimulant_terms + other_drug_terms)

    # Simple regex pattern for statistics
    stat_pattern = re.compile(
        "|".join(re.escape(term) for term in stat_terms),
        flags=re.IGNORECASE
    )

    # Construct non-overlapping patterns for each set of terms
    opioid_pattern = make_nonoverlap_pattern(opioid_terms)
    stimulant_pattern = make_nonoverlap_pattern(stimulant_terms)
    any_drug_pattern = make_nonoverlap_pattern(any_drug_terms)

    # Construct text features
    df["stat_mentions"] = text.str.count(stat_pattern)
    df["opioid_mentions"] = text.str.count(opioid_pattern)
    df["stimulant_mentions"] = text.str.count(stimulant_pattern)
    df["any_drug_mentions"] = text.str.count(any_drug_pattern)   

    # Drop the original text column
    df = df.drop(columns = ["state_doh_release"])

    return df


def remove_border(
        arr, 
        background_value = 0,
        erosion_pixels = 3,
        fill_value = np.nan
    ):
    """
    Removing background and state border from heatmaps

    Args:
        arr              : 256 x 256 Array of grayscale heatmap values
        background_value : Background img intensity
        erosion_pixels   : State border thickness
    """

    arr = np.asarray(arr)

    # Generate mask for all pixels above the background value
    state_mask = arr > background_value

    # Erode inward to remove border line and remove background + border
    interior_mask = binary_erosion(state_mask, iterations = erosion_pixels)
    cleaned = arr.astype(float).copy()
    cleaned[~interior_mask] = fill_value

    return cleaned


def create_img_features(cov_df, imgs, img_names):
    """
    Construct image features and bind to existing dataframe

    Args:
        cov_df    : Dataframe containing tabular covariates
        imgs      : Array of 256 x 256 PNGs as grayscale intensity arrays
        img_names : List of image ID names
    
    Returns:
        df : Dataframe with image features binded
    """
    # Separate image names into jurisdiction and period_id
    separate_names = [name.split("_") for name in img_names]
    jurisdiction = [split[0] for split in separate_names]
    period_id = [split[1] for split in separate_names]

    # Compute image features
    mean_intensity = [np.nanmean(arr) for arr in imgs]
    median_intensity = [np.nanmedian(arr) for arr in imgs]
    std_intensity = [np.nanstd(arr) for arr in imgs]
    max_intensity = [np.nanmax(arr) for arr in imgs]
    num_hotspots = [label(local_maxima(arr)).max() for arr in imgs]
    frac_above_mean = [
        np.sum((~np.isnan(arr)) & (arr > np.nanmean(arr))) / np.sum(~np.isnan(arr))
        for arr in imgs
    ]

    # Build into dataframe, then bind with the tabular covariates
    img_df = pd.DataFrame({
        "period_id": period_id,
        "jurisdiction": jurisdiction,
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "std_intensity": std_intensity,
        "max_intensity": max_intensity,
        "num_hotspots": num_hotspots,
        "frac_above_mean": frac_above_mean
    })
    df = pd.merge(cov_df, img_df, on = ["period_id", "jurisdiction"], how = "left")

    return df
\

def create_rolling_features(df):
    """
    Construct rolling image and std features for each variable
    """
    # Sort by jurisdiction and date
    df = df.sort_values(["jurisdiction", "date"]).copy()

    # Columns for rolling statistics
    rolling_cols = [
        "unemployment_rate",
        "temp_avg_f",
        "precip_in",
        "gtrends_overdose",
        "gtrends_fentanyl",
        "gtrends_naloxone",
        "gtrends_opioid",
        "gtrends_methamphetamine",
        "gtrends_total",
        "gtrends_max",
        "gtrends_std",
        "any_drug_mentions"
    ]

    # Rolling means by state
    for col in rolling_cols:
        for window in [3, 12]:
            df[f"{col}_rolling_mean_{window}"] = (
                df.groupby("jurisdiction")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods = 1).mean())
            )
    
    # Rolling stds by state
    for col in rolling_cols:
        for window in [3, 12]:
            df[f"{col}_rolling_std_{window}"] = (
                df.groupby("jurisdiction")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods = 1).std())
            )

    return df


def create_all_features(cov_df, imgs, img_names):
    """
    Wrapper for the entire feature engineering pipeline

    Args:
        cov_df    : Dataframe of all covariates
        imgs      : Images as 256 x 256 grayscale intensity arrays (unfiltered)
        img_names : Image ID names
    """
    # Create tabular and text features
    df = create_tabular_features(cov_df)
    df = create_text_features(df)

    # Remove image background and add image features
    cleaned_imgs = [remove_border(img) for img in imgs]
    df = create_img_features(df, cleaned_imgs, img_names)

    # Create rolling features
    df = create_rolling_features(df)

    return df


def create_rolling_features_for_validation(
    train_history_df,
    val_df,
    rolling_cols = None,
    windows = (3, 12),
    group_col = "jurisdiction",
    date_col = "date",
):
    """
    Compute rolling features for validation rows using prior training history

    Args:
        train_history_df : Feature-engineered training dataframe with past rows
        val_df           : Feature-engineered validation dataframe without rolling features
        rolling_cols     : Columns to compute rolling stats for
        windows          : Rolling window sizes
        group_col        : Usually "jurisdiction"
        date_col         : Usually "date"

    Returns:
        val_with_rolling : Validation dataframe with rolling mean/std features.
    """

    if rolling_cols is None:
        rolling_cols = [
            "unemployment_rate",
            "temp_avg_f",
            "precip_in",
            "gtrends_overdose",
            "gtrends_fentanyl",
            "gtrends_naloxone",
            "gtrends_opioid",
            "gtrends_methamphetamine",
            "gtrends_total",
            "gtrends_max",
            "gtrends_std",
            "any_drug_mentions",
        ]

    train_history_df = train_history_df.copy()
    val_df = val_df.copy()

    # Mark source so we can recover validation rows later
    train_history_df["_is_val"] = 0
    val_df["_is_val"] = 1

    # Combine train history + validation
    combined = pd.concat(
        [train_history_df, val_df],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    # Sort chronologically within each jurisdiction
    combined = combined.sort_values([group_col, date_col]).copy()

    # Compute rolling means and stds using only previous rows
    for col in rolling_cols:
        if col not in combined.columns:
            continue

        for window in windows:
            combined[f"{col}_rolling_mean_{window}"] = (
                combined.groupby(group_col)[col]
                .transform(
                    lambda s: s.shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
            )

            combined[f"{col}_rolling_std_{window}"] = (
                combined.groupby(group_col)[col]
                .transform(
                    lambda s: s.shift(1)
                    .rolling(window=window, min_periods=1)
                    .std()
                )
            )

    # Return only validation rows
    val_with_rolling = combined[combined["_is_val"] == 1].copy()

    # Clean helper column
    val_with_rolling = val_with_rolling.drop(columns=["_is_val"])

    return val_with_rolling


def create_validation_features(
    train_cov_df,
    val_cov_df,
    train_imgs,
    train_img_names,
    val_imgs,
    val_img_names,
):
    """
    Create validation features using training history for rolling statistics.

    This should be used instead of create_all_features(...) for validation,
    because validation alone does not contain enough past rows to compute
    useful rolling statistics.

    Args:
        train_cov_df     : Training covariates dataframe.
        val_cov_df       : Validation covariates dataframe.
        train_imgs       : Training images as grayscale arrays.
        train_img_names  : Training image names.
        val_imgs         : Validation images as grayscale arrays.
        val_img_names    : Validation image names.

    Returns:
        val_features : Validation dataframe with tabular, text, image,
                       and rolling features.
    """

    # Build base training features, excluding rolling features
    train_base = create_tabular_features(train_cov_df)
    train_base = create_text_features(train_base)

    cleaned_train_imgs = [remove_border(img) for img in train_imgs]
    train_base = create_img_features(
        train_base,
        cleaned_train_imgs,
        train_img_names,
    )

    # Build base validation features, excluding rolling features
    val_base = create_tabular_features(val_cov_df)
    val_base = create_text_features(val_base)

    cleaned_val_imgs = [remove_border(img) for img in val_imgs]
    val_base = create_img_features(
        val_base,
        cleaned_val_imgs,
        val_img_names,
    )

    # Compute validation rolling features using training history
    val_features = create_rolling_features_for_validation(
        train_history_df=train_base,
        val_df=val_base,
    )

    return val_features