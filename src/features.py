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

    return df


