"""Functions for loading STAI-X 2026 datasets"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image


def find_data_dir():
    """
    Find the data directory containing STAI-X 2026 datasets    
    """
    # Kaggle notebook data directory
    kaggle_dir = Path("/kaggle/input/staix-challenge")
    if kaggle_dir.exists():

        return kaggle_dir

    # Local data directory
    else:
        local_dir = Path(__file__).parent.parent / "data"
        return local_dir


def load_train_data(data_dir):
    """
    Return training covariates and target as 3 dataframe, one for each prediction category

    Args:
        data_dir : Path to data folder

    Returns:
        all_drugs_df   : Dataframe of all covariates and all_drugs target 
        all_opioids_df : Dataframe of all covariates and opioids target
        all_stims_df   : Dataframe of all covariates and stimulants target
    """
    train_dir = data_dir / "train"

    # Get covariates and target
    X = pd.read_csv(train_dir / "covariates.csv")

    # Get targets only from categories of interest
    y = pd.read_csv(train_dir / "dose_sys_train.csv")
    y_all_drugs = y.loc[y["overdose_category"] == "all_drugs"]
    y_all_opioids = y.loc[y["overdose_category"] == "all_opioids"]
    y_all_stims = y.loc[y["overdose_category"] == "all_stimulants"]
    
    # Left join using unique ID columns
    join_cols = ["period_id", "jurisdiction"]
    all_drugs_df = pd.merge(X, y_all_drugs, on = join_cols, how = "right").drop(columns = ["overdose_category"])
    all_opioids_df = pd.merge(X, y_all_opioids, on = join_cols, how = "right").drop(columns = ["overdose_category"])
    all_stims_df = pd.merge(X, y_all_stims, on = join_cols, how = "right").drop(columns = ["overdose_category"])

    return all_drugs_df, all_opioids_df, all_stims_df


def load_test_data(data_dir):
    """
    Return test covariates as a dataframe

    Args:
        data_dir : Path to data folder

    Returns:
        cov_df : Dataframe of covariates in test data
    """
    test_dir = data_dir / "val"
    cov_df = pd.read_csv(test_dir / "covariates.csv")

    return cov_df


def load_train_pngs(data_dir):
    """
    Store PNG images and image file names

    Args:
        sub_dir : Path to subdirectory containing image PNG files
    
    Returns:
        img_arrs  : Array of 256x256 gray-scale intensity arrays
        img_names : List of image ID names
    """
    # Specify sub directory
    sub_dir = data_dir / "train" / "images" / "mat_density"
    img_arrs = []

    # Store names + paths and iterate through
    img_names = os.listdir(sub_dir)
    img_paths = [sub_dir / png for png in img_names]
    for png in img_paths:
        
        # Load each image as single channel grayscale array
        img = Image.open(png).convert("L")
        arr  = np.asarray(img)
        img_arrs.append(arr)
    
    img_arrs = np.asarray(img_arrs)
    img_names = [name.replace(".png", "") for name in img_names]

    return img_arrs, img_names


def load_test_pngs(data_dir):
    """
    Store PNG images and image file names

    Args:
        sub_dir : Path to subdirectory containing image PNG files
    
    Returns:
        img_arrs  : Array of 256x256 gray-scale intensity arrays
        img_names : List of image ID names
    """
    # Specify sub directory
    sub_dir = data_dir / "val" / "images" / "mat_density"
    img_arrs = []

    # Store names + paths and iterate through
    img_names = os.listdir(sub_dir)
    img_paths = [sub_dir / png for png in img_names]
    for png in img_paths:
        
        # Load each image as single channel grayscale array
        img = Image.open(png).convert("L")
        arr  = np.asarray(img)
        img_arrs.append(arr)
    
    img_arrs = np.asarray(img_arrs)
    img_names = [name.replace(".png", "") for name in img_names]

    return img_arrs, img_names