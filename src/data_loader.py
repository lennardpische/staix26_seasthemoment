"""Functions for loading STAI-X 2026 datasets"""

import numpy as np
import pandas as pd
from pathlib import Path


def find_data_dir():
    """
    Find the data directory containing STAI-X 2026 datasets    
    """
    kaggle_dir = Path("/kaggle/input/staix-challenge")
    if kaggle_dir.exists():
        return kaggle_dir
    else:
        local_dir = Path(__file__).parent.parent
        return local_dir


def load_train_data(data_dir):
    """
    Return training covariates and target as a dataframe
    """

    return

def load_test_data(data_dir):
    """
    Return test covaraites as a dataframe
    """


def load_pngs():


    return 

