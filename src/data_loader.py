"""
Data loading utilities for STAI-X 2026 - Expert Jasmine
Auto-detects Kaggle vs local environment
"""
from pathlib import Path
import pandas as pd


def _find_data_root():
    """
    Auto-detect data location:
    1. Check Kaggle input directory
    2. Check local project directory
    3. Check parent directory
    4. Raise error if not found
    """
    # Check Kaggle
    kaggle = Path("/kaggle/input/staix-challenge")
    if kaggle.exists():
        print(f"✓ Found data at Kaggle path: {kaggle}")
        return kaggle

    # Check if running from kagglehub cache (competition download)
    kagglehub = Path("/root/.cache/kagglehub/competitions/stai-x-challenge-2026")
    if kagglehub.exists():
        print(f"✓ Found data at kagglehub cache: {kagglehub}")
        return kagglehub

    # Check current directory
    local = Path.cwd()
    if (local / "train" / "dose_sys_train.csv").exists():
        print(f"✓ Found data in current directory: {local}")
        return local

    # Check project root/data subdirectory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    if (data_dir / "train" / "dose_sys_train.csv").exists():
        print(f"✓ Found data in data subdirectory: {data_dir}")
        return data_dir

    # Check project root
    if (project_root / "train" / "dose_sys_train.csv").exists():
        print(f"✓ Found data in project root: {project_root}")
        return project_root

    # Check parent of project root
    parent = project_root.parent
    if (parent / "train" / "dose_sys_train.csv").exists():
        print(f"✓ Found data in parent directory: {parent}")
        return parent

    raise FileNotFoundError(
        "Cannot locate data root. Expected 'train/dose_sys_train.csv'.\n"
        "Please either:\n"
        "  1. Run on Kaggle (data will be at /kaggle/input/staix-challenge)\n"
        "  2. Download data and place in project directory\n"
        "  3. Set DATA_DIR environment variable"
    )


def get_data_paths():
    """
    Get all data file paths

    Returns:
        dict with keys: root, train_target, train_cov, val_cov, sample_sub,
                        train_img_dir, val_img_dir
    """
    root = _find_data_root()

    return {
        'root': root,
        'train_target': root / "train" / "dose_sys_train.csv",
        'train_cov': root / "train" / "covariates.csv",
        'val_cov': root / "val" / "covariates.csv",
        'sample_sub': root / "sample_submission.csv",
        'train_img_dir': root / "train" / "images" / "mat_density",
        'val_img_dir': root / "val" / "images" / "mat_density"
    }


def load_train_data():
    """
    Load training data (targets + covariates)

    Returns:
        tuple: (train_target_df, train_cov_df)
    """
    paths = get_data_paths()

    print("\nLoading training data...")
    train_target = pd.read_csv(paths['train_target'])
    train_cov = pd.read_csv(paths['train_cov'])

    print(f"  Train target: {len(train_target)} rows")
    print(f"  Train covariates: {len(train_cov)} rows")

    return train_target, train_cov


def load_validation_data():
    """
    Load validation data (covariates + submission template)

    Returns:
        tuple: (val_cov_df, sample_sub_df)
    """
    paths = get_data_paths()

    print("\nLoading validation data...")
    val_cov = pd.read_csv(paths['val_cov'])
    sample_sub = pd.read_csv(paths['sample_sub'])

    print(f"  Val covariates: {len(val_cov)} rows")
    print(f"  Sample submission: {len(sample_sub)} rows")

    return val_cov, sample_sub


def get_output_path(filename="expert_jasmine.csv"):
    """
    Get output file path (Kaggle working dir or current dir)

    Args:
        filename: Output filename

    Returns:
        Path object for output file
    """
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / filename
    return Path.cwd() / filename
