"""
Configuration and constants for STAI-X 2026 Competition
"""
# Note: Data paths are now auto-detected by data_loader.py
# No need to manually configure DATA_DIR!

# ===== OUTPUT =====
EXPERT_NAME = "expert_jasmine"

# ===== DOMAIN KNOWLEDGE =====
HIGH_RISK_KEYWORDS = [
    "fentanyl", "overdose", "naloxone", "opioid", "stimulant",
    "methamphetamine", "spike", "alert", "fatal", "crisis", "surge"
]

REGIONS = {
    'northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'south': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
    'west': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA'],
    'dc': ['DC']
}

REGION_ENCODING = {
    'northeast': 0,
    'midwest': 1,
    'south': 2,
    'west': 3,
    'dc': 4,
    'other': 5
}

OVERDOSE_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]

GTRENDS_COLS = [
    "gtrends_overdose",
    "gtrends_fentanyl",
    "gtrends_naloxone",
    "gtrends_opioid",
    "gtrends_methamphetamine"
]

# ===== MODEL HYPERPARAMETERS =====
LGBM_PARAMS = {
    'objective': 'mae',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.03,
    'n_estimators': 800,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42
}

# ===== VALIDATION =====
N_SPLITS = 3  # Number of folds for time-series CV

# ===== ENSEMBLE =====
CONFIDENCE_WEIGHTS = {
    'lag_availability': 0.4,
    'history_depth': 0.3,
    'consistency': 0.3
}

MIN_AGGRESSIVE_WEIGHT = 0.15
MAX_AGGRESSIVE_WEIGHT = 0.85

# ===== RECONCILIATION =====
RECONCILIATION_ALPHA = 0.6  # Weight for top-down prediction
RESIDUAL_DRUGS = 0.5  # Small residual for "other drugs" in bottom-up
