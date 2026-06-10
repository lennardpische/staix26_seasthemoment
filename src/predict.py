"""Functions to implement prediction"""

import xgboost as xgb
import pandas as pd

from src.data_loader import (
    find_data_dir,
    load_train_data,
    load_val_data,
    load_train_pngs,
    load_val_pngs,
    load_sample_submission
)

from src.features import (
    create_all_features, create_validation_features_from_train_df
)

from src.tuning import (
    return_covs_and_target
)

from src.config import (
    all_drugs_params, 
    all_opioids_params, 
    all_stims_params
)


def drop_id_cols(df):
    id_cols = ["period_id", "jurisdiction"]
    features = df.drop(columns = id_cols)
    ids = df[id_cols]

    return features, ids


def run(root, output_path):
    """
    Function to implement the entire data loading, feature engineering and prediction pipeline
    """

    # Get training data and validation covariates
    train_all_drugs, train_all_opioids, train_all_stims = load_train_data(root)
    val_cov = load_val_data(root)

    # Get image data
    train_imgs, train_img_names = load_train_pngs(root)
    val_imgs, val_img_names = load_val_pngs(root)

    # Run feature engineering pipeline on train and val
    all_drugs = create_all_features(
        train_all_drugs,
        train_imgs,
        train_img_names
    )
    all_opioids = create_all_features(
        train_all_opioids,
        train_imgs,
        train_img_names
    )
    all_stims = create_all_features(
        train_all_stims,
        train_imgs,
        train_img_names
    )

    # Compute a reference training set with text intact
    ref = create_all_features(
        train_all_stims,
        train_imgs,
        train_img_names,
        rm_text = False,
        rm_date = False
    )

    # Validation dataframe with rolling statistics
    X_val = create_validation_features_from_train_df(
        train_df = ref,
        val_cov_df = val_cov,
        train_imgs = train_imgs,
        train_img_names = train_img_names,
        val_imgs = val_imgs,
        val_img_names = val_img_names
    )
    
    # Split into features and target
    X_train_drugs_pre, y_train_drugs = return_covs_and_target(all_drugs)
    X_train_opioids_pre, y_train_opioids = return_covs_and_target(all_opioids)
    X_train_stims_pre, y_train_stims = return_covs_and_target(all_stims)

    # Fit and predict
    drugs_model = xgb.XGBRegressor(**all_drugs_params)
    opioids_model = xgb.XGBRegressor(**all_opioids_params)
    stims_model = xgb.XGBRegressor(**all_stims_params)

    # Drop the period id and jurisdiction
    X_train_drugs, _ = drop_id_cols(X_train_drugs_pre)
    X_train_opioids, _ = drop_id_cols(X_train_opioids_pre)
    X_train_stims, _ = drop_id_cols(X_train_stims_pre)

    # Sort columns
    X_train_drugs = X_train_drugs.sort_index(axis = 1)
    X_train_opioids = X_train_opioids.sort_index(axis = 1)
    X_train_stims = X_train_stims.sort_index(axis = 1)

    drugs_model.fit(X_train_drugs, y_train_drugs)
    opioids_model.fit(X_train_opioids, y_train_opioids)
    stims_model.fit(X_train_stims, y_train_stims)

    # Drop ids and predict
    X_val_data, val_ids = drop_id_cols(X_val)

    # Sort columns
    X_val_data = X_val_data.sort_index(axis = 1)

    # Construct df by dropping
    drugs_pred = drugs_model.predict(X_val_data)
    opioids_pred = opioids_model.predict(X_val_data)
    stims_pred = stims_model.predict(X_val_data)

    # Construct df using the original val ids
    drugs_df = val_ids.copy()
    opioids_df = val_ids.copy()
    stims_df = val_ids.copy()

    drugs_df["rate_per_10000_ed_visits"] = drugs_pred
    drugs_df["overdose_category"] = "all_drugs"

    opioids_df["rate_per_10000_ed_visits"] = opioids_pred
    opioids_df["overdose_category"] = "all_opioids"

    stims_df["rate_per_10000_ed_visits"] = stims_pred
    stims_df["overdose_category"] = "all_stimulants"

    # Stack
    pred_df = pd.concat((drugs_df, opioids_df, stims_df), axis = 0)
    pred_df

    # Load sample submission and bind to predictions
    sample = load_sample_submission(root)
    submission = pd.merge(sample, pred_df, on = ["period_id", "jurisdiction", "overdose_category"], how = "inner")
    submission = submission.drop(columns = ["period_id", "jurisdiction", "overdose_category"])
    
    # Write to csv
    submission.to_csv(output_path, index = False)

    return submission