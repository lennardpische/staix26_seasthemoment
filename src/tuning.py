"""Functions for XGBoost and LightGBM hyperparameter tuning"""

import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
import scipy.stats as stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from lightgbm import early_stopping


def return_covs_and_target(df):
    """
    Return covariates and prediction target
    """
    X_train = df.drop(columns = "rate_per_10000_ed_visits")
    y_train = df["rate_per_10000_ed_visits"]

    return X_train, y_train


def make_xgb_objective(X_train, y_train, num_folds):
    """
    Constructs an objective function using specified data to pass to Optuna study
    """
    # Convert to categorical just in case
    cat_cols = ["period_id", "jurisdiction", "region", "text_presence"]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")

    def xgb_objective(trial):
        """
        XGBoost regressor objective function for Optuna. Native Optuna does not support correct early stopping with k-fold CV.
        It requires that we define a fixed validation set used to eavluate early stopping for ALL folds. So, this is a
        workaround that allows each fold to use its corresponding validation set for early stopping.

        Args:
            trial : Parameter for optuna optimize function

        Returns:
            score : Mean MAE across all k folds
        """
        # Define the hyperparameter search space
        params = {
            # Fixed
            "n_estimators": 3000,
            "early_stopping_rounds": 50,
            "random_state": 111,
            "tree_method": "hist",
            "device": "cuda",
            "enable_categorical": True, # native handling

            # Tunable
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log = True),
            "subsample": trial.suggest_float("subsample", 0.5, 1, step = 0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step = 0.1),
            "max_depth": trial.suggest_int("max_depth", 2, 10, step = 1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7, step = 2),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 30, log = True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 30, log = True),

            # Eval (use MAE)
            "objective": "reg:absoluteerror",
            "eval_metric": "mae"
        }

        # Find groups
        groups = X_train["period_id"]
        X_train_new = X_train.drop(columns = "period_id")

        # Initialize k-fold CV splitter
        cv = GroupKFold(
            n_splits = num_folds,
            shuffle = True,
            random_state = 111,
        )

        # Store MAE across folds
        scores = []

        # Manual loop for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_new, y_train, groups = groups)):

            # Load the train/val sets for the current fold
            X_train_curr = X_train_new.iloc[train_idx]
            y_train_curr = y_train.iloc[train_idx]
            X_val_curr = X_train_new.iloc[val_idx]
            y_val_curr = y_train.iloc[val_idx]

            # Initialize XGB classifier model with params
            model = xgb.XGBRegressor(**params)

            # Train the model and predict on current validation set
            model.fit(X_train_curr, y_train_curr, eval_set = [(X_val_curr, y_val_curr)], verbose = False)
            y_pred = model.predict(X_val_curr)

            # Calculate MAE for current fold
            fold_score = mean_absolute_error(y_val_curr, y_pred)
            scores.append(fold_score)

            # Pruner
            intermediate_score = np.mean(scores)
            trial.report(intermediate_score, step = fold_idx)

            if trial.should_prune():
              raise optuna.TrialPruned()

        # Calculate mean MAE across folds
        score = np.mean(scores)

        return score

    return xgb_objective


def make_lgb_objective(X_train, y_train, num_folds):
    """
    Constructs an objective function using specified data to pass to Optuna study
    """
    # Convert to categorical just in case
    cat_cols = ["period_id", "jurisdiction", "region", "text_presence"]

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")

    def lgb_objective(trial):
        """
        LightGBM regressor objective function for Optuna. Native Optuna does not support correct early stopping with k-fold CV.
        It requires that we define a fixed validation set used to eavluate early stopping for ALL folds. So, this is a
        workaround that allows each fold to use its corresponding validation set for early stopping.

        Args:
            trial : Parameter for optuna optimize function

        Returns:
            score : Mean MAE across all k folds
        """
        # Define the hyperparameter search space
        params = {
            # Fixed
            "n_estimators": 2000,
            "random_state": 111,
            "bagging_freq": 1,
            "device_type": "gpu",
            "verbose": -1,

            # Tunable
            "num_leaves": trial.suggest_int("num_leaves", 8, 63, log = True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log = True),
            "subsample": trial.suggest_float("subsample", 0.5, 1, step = 0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step = 0.1),
            "max_depth": trial.suggest_int("max_depth", 6, 10, step = 1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7, step = 2),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10, log = True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10, log = True),

            # Eval (use MAE)
            "objective": "regression_l1",
            "metric": "mae"
        }

        # Find groups
        groups = X_train["period_id"]
        X_train_new = X_train.drop(columns = "period_id")

        # Initialize k-fold CV splitter
        cv = GroupKFold(
            n_splits = num_folds,
            shuffle = True,
            random_state = 111,
        )

        # Store MAE across folds
        scores = []

        # Manual loop for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_new, y_train, groups = groups)):

            # Load the train/val sets for the current fold
            X_train_curr = X_train_new.iloc[train_idx]
            y_train_curr = y_train.iloc[train_idx]
            X_val_curr = X_train_new.iloc[val_idx]
            y_val_curr = y_train.iloc[val_idx]

            # Initialize XGB classifier model with params
            model = lgb.LGBMRegressor(**params)

            # Train the model and predict on current val set w/ early stopping
            model.fit(
                X_train_curr,
                y_train_curr,
                eval_set = [(X_val_curr, y_val_curr)],
                callbacks = [early_stopping(stopping_rounds = 50, verbose = False)] # early stopping
                )
            y_pred = model.predict(X_val_curr)

            # Calculate MAE for current fold
            fold_score = mean_absolute_error(y_val_curr, y_pred)
            scores.append(fold_score)

            # Pruner
            intermediate_score = np.mean(scores)
            trial.report(intermediate_score, step = fold_idx)

            if trial.should_prune():
              raise optuna.TrialPruned()

        # Calculate mean MAE across folds
        score = np.mean(scores)

        return score

    return lgb_objective


def find_best_regressor(X_train, y_train, num_folds):
    """
    Function to find and report the best regressor out of LGB, XGB
    for a given training set

    Args:
        X_train : Features
        y_train : Prediction target
        num_folds : # folds for CV during hyperparameter tuning

    Returns:
        best_model : name of best model type
        best_mean_MAE : best score (mean MAE across k folds)
        best_params : best parameters winning model
    """

    # Create median pruner
    median_pruner = optuna.pruners.MedianPruner(
        n_startup_trials = 10, n_warmup_steps = 2, interval_steps = 1
    )

    # Initialize studies
    xgb_study = optuna.create_study(direction = "minimize", pruner = median_pruner)
    lgb_study = optuna.create_study(direction = "minimize", pruner = median_pruner)

    # Initialize objectives
    xgb_objective = make_xgb_objective(X_train, y_train, num_folds)
    lgb_objective = make_lgb_objective(X_train, y_train, num_folds)

    # Run studies
    lgb_study.optimize(lgb_objective, n_trials = 45, n_jobs = 1)
    xgb_study.optimize(xgb_objective, n_trials = 45, n_jobs = 1)

    # Store completed studies
    model_names = ["XGBoost", "LightGBM"]
    completed_studies = [xgb_study, lgb_study]
    mean_scores = [study.best_value for study in completed_studies]

    # Find best MAE and return those params
    best_idx = np.argmin(mean_scores)
    best_model = model_names[best_idx]
    best_mean_MAE = mean_scores[best_idx]
    best_params = completed_studies[best_idx].best_params

    return best_model, best_mean_MAE, best_params