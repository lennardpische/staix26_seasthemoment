"""Tuned hyperparameters for each XGBoost model"""

fixed_params = {
    "random_state": 111,
    "tree_method": "hist",
    "enable_categorical": True,
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
}

all_drugs_params = {
    **fixed_params,
    'learning_rate': 0.09706755407549365,
    'subsample': 0.5,
    'colsample_bytree': 1.0,
    'max_depth': 3,
    'min_child_weight': 3, 
    'reg_alpha': 1.5937021004273238e-05, 
    'reg_lambda': 0.5854857665993195, 
    'n_estimators': 196
}

all_opioids_params = {
    **fixed_params,
    'learning_rate': 0.021443156797638366, 
    'subsample': 0.8, 
    'colsample_bytree': 1.0, 
    'max_depth': 3, 
    'min_child_weight': 7, 
    'reg_alpha': 0.0031210149531669803, 
    'reg_lambda': 0.010209440106132521, 
    'n_estimators': 379
}

all_stims_params = {
    **fixed_params,
    'learning_rate': 0.009383187606263175, 
    'subsample': 0.7, 
    'colsample_bytree': 1.0, 
    'max_depth': 3, 
    'min_child_weight': 1, 
    'reg_alpha': 6.703287763962173e-05, 
    'reg_lambda': 1.9874822388876459, 
    'n_estimators': 1215
}