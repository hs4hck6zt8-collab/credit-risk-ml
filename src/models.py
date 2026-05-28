from gc import callbacks

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import optuna
from pyexpat import model
from sklearn.metrics import roc_auc_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, RANDOM_STATE, TARGET_CLF, TARGET_RATE, TARGET_AMOUNT,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ENGINEERED_FEATURES
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES

def get_feature_matrix(df, preprocessor, fit=False):
    available = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available]
    if fit:
        result = preprocessor.fit_transform(X)
    else:
        result = preprocessor.transform(X)

    # Возвращаем DataFrame с индексом исходного датафрейма
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result, index=df.index)
    return result


def tune_classifier(X_train, y_train, X_val, y_val, n_trials=50):

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Classifier best AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = {
        **study.best_params,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced"
    }
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    return model

def tune_regressor(X_train, y_train, X_val, y_val, name="regressor", n_trials=50):

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"{name} best MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = {
        **study.best_params,
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "random_state": RANDOM_STATE
    }
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    return model

def save_model(model, name):
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved -> {path}")

def load_model(name):
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)

