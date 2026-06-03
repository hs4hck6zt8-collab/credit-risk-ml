from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RANDOM_STATE

# Классификаторы
CLASSIFIERS = {
    "baseline_logres": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=500,
        scale_pos_weight=4,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0
    ),
    "catboost": CatBoostClassifier(
        iterations=500,
        auto_class_weights="Balanced",
        random_state=RANDOM_STATE,
        verbose=0
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
}

# Регрессоры
REGRESSORS = {
    "baseline_ridge": Ridge(),
    "lightgbm": LGBMRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        verbose=-1
    ),
    "xgboost": XGBRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        verbosity=0
    ),
    "catboost": CatBoostRegressor(
        iterations=500,
        random_state=RANDOM_STATE,
        verbose=0
    ),
    "mlp": MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
}

