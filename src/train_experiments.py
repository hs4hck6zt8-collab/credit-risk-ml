import sys
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    mean_absolute_error, r2_score
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TARGET_CLF, TARGET_RATE, TARGET_AMOUNT,
    MODELS_DIR, RANDOM_STATE
)
from src.preprocessing import (
    load_raw_data, remove_outliers,
    split_data, build_preprocessor
)
from src.feature_engineering import add_features
from src.models import ALL_FEATURES, get_feature_matrix
from src.experiment_config import CLASSIFIERS, REGRESSORS

MLFLOW_EXPERIMENT = "credit-risk-scoring"


def run_classifier_experiment(name, model, X_train, y_train, X_val, y_val):
    # Обучаем классификатор и логируем в MLflow
    with mlflow.start_run(run_name=f"clf_{name}"):
        mlflow.set_tag("model_type", "classifier")
        mlflow.set_tag("model_name", name)

        # Обучение
        model.fit(X_train, y_train)

        # Метрики на валидации
        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        y_val_binary = y_val.astype(int)

        roc_auc = roc_auc_score(y_val_binary, proba)
        pr_auc = average_precision_score(y_val_binary, proba)
        f1 = f1_score(y_val_binary, preds, zero_division=0)
        gini = 2 * roc_auc - 1

        # Логируем метрики
        mlflow.log_metrics({
            "val_roc_auc": round(roc_auc, 4),
            "val_pr_auc": round(pr_auc, 4),
            "val_f1": round(f1, 4),
            "val_gini": round(gini, 4)
        })

        # Логируем модель
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f" {name}: AUC={roc_auc:.4f} F1={f1:.4f} Gini={gini:.4f}")
        return roc_auc, model


def run_regressor_experiment(name, model, X_train, y_train, X_val, y_val, target_name):
    # Обучаем регрессор и логируем в MLflow
    with mlflow.start_run(run_name=f"regressor_{name}"):
        mlflow.set_tag("model_type", "regressor")
        mlflow.set_tag("model_name", name)
        mlflow.set_tag("target", target_name)

        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        mlflow.log_metrics({
            "val_mae": round(mae, 4),
            "val_r2": round(r2, 4)
        })

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f" {name}: MAE={mae:.4f} R2={r2:.4f}")
        return mae, model


def main():
    # Данные
    print(f"===Загрузка данных===")
    df = load_raw_data()
    df = remove_outliers(df)
    df = add_features(df)
    train, val, test = split_data(df)

    available = [f for f in ALL_FEATURES if f in train.columns]
    preprocessor = build_preprocessor(available)

    X_train = get_feature_matrix(train, preprocessor, fit=True)
    X_val = get_feature_matrix(val, preprocessor)

    # Сохраняем препроцессор
    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Классификаторы
    print(f"\n===Эксперименты: Классификация===")
    y_train_clf = train[TARGET_CLF]
    y_val_clf = val[TARGET_CLF]

    best_auc, best_clf, best_clf_name = 0, None, ""
    for name, model in CLASSIFIERS.items():
        print(f"\nОбучаю {name}...")
        auc, trained = run_classifier_experiment(
            name, model, X_train, y_train_clf, X_val, y_val_clf
        )
        if auc > best_auc:
            best_auc, best_clf, best_clf_name = auc, trained, name

    print(f"\nЛучший классификатор: {best_clf_name} (AUC={best_auc:.4f})")
    with open(MODELS_DIR / "approval_model.pkl", "wb") as f:
        pickle.dump(best_clf, f)

    # Регрессоры: Ставка
    print(f"\n===Эксперименты: Процентная ставка===")
    mask_train_rate = (train[TARGET_CLF] == 0) & train[TARGET_RATE].notna()
    mask_val_rate = (val[TARGET_CLF] == 0) & val[TARGET_RATE].notna()

    X_train_rate = X_train.loc[train.index[mask_train_rate]]
    X_val_rate = X_val.loc[val.index[mask_val_rate]]
    y_train_rate = train.loc[mask_train_rate, TARGET_RATE]
    y_val_rate = val.loc[mask_val_rate, TARGET_RATE]

    best_mae_rate, best_rate_model, best_rate_name = float("inf"), None, ""
    for name, model in REGRESSORS.items():
        print(f"\nОбучаю {name}...")
        mae, trained = run_regressor_experiment(
            name, model, X_train_rate, y_train_rate,
            X_val_rate, y_val_rate, target_name="rate"
        )
        if mae < best_mae_rate:
            best_mae_rate, best_rate_model, best_rate_name = mae, trained, name

    print(f"\nЛучший регрессор ставки: {best_rate_name} (MAE={best_mae_rate:.4f})")
    with open(MODELS_DIR / "rate_model.pkl", "wb") as f:
        pickle.dump(best_rate_model, f)

    # Регрессоры: Максимальная сумма кредита
    print(f"\n===Эксперименты: Максимальная сумма кредита===")
    mask_train_amt = train[TARGET_CLF] == 0
    mask_val_amt = val[TARGET_CLF] == 0

    X_train_amt = X_train.loc[train.index[mask_train_amt]]
    X_val_amt = X_val.loc[val.index[mask_val_amt]]
    y_train_amt = train.loc[mask_train_amt, TARGET_AMOUNT]
    y_val_amt = val.loc[mask_val_amt, TARGET_AMOUNT]

    best_mae_amt, best_amt_model, best_amt_name = float("inf"), None, ""
    for name, model in REGRESSORS.items():
        print(f"\nОбучаю {name}...")
        mae, trained = run_regressor_experiment(
            name, model, X_train_amt, y_train_amt,
            X_val_amt, y_val_amt, target_name="amount"
        )
        if mae < best_mae_amt:
            best_mae_amt, best_amt_model, best_amt_name = mae, trained, name

    print(f"\nЛучший регрессор суммы: {best_amt_name} (MAE={best_mae_amt:.4f})")
    with open(MODELS_DIR / "amount_model.pkl", "wb") as f:
        pickle.dump(best_amt_model, f)

    print(f"\nВсе эксперименты завершены!")
    print(f"Нужно запустить MLflow UI командой: mlflow ui")
    print(f"Открой браузер: http://localhost:5000")


if __name__ == "__main__":
    main()

