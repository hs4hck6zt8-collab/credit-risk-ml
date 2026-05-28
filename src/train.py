import sys
import pickle
from pathlib import Path

from matplotlib.style.core import available

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_raw_data, remove_outliers, split_data, build_preprocessor
from src.feature_engineering import add_features
from src.models import ALL_FEATURES, get_feature_matrix, tune_classifier, tune_regressor, save_model
from config import TARGET_CLF, TARGET_RATE, TARGET_AMOUNT, MODELS_DIR

def main():
    # 1. Данные
    print("=== Загрузка данных ===")
    df = load_raw_data()
    df = remove_outliers(df)
    df = add_features(df)
    train, val, test = split_data(df)

    # 2. Препроцессинг
    print("\n=== Препроцессинг ===")
    available = [f for f in ALL_FEATURES if f in train.columns]
    preprocessor = build_preprocessor(available)

    X_train = get_feature_matrix(train, preprocessor, fit=True)
    X_val = get_feature_matrix(val, preprocessor)
    X_test = get_feature_matrix(test, preprocessor)

    with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # 3. Классификатор
    print("\n=== Модель 1: Одобрение ===")
    clf = tune_classifier(
        X_train, train[TARGET_CLF],
        X_val, val[TARGET_CLF],
        n_trials=50
    )
    save_model(clf, "approval_model")

    # 4. Регрессор ставки
    # Только одобренные кредиты - без дефолтов
    print("\n=== Модель 2: Процентная ставка ===")
    mask_train = (train[TARGET_CLF] == 0) & train[TARGET_RATE].notna()
    mask_val = (val[TARGET_CLF] == 0) & val[TARGET_RATE].notna()

    mask_train_idx = train.index[mask_train]
    mask_val_idx = val.index[mask_val]

    X_train_rate = X_train.loc[mask_train_idx]
    X_val_rate = X_val.loc[mask_val_idx]

    rate_model = tune_regressor(
        X_train_rate, train.loc[mask_train_idx, TARGET_RATE],
        X_val_rate, val.loc[mask_val_idx, TARGET_RATE],
        name = "rate_model", n_trials=50
    )
    save_model(rate_model, "rate_model")

    # 5. Регрессор суммы
    print("\n=== Модель 3: Регрессор суммы ===")
    mask_train_amt_idx = train.index[train[TARGET_CLF] == 0]
    mask_val_amt_idx = val.index[val[TARGET_CLF] == 0]

    amount_model = tune_regressor(
        X_train.loc[mask_train_amt_idx], train.loc[mask_train_amt_idx, TARGET_AMOUNT],
        X_val.loc[mask_val_amt_idx], val.loc[mask_val_amt_idx, TARGET_AMOUNT],
        name = "amount_model", n_trials=50
    )
    save_model(amount_model, "amount_model")

    print("\nВсе модели обучены и сохранены в models/")

if __name__ == "__main__":
    main()

