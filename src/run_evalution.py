import sys
import pickle
import numpy as np
from pathlib import Path

from matplotlib.style.core import available

sys.path.insert(0, str((Path(__file__).parent.parent)))

from src.preprocessing import load_raw_data, remove_outliers, split_data, build_preprocessor
from src.feature_engineering import add_features
from src.models import ALL_FEATURES, get_feature_matrix
from src.evaluate import (
    load_all_models, tune_threshold,
    evaluate_classifier, evaluate_regressor, shap_analysis
)
from config import TARGET_CLF, TARGET_RATE, TARGET_AMOUNT


def main():
    # Данные
    df = load_raw_data()
    df = remove_outliers(df)
    df = add_features(df)
    train, val, test = split_data(df)

    # Препроцессинг
    available = [f for f in ALL_FEATURES if f in train.columns]
    preprocessor = load_all_models()["preprocessor"]

    X_val = get_feature_matrix(val, preprocessor)
    X_test = get_feature_matrix(test, preprocessor)

    models = load_all_models()
    clf = models["approval_model"]
    rate = models["rate_model"]
    amount = models["amount_model"]

    # Порог га валидации
    threshold = tune_threshold(clf, X_val, val[TARGET_CLF])

    # Классификатор на тесте
    evaluate_classifier(clf, X_test, test[TARGET_CLF], threshold)

    # Регрессоры на тесте (только для одобренных)
    mask_test = test[TARGET_CLF] == 0
    mask_idx = test.index[mask_test]
    X_test_nd = X_test.loc[mask_idx]

    mask_rate = mask_test & test[TARGET_RATE].notna()
    mask_rate_idx = test.index[mask_rate]

    evaluate_regressor(
        rate, X_test.loc[mask_rate_idx],
        test.loc[mask_rate_idx, TARGET_RATE],
        name="Interest Rate", unit="%"
    )
    evaluate_regressor(
        amount, X_test_nd,
        test.loc[mask_idx, TARGET_AMOUNT],
        name="Loan Amount", unit="USD"
    )

    # SHAP (на выборке из 1000 строк - побыстрее)
    sample_idx = np.random.choice(len(X_test), 1000, replace=False)
    X_sample = X_test.iloc[sample_idx]

    shap_analysis(clf, X_sample, "Approval Model")
    shap_analysis(rate, X_test_nd.iloc[:1000], "Interest Rate")
    shap_analysis(amount, X_test_nd[:1000], "Loan Amount")

    print("\nEvaluation complete. Plots saved to reports/figures/")


if __name__ == "__main__":
    main()


