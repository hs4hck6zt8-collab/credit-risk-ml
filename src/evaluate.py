import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path

from fontTools.diff import color
from matplotlib.pyplot import xlabel
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, roc_curve
)
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, FIGURES_DIR,
    TARGET_CLF, TARGET_RATE, TARGET_AMOUNT
)

plt.style.use('seaborn-v0_8-whitegrid')

def load_all_models():
    models = {}
    for name in ["approval_model", "rate_model", "amount_model"]:
        with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    with open(MODELS_DIR / f"preprocessor.pkl", "rb") as f:
        models["preprocessor"] = pickle.load(f)
        return models

def tune_threshold(model, X_val, y_val):
    # Находим порог который максимизирует F1 на валидации
    probas = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds = (probas >= thresh).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    print(f"Best threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    return best_thresh

def evaluate_classifier(model, X_test, y_test, threshold):
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, probas)
    pr_auc = average_precision_score(y_test, probas)
    gini = 2 * roc_auc - 1
    f1 = f1_score(y_test, preds, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, probas)
    ks_stat = float(np.max(tpr - fpr))

    print("\n=== Approval Model (Classification) ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"KS-stat: {ks_stat:.4f}")
    print(f"F1: {f1:.4f}")

    # ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(fpr, tpr, color="steelblue", lw=2,
                 label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Repaid", "Default"],
                yticklabels=["Repaid", "Default"],
                ax=axes[1])
    axes[1].set(title="Confusion Matrix",
                xlabel="Predicted", ylabel="Actual")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_clf_evaluation.png", dpi=150)
    plt.show()

    return {"roc_auc": roc_auc, "pr_auc": pr_auc,
            "gini": gini, "ks_stat": ks_stat, "f1": f1}


def evaluate_regressor(model, X_test, y_test, name, unit=""):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n=== {name} Model (Regression) ===")
    print(f"MAE: {mae:.4f} {unit}")
    print(f"RMSE: {rmse:.4f} {unit}")
    print(f"R^2: {r2:.4f}")

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, preds, alpha=0.2, s=8, color="steelblue")
    lims = [min(y_test.min(), preds.min()),
            max(y_test.max(), preds.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5)
    axes[0].set(title=f"{name}: Actual vs Predicted",
                xlabel=f"Actual {unit}", ylabel=f"Predicted {unit}")
    residuals = y_test.values - preds
    axes[1].hist(residuals, bins=60,
                 color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set(title=f"{name}: Residuals",
                xlabel=f"Residual {unit}", ylabel="Count")

    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"07_{slug}_evaluation.png", dpi=150)
    plt.show()

    return {"mae": mae, "rmse": rmse, "r2": r2}


def shap_analysis(model, X_sample, name):
    print(f"\nComputing SHAP for {name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Beeswarm - глобальная важность признаков
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"{name} - Feature Importance (SHAP)", fontsize=13)
    plt.tight_layout()
    slug = name.lower().replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"08_{slug}_shap.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    return shap_values

