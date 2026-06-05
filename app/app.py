import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR
from src.feature_engineering import add_features
from src.models import ALL_FEATURES, get_feature_matrix

# Намстройка страницы
st.set_page_config(
    page_title="Оценка кредитного потенциала",
    page_icon="🏦",
    layout="wide"
)

# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    for name in ["approval_model", "rate_model", "amount_model", "preprocessor"]:
        with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)
    return models

# Заголовок
st.title("🏦 Система оценки кредитного потенциала")
st.caption("ML-pipeline: loan approval * interest rate * maximum amount")
st.divider()

# Боковая панель: форма заявителя
st.sidebar.header("👤 Профиль кандидата")

with st.sidebar.expander("Персональная информация", expanded=True):
    age = st.slider("Возраст", 18, 75, 30)
    income = st.number_input("Годовой доход в $",
                             min_value=1000, max_value=300000,
                             value=60000, step=200)
    home = st.selectbox("Владение жильем",
                        ["Аренда", "Ипотека", "Собственное", "Другое"])
    emp = st.slider("Продолэительность трудовой деятельности (в годах)", 0.0, 40.0, 4.0, 0.2)

with st.sidebar.expander("Информация о кредите", expanded=True):
    intent = st.selectbox("Цель кредита",
                          ["Личное", "Образование", "Лечение",
                           "Бизнес", "Улучшение жилых условий", "Погашение кредита"])
    grade = st.selectbox("Кредитный рейтинг",
                         ["A", "B", "C", "D", "E", "F", "G"])
    pct_inc = st.slider("Процент кредита от дохода", 0.01, 0.6, 0.15, 0.005)
    loan_amnt = int(income * pct_inc)
    st.info(f"Сумма кредита: **${loan_amnt:,}**")

with st.sidebar.expander("Кредитная история", expanded=True):
    cb_default = st.radio("Default on File", ["Нет", "Да"])
    cred_hist = st.slider("Продолжительность кредитной истории (в годах)", 0.0, 40.0, 5.0)

    evaluate = st.sidebar.button("🔍 Оценка", type="primary",
                                 use_container_width=True)

# Предсказание
if not evaluate:
    st.info("👈🏻Заполните анкету заявителя и нажмите **Оценка**")
    st.stop()

try:
    models = load_models()
except FileNotFoundError:
    st.error("⚠Модели не найдены. Для начала нужно запустить 'python src/train_experiments.py'")
    st.stop()

applicant = {
    "person_age": age,
    "person_income": income,
    "person_home_ownership": home,
    "person_emp_length": emp,
    "loan_intent": intent,
    "loan_grade": grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": None,
    "loan_status": 0,
    "loan_percent_income": pct_inc,
    "cb_person_default_on_file": cb_default,
    "cb_person_cred_hist_length": cred_hist
}

with st.spinner("Анализ"):
    df = pd.DataFrame([applicant])
    df = add_features(df)
    X = get_feature_matrix(df, models["preprocessor"])

    proba = models["approval_model"].predict_proba(X)[0, 1]
    approved = proba < 0.35

    if approved:
        rate = float(models["rate_model"].predict(X)[0])
        amount = float(models["amount_model"].predict(X)[0])

# Результат
col1, col2, col3 = st.columns(3)

if approved:
    col1.success("## ✅Одобрено")
else:
    col1.error("## ❌Отклонено")

col2.metric("Вероятность дефолта", f"{proba:.2f}%")

if approved:
    col3.metric("Процентная ставка", f"{max(rate, 0):.2f}%")
    col1.metric("Максимальная сумма кредита", f"${max(amount, 0):,.2f}")

st.divider()

# Прогресс-бар риска
risk_label = "🟢Минимальный риск" if proba < 0.2 else ("🟡Средний риск" if proba < 0.4 else "🔴Высокий риск")
st.write(f"**Уровень риска:** {risk_label}")
st.progress(float(min(proba, 1.0)))

st.divider()

# SHAP объяснение
st.subheader("🧠Почему принято такое решение?")

with st.spinner("Объяснение SHAP вычесления..."):
    explainer = shap.TreeExplainer(models["approval_model"])
    shap_values = explainer(X)

fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], max_display=12, show=False)
plt.title("Вклад характеристик в вероятность дефолта", fontsize=11)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption("🔴 Красный - увеличивает риск дефолта * 🔵 Синий - снижает риск дефолта")

