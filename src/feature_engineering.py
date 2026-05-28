import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GRADE_MAP, HIGH_RISK_INTENTS

def add_features(df):
    df = df.copy()

    # Логарифм дохода - чтобы убрать правый скос
    df['log_income'] = np.log1p(df['person_income'])

    # Отношение суммы кредита к доходу
    df['debt_to_income'] = (
        df['loan_amnt'] / df['person_income'].replace(0, np.nan)
    ).clip(upper=10)

    # Доход относительно возраста - прокси карьерного роста
    df['income_per_age'] = (
        df['person_income'] / df['person_age'].replace(0, np.nan)
    )

    # Грейд как порядковое число
    df['grade_num'] = df['loan_grade'].map(GRADE_MAP).fillna(4)

    # Стаж * логарифм дохода - стабильность занятости
    emp = df['person_emp_length'].fillna(0)
    df['emp_income_interaction'] = emp * np.log1p(df['person_income'])

    # Флаг рискованной цели кредита
    df['high_risk_intent'] = (
        df['loan_intent'].isin(HIGH_RISK_INTENTS).astype(int)
    )

    # Длина кредитной истории относительно возраста
    df['cred_hist_per_age'] = (
        df['cb_person_cred_hist_length'] / df['person_age'].replace(0, np.nan)
    ).clip(upper=1.0)

    return df