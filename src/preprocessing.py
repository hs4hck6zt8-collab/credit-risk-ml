import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import sys
sys.path.insert(0,str(Path(__file__).parent.parent))
from config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    TARGET_CLF, RANDOM_STATE, TEST_SIZE, VAL_SIZE, DATA_PATH
)

def load_raw_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['loan_status'] = df['loan_status'].astype(int)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].str.upper().str.strip()
    df['loan_grade'] = df['loan_grade'].str.upper().str.strip()
    df['person_home_ownership'] = df['person_home_ownership'].str.upper().str.strip()
    df['loan_intent'] = df['loan_intent'].str.upper().str.strip()
    return df

def remove_outliers(df):
    n0 = len(df)
    df = df[df['person_age'].between(18, 75)].copy()
    df['person_emp_length'] = df['person_emp_length'].clip(upper=50)
    p99_income = df['person_income'].quantile(0.99)
    df['person_income'] = df['person_income'].clip(upper=p99_income)
    print(f"Outlier removal: {n0} - {len(df)} rows (removed {n0 - len(df)})")
    return df.reset_index(drop=True)

def split_data(df):
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_CLF], random_state=RANDOM_STATE
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_frac,
        stratify=train_val[TARGET_CLF], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def build_preprocessor(feature_names):
    num_cols = [c for c in NUMERICAL_FEATURES if c in feature_names]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_names]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ))
    ])

    return ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop"
    )

