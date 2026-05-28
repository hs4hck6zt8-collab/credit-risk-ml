from pathlib import Path

# Директории
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

DATA_PATH = DATA_DIR / "credit_risk_dataset.csv"

# Целевые переменные
TARGET_CLF = "loan_status"      # 0 - вернет, 1 - не вернет
TARGET_RATE = "loan_int_rate"   # процентная ставка
TARGET_AMOUNT = "loan_amnt"     # сумма кредита

# Признаки
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

ENGINEERED_FEATURES = [
    "log_income",
    "debt_to_income",
    "income_per_age",
    "grade_num",
    "emp_income_interaction",
    "high_risk_intent",
    "cred_hist_per_age"
]

# Const
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
HIGH_RISK_INTENTS = {"DEBTCONSOLIDATION", "MEDICAL", "VENTURE"}