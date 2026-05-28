from tkinter.constants import TRUE

import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).parent / "credit_risk_dataset.csv")

print(f"Shape: {df.shape}")
print(f"\nColumns: \n{df.dtypes}")
print(f"\nMissing Values: \n{df.isnull().sum()}")
print(f"\nTarget balance: \n{df['loan_status'].value_counts(normalize=True).round(3)}")
print(f"\nFirst rows: \n{df.head()}")

