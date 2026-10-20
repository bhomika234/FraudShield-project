
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("\nDuplicate rows:", df.duplicated().sum())

print("\nClass counts:")
print(df["Class"].value_counts())

print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

print("\nTransaction amount summary:")
print(df.groupby("Class")["Amount"].describe())

# Fraud distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class")
plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
plt.close()

# Transaction amounts
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Class", y="Amount")
plt.ylim(0, df["Amount"].quantile(0.99))
plt.title("Transaction Amounts by Class (99th percentile view)")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.savefig(REPORT_DIR / "amount_distribution.png", dpi=150)
plt.close()

print("\nEDA completed.")
print("Charts saved in:", REPORT_DIR)
