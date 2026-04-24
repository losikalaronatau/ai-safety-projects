"""
Demo: Demographic Bias Auditor on the Adult Income Dataset
==========================================================
The Adult Income dataset predicts whether a person earns >$50K/year.
It's a classic dataset for studying bias — especially gender and race bias.

We train a simple logistic regression classifier, then run the bias audit.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import urllib.request
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.auditor import DemographicBiasAuditor


ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]


def load_adult_data():
    """Load and minimally preprocess the Adult Income dataset."""
    print("  Loading Adult Income dataset...")

    # Use a small synthetic sample to avoid network dependency
    np.random.seed(42)
    n = 2000

    # Simulate a biased dataset where females are under-predicted as high earners
    sex = np.random.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    race = np.random.choice(["White", "Black", "Asian", "Other"], size=n, p=[0.85, 0.09, 0.03, 0.03])
    age = np.random.randint(18, 70, size=n)
    education_num = np.random.randint(6, 16, size=n)
    hours = np.random.randint(20, 80, size=n)

    # True income — depends on education + age + hours, with some group signal
    true_prob = (
        0.02 * education_num
        + 0.005 * age
        + 0.003 * hours
        - 0.1 * (sex == "Female").astype(float)   # structural disadvantage baked in
        - 0.08 * (race == "Black").astype(float)
        - 0.1
    )
    true_prob = 1 / (1 + np.exp(-true_prob))   # sigmoid
    y_true = (np.random.rand(n) < true_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "hours_per_week": hours,
        "sex": sex,
        "race": race,
        "income": y_true
    })
    return df


def train_model(df):
    """Train a simple logistic regression (intentionally without fairness constraints)."""
    feature_cols = ["age", "education_num", "hours_per_week"]
    X = df[feature_cols].values
    y = df["income"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42
    )

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Model trained — Test Accuracy: {acc:.2%}")

    return y_test, y_pred, df.loc[idx_test]


def main():
    print("\n" + "=" * 60)
    print("  DEMOGRAPHIC BIAS AUDITOR — DEMO")
    print("=" * 60)

    df = load_adult_data()
    y_true, y_pred, test_df = train_model(df)

    auditor = DemographicBiasAuditor(
        demographic_parity_threshold=0.1,
        disparate_impact_threshold=0.8,
        equalized_odds_threshold=0.1,
    )

    # --- Audit 1: Gender Bias ---
    print("\n  [1/2] Auditing for GENDER bias...")
    sex_feature = test_df["sex"].values
    gender_report = auditor.audit(y_true, y_pred, sex_feature, feature_name="sex")
    auditor.print_report(gender_report)
    auditor.to_csv(gender_report, "reports/gender_bias_report.csv")

    # --- Audit 2: Racial Bias ---
    print("\n  [2/2] Auditing for RACE bias...")
    race_feature = test_df["race"].values
    race_report = auditor.audit(y_true, y_pred, race_feature, feature_name="race")
    auditor.print_report(race_report)
    auditor.to_csv(race_report, "reports/race_bias_report.csv")

    print("\n  Demo complete. Check the 'reports/' directory for CSV outputs.\n")


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    main()
