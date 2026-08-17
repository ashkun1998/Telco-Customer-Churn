"""
Assignment 2 - Machine Learning (M.Tech AIML/DSE, BITS Pilani WILP)
Trains 5 classification models on the Telco Customer Churn dataset and
saves the models + evaluation metrics + test split.

Dataset: Telco Customer Churn (IBM Sample Data Sets)
Source : originally published on Kaggle -
         https://www.kaggle.com/datasets/blastchar/telco-customer-churn
         (IBM's own GitHub mirror of the same file is used here:
         https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
Task   : Binary classification - will a customer churn? (Yes/No)
Shape  : 7043 instances, 19 usable features after dropping the customerID
         identifier column (meets >=500 instances, >=12 features)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# 1. Load + clean dataset
# ---------------------------------------------------------------------------
raw = pd.read_csv(os.path.join(ROOT, "telco_raw.csv"))

df = raw.drop(columns=["customerID"]).copy()

# TotalCharges has some blank strings for brand-new customers (tenure=0); coerce + impute
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Target: Churn Yes/No -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical feature columns (label encoding keeps the feature count
# identical to the original column count, which is simplest for this app)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [c for c in df.columns if c != "Churn"]
X = df[feature_cols]
y = df["Churn"]

print(f"Dataset shape: {df.shape}  |  Features: {len(feature_cols)}  |  Classes: {sorted(y.unique())}")
print(f"Churn rate: {y.mean():.3f}")

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified) + scaling
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the raw (encoded) test split as test_data.csv for the Streamlit app's
# "upload CSV" demo feature.
test_data = X_test.copy()
test_data["Churn"] = y_test.values
test_data.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"Saved test_data.csv with {len(test_data)} rows")

# ---------------------------------------------------------------------------
# 3. Define models
#    (Logistic Regression and kNN use scaled features; tree-based / NB
#     models are trained on the raw encoded features.)
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE), False),
    "kNN": (KNeighborsClassifier(n_neighbors=15), True),
    "Naive Bayes": (GaussianNB(), False),
    "Random Forest (Ensemble)": (RandomForestClassifier(n_estimators=150, max_depth=12, random_state=RANDOM_STATE), False),
}

results = {}
os.makedirs(os.path.join(ROOT, "model", "saved_models"), exist_ok=True)

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results[name] = metrics
    print(name, metrics)

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(ROOT, "model", "saved_models", f"{fname}.joblib"))

# Save scaler + label encoders (needed at inference time in the app)
joblib.dump(scaler, os.path.join(ROOT, "model", "saved_models", "scaler.joblib"))
joblib.dump(encoders, os.path.join(ROOT, "model", "saved_models", "encoders.joblib"))

with open(os.path.join(ROOT, "model", "saved_models", "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)

results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
results_df.to_csv(os.path.join(ROOT, "model", "saved_models", "results.csv"))
print("\nComparison table:\n", results_df)

print("\nDone. Models, scaler, encoders, test_data.csv and results.csv saved.")
