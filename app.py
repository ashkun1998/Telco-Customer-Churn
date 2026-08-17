"""
Telco Customer Churn - Retention Risk Dashboard
Assignment 2 (Machine Learning, M.Tech AIML/DSE, BITS Pilani WILP)

Three-tab app built around a churn *retention* framing rather than a
generic "upload CSV / pick model / see metrics" template:

  1. Overview        - dataset story + churn-rate breakdowns
  2. Risk Checker     - build one customer profile by hand and get a live
                        churn-risk read-out from any of the 5 models
  3. Model Lab        - batch evaluation (CSV upload) + full metric/­
                        confusion-matrix comparison across all 5 models
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model", "saved_models")

ACCENT = "#0F6E5C"   # deep teal - telecom / "retain" theme, deliberately not Streamlit's default red

st.set_page_config(page_title="Telco Retention Risk Dashboard", page_icon="📶", layout="wide")

# NOTE: overall page theme (background/text/accent colors) is controlled by
# .streamlit/config.toml, NOT by CSS injection here. Hardcoding a background
# color via CSS while leaving Streamlit's own widget text colors theme-aware
# is what caused the dark-mode contrast bug (unselected tab labels, table
# text, etc. going invisible) - config.toml applies one consistent theme to
# every built-in widget, which raw CSS overrides can't guarantee.
st.markdown(
    """
    <style>
    .risk-badge {
        display: inline-block; padding: 10px 22px; border-radius: 999px;
        font-size: 1.3rem; font-weight: 700; color: white; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_FILES = {
    "Logistic Regression": ("logistic_regression.joblib", True),
    "Decision Tree": ("decision_tree.joblib", False),
    "kNN": ("knn.joblib", True),
    "Naive Bayes": ("naive_bayes.joblib", False),
    "Random Forest (Ensemble)": ("random_forest_ensemble.joblib", False),
}

CATEGORICAL_HELP = {
    "MultipleLines": "'No phone service' means the customer has no phone line at all.",
    "InternetService": "'No' means the customer has no internet service.",
    "OnlineSecurity": "'No internet service' auto-applies if InternetService = No.",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    models = {}
    for name, (fname, needs_scaling) in MODEL_FILES.items():
        models[name] = (joblib.load(os.path.join(MODEL_DIR, fname)), needs_scaling)
    results_df = pd.read_csv(os.path.join(MODEL_DIR, "results.csv"), index_col=0)
    raw_df = pd.read_csv(os.path.join(HERE, "telco_raw.csv"))
    return scaler, encoders, feature_cols, models, results_df, raw_df


scaler, encoders, feature_cols, models, results_df, raw_df = load_artifacts()
categorical_cols = list(encoders.keys())
numeric_cols = [c for c in feature_cols if c not in categorical_cols]


def encode_row(raw_values: dict) -> pd.DataFrame:
    """Turn a dict of human-readable feature values into the encoded row a model expects."""
    row = {}
    for col in feature_cols:
        if col in encoders:
            row[col] = encoders[col].transform([raw_values[col]])[0]
        else:
            row[col] = raw_values[col]
    return pd.DataFrame([row], columns=feature_cols)


def predict(model_name: str, X: pd.DataFrame):
    model, needs_scaling = models[model_name]
    X_input = scaler.transform(X) if needs_scaling else X
    pred = model.predict(X_input)
    proba = model.predict_proba(X_input)[:, 1]
    return pred, proba


st.title("📶 Telco Retention Risk Dashboard")
st.caption(
    "M.Tech (AIML/DSE) - Machine Learning, Assignment 2 · Dataset: Telco Customer Churn (Kaggle / IBM Sample Data Sets)"
)

tab_overview, tab_checker, tab_lab = st.tabs(["🏠 Overview", "🧮 Risk Checker", "🔬 Model Lab"])

# ===========================================================================
# TAB 1 — Overview
# ===========================================================================
with tab_overview:
    n_customers = len(raw_df)
    churn_rate = (raw_df["Churn"] == "Yes").mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers in dataset", f"{n_customers:,}")
    c2.metric("Historical churn rate", f"{churn_rate*100:.1f}%")
    c3.metric("Features used", f"{len(feature_cols)}")

    st.markdown("---")
    st.subheader("Where does churn concentrate?")
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        churn_by_contract = raw_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
        churn_by_contract = churn_by_contract.reindex(["Month-to-month", "One year", "Two year"])
        sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, ax=ax, color=ACCENT)
        ax.set_ylabel("Churn rate")
        ax.set_xlabel("")
        ax.set_title("Churn rate by contract type")
        for i, v in enumerate(churn_by_contract.values):
            ax.text(i, v + 0.01, f"{v*100:.0f}%", ha="center")
        st.pyplot(fig)

    with right:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        raw_df["tenure_bucket"] = pd.cut(
            raw_df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-12mo", "1-2yr", "2-4yr", "4-6yr"]
        )
        churn_by_tenure = raw_df.groupby("tenure_bucket", observed=True)["Churn"].apply(lambda s: (s == "Yes").mean())
        sns.barplot(x=churn_by_tenure.index, y=churn_by_tenure.values, ax=ax2, color="#C7622D")
        ax2.set_ylabel("Churn rate")
        ax2.set_xlabel("")
        ax2.set_title("Churn rate by tenure bucket")
        for i, v in enumerate(churn_by_tenure.values):
            ax2.text(i, v + 0.01, f"{v*100:.0f}%", ha="center")
        st.pyplot(fig2)

    st.info(
        "Month-to-month customers and customers still in their first year are the "
        "highest-risk segments — this is the pattern the models below are trying to learn."
    )

# ===========================================================================
# TAB 2 — Risk Checker (single-customer form, not a generic CSV uploader)
# ===========================================================================
with tab_checker:
    st.subheader("Build a customer profile and get a live risk read")
    model_name = st.selectbox("Scoring model", list(models.keys()), key="checker_model")

    with st.form("customer_form"):
        f1c, f2c, f3c = st.columns(3)

        with f1c:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", encoders["gender"].classes_)
            senior = st.selectbox("Senior citizen?", ["No", "Yes"])
            partner = st.selectbox("Has partner?", encoders["Partner"].classes_)
            dependents = st.selectbox("Has dependents?", encoders["Dependents"].classes_)
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with f2c:
            st.markdown("**Services**")
            phone = st.selectbox("Phone service?", encoders["PhoneService"].classes_)
            multi_lines = st.selectbox("Multiple lines?", encoders["MultipleLines"].classes_,
                                        help=CATEGORICAL_HELP["MultipleLines"])
            internet = st.selectbox("Internet service", encoders["InternetService"].classes_,
                                     help=CATEGORICAL_HELP["InternetService"])
            online_sec = st.selectbox("Online security?", encoders["OnlineSecurity"].classes_)
            online_backup = st.selectbox("Online backup?", encoders["OnlineBackup"].classes_)
            device_prot = st.selectbox("Device protection?", encoders["DeviceProtection"].classes_)
            tech_support = st.selectbox("Tech support?", encoders["TechSupport"].classes_)
            stream_tv = st.selectbox("Streaming TV?", encoders["StreamingTV"].classes_)
            stream_movies = st.selectbox("Streaming movies?", encoders["StreamingMovies"].classes_)

        with f3c:
            st.markdown("**Account**")
            contract = st.selectbox("Contract", encoders["Contract"].classes_)
            paperless = st.selectbox("Paperless billing?", encoders["PaperlessBilling"].classes_)
            payment = st.selectbox("Payment method", encoders["PaymentMethod"].classes_)
            monthly_charges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
            total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, float(tenure) * 70.0, step=10.0)

        submitted = st.form_submit_button("Check churn risk", use_container_width=True)

    if submitted:
        raw_values = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_backup,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": stream_tv, "StreamingMovies": stream_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        X_row = encode_row(raw_values)
        pred, proba = predict(model_name, X_row)
        risk = proba[0]

        color = "#C0392B" if risk >= 0.5 else ("#D68910" if risk >= 0.3 else "#1E8449")
        label = "HIGH RISK" if risk >= 0.5 else ("WATCH" if risk >= 0.3 else "LOW RISK")

        st.markdown(
            f'<span class="risk-badge" style="background-color:{color}">{label} — {risk*100:.1f}% churn probability</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Scored with **{model_name}**")

        if hasattr(models[model_name][0], "feature_importances_"):
            st.markdown("##### What's driving this model's decisions overall")
            importances = pd.Series(models[model_name][0].feature_importances_, index=feature_cols)
            importances = importances.sort_values(ascending=True).tail(8)
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            ax3.barh(importances.index, importances.values, color=ACCENT)
            ax3.set_xlabel("Feature importance")
            st.pyplot(fig3)

# ===========================================================================
# TAB 3 — Model Lab (batch evaluation across all 5 models)
# ===========================================================================
with tab_lab:
    st.subheader("Batch-evaluate a test set across all 5 models")

    src = st.radio(
        "Test data source", ["Bundled held-out test split", "Upload my own CSV"],
        horizontal=True,
    )
    if src == "Upload my own CSV":
        uploaded_file = st.file_uploader(
            "CSV with the encoded feature columns, optionally with a 'Churn' column of true labels", type=["csv"]
        )
        df = pd.read_csv(uploaded_file) if uploaded_file is not None else None
    else:
        df = pd.read_csv(os.path.join(HERE, "test_data.csv"))

    if df is None:
        st.warning("Upload a CSV to continue.")
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required feature columns: {missing}")
        else:
            has_labels = "Churn" in df.columns
            eval_model = st.selectbox("Model to evaluate", list(models.keys()), key="lab_model")
            pred, proba = predict(eval_model, df[feature_cols])

            out_df = df.copy()
            out_df["predicted_churn"] = pred
            out_df["churn_probability"] = np.round(proba, 4)

            m_col, table_col = st.columns([1, 2])
            with table_col:
                st.markdown("**Sample predictions**")
                st.dataframe(out_df.head(15), use_container_width=True)
                st.download_button(
                    "⬇ Download all predictions as CSV",
                    out_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{eval_model.replace(' ', '_')}.csv",
                )

            with m_col:
                if has_labels:
                    y_true = df["Churn"]
                    metrics = {
                        "Accuracy": accuracy_score(y_true, pred),
                        "AUC": roc_auc_score(y_true, proba) if y_true.nunique() > 1 else float("nan"),
                        "Precision": precision_score(y_true, pred),
                        "Recall": recall_score(y_true, pred),
                        "F1": f1_score(y_true, pred),
                        "MCC": matthews_corrcoef(y_true, pred),
                    }
                    st.markdown("**Metrics on this data**")
                    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).assign(
                        Value=lambda d: d["Value"].round(4)
                    ))
                else:
                    st.info("No 'Churn' column — showing predictions only.")

            if has_labels:
                st.markdown("---")
                cm_col, report_col = st.columns(2)
                with cm_col:
                    cm = confusion_matrix(y_true, pred)
                    fig4, ax4 = plt.subplots(figsize=(4, 3.5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn",
                                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax4)
                    ax4.set_xlabel("Predicted")
                    ax4.set_ylabel("Actual")
                    st.pyplot(fig4)
                with report_col:
                    st.text("Classification report")
                    st.code(classification_report(y_true, pred, target_names=["No Churn", "Churn"]))

        st.markdown("---")
        st.subheader("All 5 models, side by side")
        st.dataframe(results_df.style.highlight_max(axis=0, color="#D5F5E3"), use_container_width=True)

        fig5, ax5 = plt.subplots(figsize=(9, 3.5))
        results_df.plot(kind="bar", ax=ax5, colormap="viridis")
        ax5.set_ylabel("Score")
        ax5.legend(loc="lower right", ncol=3, fontsize=8)
        plt.xticks(rotation=15, ha="right")
        st.pyplot(fig5)

st.markdown("---")
st.caption("Built for BITS Pilani WILP — M.Tech (AIML/DSE) Machine Learning, Assignment 2.")
