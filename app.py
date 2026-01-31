import streamlit as st
import numpy as np
import pandas as pd
import joblib
#from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# =====================================================
# Load trained models
# =====================================================
try:
    # Adjust paths if your models are inside a folder, e.g., "models/logistic_regression.pkl"
    lr_model = joblib.load("models/logistic_regression.pkl")
    rf_model = joblib.load("models/random_forest.joblib")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_model.json")
   # mlp_model = load_model("models/mlp.keras", compile=False)

    MODELS = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
       # "MLP - Deep Learning": mlp_model
    }
except Exception as e:
    st.error(f"Error loading models: {e}. Ensure model files are in the root directory.")
    MODELS = {}

# =====================================================
# Helper functions
# =====================================================
def neural_net_predictions(model, X):
    probs = model.predict(X).flatten()
    preds = (probs > 0.5).astype(int)
    return probs, preds


def classification_df(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Not fraud", "Fraud"],
        output_dict=True
    )
    df_res = pd.DataFrame(report).T
    keep_rows = [r for r in ["Not fraud", "Fraud", "accuracy", "macro avg", "weighted avg"] if r in df_res.index]
    df_res = df_res.loc[keep_rows]

    for col in ["precision", "recall", "f1-score", "accuracy", "support"]:
        if col in df_res.columns:
            df_res[col] = df_res[col].apply(lambda x: int(x) if float(x).is_integer() else round(x, 4))
    return df_res


def run_models(X_test, y_test):
    results = {}
    for name, model in MODELS.items():
        try:
            # Check for corrupted RF models (Version Mismatch)
            if hasattr(model, "estimators_") and len(model.estimators_) == 0:
                continue

            # --- LOGIC SEPARATION ---
            if name == "MLP - Deep Learning":
                probs, preds = neural_net_predictions(model, X_test)
            else:
                # Scikit-Learn and XGBoost use predict_proba
                probs = model.predict_proba(X_test)[:, 1]
                preds = model.predict(X_test)

            results[name] = {
                "df": classification_df(y_test, preds),
                "probs": probs,
                "preds": preds
            }
        except Exception as e:
            st.error(f"⚠️ Error running {name}: {str(e)}")
            continue
    return results

# =====================================================
# Plotting functions
# =====================================================
FIG_SIZE = (4, 4)

def plot_roc(y_true, probs, title):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig


def plot_pr(y_true, probs):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    return fig


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
        xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


def highlight_fraud_best(s):
    styles = [""] * len(s)
    if s.name != "Fraud":
        return styles
    for metric in ["precision", "recall", "f1-score"]:
        metric_cols = [c for c in s.index if c[1] == metric]
        values = {c: round(float(s[c]), 4) for c in metric_cols}
        if not values:
            continue
        max_val = max(values.values())
        for c, v in values.items():
            if v == max_val:
                idx = list(s.index).index(c)
                styles[idx] = "background-color:#2ecc71; color:black; font-weight:bold"
    return styles

# =====================================================
# Streamlit UI
# =====================================================
st.title("💳 Credit Card Fraud Detection – Model Evaluation")

st.markdown(
    """
    **Purpose of this application**

    This app simulates a **real business scenario** where a bank or financial institution
    analyzes a dataset of **credit card transactions** to detect **fraudulent activity**.

      The app evaluates:
    - Classification metrics
    - ROC curves
    - Precision–Recall curves
    - Confusion matrices
    """
)

# =====================================================
# Demo dataset vs Upload
# =====================================================
st.markdown("### Choose data source")

data_source = st.radio(
    "Select how you want to load the dataset:",
    ("Use demo dataset (recommended)", "Upload my own CSV"),
    horizontal=True,
    index=None
)

df = None

if data_source == "Use demo dataset (recommended)":
    try:
        df = pd.read_csv("data/test_data.csv")
        st.success("✅ Demo dataset loaded successfully.")
        st.info("ℹ️ You are viewing results using the built-in demo dataset.")
    except Exception as e:
        st.error(f"❌ Could not load demo dataset: {e}")

else:
    uploaded_file = st.file_uploader("Upload your testing dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read uploaded CSV: {e}")

# =====================================================
# Restul codului tău rămâne identic, doar pornește dacă df există
# =====================================================
if df is not None:
    try:
        if "target" not in df.columns:
            st.error("❌ Error: The file must contain a 'target' column (0 for normal, 1 for fraud).")
        else:
            # 1. DATA INSPECTION (Head and Count)
            st.markdown("---")
            st.header("🔍 1. Initial Data Inspection")

            col_head, col_stats = st.columns([2, 1])

            with col_head:
                st.write("**Data Preview (First 5 Rows)**")
                st.dataframe(df.head(5), use_container_width=True)

            with col_stats:
                st.write("**Class Distribution**")
                counts = df['target'].value_counts()
                total = len(df)

                # Calculation for labels with percentages
                for val, label in zip([0, 1], ["Not Fraud", "Fraud"]):
                    count = counts.get(val, 0)
                    percent = (count / total) * 100
                    st.metric(label, f"{count} tx", f"{percent:.2f}% of total", delta_color="off")

            # DATA PREP FOR MODELS
            y_test = df["target"].values
            X_test = df.drop(columns=["target"]).values

            # 2. RUN MODELS
            st.markdown("---")
            st.header(" Model Evaluation & Comparison")

            with st.spinner("Analyzing data through all models..."):
                results = run_models(X_test, y_test)

            if results:
                # Comparison Table
                st.subheader("Metrics Summary Table")
                st.info("The table below highlights the best performing model for 'Fraud' detection in green.")

                combined_df = pd.concat({k: v["df"] for k, v in results.items()}, axis=1)

                def format_cell(x):
                    return f"{int(x)}" if float(x).is_integer() else f"{round(x,4):.4f}"

                styled_df = combined_df.style.format(format_cell).apply(highlight_fraud_best, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # Visualizations
                st.subheader("Performance Visualizations")
                tabs = st.tabs(list(results.keys()))

                for tab, (name, output) in zip(tabs, results.items()):
                    with tab:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.pyplot(plot_roc(y_test, output["probs"], name))
                        with c2:
                            st.pyplot(plot_pr(y_test, output["probs"]))
                        with c3:
                            st.pyplot(plot_cm(y_test, output["preds"]))

                # 3. PREDICTION EXPORT
                st.markdown("---")
                st.header("📤Export Fraud Predictions")
                st.write("Select a model to generate a downloadable report of its findings.")

                selected_model_name = st.selectbox("Choose model for export:", list(results.keys()))

                # Create export dataframe
                export_df = df.copy()
                export_df["predicted_fraud"] = results[selected_model_name]["preds"]
                export_df["fraud_probability"] = results[selected_model_name]["probs"]

                # UI/UX Message for export
                fraud_detected = export_df["predicted_fraud"].sum()
                st.success(
                    f"**Analysis Complete!** The {selected_model_name} model flagged "
                    f"**{fraud_detected}** transactions as potentially fraudulent."
                )

                csv = export_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label=f"Download Predictions ({selected_model_name})",
                    data=csv,
                    file_name=f"fraud_predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    help="Click to download the full dataset with a new 'predicted_fraud' column."
                )
            else:
                st.warning("No models were able to run. Please check your .pkl and .json files.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    if data_source == "Upload my own CSV":
        st.info("Please upload a CSV file to begin. The file should contain transaction features and a 'target' column.")
    else:
        st.warning("Demo dataset is not available. Please check that `data/test_data.csv` exists.")
