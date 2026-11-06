
# ==========================================================
# STUDENT DROPOUT RISK PREDICTION
# Streamlit Dashboard
# Based on Existing CSV Prediction Results
# ==========================================================

# ----- 1. Import Libraries -----

import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ----- 2. Streamlit Page Configuration -----

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Dropout Risk Prediction")

st.subheader(
    "Machine Learning Based Student Risk Analysis"
)

st.write(
    "This application displays student dropout predictions "
    "using previously generated Machine Learning results."
)

# ----- 3. Load CSV Dataset -----

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "student_dropout_predictions.csv"

if not DATA_FILE.exists():

    st.error(
        "Dataset not found: student_dropout_predictions.csv"
    )

    st.info(
        "Upload student_dropout_predictions.csv "
        "into the same GitHub folder as project3.py."
    )

    st.stop()

try:

    df = pd.read_csv(DATA_FILE)

except Exception as e:

    st.error(f"Error reading CSV file: {e}")

    st.stop()

st.success("Student prediction dataset loaded successfully!")

# ----- 4. Check Required Columns -----

REQUIRED = [
    "student_id",
    "actual_dropout",
    "dropout_probability",
    "risk_level"
]

missing = [
    col for col in REQUIRED
    if col not in df.columns
]

if missing:

    st.error(f"Missing columns: {missing}")

    st.stop()

# ----- 5. Data Cleaning -----

df = df.drop_duplicates().copy()

df["actual_dropout"] = pd.to_numeric(
    df["actual_dropout"],
    errors="coerce"
)

df["dropout_probability"] = pd.to_numeric(
    df["dropout_probability"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "student_id",
        "actual_dropout",
        "dropout_probability"
    ]
).copy()

if df.empty:

    st.error("No valid student prediction records found.")

    st.stop()

if not set(df["actual_dropout"].unique()).issubset({0, 1}):

    st.error("Actual dropout must contain only 0 and 1.")

    st.stop()

if not df["dropout_probability"].between(0, 1).all():

    st.error("Dropout probabilities must be between 0 and 1.")

    st.stop()

df["actual_dropout"] = df["actual_dropout"].astype(int)

# ----- 6. Student Risk Classification -----

def band(probability):

    if probability < 0.3:
        return "Low"

    elif probability < 0.6:
        return "Medium"

    else:
        return "High"


# Recalculate risk level from probability
df["risk_level"] = df["dropout_probability"].apply(band)

df["predicted_dropout"] = (
    df["dropout_probability"] >= 0.5
).astype(int)

# ----- 7. Dataset Overview -----

st.header("📊 Dataset Overview")

total_students = len(df)

actual_dropouts = int(
    (df["actual_dropout"] == 1).sum()
)

actual_non_dropouts = int(
    (df["actual_dropout"] == 0).sum()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Students",
    total_students
)

col2.metric(
    "Actual Dropouts",
    actual_dropouts
)

col3.metric(
    "Non-Dropouts",
    actual_non_dropouts
)

# ----- 8. Display Student Data -----

st.header("👨‍🎓 Student Prediction Details")

st.dataframe(
    df[
        [
            "student_id",
            "actual_dropout",
            "dropout_probability",
            "predicted_dropout",
            "risk_level"
        ]
    ],
    use_container_width=True
)

# ----- 9. Risk Level Summary -----

st.header("⚠️ Student Risk Level Summary")

risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Low Risk",
    int(risk_counts["Low"])
)

col2.metric(
    "Medium Risk",
    int(risk_counts["Medium"])
)

col3.metric(
    "High Risk",
    int(risk_counts["High"])
)

# ----- 10. Risk Level Chart -----

st.subheader("Student Risk Distribution")

st.bar_chart(risk_counts)

# ----- 11. Dropout Probability Chart -----

st.subheader("Student Dropout Probability")

prob_chart = df.set_index("student_id")[
    "dropout_probability"
]

st.bar_chart(prob_chart)

# ----- 12. Model Performance -----

st.header("📈 Model Performance")

y_actual = df["actual_dropout"]

y_predicted = df["predicted_dropout"]

probabilities = df["dropout_probability"]

accuracy = accuracy_score(
    y_actual,
    y_predicted
)

precision = precision_score(
    y_actual,
    y_predicted,
    zero_division=0
)

recall = recall_score(
    y_actual,
    y_predicted,
    zero_division=0
)

f1 = f1_score(
    y_actual,
    y_predicted,
    zero_division=0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{precision * 100:.2f}%"
)

col3.metric(
    "Recall",
    f"{recall * 100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{f1 * 100:.2f}%"
)

if y_actual.nunique() == 2:

    auc = roc_auc_score(
        y_actual,
        probabilities
    )

    st.metric(
        "ROC-AUC Score",
        f"{auc:.4f}"
    )

else:

    st.info(
        "ROC-AUC requires both actual dropout classes."
    )

# ----- 13. Confusion Matrix -----

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_actual,
    y_predicted,
    labels=[0, 1]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Dropout", "Actual Dropout"],
    columns=["Predicted Non-Dropout", "Predicted Dropout"]
)

st.dataframe(cm_df)

# ----- 14. Classification Report -----

st.subheader("Classification Report")

report = classification_report(
    y_actual,
    y_predicted,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

# ----- 15. Download Results -----

st.header("📥 Download Results")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Student Dropout Predictions",
    data=csv_data,
    file_name="student_dropout_predictions.csv",
    mime="text/csv"
)

# ----- 16. Project Completion -----

st.success(
    "STUDENT DROPOUT RISK PREDICTION DASHBOARD "
    "COMPLETED SUCCESSFULLY!"
)

st.caption(
    "Project: Student Dropout Risk Prediction"
)

