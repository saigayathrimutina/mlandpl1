
# ==========================================================
# STUDENT DROPOUT RISK PREDICTION
# Logistic Regression vs Random Forest
# GitHub + Streamlit + Excel Dataset
# ==========================================================

# ----- 1. Import Libraries -----

import streamlit as st
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    "Logistic Regression vs Random Forest"
)

st.write(
    "This project predicts student dropout risk "
    "using Machine Learning algorithms."
)

SEED = 42

# ----- 3. Load Excel Dataset -----

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "student_dropout_predictions.xlsx"

if not DATA_FILE.exists():
    st.error(
        f"Excel dataset not found: {DATA_FILE.name}"
    )
    st.stop()

try:
    raw = pd.read_excel(DATA_FILE)

except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

st.success("Dataset loaded successfully!")

st.write("### Dataset Information")

col1, col2 = st.columns(2)

col1.metric("Total Rows", raw.shape[0])
col2.metric("Total Columns", raw.shape[1])

st.write("### Dataset Preview")

st.dataframe(raw.head(10), use_container_width=True)

# ----- 4. Required Columns -----

REQUIRED = [
    "student_id",
    "gender",
    "age",
    "entrance_rank_pct",
    "inter_pct",
    "family_income",
    "first_generation",
    "scholarship",
    "distance_km",
    "hostel_resident",
    "attendance_pct",
    "internal_marks",
    "sgpa_sem1",
    "sgpa_sem2",
    "backlogs",
    "assignment_pct",
    "fee_due_inr",
    "dropout"
]

missing = [
    c for c in REQUIRED
    if c not in raw.columns
]

if missing:
    st.error(
        f"Missing required columns: {missing}"
    )
    st.stop()

# ----- 5. Data Cleaning -----

n_raw = len(raw)

n_dup = int(raw.duplicated().sum())

df = raw.drop_duplicates().reset_index(drop=True)

n_missing = int(df.isna().sum().sum())

missing_by_col = {
    k: int(v)
    for k, v in df.isna().sum().items()
    if v > 0
}

st.write("### Data Cleaning Report")

clean1, clean2, clean3 = st.columns(3)

clean1.metric("Original Rows", n_raw)
clean2.metric("Duplicate Rows", n_dup)
clean3.metric("Rows After Cleaning", len(df))

st.write("Total Missing Values:", n_missing)

st.write("Missing Values by Column:", missing_by_col)

# ----- 6. Feature Engineering -----

df["sgpa_trend"] = (
    df["sgpa_sem2"] - df["sgpa_sem1"]
)

df["low_attendance_flag"] = (
    df["attendance_pct"] < 65
).astype(int)

num = [
    "age",
    "entrance_rank_pct",
    "inter_pct",
    "distance_km",
    "attendance_pct",
    "internal_marks",
    "sgpa_sem1",
    "sgpa_sem2",
    "backlogs",
    "assignment_pct",
    "fee_due_inr",
    "sgpa_trend",
    "first_generation",
    "scholarship",
    "hostel_resident",
    "low_attendance_flag"
]

cat = [
    "gender",
    "family_income"
]

# ----- 7. Prepare Input and Target -----

df["dropout"] = pd.to_numeric(
    df["dropout"],
    errors="coerce"
)

df = df.dropna(
    subset=["dropout"]
).reset_index(drop=True)

if not set(df["dropout"].unique()).issubset({0, 1}):
    st.error(
        "Dropout column must contain only 0 and 1."
    )
    st.stop()

df["dropout"] = df["dropout"].astype(int)

if df["dropout"].nunique() != 2:
    st.error(
        "Both dropout classes (0 and 1) are required."
    )
    st.stop()

if df["dropout"].value_counts().min() < 5:
    st.error(
        "At least 5 records in each dropout class "
        "are required for 5-fold cross-validation."
    )
    st.stop()

X = df[num + cat]

y = df["dropout"]

ids = df["student_id"]

# ----- 8. Train-Test Split -----

Xtr, Xte, ytr, yte, idtr, idte = train_test_split(
    X,
    y,
    ids,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# ----- 9. Data Preprocessing -----

def prep(scale):

    steps = [
        ("imp", SimpleImputer(strategy="median"))
    ]

    if scale:
        steps.append(
            ("sc", StandardScaler())
        )

    return ColumnTransformer([
        (
            "num",
            Pipeline(steps),
            num
        ),

        (
            "cat",
            Pipeline([
                (
                    "imp",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "oh",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            cat
        )
    ])

# ----- 10. Cross Validation -----

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# ----- 11. Logistic Regression -----

with st.spinner("Training Logistic Regression..."):

    lr_pipe = Pipeline([
        ("prep", prep(True)),
        (
            "m",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000
            )
        )
    ])

    lr_grid = {
        "m__C": [0.01, 0.1, 1, 10]
    }

    lr = GridSearchCV(
        lr_pipe,
        lr_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    lr.fit(Xtr, ytr)

st.success("Logistic Regression training completed!")

# ----- 12. Random Forest -----

with st.spinner("Training Random Forest..."):

    rf_pipe = Pipeline([
        ("prep", prep(False)),
        (
            "m",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=1
            )
        )
    ])

    rf_grid = {
        "m__n_estimators": [100, 200],
        "m__max_depth": [6, None],
        "m__min_samples_leaf": [1, 5]
    }

    rf = GridSearchCV(
        rf_pipe,
        rf_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    rf.fit(Xtr, ytr)

st.success("Random Forest training completed!")

# ----- 13. Model Evaluation -----

def evaluate(model):

    p = model.predict_proba(Xte)[:, 1]

    yh = (p >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        yte,
        yh,
        labels=[0, 1]
    ).ravel()

    return dict(
        accuracy=accuracy_score(yte, yh),

        precision=precision_score(
            yte, yh, zero_division=0
        ),

        recall=recall_score(
            yte, yh, zero_division=0
        ),

        f1=f1_score(
            yte, yh, zero_division=0
        ),

        auc=roc_auc_score(yte, p),

        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp)
    ), p


res_lr, p_lr = evaluate(lr)

res_rf, p_rf = evaluate(rf)

# ----- 14. Display Model Performance -----

st.write("## Model Performance Comparison")

results = pd.DataFrame(
    [res_lr, res_rf],
    index=[
        "Logistic Regression",
        "Random Forest"
    ]
)

st.dataframe(
    results.round(4),
    use_container_width=True
)

st.write("### Best Model Parameters")

st.write("Logistic Regression:", lr.best_params_)

st.write("Random Forest:", rf.best_params_)

# ----- 15. Classification Reports -----

st.write("## Classification Reports")

with st.expander("Logistic Regression Report"):

    st.text(
        classification_report(
            yte,
            (p_lr >= 0.5).astype(int),
            zero_division=0
        )
    )

with st.expander("Random Forest Report"):

    st.text(
        classification_report(
            yte,
            (p_rf >= 0.5).astype(int),
            zero_division=0
        )
    )

# ----- 16. Risk Bands -----

def band(p):

    return (
        "Low" if p < 0.3
        else "Medium" if p < 0.6
        else "High"
    )

# ----- 17. Select Best Model -----

best_model = (
    rf if res_rf["auc"] >= res_lr["auc"]
    else lr
)

selected_name = (
    "Random Forest"
    if best_model is rf
    else "Logistic Regression"
)

st.write("## Selected Best Model")

st.success(selected_name)

# ----- 18. Student Risk Prediction -----

probabilities = best_model.predict_proba(Xte)[:, 1]

predictions = pd.DataFrame({

    "student_id": idte.values,

    "actual_dropout": yte.values,

    "dropout_probability": probabilities,

    "risk_level": [
        band(p) for p in probabilities
    ]
})

st.write("## Student Dropout Risk Predictions")

st.dataframe(
    predictions.head(20),
    use_container_width=True
)

# ----- 19. Risk Level Summary -----

st.write("## Risk Level Summary")

risk_counts = (
    predictions["risk_level"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )
)

risk1, risk2, risk3 = st.columns(3)

risk1.metric("Low Risk", int(risk_counts["Low"]))
risk2.metric("Medium Risk", int(risk_counts["Medium"]))
risk3.metric("High Risk", int(risk_counts["High"]))

st.bar_chart(risk_counts)

# ----- 20. Save and Download Results -----

predictions.to_csv(
    "student_dropout_predictions.csv",
    index=False
)

results.to_csv(
    "model_performance.csv"
)

st.write("## Download Project Results")

st.download_button(
    label="Download Student Risk Predictions",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="student_dropout_predictions.csv",
    mime="text/csv"
)

st.download_button(
    label="Download Model Performance",
    data=results.to_csv().encode("utf-8"),
    file_name="model_performance.csv",
    mime="text/csv"
)

st.success("PROJECT COMPLETED SUCCESSFULLY!")




