
# ==========================================================
# STUDENT DROPOUT RISK PREDICTION
# Logistic Regression vs Random Forest
# GitHub Compatible Version
# ==========================================================

# ----- 1. Import Libraries -----

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

SEED = 42

# ----- 2. Load Dataset -----

DATA_FILE = Path("student_dropout_dataset.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found! Place student_dropout_dataset.csv "
        "in the same folder as main.py."
    )

raw = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", raw.shape)
print(raw.head())

# ----- 3. Required Columns -----

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

missing = [c for c in REQUIRED if c not in raw.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ----- 4. Cleaning -----

n_raw = len(raw)

n_dup = int(raw.duplicated().sum())

df = raw.drop_duplicates().reset_index(drop=True)

n_missing = int(df.isna().sum().sum())

missing_by_col = {
    k: int(v)
    for k, v in df.isna().sum().items()
    if v > 0
}

print("\nDATA CLEANING REPORT")

print("Original rows:", n_raw)
print("Duplicate rows:", n_dup)
print("Rows after cleaning:", len(df))
print("Total missing values:", n_missing)
print("Missing values by column:", missing_by_col)

# ----- 5. Feature Engineering -----

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

# ----- 6. Prepare Input and Target -----

df["dropout"] = pd.to_numeric(
    df["dropout"],
    errors="coerce"
)

df = df.dropna(
    subset=["dropout"]
).reset_index(drop=True)

if not set(df["dropout"].unique()).issubset({0, 1}):
    raise ValueError(
        "Dropout must contain only 0 and 1."
    )

df["dropout"] = df["dropout"].astype(int)

if df["dropout"].nunique() != 2:
    raise ValueError(
        "Both dropout classes are required."
    )

if df["dropout"].value_counts().min() < 5:
    raise ValueError(
        "At least 5 students in each dropout class "
        "are required for 5-fold cross-validation."
    )

X = df[num + cat]

y = df["dropout"]

ids = df["student_id"]

# ----- 7. Train-Test Split -----

Xtr, Xte, ytr, yte, idtr, idte = train_test_split(
    X,
    y,
    ids,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# ----- 8. Data Preprocessing -----

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

# ----- 9. Cross Validation -----

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# ----- 10. Logistic Regression -----

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

print("\nLogistic Regression training completed!")

# ----- 11. Random Forest -----

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

print("\nRandom Forest training completed!")

print("Best parameters:", rf.best_params_)

# ----- 12. Model Evaluation -----

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

# ----- 13. Display Results -----

results = pd.DataFrame(
    [res_lr, res_rf],
    index=[
        "Logistic Regression",
        "Random Forest"
    ]
)

print("\nMODEL PERFORMANCE")

print(results.round(4))

print("\nBest Logistic Regression Parameters:")
print(lr.best_params_)

print("\nBest Random Forest Parameters:")
print(rf.best_params_)

# ----- 14. Classification Report -----

print("\nLOGISTIC REGRESSION REPORT")

print(classification_report(
    yte,
    (p_lr >= 0.5).astype(int),
    zero_division=0
))

print("\nRANDOM FOREST REPORT")

print(classification_report(
    yte,
    (p_rf >= 0.5).astype(int),
    zero_division=0
))

# ----- 15. Risk Bands -----

def band(p):

    return (
        "Low" if p < 0.3
        else "Medium" if p < 0.6
        else "High"
    )

# ----- 16. Select Best Model -----

best_model = (
    rf if res_rf["auc"] >= res_lr["auc"]
    else lr
)

print("\nSelected Model:")

print(
    "Random Forest"
    if best_model is rf
    else "Logistic Regression"
)

# ----- 17. Student Risk Prediction -----

probabilities = best_model.predict_proba(Xte)[:, 1]

predictions = pd.DataFrame({

    "student_id": idte.values,

    "actual_dropout": yte.values,

    "dropout_probability": probabilities,

    "risk_level": [
        band(p) for p in probabilities
    ]
})

print("\nSTUDENT DROPOUT RISK PREDICTIONS")

print(
    predictions.head(20).to_string(index=False)
)

# ----- 18. Risk Summary -----

print("\nRISK LEVEL SUMMARY")

print(
    predictions["risk_level"].value_counts()
)

# ----- 19. Save Results -----

predictions.to_csv(
    "student_dropout_predictions.csv",
    index=False
)

results.to_csv(
    "model_performance.csv"
)

print("\nBoth result files saved successfully!")

# ----- 20. Project Completed -----

print("\nPROJECT COMPLETED SUCCESSFULLY!")
