
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "risk_prediction_model.pkl"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ATMOSYNC MICRO-CLIMATE RISK PREDICTION")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("\nAVAILABLE DATASET COLUMNS:")
print(df.columns.tolist())
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

print("\nDataset loaded successfully!")
print("Total rows:", len(df))


# ============================================================
# 3. CREATE NEXT-READING TARGET
# ============================================================

df["TARGET_RISK"] = (
    df.groupby("CONTAINER_ID")["RISK_LABEL"].shift(-1)
)

# Exclude Unknown labels from this risk-classification experiment
df = df.dropna(subset=["TARGET_RISK"]).copy()

df = df[
    df["TARGET_RISK"].isin(["Low", "Medium", "High"])
].copy()

print("\nUnknown target labels excluded.")
print("\nUpdated target distribution:")
print(df["TARGET_RISK"].value_counts())

# Remove rows where the next reading is unavailable
df = df.dropna(subset=["TARGET_RISK"]).copy()

print("\nTarget created: Predicting next reading's risk")


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

numeric_features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",

    # Rolling averages
    "TEMPERATURE_C_ROLLING_MEAN_4",
    "HUMIDITY_PCT_ROLLING_MEAN_4",
    "VIBRATION_G_ROLLING_MEAN_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",

    # Rolling standard deviations
    "TEMPERATURE_C_ROLLING_STD_4",
    "HUMIDITY_PCT_ROLLING_STD_4",
    "VIBRATION_G_ROLLING_STD_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_STD_4",

    # Previous readings and changes
    "TEMPERATURE_C_PREVIOUS",
    "TEMPERATURE_C_CHANGE",
    "HUMIDITY_PCT_PREVIOUS",
    "HUMIDITY_PCT_CHANGE",
    "VIBRATION_G_PREVIOUS",
    "VIBRATION_G_CHANGE",

    # Time features
    "HOUR",
    "DAY_OF_WEEK"
]

categorical_features = [
    "SENSOR_STATUS"
]

required_columns = (
    numeric_features
    + categorical_features
    + ["TIMESTAMP", "RISK_LABEL", "TARGET_RISK"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

X = df[numeric_features + categorical_features].copy()
y = df["TARGET_RISK"].copy()


# ============================================================
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

# Preserve chronological order instead of random splitting
unique_timestamps = np.sort(df["TIMESTAMP"].unique())

split_index = int(len(unique_timestamps) * 0.80)

split_timestamp = unique_timestamps[split_index]

train_mask = df["TIMESTAMP"] < split_timestamp
test_mask = df["TIMESTAMP"] >= split_timestamp

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining risk distribution:")
print(y_train.value_counts())

print("\nTesting risk distribution:")
print(y_test.value_counts())


# ============================================================
# 6. DATA PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")
print("\nModel configuration:")
print(classifier.get_params())

model.fit(X_train, y_train)

print("\nTraining completed successfully!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print(f"Macro F1: {macro_f1:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print("\nActual test distribution:")
print(y_test.value_counts())

print("\nPredicted test distribution:")
print(pd.Series(y_pred).value_counts())

print("\nPredicted percentages:")

print(
    pd.Series(y_pred)
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(f"\nModel location: {MODEL_PATH}")


