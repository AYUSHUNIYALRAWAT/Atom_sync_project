
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_risk_model.pkl"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ATMOSYNC XGBOOST RISK PREDICTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

print("\nDataset loaded:", len(df))


# ============================================================
# 3. CREATE NEXT-READING TARGET
# ============================================================

df["TARGET_RISK"] = (
    df.groupby("CONTAINER_ID")["RISK_LABEL"].shift(-1)
)

df = df.dropna(subset=["TARGET_RISK"]).copy()

df = df[
    df["TARGET_RISK"].isin(["Low", "Medium", "High"])
].copy()

print("\nTarget distribution:")
print(df["TARGET_RISK"].value_counts())


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

numeric_features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",

    "TEMPERATURE_C_ROLLING_MEAN_4",
    "HUMIDITY_PCT_ROLLING_MEAN_4",
    "VIBRATION_G_ROLLING_MEAN_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",

    "TEMPERATURE_C_ROLLING_STD_4",
    "HUMIDITY_PCT_ROLLING_STD_4",
    "VIBRATION_G_ROLLING_STD_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_STD_4",

    "TEMPERATURE_C_PREVIOUS",
    "TEMPERATURE_C_CHANGE",
    "HUMIDITY_PCT_PREVIOUS",
    "HUMIDITY_PCT_CHANGE",
    "VIBRATION_G_PREVIOUS",
    "VIBRATION_G_CHANGE",

    "HOUR",
    "DAY_OF_WEEK"
]

categorical_features = [
    "SENSOR_STATUS"
]

required_columns = (
    numeric_features
    + categorical_features
    + ["TIMESTAMP", "TARGET_RISK"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

X = df[numeric_features + categorical_features].copy()
y = df["TARGET_RISK"].copy()


# ============================================================
# 5. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget encoding:")
for index, label in enumerate(label_encoder.classes_):
    print(label, "=", index)


# ============================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

unique_timestamps = np.sort(
    df["TIMESTAMP"].unique()
)

split_index = int(len(unique_timestamps) * 0.80)

split_timestamp = unique_timestamps[split_index]

train_mask = df["TIMESTAMP"] < split_timestamp
test_mask = df["TIMESTAMP"] >= split_timestamp

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y_encoded[train_mask.to_numpy()]
y_test = y_encoded[test_mask.to_numpy()]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 8. XGBOOST MODEL
# ============================================================

classifier = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=2,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 10. PREDICTIONS
# ============================================================

y_pred_encoded = model.predict(X_test)

y_pred = label_encoder.inverse_transform(
    y_pred_encoded.astype(int)
)

y_test_labels = label_encoder.inverse_transform(y_test)


# ============================================================
# 11. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test_labels,
    y_pred
)

macro_f1 = f1_score(
    y_test_labels,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("XGBOOST MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test_labels,
        y_pred,
        labels=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_labels,
        y_pred,
        labels=label_encoder.classes_
    )
)


# ============================================================
# 12. PREDICTION DISTRIBUTION
# ============================================================

print("\nPrediction distribution:")

print(pd.Series(y_pred).value_counts())

print("\nPrediction percentages:")

print(
    pd.Series(y_pred)
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    },
    MODEL_PATH
)

print("\nXGBoost model saved:")
print(MODEL_PATH)
