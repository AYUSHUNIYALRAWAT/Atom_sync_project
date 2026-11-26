
from pathlib import Path
from itertools import product

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

MODEL_PATH = MODEL_DIR / "xgboost_threshold_optimized_model.pkl"
RESULTS_PATH = MODEL_DIR / "xgboost_threshold_results.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("ATMOSYNC XGBOOST PROBABILITY THRESHOLD OPTIMIZATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)


# ============================================================
# 3. CREATE NEXT-READING TARGET
# ============================================================

grouped = df.groupby("CONTAINER_ID")

df["TARGET_RISK"] = grouped["RISK_LABEL"].shift(-1)
df["TARGET_TIMESTAMP"] = grouped["TIMESTAMP"].shift(-1)

df = df.dropna(
    subset=["TARGET_RISK", "TARGET_TIMESTAMP"]
).copy()

df = df[
    df["TARGET_RISK"].isin(["Low", "Medium", "High"])
].copy()

df["TARGET_TIMESTAMP"] = pd.to_datetime(
    df["TARGET_TIMESTAMP"]
)

print("\nDataset loaded:", len(df))
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
    "DAY_OF_WEEK",

    "TEMP_DEV_LAG_2",
    "TEMP_DEV_LAG_3",
    "TEMP_DEV_ROLLING_MAX_4",
    "TEMP_DEV_ROLLING_MIN_4",
    "TEMP_DEV_ABSOLUTE_CHANGE",
    "TEMP_DEV_ACCELERATION"
]

categorical_features = ["SENSOR_STATUS"]

required_columns = (
    numeric_features
    + categorical_features
    + ["TARGET_TIMESTAMP", "TARGET_RISK"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

X = df[numeric_features + categorical_features].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["TARGET_RISK"])

print("\nTarget encoding:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} = {i}")


# ============================================================
# 5. CHRONOLOGICAL DATA SPLIT
# ============================================================

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)
test_start_time = unique_timestamps[test_start_index]

development_mask = (
    df["TARGET_TIMESTAMP"] < test_start_time
)

test_mask = (
    df["TARGET_TIMESTAMP"] >= test_start_time
)

development_df = df.loc[development_mask].copy()
test_df = df.loc[test_mask].copy()

development_timestamps = np.sort(
    development_df["TARGET_TIMESTAMP"].unique()
)

validation_start_index = int(
    len(development_timestamps) * 0.80
)

validation_start_time = (
    development_timestamps[validation_start_index]
)

train_mask = (
    development_df["TARGET_TIMESTAMP"] < validation_start_time
)

validation_mask = (
    development_df["TARGET_TIMESTAMP"] >= validation_start_time
)

X_train = X.loc[development_df.index[train_mask]]
X_val = X.loc[development_df.index[validation_mask]]
X_test = X.loc[test_df.index]

y_series = pd.Series(y, index=df.index)

y_train = y_series.loc[X_train.index].to_numpy()
y_val = y_series.loc[X_val.index].to_numpy()
y_test = y_series.loc[X_test.index].to_numpy()

print("\nChronological split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# ============================================================
# 6. PREPROCESSING
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
# 7. CREATE CUSTOM-WEIGHTED MODEL
# ============================================================

best_weights = {
    "High": 2.5,
    "Low": 1.0,
    "Medium": 1.75
}

def create_model():

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),

        n_estimators=300,
        max_depth=4,
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


# ============================================================
# 8. TRAIN USING TRAINING DATA ONLY
# ============================================================

print("\nTraining model for threshold selection...")

train_labels = label_encoder.inverse_transform(y_train)

train_weights = np.array([
    best_weights[label] for label in train_labels
])

model = create_model()

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=train_weights
)

print("Training completed.")


# ============================================================
# 9. GENERATE VALIDATION PROBABILITIES
# ============================================================

val_probabilities = model.predict_proba(X_val)

# LabelEncoder order is High, Low, Medium.
class_names = list(label_encoder.classes_)

high_index = class_names.index("High")
low_index = class_names.index("Low")
medium_index = class_names.index("Medium")

y_val_labels = label_encoder.inverse_transform(y_val)


# ============================================================
# 10. THRESHOLD PREDICTION FUNCTION
# ============================================================

def apply_thresholds(
    probabilities,
    high_threshold,
    medium_threshold
):

    predictions = []

    for row in probabilities:

        eligible = []

        if row[high_index] >= high_threshold:
            eligible.append(high_index)

        if row[medium_index] >= medium_threshold:
            eligible.append(medium_index)

        # Low is the fallback class
        eligible.append(low_index)

        # Choose the eligible class with the highest probability
        selected_class = max(
            eligible,
            key=lambda class_index: row[class_index]
        )

        predictions.append(selected_class)

    return np.array(predictions)


# ============================================================
# 11. SEARCH THRESHOLD COMBINATIONS
# ============================================================

threshold_values = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]

threshold_combinations = list(
    product(threshold_values, repeat=2)
)

results = []

best_macro_f1 = -1
best_thresholds = None

print("\nStarting threshold optimization...")

for trial, (high_threshold, medium_threshold) in enumerate(
    threshold_combinations,
    start=1
):

    predictions = apply_thresholds(
        val_probabilities,
        high_threshold,
        medium_threshold
    )

    predicted_labels = label_encoder.inverse_transform(
        predictions
    )

    accuracy = accuracy_score(
        y_val_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        y_val_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_val_labels,
        predicted_labels,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    high_recall = report["High"]["recall"]
    medium_recall = report["Medium"]["recall"]
    low_recall = report["Low"]["recall"]

    results.append({
        "trial": trial,
        "high_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "high_recall": high_recall,
        "medium_recall": medium_recall,
        "low_recall": low_recall
    })

    print(
        f"Trial {trial}/81 | "
        f"High: {high_threshold:.2f} | "
        f"Medium: {medium_threshold:.2f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_thresholds = {
            "High": high_threshold,
            "Medium": medium_threshold
        }


# ============================================================
# 12. SELECT BEST THRESHOLDS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_macro_f1",
    ascending=False
)

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 70)

print(results_df.head(15).to_string(index=False))

print("\nBest thresholds:")
print(best_thresholds)

print("\nBest validation Macro F1:", round(best_macro_f1, 4))


# ============================================================
# 13. RETRAIN USING TRAIN + VALIDATION
# ============================================================

print("\nRetraining model on training + validation data...")

X_development = pd.concat([X_train, X_val])
y_development = np.concatenate([y_train, y_val])

development_labels = label_encoder.inverse_transform(
    y_development
)

development_weights = np.array([
    best_weights[label] for label in development_labels
])

final_model = create_model()

final_model.fit(
    X_development,
    y_development,
    classifier__sample_weight=development_weights
)

print("Final model training completed.")


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

test_probabilities = final_model.predict_proba(X_test)

test_predictions = apply_thresholds(
    test_probabilities,
    best_thresholds["High"],
    best_thresholds["Medium"]
)

y_test_labels = label_encoder.inverse_transform(y_test)

test_pred_labels = label_encoder.inverse_transform(
    test_predictions
)

test_accuracy = accuracy_score(
    y_test_labels,
    test_pred_labels
)

test_macro_f1 = f1_score(
    y_test_labels,
    test_pred_labels,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL THRESHOLD-OPTIMIZED XGBOOST RESULTS")
print("=" * 70)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test_labels,
        test_pred_labels,
        labels=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_labels,
        test_pred_labels,
        labels=class_names
    )
)

print("\nPrediction Distribution:")
print(pd.Series(test_pred_labels).value_counts())


# ============================================================
# 15. SAVE MODEL AND THRESHOLDS
# ============================================================

joblib.dump(
    {
        "model": final_model,
        "label_encoder": label_encoder,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "class_weights": best_weights,
        "decision_thresholds": best_thresholds,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1
    },
    MODEL_PATH
)

results_df.to_csv(RESULTS_PATH, index=False)

print("\nOptimized model saved:")
print(MODEL_PATH)

print("\nThreshold results saved:")
print(RESULTS_PATH)

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 70)
