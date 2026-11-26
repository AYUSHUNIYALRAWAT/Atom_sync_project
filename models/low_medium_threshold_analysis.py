
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "saved_models"
    / "low_medium_binary_model.pkl"
)

DATA_PATH = (
    BASE_DIR / "data" / "analytics"
    / "weather_features.csv"
)

OUTPUT_DIR = BASE_DIR / "models" / "low_medium_threshold_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_PATH = OUTPUT_DIR / "threshold_trials.csv"
RESULTS_PATH = OUTPUT_DIR / "threshold_test_results.csv"


# ============================================================
# 2. LOAD DATA AND SAVED MODEL CONFIGURATION
# ============================================================

print("=" * 70)
print("ATMOSYNC LOW VS MEDIUM THRESHOLD OPTIMIZATION")
print("=" * 70)

saved = joblib.load(MODEL_PATH)

best_weight = saved["best_medium_weight"]

numeric_features = saved["numeric_features"]
categorical_features = saved["categorical_features"]

feature_columns = numeric_features + categorical_features

model = saved["model"]

label_encoder = saved["label_encoder"]

low_index = list(label_encoder.classes_).index("Low")
medium_index = list(label_encoder.classes_).index("Medium")

print("\nLoaded binary model.")
print("Selected Medium weight:", best_weight)


# ============================================================
# 3. RECREATE THE ORIGINAL CHRONOLOGICAL SPLIT
# ============================================================

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

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

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)

test_start_time = unique_timestamps[test_start_index]

development_df = df[
    df["TARGET_TIMESTAMP"] < test_start_time
].copy()

test_df = df[
    df["TARGET_TIMESTAMP"] >= test_start_time
].copy()

development_timestamps = np.sort(
    development_df["TARGET_TIMESTAMP"].unique()
)

validation_start_index = int(
    len(development_timestamps) * 0.80
)

validation_start_time = (
    development_timestamps[validation_start_index]
)

train_df = development_df[
    development_df["TARGET_TIMESTAMP"] < validation_start_time
].copy()

validation_df = development_df[
    development_df["TARGET_TIMESTAMP"] >= validation_start_time
].copy()


# ============================================================
# 4. FILTER LOW AND MEDIUM
# ============================================================

allowed_classes = ["Low", "Medium"]

train_df = train_df[
    train_df["TARGET_RISK"].isin(allowed_classes)
].copy()

validation_df = validation_df[
    validation_df["TARGET_RISK"].isin(allowed_classes)
].copy()

test_df = test_df[
    test_df["TARGET_RISK"].isin(allowed_classes)
].copy()

print("\nDataset sizes:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# ============================================================
# 5. PREPARE FEATURES AND LABELS
# ============================================================

X_train = train_df[feature_columns]
X_val = validation_df[feature_columns]
X_test = test_df[feature_columns]

y_train = label_encoder.transform(
    train_df["TARGET_RISK"]
)

y_val = label_encoder.transform(
    validation_df["TARGET_RISK"]
)

y_test = label_encoder.transform(
    test_df["TARGET_RISK"]
)


# ============================================================
# 6. TRAIN MODEL ON TRAINING DATA ONLY
# ============================================================

print("\nTraining model on training split only...")

sample_weights = np.where(
    y_train == medium_index,
    best_weight,
    1.0
)

model.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weights
)

print("Training completed.")


# ============================================================
# 7. GENERATE VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(X_val)

medium_probabilities = validation_probabilities[:, medium_index]

print("\nValidation probabilities generated.")


# ============================================================
# 8. TEST THRESHOLDS ON VALIDATION DATA
# ============================================================

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

trial_results = []

best_threshold = None
best_validation_f1 = -1

print("\nTesting thresholds...")

for threshold in thresholds:

    predictions = np.where(
        medium_probabilities >= threshold,
        medium_index,
        low_index
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    medium_precision = precision_score(
        y_val,
        predictions,
        pos_label=medium_index,
        zero_division=0
    )

    medium_recall = recall_score(
        y_val,
        predictions,
        pos_label=medium_index,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    trial_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "medium_precision": medium_precision,
        "medium_recall": medium_recall
    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Medium Precision: {medium_precision:.4f} | "
        f"Medium Recall: {medium_recall:.4f}"
    )

    if macro_f1 > best_validation_f1:

        best_validation_f1 = macro_f1
        best_threshold = threshold


# ============================================================
# 9. SAVE VALIDATION RESULTS
# ============================================================

trials_df = pd.DataFrame(trial_results)

trials_df = trials_df.sort_values(
    "macro_f1",
    ascending=False
)

trials_df.to_csv(
    TRIALS_PATH,
    index=False
)

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD RESULTS")
print("=" * 70)

print(trials_df.to_string(index=False))

print("\nSelected threshold:", best_threshold)
print("Best validation Macro F1:", round(best_validation_f1, 4))


# ============================================================
# 10. RETRAIN MODEL ON TRAIN + VALIDATION
# ============================================================

print("\nRetraining on training + validation data...")

X_development = pd.concat(
    [X_train, X_val],
    ignore_index=True
)

y_development = np.concatenate(
    [y_train, y_val]
)

development_weights = np.where(
    y_development == medium_index,
    best_weight,
    1.0
)

model.fit(
    X_development,
    y_development,
    classifier__sample_weight=development_weights
)

print("Final model retraining completed.")


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

test_probabilities = model.predict_proba(X_test)

test_medium_probabilities = test_probabilities[:, medium_index]

test_predictions = np.where(
    test_medium_probabilities >= best_threshold,
    medium_index,
    low_index
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

test_medium_precision = precision_score(
    y_test,
    test_predictions,
    pos_label=medium_index,
    zero_division=0
)

test_medium_recall = recall_score(
    y_test,
    test_predictions,
    pos_label=medium_index,
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Medium Precision: {test_medium_precision:.4f}")
print(f"Medium Recall: {test_medium_recall:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[low_index, medium_index],
        target_names=["Low", "Medium"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[low_index, medium_index]
)

print(cm)


# ============================================================
# 12. SAVE FINAL TEST RESULTS
# ============================================================

test_results = pd.DataFrame({
    "ACTUAL_RISK": label_encoder.inverse_transform(y_test),
    "PREDICTED_RISK": label_encoder.inverse_transform(
        test_predictions
    ),
    "PROBABILITY_LOW": test_probabilities[:, low_index],
    "PROBABILITY_MEDIUM": test_probabilities[:, medium_index],
    "SELECTED_THRESHOLD": best_threshold
})

test_results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# 13. SAVE THRESHOLD-OPTIMIZED MODEL
# ============================================================

model_save_path = (
    OUTPUT_DIR / "low_medium_threshold_model.pkl"
)

joblib.dump(
    {
        "model": model,
        "label_encoder": label_encoder,
        "threshold": best_threshold,
        "medium_weight": best_weight,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_medium_precision": test_medium_precision,
        "test_medium_recall": test_medium_recall
    },
    model_save_path
)

print("\nSaved validation trials:", TRIALS_PATH)
print("Saved test predictions:", RESULTS_PATH)
print("Saved optimized model:", model_save_path)

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 70)
