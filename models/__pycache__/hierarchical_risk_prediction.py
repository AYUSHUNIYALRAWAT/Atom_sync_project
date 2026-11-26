
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

DATA_PATH = (
    BASE_DIR / "data" / "analytics"
    / "weather_features.csv"
)

STAGE1_PATH = (
    BASE_DIR / "models" / "saved_models"
    / "xgboost_custom_weights_model.pkl"
)

STAGE2_PATH = (
    BASE_DIR / "models" / "low_medium_threshold_analysis"
    / "low_medium_threshold_model.pkl"
)

OUTPUT_DIR = (
    BASE_DIR / "models" / "hierarchical_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD EXISTING MODELS
# ============================================================

print("=" * 70)
print("ATMOSYNC HIERARCHICAL RISK PREDICTION")
print("=" * 70)

stage1_bundle = joblib.load(STAGE1_PATH)
stage2_bundle = joblib.load(STAGE2_PATH)


def extract_model(bundle):
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle


stage1_model = extract_model(stage1_bundle)
stage2_model = extract_model(stage2_bundle)

stage1_encoder = (
    stage1_bundle.get("label_encoder")
    if isinstance(stage1_bundle, dict)
    else None
)

stage2_encoder = stage2_bundle["label_encoder"]

threshold = stage2_bundle.get("threshold", 0.50)

numeric_features = stage2_bundle["numeric_features"]
categorical_features = stage2_bundle["categorical_features"]

feature_columns = numeric_features + categorical_features

print("\nBoth models loaded successfully.")
print("Stage 1: Multiclass XGBoost")
print("Stage 2: Low-vs-Medium binary XGBoost")
print("Stage 2 threshold:", threshold)


# ============================================================
# 3. LOAD DATA AND CREATE NEXT-READING TARGET
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

print("\nDataset loaded:", len(df))


# ============================================================
# 4. RECREATE SAME CHRONOLOGICAL TEST PERIOD
# ============================================================

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)

test_start_time = unique_timestamps[test_start_index]

test_df = df[
    df["TARGET_TIMESTAMP"] >= test_start_time
].copy()

print("\nTest period starts:", test_start_time)
print("Test observations:", len(test_df))


# ============================================================
# 5. PREPARE TEST FEATURES
# ============================================================

X_test = test_df[feature_columns].copy()

y_actual = test_df["TARGET_RISK"].to_numpy()

print("\nActual test distribution:")
print(pd.Series(y_actual).value_counts())


# ============================================================
# 6. STAGE 1: MULTICLASS PREDICTION
# ============================================================

print("\nRunning Stage 1 multiclass model...")

stage1_raw_predictions = stage1_model.predict(X_test)

# Decode numeric predictions using the saved encoder.
if stage1_encoder is not None:
    if np.issubdtype(
        np.asarray(stage1_raw_predictions).dtype,
        np.number
    ):
        stage1_predictions = (
            stage1_encoder.inverse_transform(
                np.asarray(stage1_raw_predictions).astype(int)
            )
        )
    else:
        stage1_predictions = np.asarray(
            stage1_raw_predictions
        ).astype(str)

else:
    # LabelEncoder sorts classes alphabetically:
    # High = 0, Low = 1, Medium = 2.
    if np.issubdtype(
        np.asarray(stage1_raw_predictions).dtype,
        np.number
    ):
        stage1_classes = np.array(
            ["High", "Low", "Medium"]
        )

        stage1_predictions = stage1_classes[
            np.asarray(stage1_raw_predictions).astype(int)
        ]
    else:
        stage1_predictions = np.asarray(
            stage1_raw_predictions
        ).astype(str)

print("Stage 1 predictions completed.")


# ============================================================
# 7. STAGE 2: REFINE LOW AND MEDIUM PREDICTIONS
# ============================================================

print("\nRunning Stage 2 binary model...")

hierarchical_predictions = stage1_predictions.copy()

# Only observations predicted as Low or Medium
# by Stage 1 are passed to Stage 2.

refinement_mask = np.isin(
    stage1_predictions,
    ["Low", "Medium"]
)

refinement_indices = np.where(
    refinement_mask
)[0]

print(
    "Observations sent to Stage 2:",
    len(refinement_indices)
)

print(
    "High predictions retained from Stage 1:",
    np.sum(stage1_predictions == "High")
)

if len(refinement_indices) > 0:

    X_refinement = X_test.iloc[
        refinement_indices
    ]

    binary_probabilities = (
        stage2_model.predict_proba(X_refinement)
    )

    medium_index = list(
        stage2_encoder.classes_
    ).index("Medium")

    medium_probabilities = (
        binary_probabilities[:, medium_index]
    )

    refined_predictions = np.where(
        medium_probabilities >= threshold,
        "Medium",
        "Low"
    )

    hierarchical_predictions[
        refinement_indices
    ] = refined_predictions

print("Stage 2 refinement completed.")


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, actual, predictions):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(
        actual,
        predictions
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        labels=["High", "Low", "Medium"],
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predictions,
            labels=["High", "Low", "Medium"],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        actual,
        predictions,
        labels=["High", "Low", "Medium"]
    )

    print("\nConfusion Matrix")
    print("Class order: High, Low, Medium")
    print(cm)

    return accuracy, macro_f1, cm


# ============================================================
# 9. BASELINE MULTICLASS EVALUATION
# ============================================================

baseline_accuracy, baseline_macro_f1, baseline_cm = (
    evaluate_model(
        "BASELINE MULTICLASS MODEL",
        y_actual,
        stage1_predictions
    )
)


# ============================================================
# 10. HIERARCHICAL MODEL EVALUATION
# ============================================================

hierarchical_accuracy, hierarchical_macro_f1, hierarchical_cm = (
    evaluate_model(
        "HIERARCHICAL RISK MODEL",
        y_actual,
        hierarchical_predictions
    )
)


# ============================================================
# 11. COMPARE BOTH MODELS
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Baseline Multiclass XGBoost",
        "Hierarchical XGBoost"
    ],
    "Accuracy": [
        baseline_accuracy,
        hierarchical_accuracy
    ],
    "Macro_F1": [
        baseline_macro_f1,
        hierarchical_macro_f1
    ]
})

comparison["Macro_F1_Change"] = (
    comparison["Macro_F1"] - baseline_macro_f1
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "CONTAINER_ID": test_df["CONTAINER_ID"].to_numpy(),
    "TARGET_TIMESTAMP": test_df["TARGET_TIMESTAMP"].to_numpy(),
    "ACTUAL_RISK": y_actual,
    "STAGE1_PREDICTION": stage1_predictions,
    "HIERARCHICAL_PREDICTION": hierarchical_predictions
})

results.to_csv(
    OUTPUT_DIR / "hierarchical_test_predictions.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nSaved predictions:")
print(OUTPUT_DIR / "hierarchical_test_predictions.csv")

print("\nSaved comparison:")
print(OUTPUT_DIR / "model_comparison.csv")

print("\n" + "=" * 70)
print("HIERARCHICAL EXPERIMENT COMPLETED")
print("=" * 70)
