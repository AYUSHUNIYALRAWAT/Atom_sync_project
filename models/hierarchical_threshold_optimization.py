
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
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
    BASE_DIR / "models" / "hierarchical_threshold_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD MODELS
# ============================================================

print("=" * 70)
print("ATMOSYNC HIERARCHICAL THRESHOLD OPTIMIZATION")
print("=" * 70)

stage1_bundle = joblib.load(STAGE1_PATH)
stage2_bundle = joblib.load(STAGE2_PATH)


def extract_model(bundle):
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle


stage1_model = extract_model(stage1_bundle)
stage2_model = extract_model(stage2_bundle)

# The original multiclass model used alphabetical label encoding.
stage1_encoder = stage1_bundle.get("label_encoder")

if stage1_encoder is None:
    stage1_encoder = LabelEncoder()
    stage1_encoder.fit(["High", "Low", "Medium"])

stage2_encoder = stage2_bundle["label_encoder"]

numeric_features = stage2_bundle["numeric_features"]
categorical_features = stage2_bundle["categorical_features"]

feature_columns = numeric_features + categorical_features

stage2_medium_weight = stage2_bundle.get(
    "medium_weight", 3.0
)

print("\nBoth models loaded.")
print("Stage 2 Medium weight:", stage2_medium_weight)


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
    df["TARGET_RISK"].isin(["High", "Low", "Medium"])
].copy()

df["TARGET_TIMESTAMP"] = pd.to_datetime(
    df["TARGET_TIMESTAMP"]
)

print("\nDataset:", len(df))


# ============================================================
# 4. RECREATE CHRONOLOGICAL SPLIT
# ============================================================

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

print("\nChronological split:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# ============================================================
# 5. PREPARE FEATURES AND TARGETS
# ============================================================

X_train = train_df[feature_columns].copy()
X_val = validation_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

y_train_text = train_df["TARGET_RISK"].to_numpy()
y_val_text = validation_df["TARGET_RISK"].to_numpy()
y_test_text = test_df["TARGET_RISK"].to_numpy()

y_train_stage1 = stage1_encoder.transform(y_train_text)
y_val_stage1 = stage1_encoder.transform(y_val_text)
y_test_stage1 = stage1_encoder.transform(y_test_text)

high_index = list(stage1_encoder.classes_).index("High")
low_index = list(stage1_encoder.classes_).index("Low")
medium_index = list(stage1_encoder.classes_).index("Medium")

stage2_low_index = list(stage2_encoder.classes_).index("Low")
stage2_medium_index = list(stage2_encoder.classes_).index("Medium")


# ============================================================
# 6. TRAIN STAGE 1 ON TRAINING DATA ONLY
# ============================================================

print("\nTraining Stage 1 on training data...")

stage1_weights = {
    "High": 2.5,
    "Low": 1.0,
    "Medium": 1.75
}

sample_weights_stage1 = np.array([
    stage1_weights[label]
    for label in y_train_text
])

stage1_model.fit(
    X_train,
    y_train_stage1,
    classifier__sample_weight=sample_weights_stage1
)

print("Stage 1 training completed.")


# ============================================================
# 7. TRAIN STAGE 2 ON LOW AND MEDIUM ONLY
# ============================================================

print("\nTraining Stage 2 on Low and Medium observations...")

train_lm_mask = train_df["TARGET_RISK"].isin(
    ["Low", "Medium"]
).to_numpy()

X_train_lm = X_train.loc[train_lm_mask]

y_train_lm_text = y_train_text[train_lm_mask]

y_train_lm = stage2_encoder.transform(
    y_train_lm_text
)

sample_weights_stage2 = np.where(
    y_train_lm == stage2_medium_index,
    stage2_medium_weight,
    1.0
)

stage2_model.fit(
    X_train_lm,
    y_train_lm,
    classifier__sample_weight=sample_weights_stage2
)

print("Stage 2 training completed.")


# ============================================================
# 8. GENERATE VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

stage1_val_probabilities = stage1_model.predict_proba(X_val)

stage1_classes = np.asarray(stage1_model.classes_)

if np.issubdtype(stage1_classes.dtype, np.number):
    stage1_probability_labels = stage1_encoder.inverse_transform(
        stage1_classes.astype(int)
    )
else:
    stage1_probability_labels = stage1_classes.astype(str)


def get_probability(probabilities, labels, target):
    index = np.where(labels == target)[0][0]
    return probabilities[:, index]


high_probabilities_val = get_probability(
    stage1_val_probabilities,
    stage1_probability_labels,
    "High"
)

stage2_val_probabilities = np.zeros(len(X_val))

val_lm_mask = np.isin(
    stage1_val_probabilities.argmax(axis=1),
    np.arange(len(stage1_probability_labels))
)

# Stage 2 probabilities are calculated for every validation row.
# This lets us refine all non-High decisions for each tested threshold.

stage2_val_all = stage2_model.predict_proba(X_val)

stage2_probability_labels = np.asarray(stage2_model.classes_)

if np.issubdtype(stage2_probability_labels.dtype, np.number):
    stage2_probability_labels = stage2_encoder.inverse_transform(
        stage2_probability_labels.astype(int)
    )
else:
    stage2_probability_labels = stage2_probability_labels.astype(str)

medium_probabilities_val = get_probability(
    stage2_val_all,
    stage2_probability_labels,
    "Medium"
)

print("Validation probabilities generated.")


# ============================================================
# 9. DEFINE HIERARCHICAL PREDICTION FUNCTION
# ============================================================

def hierarchical_predict(
    high_probabilities,
    medium_probabilities,
    high_threshold,
    medium_threshold
):

    predictions = np.where(
        high_probabilities >= high_threshold,
        "High",
        np.where(
            medium_probabilities >= medium_threshold,
            "Medium",
            "Low"
        )
    )

    return predictions


# ============================================================
# 10. OPTIMIZE BOTH THRESHOLDS ON VALIDATION
# ============================================================

high_thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50
]

medium_thresholds = [
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60
]

trial_results = []

best_macro_f1 = -1
best_high_threshold = None
best_medium_threshold = None

print("\nTesting hierarchical threshold combinations...")

for high_threshold in high_thresholds:

    for medium_threshold in medium_thresholds:

        predictions = hierarchical_predict(
            high_probabilities_val,
            medium_probabilities_val,
            high_threshold,
            medium_threshold
        )

        macro_f1 = f1_score(
            y_val_text,
            predictions,
            labels=["High", "Low", "Medium"],
            average="macro",
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val_text,
            predictions
        )

        report = classification_report(
            y_val_text,
            predictions,
            labels=["High", "Low", "Medium"],
            output_dict=True,
            zero_division=0
        )

        high_recall = report["High"]["recall"]
        medium_recall = report["Medium"]["recall"]

        trial_results.append({
            "high_threshold": high_threshold,
            "medium_threshold": medium_threshold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "high_recall": high_recall,
            "medium_recall": medium_recall
        })

        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1
            best_high_threshold = high_threshold
            best_medium_threshold = medium_threshold


# ============================================================
# 11. SAVE VALIDATION RESULTS
# ============================================================

trials_df = pd.DataFrame(trial_results)

trials_df = trials_df.sort_values(
    "macro_f1",
    ascending=False
)

trials_path = OUTPUT_DIR / "hierarchical_threshold_trials.csv"

trials_df.to_csv(
    trials_path,
    index=False
)

print("\n" + "=" * 70)
print("TOP 10 VALIDATION COMBINATIONS")
print("=" * 70)

print(
    trials_df.head(10).to_string(index=False)
)

print("\nSelected High threshold:", best_high_threshold)
print("Selected Medium threshold:", best_medium_threshold)
print("Best validation Macro F1:", round(best_macro_f1, 4))


# ============================================================
# 12. RETRAIN BOTH MODELS ON TRAIN + VALIDATION
# ============================================================

print("\nRetraining both models on development data...")

X_development = pd.concat(
    [X_train, X_val],
    ignore_index=True
)

y_development_text = np.concatenate(
    [y_train_text, y_val_text]
)

y_development_stage1 = stage1_encoder.transform(
    y_development_text
)

development_weights_stage1 = np.array([
    stage1_weights[label]
    for label in y_development_text
])

stage1_model.fit(
    X_development,
    y_development_stage1,
    classifier__sample_weight=development_weights_stage1
)

development_lm_mask = np.isin(
    y_development_text,
    ["Low", "Medium"]
)

X_development_lm = X_development.loc[
    development_lm_mask
]

y_development_lm = stage2_encoder.transform(
    y_development_text[development_lm_mask]
)

development_weights_stage2 = np.where(
    y_development_lm == stage2_medium_index,
    stage2_medium_weight,
    1.0
)

stage2_model.fit(
    X_development_lm,
    y_development_lm,
    classifier__sample_weight=development_weights_stage2
)

print("Final model retraining completed.")


# ============================================================
# 13. FINAL TEST PROBABILITIES
# ============================================================

print("\nGenerating final test probabilities...")

stage1_test_probabilities = stage1_model.predict_proba(X_test)

high_probabilities_test = get_probability(
    stage1_test_probabilities,
    stage1_probability_labels,
    "High"
)

stage2_test_probabilities = stage2_model.predict_proba(X_test)

medium_probabilities_test = get_probability(
    stage2_test_probabilities,
    stage2_probability_labels,
    "Medium"
)


# ============================================================
# 14. FINAL TEST PREDICTIONS
# ============================================================

final_predictions = hierarchical_predict(
    high_probabilities_test,
    medium_probabilities_test,
    best_high_threshold,
    best_medium_threshold
)


# ============================================================
# 15. FINAL TEST EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test_text,
    final_predictions
)

macro_f1 = f1_score(
    y_test_text,
    final_predictions,
    labels=["High", "Low", "Medium"],
    average="macro",
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL HIERARCHICAL TEST RESULTS")
print("=" * 70)

print("High threshold:", best_high_threshold)
print("Medium threshold:", best_medium_threshold)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test_text,
        final_predictions,
        labels=["High", "Low", "Medium"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test_text,
    final_predictions,
    labels=["High", "Low", "Medium"]
)

print("Class order: High, Low, Medium")
print(cm)


# ============================================================
# 16. SAVE TEST RESULTS
# ============================================================

results = pd.DataFrame({
    "CONTAINER_ID": test_df["CONTAINER_ID"].to_numpy(),
    "TARGET_TIMESTAMP": test_df["TARGET_TIMESTAMP"].to_numpy(),
    "ACTUAL_RISK": y_test_text,
    "PREDICTED_RISK": final_predictions,
    "HIGH_PROBABILITY": high_probabilities_test,
    "MEDIUM_PROBABILITY": medium_probabilities_test,
    "HIGH_THRESHOLD": best_high_threshold,
    "MEDIUM_THRESHOLD": best_medium_threshold
})

results.to_csv(
    OUTPUT_DIR / "hierarchical_optimized_test_results.csv",
    index=False
)


# ============================================================
# 17. SAVE OPTIMIZED MODELS
# ============================================================

joblib.dump(
    {
        "stage1_model": stage1_model,
        "stage1_encoder": stage1_encoder,
        "stage2_model": stage2_model,
        "stage2_encoder": stage2_encoder,
        "high_threshold": best_high_threshold,
        "medium_threshold": best_medium_threshold,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1
    },
    OUTPUT_DIR / "hierarchical_optimized_model.pkl"
)

print("\nSaved threshold trials:", trials_path)

print(
    "Saved test results:",
    OUTPUT_DIR / "hierarchical_optimized_test_results.csv"
)

print(
    "Saved optimized model:",
    OUTPUT_DIR / "hierarchical_optimized_model.pkl"
)

print("\n" + "=" * 70)
print("HIERARCHICAL THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 70)

