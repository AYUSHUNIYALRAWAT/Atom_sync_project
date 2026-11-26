
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
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
    BASE_DIR / "models" / "temporal_validation_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ATMOSYNC TEMPORAL MODEL VALIDATION")
print("=" * 70)


# ============================================================
# 2. LOAD SAVED MODELS
# ============================================================

stage1_bundle = joblib.load(STAGE1_PATH)
stage2_bundle = joblib.load(STAGE2_PATH)


def extract_model(bundle):
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle


stage1_original = extract_model(stage1_bundle)
stage2_original = extract_model(stage2_bundle)

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

# Keep the original hierarchical decision rule.
# Stage 1 High predictions are retained.
# All other predictions are refined by Stage 2.
MEDIUM_THRESHOLD = 0.50

print("\nModels loaded successfully.")


# ============================================================
# 3. LOAD DATA
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

missing_columns = [
    column for column in feature_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing feature columns: {missing_columns}"
    )

print("\nTotal observations:", len(df))


# ============================================================
# 4. RESERVE FINAL 20% AS UNTOUCHED TEST PERIOD
# ============================================================

all_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(all_timestamps) * 0.80)

test_start_time = all_timestamps[test_start_index]

development_df = df[
    df["TARGET_TIMESTAMP"] < test_start_time
].copy()

print("\nDevelopment observations:", len(development_df))
print("Final untouched test period starts:", test_start_time)


# ============================================================
# 5. CREATE THREE EXPANDING TEMPORAL FOLDS
# ============================================================

development_timestamps = np.sort(
    development_df["TARGET_TIMESTAMP"].unique()
)

n_timestamps = len(development_timestamps)

cut_40 = int(n_timestamps * 0.40)
cut_60 = int(n_timestamps * 0.60)
cut_80 = int(n_timestamps * 0.80)

fold_boundaries = [
    (cut_40, cut_60),
    (cut_60, cut_80),
    (cut_80, n_timestamps)
]

folds = []

for fold_number, (train_end, val_end) in enumerate(
    fold_boundaries, start=1
):

    train_end_time = development_timestamps[train_end]

    if val_end < n_timestamps:
        val_end_time = development_timestamps[val_end]
    else:
        val_end_time = None

    fold_train = development_df[
        development_df["TARGET_TIMESTAMP"] < train_end_time
    ].copy()

    if val_end_time is not None:
        fold_val = development_df[
            (development_df["TARGET_TIMESTAMP"] >= train_end_time)
            & (development_df["TARGET_TIMESTAMP"] < val_end_time)
        ].copy()
    else:
        fold_val = development_df[
            development_df["TARGET_TIMESTAMP"] >= train_end_time
        ].copy()

    folds.append((fold_number, fold_train, fold_val))

    print(
        f"\nFold {fold_number}: "
        f"Train={len(fold_train)}, "
        f"Validation={len(fold_val)}"
    )


# ============================================================
# 6. MODEL CONFIGURATION
# ============================================================

stage1_weights = {
    "High": 2.5,
    "Low": 1.0,
    "Medium": 1.75
}

stage2_medium_index = stage2_encoder.transform(
    ["Medium"]
)[0]

class_order = ["High", "Low", "Medium"]

fold_metrics = []
all_predictions = []


# ============================================================
# 7. RUN TEMPORAL VALIDATION
# ============================================================

for fold_number, train_df, val_df in folds:

    print("\n" + "=" * 70)
    print(f"RUNNING TEMPORAL FOLD {fold_number}")
    print("=" * 70)

    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()

    y_train_text = train_df["TARGET_RISK"].to_numpy()
    y_val_text = val_df["TARGET_RISK"].to_numpy()

    # --------------------------------------------------------
    # STAGE 1: MULTICLASS MODEL
    # --------------------------------------------------------

    model1 = clone(stage1_original)

    y_train_stage1 = stage1_encoder.transform(
        y_train_text
    )

    weights1 = np.array([
        stage1_weights[label]
        for label in y_train_text
    ])

    model1.fit(
        X_train,
        y_train_stage1,
        classifier__sample_weight=weights1
    )

    stage1_encoded_predictions = model1.predict(
        X_val
    ).astype(int)

    stage1_predictions = (
        stage1_encoder.inverse_transform(
            stage1_encoded_predictions
        )
    )

    # --------------------------------------------------------
    # BASELINE PREDICTIONS
    # --------------------------------------------------------

    baseline_predictions = stage1_predictions.copy()

    # --------------------------------------------------------
    # STAGE 2: BINARY LOW VS MEDIUM MODEL
    # --------------------------------------------------------

    train_lm_mask = np.isin(
        y_train_text,
        ["Low", "Medium"]
    )

    X_train_lm = X_train.loc[train_lm_mask]

    y_train_lm_text = y_train_text[train_lm_mask]

    y_train_lm = stage2_encoder.transform(
        y_train_lm_text
    )

    weights2 = np.where(
        y_train_lm == stage2_medium_index,
        stage2_medium_weight,
        1.0
    )

    model2 = clone(stage2_original)

    model2.fit(
        X_train_lm,
        y_train_lm,
        classifier__sample_weight=weights2
    )

    # --------------------------------------------------------
    # HIERARCHICAL PREDICTIONS
    # --------------------------------------------------------

    hierarchical_predictions = stage1_predictions.copy()

    refine_mask = np.isin(
        stage1_predictions,
        ["Low", "Medium"]
    )

    refine_indices = np.where(refine_mask)[0]

    if len(refine_indices) > 0:

        X_refine = X_val.iloc[refine_indices]

        probabilities = model2.predict_proba(
            X_refine
        )

        model2_classes = (
            model2.named_steps["classifier"].classes_
        )

        medium_probability_index = np.where(
            model2_classes == stage2_medium_index
        )[0][0]

        medium_probabilities = probabilities[
            :, medium_probability_index
        ]

        refined_predictions = np.where(
            medium_probabilities >= MEDIUM_THRESHOLD,
            "Medium",
            "Low"
        )

        hierarchical_predictions[
            refine_indices
        ] = refined_predictions

    # --------------------------------------------------------
    # EVALUATE BOTH MODELS
    # --------------------------------------------------------

    baseline_accuracy = accuracy_score(
        y_val_text,
        baseline_predictions
    )

    baseline_macro_f1 = f1_score(
        y_val_text,
        baseline_predictions,
        labels=class_order,
        average="macro",
        zero_division=0
    )

    hierarchical_accuracy = accuracy_score(
        y_val_text,
        hierarchical_predictions
    )

    hierarchical_macro_f1 = f1_score(
        y_val_text,
        hierarchical_predictions,
        labels=class_order,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_val_text,
        hierarchical_predictions,
        labels=class_order,
        output_dict=True,
        zero_division=0
    )

    fold_result = {
        "fold": fold_number,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "validation_start": val_df["TARGET_TIMESTAMP"].min(),
        "validation_end": val_df["TARGET_TIMESTAMP"].max(),

        "baseline_accuracy": baseline_accuracy,
        "baseline_macro_f1": baseline_macro_f1,

        "hierarchical_accuracy": hierarchical_accuracy,
        "hierarchical_macro_f1": hierarchical_macro_f1,

        "high_precision": report["High"]["precision"],
        "high_recall": report["High"]["recall"],
        "high_f1": report["High"]["f1-score"],

        "low_precision": report["Low"]["precision"],
        "low_recall": report["Low"]["recall"],
        "low_f1": report["Low"]["f1-score"],

        "medium_precision": report["Medium"]["precision"],
        "medium_recall": report["Medium"]["recall"],
        "medium_f1": report["Medium"]["f1-score"]
    }

    fold_metrics.append(fold_result)

    print("\nBaseline Accuracy:", round(baseline_accuracy, 4))
    print("Baseline Macro F1:", round(baseline_macro_f1, 4))

    print(
        "\nHierarchical Accuracy:",
        round(hierarchical_accuracy, 4)
    )

    print(
        "Hierarchical Macro F1:",
        round(hierarchical_macro_f1, 4)
    )

    print("\nHierarchical classification report:")

    print(
        classification_report(
            y_val_text,
            hierarchical_predictions,
            labels=class_order,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # SAVE ROW-LEVEL PREDICTIONS
    # --------------------------------------------------------

    fold_predictions = pd.DataFrame({
        "FOLD": fold_number,
        "CONTAINER_ID": val_df["CONTAINER_ID"].to_numpy(),
        "TARGET_TIMESTAMP": val_df["TARGET_TIMESTAMP"].to_numpy(),
        "ACTUAL_RISK": y_val_text,
        "BASELINE_PREDICTION": baseline_predictions,
        "HIERARCHICAL_PREDICTION": hierarchical_predictions
    })

    all_predictions.append(fold_predictions)


# ============================================================
# 8. SAVE FOLD METRICS
# ============================================================

metrics_df = pd.DataFrame(fold_metrics)

metrics_df.to_csv(
    OUTPUT_DIR / "temporal_fold_metrics.csv",
    index=False
)

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION SUMMARY")
print("=" * 70)

print(metrics_df.to_string(index=False))


# ============================================================
# 9. CALCULATE OVERALL STABILITY
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL MACRO F1 STABILITY")
print("=" * 70)

print(
    "Mean Macro F1:",
    round(metrics_df["hierarchical_macro_f1"].mean(), 4)
)

print(
    "Standard deviation:",
    round(metrics_df["hierarchical_macro_f1"].std(), 4)
)

print(
    "Minimum Macro F1:",
    round(metrics_df["hierarchical_macro_f1"].min(), 4)
)

print(
    "Maximum Macro F1:",
    round(metrics_df["hierarchical_macro_f1"].max(), 4)
)


# ============================================================
# 10. SAVE ALL VALIDATION PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    OUTPUT_DIR / "temporal_predictions.csv",
    index=False
)


# ============================================================
# 11. CONTAINER-WISE ERROR ANALYSIS
# ============================================================

predictions_df["IS_ERROR"] = (
    predictions_df["ACTUAL_RISK"]
    != predictions_df["HIERARCHICAL_PREDICTION"]
)

container_summary = (
    predictions_df.groupby("CONTAINER_ID")
    .agg(
        total_observations=("ACTUAL_RISK", "size"),
        errors=("IS_ERROR", "sum")
    )
    .reset_index()
)

container_summary["error_rate"] = (
    container_summary["errors"]
    / container_summary["total_observations"]
)

container_summary = container_summary.sort_values(
    "error_rate",
    ascending=False
)

container_summary.to_csv(
    OUTPUT_DIR / "container_error_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("TOP 10 CONTAINERS BY ERROR RATE")
print("=" * 70)

print(
    container_summary.head(10).to_string(index=False)
)


# ============================================================
# 12. SAVE FINAL SUMMARY
# ============================================================

print("\nSaved files:")

print(
    OUTPUT_DIR / "temporal_fold_metrics.csv"
)

print(
    OUTPUT_DIR / "temporal_predictions.csv"
)

print(
    OUTPUT_DIR / "container_error_summary.csv"
)

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION COMPLETED")
print("=" * 70)

