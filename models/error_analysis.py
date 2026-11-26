
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

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "saved_models"
    / "xgboost_custom_weights_model.pkl"
)

OUTPUT_DIR = BASE_DIR / "models" / "error_analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("ATMOSYNC MODEL ERROR ANALYSIS")
print("=" * 70)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
label_encoder = saved_data["label_encoder"]

numeric_features = saved_data["numeric_features"]
categorical_features = saved_data["categorical_features"]

print("\nCustom-weighted XGBoost model loaded.")


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

print("Dataset rows:", len(df))


# ============================================================
# 4. CREATE NEXT-READING TARGET
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


# ============================================================
# 5. RECREATE CHRONOLOGICAL TEST SET
# ============================================================

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)

test_start_time = unique_timestamps[test_start_index]

test_df = df[
    df["TARGET_TIMESTAMP"] >= test_start_time
].copy()

X_test = test_df[
    numeric_features + categorical_features
].copy()

y_test = test_df["TARGET_RISK"].to_numpy()

print("\nTest rows:", len(X_test))


# ============================================================
# 6. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

predictions = model.predict(X_test).astype(int)

predicted_labels = label_encoder.inverse_transform(
    predictions
)

# Convert numeric predictions into class names
predictions = predicted_labels

probabilities = model.predict_proba(X_test)

class_names = list(label_encoder.classes_)

probability_columns = {
    f"PROBABILITY_{class_name.upper()}":
        probabilities[:, class_index]
    for class_index, class_name in enumerate(class_names)
}


# ============================================================
# 7. CREATE ERROR ANALYSIS DATAFRAME
# ============================================================

results = test_df[
    [
        "CONTAINER_ID",
        "TIMESTAMP",
        "TARGET_TIMESTAMP",
        "TEMPERATURE_C",
        "HUMIDITY_PCT",
        "VIBRATION_G",
        "TEMPERATURE_DEVIATION_C",
        "SENSOR_STATUS"
    ]
].copy()

results["ACTUAL_RISK"] = y_test
results["PREDICTED_RISK"] = predicted_labels

for column, values in probability_columns.items():
    results[column] = values

results["IS_CORRECT"] = (
    results["ACTUAL_RISK"] == results["PREDICTED_RISK"]
)

results["ERROR_TYPE"] = np.select(
    [
        (results["ACTUAL_RISK"] == "High") &
        (results["PREDICTED_RISK"] == "Low"),

        (results["ACTUAL_RISK"] == "High") &
        (results["PREDICTED_RISK"] == "Medium"),

        (results["ACTUAL_RISK"] == "Medium") &
        (results["PREDICTED_RISK"] == "Low"),

        (results["ACTUAL_RISK"] == "Medium") &
        (results["PREDICTED_RISK"] == "High"),

        (results["ACTUAL_RISK"] == "Low") &
        (results["PREDICTED_RISK"] == "High"),

        (results["ACTUAL_RISK"] == "Low") &
        (results["PREDICTED_RISK"] == "Medium")
    ],
    [
        "MISSED_HIGH_PREDICTED_LOW",
        "MISSED_HIGH_PREDICTED_MEDIUM",
        "MISSED_MEDIUM_PREDICTED_LOW",
        "FALSE_HIGH_ALERT",
        "FALSE_HIGH_ALERT",
        "FALSE_MEDIUM_ALERT"
    ],
    default="CORRECT"
)


# ============================================================
# 8. OVERALL MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TEST PERFORMANCE")
print("=" * 70)

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions), 4)
)

print(
    "Macro F1:",
    round(
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        4
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predicted_labels,
        labels=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predicted_labels,
        labels=class_names
    )
)


# ============================================================
# 9. ERROR TYPE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR TYPE ANALYSIS")
print("=" * 70)

error_summary = (
    results["ERROR_TYPE"]
    .value_counts()
    .rename_axis("ERROR_TYPE")
    .reset_index(name="COUNT")
)

error_summary["PERCENTAGE"] = (
    error_summary["COUNT"] / len(results) * 100
)

print(error_summary.to_string(index=False))

error_summary.to_csv(
    OUTPUT_DIR / "error_type_summary.csv",
    index=False
)


# ============================================================
# 10. MISSED HIGH-RISK CASES
# ============================================================

missed_high = results[
    (results["ACTUAL_RISK"] == "High") &
    (results["PREDICTED_RISK"] != "High")
].copy()

print("\n" + "=" * 70)
print("MISSED HIGH-RISK CASES")
print("=" * 70)

print("Total missed High-risk cases:", len(missed_high))

if len(missed_high) > 0:

    print("\nAverage sensor conditions in missed High-risk cases:")

    print(
        missed_high[
            [
                "TEMPERATURE_C",
                "HUMIDITY_PCT",
                "VIBRATION_G",
                "TEMPERATURE_DEVIATION_C"
            ]
        ].mean().round(3)
    )

missed_high.to_csv(
    OUTPUT_DIR / "missed_high_risk_cases.csv",
    index=False
)


# ============================================================
# 11. MISSED MEDIUM-RISK CASES
# ============================================================

missed_medium = results[
    (results["ACTUAL_RISK"] == "Medium") &
    (results["PREDICTED_RISK"] != "Medium")
].copy()

print("\n" + "=" * 70)
print("MISSED MEDIUM-RISK CASES")
print("=" * 70)

print("Total missed Medium-risk cases:", len(missed_medium))

if len(missed_medium) > 0:

    print("\nAverage sensor conditions in missed Medium-risk cases:")

    print(
        missed_medium[
            [
                "TEMPERATURE_C",
                "HUMIDITY_PCT",
                "VIBRATION_G",
                "TEMPERATURE_DEVIATION_C"
            ]
        ].mean().round(3)
    )

missed_medium.to_csv(
    OUTPUT_DIR / "missed_medium_risk_cases.csv",
    index=False
)


# ============================================================
# 12. CONTAINER-WISE ERROR ANALYSIS
# ============================================================

incorrect_predictions = results[
    ~results["IS_CORRECT"]
].copy()

container_errors = (
    incorrect_predictions
    .groupby("CONTAINER_ID")
    .agg(
        total_errors=("IS_CORRECT", "size"),
        missed_high=("ERROR_TYPE", lambda x:
            (x == "MISSED_HIGH_PREDICTED_LOW").sum()
            + (x == "MISSED_HIGH_PREDICTED_MEDIUM").sum()
        ),
        missed_medium=("ERROR_TYPE", lambda x:
            (x == "MISSED_MEDIUM_PREDICTED_LOW").sum()
        )
    )
    .reset_index()
    .sort_values("total_errors", ascending=False)
)

print("\n" + "=" * 70)
print("CONTAINER-WISE ERROR ANALYSIS")
print("=" * 70)

print("\nTop 15 containers with prediction errors:")

print(container_errors.head(15).to_string(index=False))

container_errors.to_csv(
    OUTPUT_DIR / "container_error_summary.csv",
    index=False
)


# ============================================================
# 13. COMPARE CORRECT AND INCORRECT PREDICTIONS
# ============================================================

comparison_features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C"
]

comparison = results.groupby("IS_CORRECT")[
    comparison_features
].mean().round(3)

comparison.index = comparison.index.map({
    True: "Correct Predictions",
    False: "Incorrect Predictions"
})

print("\n" + "=" * 70)
print("CORRECT VS INCORRECT PREDICTIONS")
print("=" * 70)

print(comparison)

comparison.to_csv(
    OUTPUT_DIR / "correct_vs_incorrect_conditions.csv"
)


# ============================================================
# 14. SAVE COMPLETE PREDICTION RESULTS
# ============================================================

results.to_csv(
    OUTPUT_DIR / "all_test_predictions.csv",
    index=False
)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print("\nSaved files:")

for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file_path.name)

print("\nError analysis completed successfully.")
