
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

OUTPUT_DIR = BASE_DIR / "models" / "medium_risk_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("=" * 70)
print("ATMOSYNC MEDIUM-RISK INVESTIGATION")
print("=" * 70)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
label_encoder = saved_data["label_encoder"]

numeric_features = saved_data["numeric_features"]
categorical_features = saved_data["categorical_features"]

print("\nModel loaded successfully.")


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
    df["TARGET_RISK"].isin(["Low", "Medium", "High"])
].copy()

df["TARGET_TIMESTAMP"] = pd.to_datetime(
    df["TARGET_TIMESTAMP"]
)


# ============================================================
# 4. RECREATE TEST DATASET
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
# 5. PREDICTIONS AND PROBABILITIES
# ============================================================

predictions = model.predict(X_test).astype(int)

predicted_labels = label_encoder.inverse_transform(
    predictions
)

probabilities = model.predict_proba(X_test)

class_names = list(label_encoder.classes_)

probability_data = {
    f"PROB_{name.upper()}": probabilities[:, index]
    for index, name in enumerate(class_names)
}


# ============================================================
# 6. CREATE INVESTIGATION DATAFRAME
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
        "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",
        "TEMP_DEV_ROLLING_MAX_4",
        "TEMP_DEV_ROLLING_MIN_4",
        "TEMPERATURE_C_ROLLING_MEAN_4",
        "HUMIDITY_PCT_ROLLING_MEAN_4",
        "SENSOR_STATUS"
    ]
].copy()

results["ACTUAL_RISK"] = y_test
results["PREDICTED_RISK"] = predicted_labels

for column, values in probability_data.items():
    results[column] = values

results["IS_CORRECT"] = (
    results["ACTUAL_RISK"] == results["PREDICTED_RISK"]
)


# ============================================================
# 7. ISOLATE MEDIUM-RISK CASES
# ============================================================

medium_df = results[
    results["ACTUAL_RISK"] == "Medium"
].copy()

correct_medium = medium_df[
    medium_df["PREDICTED_RISK"] == "Medium"
].copy()

missed_medium = medium_df[
    medium_df["PREDICTED_RISK"] != "Medium"
].copy()

missed_medium_low = medium_df[
    medium_df["PREDICTED_RISK"] == "Low"
].copy()

missed_medium_high = medium_df[
    medium_df["PREDICTED_RISK"] == "High"
].copy()

print("\n" + "=" * 70)
print("MEDIUM-RISK PREDICTION SUMMARY")
print("=" * 70)

print("Actual Medium cases:", len(medium_df))
print("Correct Medium predictions:", len(correct_medium))
print("Missed Medium predictions:", len(missed_medium))
print("Predicted Low instead:", len(missed_medium_low))
print("Predicted High instead:", len(missed_medium_high))


# ============================================================
# 8. COMPARE SENSOR CONDITIONS
# ============================================================

comparison_features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",
    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",
    "TEMP_DEV_ROLLING_MAX_4",
    "TEMP_DEV_ROLLING_MIN_4",
    "TEMPERATURE_C_ROLLING_MEAN_4",
    "HUMIDITY_PCT_ROLLING_MEAN_4"
]

comparison = pd.DataFrame({
    "Correct_Medium": correct_medium[
        comparison_features
    ].mean(),

    "Missed_Medium": missed_medium[
        comparison_features
    ].mean()
})

comparison["Difference"] = (
    comparison["Missed_Medium"]
    - comparison["Correct_Medium"]
)

comparison = comparison.round(4)

print("\n" + "=" * 70)
print("CORRECT VS MISSED MEDIUM SENSOR CONDITIONS")
print("=" * 70)

print(comparison.to_string())

comparison.to_csv(
    OUTPUT_DIR / "medium_sensor_comparison.csv"
)


# ============================================================
# 9. ANALYZE MODEL PROBABILITIES
# ============================================================

probability_columns = [
    "PROB_HIGH",
    "PROB_LOW",
    "PROB_MEDIUM"
]

probability_comparison = pd.DataFrame({
    "Correct_Medium": correct_medium[
        probability_columns
    ].mean(),

    "Missed_Medium": missed_medium[
        probability_columns
    ].mean()
}).round(4)

print("\n" + "=" * 70)
print("MODEL PROBABILITY COMPARISON")
print("=" * 70)

print(probability_comparison.to_string())

probability_comparison.to_csv(
    OUTPUT_DIR / "medium_probability_comparison.csv"
)


# ============================================================
# 10. CONTAINER-WISE MEDIUM-RISK ERRORS
# ============================================================

container_summary = (
    missed_medium.groupby("CONTAINER_ID")
    .agg(
        missed_medium_cases=("ACTUAL_RISK", "size"),
        avg_temperature=("TEMPERATURE_C", "mean"),
        avg_temperature_deviation=(
            "TEMPERATURE_DEVIATION_C",
            "mean"
        ),
        avg_rolling_deviation=(
            "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "missed_medium_cases",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("TOP 15 CONTAINERS WITH MISSED MEDIUM RISK")
print("=" * 70)

print(container_summary.head(15).round(3).to_string(index=False))

container_summary.to_csv(
    OUTPUT_DIR / "medium_container_errors.csv",
    index=False
)


# ============================================================
# 11. CONFIDENCE ANALYSIS
# ============================================================

missed_medium["PREDICTED_CLASS_CONFIDENCE"] = (
    missed_medium[
        ["PROB_HIGH", "PROB_LOW", "PROB_MEDIUM"]
    ].max(axis=1)
)

print("\n" + "=" * 70)
print("MISSED MEDIUM CONFIDENCE")
print("=" * 70)

print(
    missed_medium[
        [
            "PROB_HIGH",
            "PROB_LOW",
            "PROB_MEDIUM",
            "PREDICTED_CLASS_CONFIDENCE"
        ]
    ].describe().round(4)
)


# ============================================================
# 12. SAVE INVESTIGATION DATA
# ============================================================

correct_medium.to_csv(
    OUTPUT_DIR / "correct_medium_predictions.csv",
    index=False
)

missed_medium.to_csv(
    OUTPUT_DIR / "missed_medium_predictions.csv",
    index=False
)

missed_medium_low.to_csv(
    OUTPUT_DIR / "medium_predicted_low.csv",
    index=False
)

missed_medium_high.to_csv(
    OUTPUT_DIR / "medium_predicted_high.csv",
    index=False
)

results.to_csv(
    OUTPUT_DIR / "all_test_predictions.csv",
    index=False
)


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MEDIUM-RISK INVESTIGATION COMPLETED")
print("=" * 70)

print("\nSaved files:")

for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file_path.name)

print("\nInvestigation completed successfully.")
