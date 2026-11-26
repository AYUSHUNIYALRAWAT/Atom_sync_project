
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "analytics"
    / "weather_features.csv"
)

PREDICTIONS_PATH = (
    BASE_DIR / "models" / "temporal_validation_results"
    / "temporal_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "models" / "container_investigation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("ATMOSYNC CONTAINER ERROR INVESTIGATION")
print("=" * 70)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

predictions = pd.read_csv(PREDICTIONS_PATH)

predictions["TARGET_TIMESTAMP"] = pd.to_datetime(
    predictions["TARGET_TIMESTAMP"]
)

print("\nDataset loaded:", len(df))
print("Validation predictions:", len(predictions))


# ============================================================
# 3. PREPARE NEXT-READING TARGET DATA
# ============================================================

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
# 4. MERGE SENSOR DATA WITH PREDICTIONS
# ============================================================

merged = predictions.merge(
    df,
    on=["CONTAINER_ID", "TARGET_TIMESTAMP"],
    how="left",
    suffixes=("_PREDICTION", "_SOURCE")
)

if merged["RISK_LABEL"].isna().any():
    raise ValueError(
        "Some prediction rows could not be matched "
        "to the source dataset."
    )

merged["IS_ERROR"] = (
    merged["ACTUAL_RISK"]
    != merged["HIERARCHICAL_PREDICTION"]
)

merged["MISSED_HIGH"] = (
    (merged["ACTUAL_RISK"] == "High")
    & (merged["HIERARCHICAL_PREDICTION"] != "High")
)

merged["MISSED_MEDIUM"] = (
    (merged["ACTUAL_RISK"] == "Medium")
    & (merged["HIERARCHICAL_PREDICTION"] != "Medium")
)


# ============================================================
# 5. CONTAINER-LEVEL PERFORMANCE
# ============================================================

container_summary = (
    merged.groupby("CONTAINER_ID")
    .agg(
        observations=("ACTUAL_RISK", "size"),
        errors=("IS_ERROR", "sum"),
        missed_high=("MISSED_HIGH", "sum"),
        missed_medium=("MISSED_MEDIUM", "sum"),

        average_temperature=("TEMPERATURE_C", "mean"),
        average_humidity=("HUMIDITY_PCT", "mean"),
        average_vibration=("VIBRATION_G", "mean"),
        average_temp_deviation=(
            "TEMPERATURE_DEVIATION_C", "mean"
        )
    )
    .reset_index()
)

container_summary["error_rate"] = (
    container_summary["errors"]
    / container_summary["observations"]
)

container_summary.to_csv(
    OUTPUT_DIR / "container_performance.csv",
    index=False
)

print("\nTop 15 containers by error rate:")

print(
    container_summary.sort_values(
        "error_rate",
        ascending=False
    ).head(15).to_string(index=False)
)


# ============================================================
# 6. ACTUAL RISK DISTRIBUTION BY CONTAINER
# ============================================================

risk_distribution = pd.crosstab(
    merged["CONTAINER_ID"],
    merged["ACTUAL_RISK"],
    normalize="index"
)

risk_distribution.columns = [
    f"{column}_percentage"
    for column in risk_distribution.columns
]

risk_distribution.to_csv(
    OUTPUT_DIR / "container_risk_distribution.csv"
)


# ============================================================
# 7. MISSED HIGH AND MEDIUM INVESTIGATION
# ============================================================

missed_high = merged[
    merged["MISSED_HIGH"]
].copy()

missed_medium = merged[
    merged["MISSED_MEDIUM"]
].copy()

sensor_columns = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",
    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",
    "TEMP_DEV_ROLLING_MAX_4",
    "TEMP_DEV_ROLLING_MIN_4"
]

available_sensor_columns = [
    column for column in sensor_columns
    if column in merged.columns
]

high_summary = missed_high[
    available_sensor_columns
].describe().T

medium_summary = missed_medium[
    available_sensor_columns
].describe().T

high_summary.to_csv(
    OUTPUT_DIR / "missed_high_sensor_summary.csv"
)

medium_summary.to_csv(
    OUTPUT_DIR / "missed_medium_sensor_summary.csv"
)


# ============================================================
# 8. COMPARE CORRECT AND INCORRECT PREDICTIONS
# ============================================================

correct = merged[
    ~merged["IS_ERROR"]
]

incorrect = merged[
    merged["IS_ERROR"]
]

comparison = pd.DataFrame({
    "correct_mean": correct[
        available_sensor_columns
    ].mean(),

    "incorrect_mean": incorrect[
        available_sensor_columns
    ].mean()
})

comparison["difference"] = (
    comparison["incorrect_mean"]
    - comparison["correct_mean"]
)

comparison.to_csv(
    OUTPUT_DIR / "correct_vs_incorrect_sensor_conditions.csv"
)


# ============================================================
# 9. FOLD-LEVEL CONTAINER ERROR ANALYSIS
# ============================================================

fold_container = (
    merged.groupby(["FOLD", "CONTAINER_ID"])
    .agg(
        observations=("IS_ERROR", "size"),
        errors=("IS_ERROR", "sum")
    )
    .reset_index()
)

fold_container["error_rate"] = (
    fold_container["errors"]
    / fold_container["observations"]
)

fold_container.to_csv(
    OUTPUT_DIR / "fold_container_errors.csv",
    index=False
)


# ============================================================
# 10. RANK CONTAINERS WITH REPEATED HIGH ERROR RATES
# ============================================================

error_pivot = fold_container.pivot(
    index="CONTAINER_ID",
    columns="FOLD",
    values="error_rate"
)

error_pivot.columns = [
    f"fold_{column}_error_rate"
    for column in error_pivot.columns
]

error_pivot["average_fold_error_rate"] = (
    error_pivot.mean(axis=1)
)

fold_rate_columns = [
    column for column in error_pivot.columns
    if column.startswith("fold_")
    and column.endswith("_error_rate")
]

error_pivot["folds_above_50_percent"] = (
    error_pivot[fold_rate_columns] > 0.50
).sum(axis=1)

error_pivot = error_pivot.sort_values(
    "average_fold_error_rate",
    ascending=False
)

error_pivot.to_csv(
    OUTPUT_DIR / "repeated_container_errors.csv"
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CONTAINER INVESTIGATION SUMMARY")
print("=" * 70)

print("Missed High observations:", len(missed_high))
print("Missed Medium observations:", len(missed_medium))

print("\nTop 10 repeated high-error containers:")

print(
    error_pivot.head(10).to_string()
)

print("\nSaved investigation files in:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)
print("CONTAINER ERROR INVESTIGATION COMPLETED")
print("=" * 70)
