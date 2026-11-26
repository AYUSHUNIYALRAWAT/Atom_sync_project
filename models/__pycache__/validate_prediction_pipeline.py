
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

PIPELINE_PATH = (
    BASE_DIR / "models" / "prediction_pipeline_results"
    / "risk_predictions.csv"
)

REFERENCE_PATH = (
    BASE_DIR / "models" / "hierarchical_results"
    / "hierarchical_test_predictions.csv"
)

from prediction_pipeline import predict_risk


# Load data
df = pd.read_csv(DATA_PATH)
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

# Recreate the original chronological test period
df["TARGET_RISK"] = (
    df.groupby("CONTAINER_ID")["RISK_LABEL"].shift(-1)
)

df["TARGET_TIMESTAMP"] = (
    df.groupby("CONTAINER_ID")["TIMESTAMP"].shift(-1)
)

df = df.dropna(
    subset=["TARGET_RISK", "TARGET_TIMESTAMP"]
).copy()

df = df[
    df["TARGET_RISK"].isin(["Low", "Medium", "High"])
].copy()

unique_timestamps = sorted(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)

test_start = unique_timestamps[test_start_index]

test_df = df[
    df["TARGET_TIMESTAMP"] >= test_start
].copy()

# Run the reusable pipeline
predictions = predict_risk(test_df)

predictions["TARGET_TIMESTAMP"] = test_df[
    "TARGET_TIMESTAMP"
].to_numpy()

predictions["ACTUAL_RISK"] = test_df[
    "TARGET_RISK"
].to_numpy()

# Load original hierarchical predictions
reference = pd.read_csv(REFERENCE_PATH)

# Convert reference timestamp to datetime
reference["TARGET_TIMESTAMP"] = pd.to_datetime(
    reference["TARGET_TIMESTAMP"]
)

# Ensure both sides use the same datetime format
predictions["TARGET_TIMESTAMP"] = pd.to_datetime(
    predictions["TARGET_TIMESTAMP"]
)
# Align records
comparison = predictions.merge(
    reference[
        [
            "CONTAINER_ID",
            "TARGET_TIMESTAMP",
            "ACTUAL_RISK",
            "HIERARCHICAL_PREDICTION",
        ]
    ],
    on=[
        "CONTAINER_ID",
        "TARGET_TIMESTAMP",
        "ACTUAL_RISK",
    ],
    how="inner",
    validate="one_to_one",
)

if len(comparison) != len(reference):
    raise ValueError(
        "Prediction observations do not match the reference test set."
    )

# Compare predictions
comparison["MATCH"] = (
    comparison["PREDICTED_RISK"]
    == comparison["HIERARCHICAL_PREDICTION"]
)

agreement = comparison["MATCH"].mean()

print("\nPREDICTION CONSISTENCY VALIDATION")
print("=" * 55)

print("Reference observations:", len(reference))
print("Matched observations:", len(comparison))
print("Prediction agreement:", round(agreement * 100, 2), "%")

print("\nOriginal model Macro F1:")
print(
    f1_score(
        comparison["ACTUAL_RISK"],
        comparison["HIERARCHICAL_PREDICTION"],
        average="macro",
    )
)

print("\nReusable pipeline Macro F1:")
print(
    f1_score(
        comparison["ACTUAL_RISK"],
        comparison["PREDICTED_RISK"],
        average="macro",
    )
)

print("\nPrediction differences:")
print(
    comparison.loc[
        ~comparison["MATCH"],
        [
            "ACTUAL_RISK",
            "PREDICTED_RISK",
            "HIERARCHICAL_PREDICTION",
        ],
    ].head(10).to_string(index=False)
)

output_path = (
    BASE_DIR / "models" / "prediction_pipeline_results"
    / "consistency_validation.csv"
)

comparison.to_csv(output_path, index=False)

print("\nSaved:", output_path)

if agreement == 1.0:
    print("\nSUCCESS: Predictions match exactly.")
else:
    print("\nWARNING: Investigate prediction differences.")
