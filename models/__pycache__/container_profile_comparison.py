
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "models" / "container_investigation"
OUTPUT_DIR = BASE_DIR / "models" / "container_profile_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

performance = pd.read_csv(
    INPUT_DIR / "container_performance.csv"
)

risk = pd.read_csv(
    INPUT_DIR / "container_risk_distribution.csv"
)

# Select the 10 containers with the highest error rates.
top_containers = (
    performance.sort_values("error_rate", ascending=False)
    .head(10)["CONTAINER_ID"]
    .tolist()
)

performance["GROUP"] = performance["CONTAINER_ID"].apply(
    lambda x: "Top 10 high-error containers"
    if x in top_containers
    else "Remaining containers"
)

# ------------------------------------------------------------
# 1. COMPARE SENSOR CONDITIONS
# ------------------------------------------------------------

sensor_columns = [
    "average_temperature",
    "average_humidity",
    "average_vibration",
    "average_temp_deviation",
    "error_rate"
]

sensor_comparison = (
    performance.groupby("GROUP")[sensor_columns]
    .mean()
    .T
)

sensor_comparison["difference"] = (
    sensor_comparison["Top 10 high-error containers"]
    - sensor_comparison["Remaining containers"]
)

sensor_comparison.to_csv(
    OUTPUT_DIR / "sensor_condition_comparison.csv"
)

print("\nSENSOR CONDITION COMPARISON")
print(sensor_comparison.round(4).to_string())


# ------------------------------------------------------------
# 2. COMPARE ACTUAL RISK DISTRIBUTION
# ------------------------------------------------------------

risk = risk.merge(
    performance[["CONTAINER_ID", "GROUP"]],
    on="CONTAINER_ID",
    how="inner"
)

risk_columns = [
    column for column in risk.columns
    if column.endswith("_percentage")
]

risk_comparison = (
    risk.groupby("GROUP")[risk_columns]
    .mean()
    .T
)

risk_comparison["difference"] = (
    risk_comparison["Top 10 high-error containers"]
    - risk_comparison["Remaining containers"]
)

risk_comparison.to_csv(
    OUTPUT_DIR / "risk_distribution_comparison.csv"
)

print("\nACTUAL RISK DISTRIBUTION COMPARISON")
print(risk_comparison.round(4).to_string())


# ------------------------------------------------------------
# 3. SAVE TOP CONTAINER DETAILS
# ------------------------------------------------------------

top_details = performance[
    performance["CONTAINER_ID"].isin(top_containers)
].sort_values("error_rate", ascending=False)

top_details.to_csv(
    OUTPUT_DIR / "top_10_container_details.csv",
    index=False
)

print("\nTOP 10 HIGH-ERROR CONTAINERS")
print(top_details.to_string(index=False))

print("\nSaved results in:", OUTPUT_DIR)
print("\nCONTAINER PROFILE COMPARISON COMPLETED")
