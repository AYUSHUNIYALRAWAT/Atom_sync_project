
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "analytics"

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

# Predict the next reading's risk
df["TARGET_RISK"] = (
    df.groupby("CONTAINER_ID")["RISK_LABEL"].shift(-1)
)

df = df.dropna(subset=["TARGET_RISK"]).copy()

df = df[
    df["TARGET_RISK"].isin(["Low", "Medium", "High"])
].copy()

print("=" * 60)
print("NEXT-READING TEMPERATURE RISK ANALYSIS")
print("=" * 60)

# Statistics using current features and next-reading labels
print("\nCurrent temperature deviation by NEXT risk:")

print(
    df.groupby("TARGET_RISK")["TEMPERATURE_DEVIATION_C"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(3)
)

print("\nRolling temperature deviation by NEXT risk:")

print(
    df.groupby("TARGET_RISK")[
        "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4"
    ]
    .agg(["mean", "median", "std", "min", "max"])
    .round(3)
)

# Plot next-reading risk against current temperature deviation
plt.figure(figsize=(10, 6))

df.boxplot(
    column="TEMPERATURE_DEVIATION_C",
    by="TARGET_RISK"
)

plt.title("Current Temperature Deviation vs Next Reading Risk")
plt.suptitle("")
plt.xlabel("Next Reading Risk")
plt.ylabel("Current Temperature Deviation (C)")
plt.tight_layout()

chart_path = OUTPUT_DIR / "next_reading_temperature_risk.png"

plt.savefig(chart_path, dpi=300)
plt.show()

print("\nChart saved:", chart_path)

print("\nTEMPERATURE DEVIATION RANGES BY NEXT RISK")

for risk in ["Low", "Medium", "High"]:

    values = df.loc[
        df["TARGET_RISK"] == risk,
        "TEMPERATURE_DEVIATION_C"
    ]

    print(f"\n{risk} Risk")

    print("25th percentile:", round(values.quantile(0.25), 3))
    print("50th percentile:", round(values.quantile(0.50), 3))
    print("75th percentile:", round(values.quantile(0.75), 3))
    print("90th percentile:", round(values.quantile(0.90), 3))


