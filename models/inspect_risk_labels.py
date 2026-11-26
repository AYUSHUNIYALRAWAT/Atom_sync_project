
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ATMOSYNC RISK LABEL INVESTIGATION")
print("=" * 60)

# 1. Risk distribution
print("\n1. RISK LABEL DISTRIBUTION")
print(df["RISK_LABEL"].value_counts(dropna=False))

print("\nRisk percentages:")
print(
    df["RISK_LABEL"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

# 2. Sensor patterns by risk category
features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C"
]

print("\n2. SENSOR STATISTICS BY RISK")
print(
    df.groupby("RISK_LABEL")[features]
    .agg(["mean", "median", "min", "max"])
    .round(3)
)

# 3. Sensor status by risk
print("\n3. SENSOR STATUS BY RISK")
print(
    pd.crosstab(
        df["RISK_LABEL"],
        df["SENSOR_STATUS"],
        normalize="index"
    ).round(3)
)

# 4. Missing values
print("\n4. MISSING VALUES")
print(df[features + ["RISK_LABEL"]].isnull().sum())

print("\nInvestigation completed.")
