
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

OUTPUT_DIR = BASE_DIR / "models" / "low_medium_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "low_medium_separability.csv"

print("=" * 70)
print("ATMOSYNC LOW VS MEDIUM RISK SEPARABILITY")
print("=" * 70)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

print("\nDataset loaded:", len(df))


# ============================================================
# 3. CREATE NEXT-READING TARGET
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

print("\nTarget distribution:")
print(df["TARGET_RISK"].value_counts())


# ============================================================
# 4. KEEP DEVELOPMENT DATA ONLY
# ============================================================

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)

test_start_time = unique_timestamps[test_start_index]

development_df = df[
    df["TARGET_TIMESTAMP"] < test_start_time
].copy()

print("\nDevelopment rows:", len(development_df))
print("Final test period excluded:", test_start_time)


# ============================================================
# 5. SELECT LOW AND MEDIUM CLASSES
# ============================================================

analysis_df = development_df[
    development_df["TARGET_RISK"].isin(["Low", "Medium"])
].copy()

low_df = analysis_df[
    analysis_df["TARGET_RISK"] == "Low"
]

medium_df = analysis_df[
    analysis_df["TARGET_RISK"] == "Medium"
]

print("\nLow observations:", len(low_df))
print("Medium observations:", len(medium_df))


# ============================================================
# 6. SELECT SENSOR FEATURES
# ============================================================

features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",

    "TEMPERATURE_C_ROLLING_MEAN_4",
    "HUMIDITY_PCT_ROLLING_MEAN_4",
    "VIBRATION_G_ROLLING_MEAN_4",

    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",
    "TEMP_DEV_ROLLING_MAX_4",
    "TEMP_DEV_ROLLING_MIN_4",

    "TEMP_DEV_ABSOLUTE_CHANGE",
    "TEMP_DEV_ACCELERATION",

    "TEMPERATURE_C_CHANGE",
    "HUMIDITY_PCT_CHANGE",
    "VIBRATION_G_CHANGE",

    "TEMPERATURE_C_ROLLING_STD_4",
    "HUMIDITY_PCT_ROLLING_STD_4",
    "VIBRATION_G_ROLLING_STD_4"
]

missing_features = [
    feature for feature in features
    if feature not in analysis_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# 7. CALCULATE CLASS DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOW VS MEDIUM FEATURE DISTRIBUTIONS")
print("=" * 70)

distribution = pd.DataFrame({
    "Low_Mean": low_df[features].mean(),
    "Medium_Mean": medium_df[features].mean(),

    "Low_Median": low_df[features].median(),
    "Medium_Median": medium_df[features].median(),

    "Low_Std": low_df[features].std(),
    "Medium_Std": medium_df[features].std()
})

distribution["Mean_Difference"] = (
    distribution["Medium_Mean"]
    - distribution["Low_Mean"]
)

distribution["Absolute_Mean_Difference"] = (
    distribution["Mean_Difference"].abs()
)

distribution.to_csv(
    OUTPUT_DIR / "feature_distributions.csv"
)

print(distribution.round(4).to_string())


# ============================================================
# 8. CALCULATE FEATURE SEPARABILITY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SEPARABILITY ANALYSIS")
print("=" * 70)

y_binary = (
    analysis_df["TARGET_RISK"] == "Medium"
).astype(int)

separability_results = []

for feature in features:

    valid_mask = analysis_df[feature].notna()

    feature_values = analysis_df.loc[
        valid_mask, feature
    ]

    feature_target = y_binary.loc[valid_mask]

    low_values = low_df[feature].dropna()
    medium_values = medium_df[feature].dropna()

    if len(low_values) == 0 or len(medium_values) == 0:
        continue

    # Kolmogorov-Smirnov statistic
    ks_statistic, ks_pvalue = ks_2samp(
        low_values,
        medium_values
    )

    # Single-feature ROC AUC
    # AUC near 0.5 means weak separation.
    # AUC near 1 or 0 means stronger separation.
    if feature_target.nunique() == 2:

        auc = roc_auc_score(
            feature_target,
            feature_values
        )

    else:
        auc = np.nan

    auc_separation = (
        abs(auc - 0.5) * 2
        if not np.isnan(auc)
        else np.nan
    )

    pooled_std = np.sqrt(
        (
            low_values.var()
            + medium_values.var()
        ) / 2
    )

    if pooled_std > 0:

        standardized_difference = (
            medium_values.mean() - low_values.mean()
        ) / pooled_std

    else:
        standardized_difference = np.nan

    separability_results.append({
        "feature": feature,
        "low_mean": low_values.mean(),
        "medium_mean": medium_values.mean(),
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "roc_auc": auc,
        "auc_separation": auc_separation,
        "standardized_mean_difference":
            standardized_difference
    })


separability_df = pd.DataFrame(
    separability_results
)

separability_df = separability_df.sort_values(
    "auc_separation",
    ascending=False
)

separability_df.to_csv(
    REPORT_PATH,
    index=False
)

print("\nFeatures ranked by univariate separation:")

print(
    separability_df.round(4).to_string(index=False)
)


# ============================================================
# 9. VISUALIZE TOP FEATURES
# ============================================================

top_features = separability_df.head(6)["feature"].tolist()

print("\nGenerating feature distribution plots...")

for feature in top_features:

    plt.figure(figsize=(10, 6))

    plt.hist(
        low_df[feature].dropna(),
        bins=40,
        alpha=0.6,
        density=True,
        label="Low Risk"
    )

    plt.hist(
        medium_df[feature].dropna(),
        bins=40,
        alpha=0.6,
        density=True,
        label="Medium Risk"
    )

    plt.xlabel(feature)
    plt.ylabel("Density")

    plt.title(
        f"Low vs Medium Risk: {feature}"
    )

    plt.legend()
    plt.tight_layout()

    safe_name = feature.lower()

    plt.savefig(
        OUTPUT_DIR / f"{safe_name}_distribution.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. VISUALIZE TOP FEATURE SEPARABILITY
# ============================================================

plot_data = separability_df.head(10).sort_values(
    "auc_separation"
)

plt.figure(figsize=(12, 7))

plt.barh(
    plot_data["feature"],
    plot_data["auc_separation"]
)

plt.xlabel("Univariate AUC Separation (0 to 1)")
plt.ylabel("Feature")

plt.title(
    "AtmoSync - Low vs Medium Feature Separability"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_separability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LOW VS MEDIUM ANALYSIS COMPLETED")
print("=" * 70)

print("\nTop 5 distinguishing features:")

print(
    separability_df[
        [
            "feature",
            "roc_auc",
            "auc_separation",
            "ks_statistic"
        ]
    ].head(5).round(4).to_string(index=False)
)

print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nAnalysis completed successfully.")
