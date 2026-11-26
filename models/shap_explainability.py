
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


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

OUTPUT_DIR = BASE_DIR / "models" / "shap_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_RESULTS_PATH = OUTPUT_DIR / "global_shap_importance.csv"

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("ATMOSYNC SHAP EXPLAINABILITY")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
label_encoder = saved_data["label_encoder"]

numeric_features = saved_data["numeric_features"]
categorical_features = saved_data["categorical_features"]

print("\nTrained XGBoost model loaded.")
print("Model type:", saved_data.get("model_type"))


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

df = df.sort_values(
    ["CONTAINER_ID", "TIMESTAMP"]
).reset_index(drop=True)

print("\nDataset loaded:", len(df))


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

print("\nTest dataset rows:", len(X_test))


# ============================================================
# 6. SELECT TEST SAMPLE
# ============================================================

if len(X_test) > SAMPLE_SIZE:

    X_sample = X_test.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

else:
    X_sample = X_test.copy()

print("SHAP sample size:", len(X_sample))


# ============================================================
# 7. TRANSFORM FEATURES
# ============================================================

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_sample)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_sample.index
)

print("\nTransformed feature count:", len(feature_names))


# ============================================================
# 8. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(
    X_transformed_df
)

print("SHAP calculation completed.")


# ============================================================
# 9. NORMALIZE MULTICLASS SHAP OUTPUT
# ============================================================

# Different SHAP versions can return different shapes.

if isinstance(shap_values, list):

    class_shap_values = [
        np.asarray(values)
        for values in shap_values
    ]

else:

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:

        # Expected shape:
        # samples x features x classes

        class_shap_values = [
            shap_array[:, :, i]
            for i in range(shap_array.shape[2])
        ]

    elif shap_array.ndim == 2:

        class_shap_values = [shap_array]

    else:

        raise ValueError(
            f"Unexpected SHAP output shape: {shap_array.shape}"
        )


class_names = list(label_encoder.classes_)

if len(class_shap_values) != len(class_names):

    raise ValueError(
        "Number of SHAP outputs does not match target classes. "
        f"SHAP outputs: {len(class_shap_values)}, "
        f"classes: {class_names}"
    )


# ============================================================
# 10. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

print("\nCalculating global feature importance...")

importance_by_class = {}

for class_index, class_name in enumerate(class_names):

    importance_by_class[class_name] = np.mean(
        np.abs(class_shap_values[class_index]),
        axis=0
    )

importance_df = pd.DataFrame(
    importance_by_class,
    index=feature_names
)

importance_df["overall_mean_abs_shap"] = (
    importance_df.mean(axis=1)
)

importance_df = importance_df.sort_values(
    "overall_mean_abs_shap",
    ascending=False
)

importance_df.to_csv(GLOBAL_RESULTS_PATH)

print("\nTop 20 Global SHAP Features:")

print(
    importance_df.head(20).to_string()
)


# ============================================================
# 11. GLOBAL FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(12, 8))

top_features = importance_df.head(20).sort_values(
    "overall_mean_abs_shap"
)

plt.barh(
    top_features.index,
    top_features["overall_mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Features")

plt.title(
    "AtmoSync - Global SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "global_shap_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. CLASS-SPECIFIC SHAP SUMMARY PLOTS
# ============================================================

for class_index, class_name in enumerate(class_names):

    print(f"\nGenerating SHAP plot for {class_name} risk...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        class_shap_values[class_index],
        X_transformed_df,
        max_display=20,
        show=False
    )

    plt.title(
        f"AtmoSync - SHAP Explanation: {class_name} Risk"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"shap_{class_name.lower()}_risk.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. SAVE TOP FEATURES BY CLASS
# ============================================================

for class_name in class_names:

    class_results = importance_df[
        [class_name]
    ].sort_values(
        class_name,
        ascending=False
    )

    class_results.to_csv(
        OUTPUT_DIR / f"{class_name.lower()}_risk_importance.csv"
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 70)

print("\nSaved files:")

for file_path in sorted(OUTPUT_DIR.glob("*")):

    print(file_path.name)

print("\nSHAP analysis completed successfully.")

