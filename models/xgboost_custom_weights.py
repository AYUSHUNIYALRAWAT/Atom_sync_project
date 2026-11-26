
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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

MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_custom_weights_model.pkl"
TUNING_PATH = MODEL_DIR / "xgboost_custom_weights_trials.csv"
RESULTS_PATH = MODEL_DIR / "xgboost_custom_weights_results.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("ATMOSYNC CUSTOM CLASS-WEIGHTED XGBOOST")
print("=" * 70)

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
# 4. DEFINE FEATURES
# ============================================================

numeric_features = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",

    "TEMPERATURE_C_ROLLING_MEAN_4",
    "HUMIDITY_PCT_ROLLING_MEAN_4",
    "VIBRATION_G_ROLLING_MEAN_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",

    "TEMPERATURE_C_ROLLING_STD_4",
    "HUMIDITY_PCT_ROLLING_STD_4",
    "VIBRATION_G_ROLLING_STD_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_STD_4",

    "TEMPERATURE_C_PREVIOUS",
    "TEMPERATURE_C_CHANGE",
    "HUMIDITY_PCT_PREVIOUS",
    "HUMIDITY_PCT_CHANGE",
    "VIBRATION_G_PREVIOUS",
    "VIBRATION_G_CHANGE",

    "HOUR",
    "DAY_OF_WEEK",

    "TEMP_DEV_LAG_2",
    "TEMP_DEV_LAG_3",
    "TEMP_DEV_ROLLING_MAX_4",
    "TEMP_DEV_ROLLING_MIN_4",
    "TEMP_DEV_ABSOLUTE_CHANGE",
    "TEMP_DEV_ACCELERATION"
]

categorical_features = ["SENSOR_STATUS"]

required_columns = (
    numeric_features
    + categorical_features
    + ["TARGET_TIMESTAMP", "TARGET_RISK"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

X = df[numeric_features + categorical_features].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["TARGET_RISK"])

print("\nTarget encoding:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{label} = {i}")


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)
test_start_time = unique_timestamps[test_start_index]

development_mask = (
    df["TARGET_TIMESTAMP"] < test_start_time
)

test_mask = (
    df["TARGET_TIMESTAMP"] >= test_start_time
)

development_df = df.loc[development_mask].copy()
test_df = df.loc[test_mask].copy()

development_timestamps = np.sort(
    development_df["TARGET_TIMESTAMP"].unique()
)

validation_start_index = int(
    len(development_timestamps) * 0.80
)

validation_start_time = (
    development_timestamps[validation_start_index]
)

train_mask = (
    development_df["TARGET_TIMESTAMP"] < validation_start_time
)

validation_mask = (
    development_df["TARGET_TIMESTAMP"] >= validation_start_time
)

X_train = X.loc[development_df.index[train_mask]]
X_val = X.loc[development_df.index[validation_mask]]
X_test = X.loc[test_df.index]

y_series = pd.Series(y, index=df.index)

y_train = y_series.loc[X_train.index].to_numpy()
y_val = y_series.loc[X_val.index].to_numpy()
y_test = y_series.loc[X_test.index].to_numpy()

print("\nChronological split:")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 7. DEFINE CUSTOM WEIGHT COMBINATIONS
# ============================================================

# LabelEncoder order:
# High = 0
# Low = 1
# Medium = 2

# Each tuple represents:
# (High weight, Medium weight)
# Low weight remains 1.0.

weight_combinations = list(product(
    [1.5, 2.0, 2.5],
    [1.25, 1.5, 1.75]
))

print("\nCustom weight combinations:")
for trial, (high_weight, medium_weight) in enumerate(
    weight_combinations, start=1
):
    print(
        f"Trial {trial}: "
        f"High={high_weight}, "
        f"Low=1.0, "
        f"Medium={medium_weight}"
    )


# ============================================================
# 8. CREATE BASE MODEL
# ============================================================

def create_model():

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),

        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2,

        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


# ============================================================
# 9. CUSTOM WEIGHT SEARCH
# ============================================================

results = []

best_macro_f1 = -1
best_model = None
best_weights = None

y_val_labels = label_encoder.inverse_transform(y_val)

print("\nStarting custom class-weight optimization...")

for trial, (high_weight, medium_weight) in enumerate(
    weight_combinations, start=1
):

    class_weights = {
        "High": high_weight,
        "Low": 1.0,
        "Medium": medium_weight
    }

    sample_weights = np.array([
        class_weights[label]
        for label in label_encoder.inverse_transform(y_train)
    ])

    model = create_model()

    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights
    )

    val_predictions = model.predict(X_val).astype(int)

    val_pred_labels = label_encoder.inverse_transform(
        val_predictions
    )

    val_accuracy = accuracy_score(
        y_val_labels,
        val_pred_labels
    )

    val_macro_f1 = f1_score(
        y_val_labels,
        val_pred_labels,
        average="macro",
        zero_division=0
    )

    val_report = classification_report(
        y_val_labels,
        val_pred_labels,
        labels=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    high_recall = val_report["High"]["recall"]
    low_recall = val_report["Low"]["recall"]
    medium_recall = val_report["Medium"]["recall"]

    results.append({
        "trial": trial,
        "high_weight": high_weight,
        "low_weight": 1.0,
        "medium_weight": medium_weight,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "high_recall": high_recall,
        "low_recall": low_recall,
        "medium_recall": medium_recall
    })

    print(
        f"Trial {trial}/9 | "
        f"Weights H:{high_weight}, M:{medium_weight} | "
        f"Accuracy: {val_accuracy:.4f} | "
        f"Macro F1: {val_macro_f1:.4f}"
    )

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1
        best_model = model

        best_weights = class_weights.copy()


# ============================================================
# 10. SHOW OPTIMIZATION RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_macro_f1",
    ascending=False
)

print("\n" + "=" * 70)
print("CUSTOM WEIGHT OPTIMIZATION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nBest class weights:")
print(best_weights)

print("\nBest validation Macro F1:", round(best_macro_f1, 4))

results_df.to_csv(TUNING_PATH, index=False)


# ============================================================
# 11. RETRAIN ON TRAIN + VALIDATION
# ============================================================

print("\nRetraining best model on training + validation...")

X_development = pd.concat([X_train, X_val])
y_development = np.concatenate([y_train, y_val])

development_labels = label_encoder.inverse_transform(
    y_development
)

development_weights = np.array([
    best_weights[label]
    for label in development_labels
])

best_model.fit(
    X_development,
    y_development,
    classifier__sample_weight=development_weights
)

print("Final retraining completed.")


# ============================================================
# 12. FINAL TEST EVALUATION
# ============================================================

test_predictions = best_model.predict(X_test).astype(int)

y_test_labels = label_encoder.inverse_transform(y_test)

test_pred_labels = label_encoder.inverse_transform(
    test_predictions
)

test_accuracy = accuracy_score(
    y_test_labels,
    test_pred_labels
)

test_macro_f1 = f1_score(
    y_test_labels,
    test_pred_labels,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL CUSTOM-WEIGHTED XGBOOST EVALUATION")
print("=" * 70)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test_labels,
        test_pred_labels,
        labels=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test_labels,
    test_pred_labels,
    labels=label_encoder.classes_
)

print(cm)

print("\nPrediction Distribution:")
print(pd.Series(test_pred_labels).value_counts())


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

joblib.dump(
    {
        "model": best_model,
        "label_encoder": label_encoder,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_type": "custom_weighted_xgboost",
        "best_weights": best_weights,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1
    },
    MODEL_PATH
)

print("\nBest model saved:")
print(MODEL_PATH)


# ============================================================
# 14. SAVE FINAL EVALUATION
# ============================================================

report = classification_report(
    y_test_labels,
    test_pred_labels,
    labels=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

results_report = pd.DataFrame(report).transpose()

results_report.to_csv(RESULTS_PATH)

print("\nOptimization trials saved:")
print(TUNING_PATH)

print("\nFinal evaluation saved:")
print(RESULTS_PATH)

print("\n" + "=" * 70)
print("CUSTOM-WEIGHTED EXPERIMENT COMPLETED")
print("=" * 70)
