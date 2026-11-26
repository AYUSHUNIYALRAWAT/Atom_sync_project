
from pathlib import Path

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
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "analytics" / "weather_features.csv"

MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "low_medium_binary_model.pkl"
TRIALS_PATH = MODEL_DIR / "low_medium_binary_trials.csv"
RESULTS_PATH = MODEL_DIR / "low_medium_binary_results.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("ATMOSYNC LOW VS MEDIUM BINARY CLASSIFICATION")
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

print("\nOriginal target distribution:")
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
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

unique_timestamps = np.sort(
    df["TARGET_TIMESTAMP"].unique()
)

test_start_index = int(len(unique_timestamps) * 0.80)

test_start_time = unique_timestamps[test_start_index]

development_df = df[
    df["TARGET_TIMESTAMP"] < test_start_time
].copy()

test_df = df[
    df["TARGET_TIMESTAMP"] >= test_start_time
].copy()

development_timestamps = np.sort(
    development_df["TARGET_TIMESTAMP"].unique()
)

validation_start_index = int(
    len(development_timestamps) * 0.80
)

validation_start_time = (
    development_timestamps[validation_start_index]
)

train_df = development_df[
    development_df["TARGET_TIMESTAMP"] < validation_start_time
].copy()

validation_df = development_df[
    development_df["TARGET_TIMESTAMP"] >= validation_start_time
].copy()

print("\nChronological split before class filtering:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# ============================================================
# 6. FILTER LOW AND MEDIUM CLASSES
# ============================================================

allowed_classes = ["Low", "Medium"]

train_df = train_df[
    train_df["TARGET_RISK"].isin(allowed_classes)
].copy()

validation_df = validation_df[
    validation_df["TARGET_RISK"].isin(allowed_classes)
].copy()

test_df = test_df[
    test_df["TARGET_RISK"].isin(allowed_classes)
].copy()

print("\nLow vs Medium observations:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# ============================================================
# 7. PREPARE FEATURES AND TARGET
# ============================================================

feature_columns = numeric_features + categorical_features

X_train = train_df[feature_columns].copy()
X_val = validation_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df["TARGET_RISK"]
)

y_val = label_encoder.transform(
    validation_df["TARGET_RISK"]
)

y_test = label_encoder.transform(
    test_df["TARGET_RISK"]
)

print("\nBinary target encoding:")

for index, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name} = {index}")


# ============================================================
# 8. PREPROCESSING
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
# 9. DEFINE BASE BINARY MODEL
# ============================================================

def create_model():

    classifier = XGBClassifier(
        objective="binary:logistic",

        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=2,

        eval_metric="logloss",
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
# 10. TEST MEDIUM CLASS WEIGHTS
# ============================================================

medium_weights = [1.0, 1.5, 2.0, 2.5, 3.0]

best_macro_f1 = -1
best_model = None
best_weight = None

trial_results = []

y_val_labels = label_encoder.inverse_transform(y_val)

print("\nStarting binary class-weight optimization...")

for trial, medium_weight in enumerate(
    medium_weights, start=1
):

    sample_weights = np.where(
        y_train == label_encoder.transform(["Medium"])[0],
        medium_weight,
        1.0
    )

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

    val_medium_recall = val_report["Medium"]["recall"]
    val_medium_precision = val_report["Medium"]["precision"]

    trial_results.append({
        "trial": trial,
        "medium_weight": medium_weight,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "medium_precision": val_medium_precision,
        "medium_recall": val_medium_recall
    })

    print(
        f"Trial {trial}/5 | "
        f"Medium weight: {medium_weight} | "
        f"Accuracy: {val_accuracy:.4f} | "
        f"Macro F1: {val_macro_f1:.4f} | "
        f"Medium Recall: {val_medium_recall:.4f}"
    )

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1
        best_model = model
        best_weight = medium_weight


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

trials_df = pd.DataFrame(trial_results)

trials_df = trials_df.sort_values(
    "validation_macro_f1",
    ascending=False
)

trials_df.to_csv(TRIALS_PATH, index=False)

print("\n" + "=" * 70)
print("BINARY MODEL OPTIMIZATION RESULTS")
print("=" * 70)

print(trials_df.to_string(index=False))

print("\nBest Medium weight:", best_weight)
print("Best validation Macro F1:", round(best_macro_f1, 4))


# ============================================================
# 12. RETRAIN ON TRAIN + VALIDATION
# ============================================================

print("\nRetraining selected model...")

X_development = pd.concat([X_train, X_val])

y_development = np.concatenate([y_train, y_val])

development_weights = np.where(
    y_development == label_encoder.transform(["Medium"])[0],
    best_weight,
    1.0
)

best_model.fit(
    X_development,
    y_development,
    classifier__sample_weight=development_weights
)

print("Retraining completed.")


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

test_predictions = best_model.predict(X_test).astype(int)

test_probabilities = best_model.predict_proba(X_test)

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

medium_index = list(label_encoder.classes_).index("Medium")

test_medium_probabilities = test_probabilities[:, medium_index]

test_roc_auc = roc_auc_score(
    y_test,
    test_medium_probabilities
)

test_average_precision = average_precision_score(
    y_test,
    test_medium_probabilities
)

print("\n" + "=" * 70)
print("FINAL LOW VS MEDIUM BINARY EVALUATION")
print("=" * 70)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test ROC AUC: {test_roc_auc:.4f}")
print(f"Test Medium Average Precision: {test_average_precision:.4f}")

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


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": best_model,
        "label_encoder": label_encoder,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_type": "low_medium_binary_xgboost",
        "best_medium_weight": best_weight,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_roc_auc": test_roc_auc,
        "test_average_precision": test_average_precision
    },
    MODEL_PATH
)


# ============================================================
# 15. SAVE TEST RESULTS
# ============================================================

test_results = pd.DataFrame({
    "ACTUAL_RISK": y_test_labels,
    "PREDICTED_RISK": test_pred_labels,
    "PROBABILITY_LOW": test_probabilities[:, 0],
    "PROBABILITY_MEDIUM": test_probabilities[:, 1]
})

test_results.to_csv(RESULTS_PATH, index=False)

print("\nSaved model:", MODEL_PATH)
print("Saved trials:", TRIALS_PATH)
print("Saved test predictions:", RESULTS_PATH)

print("\n" + "=" * 70)
print("LOW VS MEDIUM BINARY EXPERIMENT COMPLETED")
print("=" * 70)
