
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

STAGE1_PATH = (
    BASE_DIR / "models" / "saved_models"
    / "xgboost_custom_weights_model.pkl"
)

STAGE2_PATH = (
    BASE_DIR / "models" / "low_medium_threshold_analysis"
    / "low_medium_threshold_model.pkl"
)

OUTPUT_DIR = BASE_DIR / "models" / "prediction_pipeline_results"


# ============================================================
# 2. LOAD MODEL BUNDLES
# ============================================================

def load_models():

    if not STAGE1_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found: {STAGE1_PATH}"
        )

    if not STAGE2_PATH.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found: {STAGE2_PATH}"
        )

    stage1_bundle = joblib.load(STAGE1_PATH)
    stage2_bundle = joblib.load(STAGE2_PATH)

    required_stage1 = {
        "model",
        "label_encoder",
        "numeric_features",
        "categorical_features",
    }

    required_stage2 = {
        "model",
        "label_encoder",
        "numeric_features",
        "categorical_features",
        "threshold",
    }

    if not required_stage1.issubset(stage1_bundle):
        raise ValueError("Stage 1 model bundle is incomplete.")

    if not required_stage2.issubset(stage2_bundle):
        raise ValueError("Stage 2 model bundle is incomplete.")

    if (
        stage1_bundle["numeric_features"]
        != stage2_bundle["numeric_features"]
    ):
        raise ValueError("Numeric feature definitions do not match.")

    if (
        stage1_bundle["categorical_features"]
        != stage2_bundle["categorical_features"]
    ):
        raise ValueError("Categorical feature definitions do not match.")

    threshold = float(stage2_bundle["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid Stage 2 threshold.")

    return stage1_bundle, stage2_bundle


# ============================================================
# 3. INITIALIZE MODELS
# ============================================================

STAGE1_BUNDLE, STAGE2_BUNDLE = load_models()

STAGE1_MODEL = STAGE1_BUNDLE["model"]
STAGE2_MODEL = STAGE2_BUNDLE["model"]

STAGE1_ENCODER = STAGE1_BUNDLE["label_encoder"]
STAGE2_ENCODER = STAGE2_BUNDLE["label_encoder"]

NUMERIC_FEATURES = STAGE1_BUNDLE["numeric_features"]
CATEGORICAL_FEATURES = STAGE1_BUNDLE["categorical_features"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MEDIUM_THRESHOLD = float(STAGE2_BUNDLE["threshold"])


# ============================================================
# 4. VALIDATE INPUT DATA
# ============================================================

def validate_input(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = df[FEATURE_COLUMNS].copy()

    # Numeric features must contain numeric values.
    for column in NUMERIC_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        if np.isinf(X[column].dropna()).any():
            raise ValueError(
                f"Infinite values found in feature: {column}"
            )

    # Normalize the categorical sensor status.
    for column in CATEGORICAL_FEATURES:

        X[column] = (
            X[column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    return X


# ============================================================
# 5. STAGE 1: MULTICLASS PREDICTION
# ============================================================

def predict_stage1(X):

    raw_predictions = STAGE1_MODEL.predict(X)

    predictions = STAGE1_ENCODER.inverse_transform(
        np.asarray(raw_predictions).astype(int)
    )

    raw_probabilities = STAGE1_MODEL.predict_proba(X)

    model_classes = np.asarray(
        STAGE1_MODEL.named_steps["classifier"].classes_
    ).astype(int)

    probability_df = pd.DataFrame(
        raw_probabilities,
        columns=STAGE1_ENCODER.inverse_transform(model_classes),
        index=X.index,
    )

    return predictions, probability_df


# ============================================================
# 6. STAGE 2: LOW VS MEDIUM REFINEMENT
# ============================================================

def predict_stage2(X, refinement_indices):

    if len(refinement_indices) == 0:
        return np.array([], dtype=object), np.array([])

    X_refinement = X.iloc[refinement_indices]

    raw_probabilities = STAGE2_MODEL.predict_proba(
        X_refinement
    )

    model_classes = np.asarray(
        STAGE2_MODEL.named_steps["classifier"].classes_
    ).astype(int)

    decoded_classes = STAGE2_ENCODER.inverse_transform(
        model_classes
    )

    medium_positions = np.where(
        decoded_classes == "Medium"
    )[0]

    if len(medium_positions) != 1:
        raise ValueError(
            "Could not identify the Medium class in Stage 2."
        )

    medium_index = medium_positions[0]

    medium_probabilities = raw_probabilities[:, medium_index]

    refined_predictions = np.where(
        medium_probabilities >= MEDIUM_THRESHOLD,
        "Medium",
        "Low",
    )

    return refined_predictions, medium_probabilities


# ============================================================
# 7. COMPLETE HIERARCHICAL PREDICTION
# ============================================================

def predict_risk(df):

    X = validate_input(df)

    # Stage 1
    stage1_predictions, stage1_probabilities = predict_stage1(X)

    final_predictions = stage1_predictions.copy()

    # Only Low/Medium predictions go to Stage 2.
    refinement_mask = np.isin(
        stage1_predictions,
        ["Low", "Medium"],
    )

    refinement_indices = np.where(
        refinement_mask
    )[0]

    # Stage 2
    refined_predictions, medium_probabilities = predict_stage2(
        X,
        refinement_indices,
    )

    if len(refinement_indices) > 0:
        final_predictions[refinement_indices] = refined_predictions

    # Prepare results
    results = pd.DataFrame(index=df.index)

    metadata_columns = [
        column
        for column in ["CONTAINER_ID", "TIMESTAMP"]
        if column in df.columns
    ]

    for column in metadata_columns:
        results[column] = df[column]

    results["STAGE1_PREDICTION"] = stage1_predictions

    results["PREDICTED_RISK"] = final_predictions

    # Stage 1 probabilities
    for risk in ["High", "Low", "Medium"]:
        results[f"STAGE1_PROBABILITY_{risk.upper()}"] = (
            stage1_probabilities[risk].to_numpy()
        )

    # Stage 2 probability is conditional on being sent to Stage 2.
    results["STAGE2_MEDIUM_PROBABILITY"] = np.nan

    if len(refinement_indices) > 0:
        results.iloc[
            refinement_indices,
            results.columns.get_loc(
                "STAGE2_MEDIUM_PROBABILITY"
            ),
        ] = medium_probabilities

    return results


# ============================================================
# 8. RUN A TEST USING ENGINEERED DATA
# ============================================================

def main():

    print("=" * 65)
    print("ATMOSYNC REUSABLE RISK PREDICTION PIPELINE")
    print("=" * 65)

    print("\nStage 1 model loaded.")
    print("Stage 2 model loaded.")

    print("Required features:", len(FEATURE_COLUMNS))
    print("Stage 2 Medium threshold:", MEDIUM_THRESHOLD)

    input_path = (
        BASE_DIR / "data" / "analytics"
        / "weather_features.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        errors="coerce",
    )

    print("\nInput observations:", len(df))

    predictions = predict_risk(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / "risk_predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print("\nPrediction distribution:")
    print(
        predictions["PREDICTED_RISK"]
        .value_counts()
        .to_string()
    )

    print("\nPrediction sample:")
    print(
        predictions.head(10).to_string(index=False)
    )

    print("\nSaved predictions:", output_path)

    print("\nPREDICTION PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
