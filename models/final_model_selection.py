
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR = MODELS_DIR / "final_model_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["High", "Low", "Medium"]

# --------------------------------------------------
# 2. LOAD FINAL TEST PREDICTIONS
# --------------------------------------------------

baseline = pd.read_csv(
    MODELS_DIR / "error_analysis_results" / "all_test_predictions.csv"
)

hierarchical = pd.read_csv(
    MODELS_DIR / "hierarchical_results" / "hierarchical_test_predictions.csv"
)

optimized = pd.read_csv(
    MODELS_DIR / "hierarchical_threshold_results"
    / "hierarchical_optimized_test_results.csv"
)

# Standardize prediction column names
baseline = baseline[
    ["CONTAINER_ID", "TARGET_TIMESTAMP", "ACTUAL_RISK", "PREDICTED_RISK"]
].rename(columns={"PREDICTED_RISK": "Baseline_XGBoost"})

hierarchical = hierarchical[
    [
        "CONTAINER_ID",
        "TARGET_TIMESTAMP",
        "ACTUAL_RISK",
        "HIERARCHICAL_PREDICTION",
    ]
].rename(columns={"HIERARCHICAL_PREDICTION": "Hierarchical_XGBoost"})

optimized = optimized[
    ["CONTAINER_ID", "TARGET_TIMESTAMP", "ACTUAL_RISK", "PREDICTED_RISK"]
].rename(columns={"PREDICTED_RISK": "Optimized_Hierarchical"})

# --------------------------------------------------
# 3. ALIGN THE SAME TEST OBSERVATIONS
# --------------------------------------------------

keys = ["CONTAINER_ID", "TARGET_TIMESTAMP", "ACTUAL_RISK"]

comparison = baseline.merge(
    hierarchical,
    on=keys,
    how="inner",
    validate="one_to_one",
)

comparison = comparison.merge(
    optimized,
    on=keys,
    how="inner",
    validate="one_to_one",
)

if len(comparison) != len(baseline):
    raise ValueError(
        "Test prediction files do not contain identical observations. "
        "Check missing or duplicated records before comparing models."
    )

if len(comparison) != len(hierarchical) or len(comparison) != len(optimized):
    raise ValueError("The three models do not have the same test observations.")

y_true = comparison["ACTUAL_RISK"]

# --------------------------------------------------
# 4. EVALUATE ALL THREE MODELS
# --------------------------------------------------

prediction_columns = {
    "Baseline XGBoost": "Baseline_XGBoost",
    "Hierarchical XGBoost": "Hierarchical_XGBoost",
    "Optimized Hierarchical XGBoost": "Optimized_Hierarchical",
}

overall_results = []
class_results = []

for model_name, prediction_column in prediction_columns.items():

    y_pred = comparison[prediction_column]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    overall_results.append({
        "Model": model_name,
        "Test_Observations": len(comparison),
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
    })

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    for risk in LABELS:
        class_results.append({
            "Model": model_name,
            "Risk_Class": risk,
            "Precision": report[risk]["precision"],
            "Recall": report[risk]["recall"],
            "F1_Score": report[risk]["f1-score"],
            "Support": report[risk]["support"],
        })

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{x}" for x in LABELS],
        columns=[f"Predicted_{x}" for x in LABELS],
    )

    matrix_df.to_csv(
        OUTPUT_DIR / f"{prediction_column}_confusion_matrix.csv"
    )

# --------------------------------------------------
# 5. TEMPORAL VALIDATION COMPARISON
# --------------------------------------------------

temporal = pd.read_csv(
    MODELS_DIR / "temporal_validation_results" / "temporal_fold_metrics.csv"
)

temporal_columns = {
    "Baseline XGBoost": "baseline_macro_f1",
    "Hierarchical XGBoost": "hierarchical_macro_f1",
}

temporal_results = []

for model_name, column in temporal_columns.items():

    temporal_results.append({
        "Model": model_name,
        "Mean_Temporal_Macro_F1": temporal[column].mean(),
        "Std_Temporal_Macro_F1": temporal[column].std(),
        "Minimum_Temporal_Macro_F1": temporal[column].min(),
        "Maximum_Temporal_Macro_F1": temporal[column].max(),
        "Number_of_Folds": temporal[column].count(),
    })

temporal_summary = pd.DataFrame(temporal_results)

# --------------------------------------------------
# 6. SAVE COMPARISON RESULTS
# --------------------------------------------------

overall_df = pd.DataFrame(overall_results)
class_df = pd.DataFrame(class_results)

overall_df.to_csv(
    OUTPUT_DIR / "overall_model_comparison.csv",
    index=False,
)

class_df.to_csv(
    OUTPUT_DIR / "class_level_model_comparison.csv",
    index=False,
)

temporal_summary.to_csv(
    OUTPUT_DIR / "temporal_robustness_comparison.csv",
    index=False,
)

comparison.to_csv(
    OUTPUT_DIR / "aligned_test_predictions.csv",
    index=False,
)

# --------------------------------------------------
# 7. PRINT FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION REPORT")
print("=" * 70)

print("\nTEST SET OVERALL COMPARISON")
print(overall_df.round(4).to_string(index=False))

print("\nCLASS-LEVEL TEST COMPARISON")
print(class_df.round(4).to_string(index=False))

print("\nTEMPORAL VALIDATION ROBUSTNESS")
print(temporal_summary.round(4).to_string(index=False))

print("\nMODEL SELECTION INTERPRETATION")
print("-" * 70)

baseline_mean = temporal_summary.loc[
    temporal_summary["Model"] == "Baseline XGBoost",
    "Mean_Temporal_Macro_F1",
].iloc[0]

hierarchical_mean = temporal_summary.loc[
    temporal_summary["Model"] == "Hierarchical XGBoost",
    "Mean_Temporal_Macro_F1",
].iloc[0]

difference = hierarchical_mean - baseline_mean

print(f"Baseline temporal Macro F1:     {baseline_mean:.4f}")
print(f"Hierarchical temporal Macro F1:  {hierarchical_mean:.4f}")
print(f"Temporal Macro F1 improvement:   {difference:+.4f}")

if difference > 0:
    print(
        "\nPRELIMINARY RECOMMENDATION: "
        "Hierarchical XGBoost."
    )
    print(
        "It has better mean temporal Macro F1 than the baseline. "
        "Review High and Medium recall before deployment."
    )
else:
    print(
        "\nPRELIMINARY RECOMMENDATION: "
        "Baseline XGBoost."
    )
    print(
        "It has better mean temporal Macro F1 than the hierarchical model."
    )

print(
    "\nIMPORTANT: Optimized Hierarchical XGBoost has not been evaluated "
    "across the same three temporal folds."
)
print(
    "Its final test performance alone is not sufficient evidence "
    "of temporal robustness."
)

print("\nResults saved in:", OUTPUT_DIR)
print("\nFINAL MODEL COMPARISON COMPLETED")
