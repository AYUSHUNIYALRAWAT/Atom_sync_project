
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "saved_models"
    / "xgboost_tuned_model.pkl"
)

OUTPUT_DIR = BASE_DIR / "data" / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load saved model bundle
saved = joblib.load(MODEL_PATH)

model = saved["model"]

numeric_features = saved["numeric_features"]
categorical_features = saved["categorical_features"]

# Get transformed feature names
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

# Get XGBoost importance
classifier = model.named_steps["classifier"]

importance_values = classifier.feature_importances_

importance_df = pd.DataFrame({
    "FEATURE": feature_names,
    "IMPORTANCE": importance_values
})

importance_df = importance_df.sort_values(
    by="IMPORTANCE",
    ascending=False
)

print("=" * 60)
print("ATMOSYNC FEATURE IMPORTANCE")
print("=" * 60)

print("\nTop 20 Important Features:")

print(
    importance_df.head(20).to_string(index=False)
)

# Save importance report
output_path = OUTPUT_DIR / "feature_importance.csv"

importance_df.to_csv(output_path, index=False)

print("\nFeature importance saved:")
print(output_path)

# Plot top 15 features
top_features = importance_df.head(15).sort_values(
    by="IMPORTANCE"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["FEATURE"],
    top_features["IMPORTANCE"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("AtmoSync - XGBoost Feature Importance")

plt.tight_layout()

plot_path = OUTPUT_DIR / "feature_importance.png"

plt.savefig(plot_path, dpi=300)
plt.show()

print("\nChart saved:")
print(plot_path)
