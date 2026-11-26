
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "profiling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "containers_cleaned.csv",
    "weather_raw_cleaned.csv",
    "commodity_market_price_cleaned.csv",
    "market_routes_cleaned.csv"
]


def profile_dataset(filename):

    file_path = INPUT_DIR / filename

    if not file_path.exists():
        print(f"File not found: {filename}")
        return None

    df = pd.read_csv(file_path)

    print(f"\nProfiling: {filename}")

    profile = []

    for column in df.columns:

        profile.append({
            "Dataset": filename,
            "Column": column,
            "Data_Type": str(df[column].dtype),
            "Missing_Count": df[column].isnull().sum(),
            "Missing_Percentage": round(
                df[column].isnull().mean() * 100, 2
            ),
            "Unique_Values": df[column].nunique(),
            "Duplicate_Count": df[column].duplicated().sum(),
            "Minimum": (
                df[column].min()
                if pd.api.types.is_numeric_dtype(df[column])
                else None
            ),
            "Maximum": (
                df[column].max()
                if pd.api.types.is_numeric_dtype(df[column])
                else None
            )
        })

    profile_df = pd.DataFrame(profile)

    return profile_df


def main():

    all_profiles = []

    for filename in DATASETS:

        result = profile_dataset(filename)

        if result is not None:
            all_profiles.append(result)

    if all_profiles:

        final_report = pd.concat(
            all_profiles,
            ignore_index=True
        )

        output_file = OUTPUT_DIR / "data_profiling_report.csv"

        final_report.to_csv(output_file, index=False)

        print("\nData profiling completed!")
        print("Report saved:", output_file)


if __name__ == "__main__":
    main()
