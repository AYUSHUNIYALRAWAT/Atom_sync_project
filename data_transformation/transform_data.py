
import pandas as pd
from pathlib import Path


# -----------------------------------------
# 1. PROJECT PATHS
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "extracted"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. DATASETS
# -----------------------------------------

DATASETS = [
    "containers",
    "weather_raw",
    "commodity_market_price",
    "market_routes"
]


# -----------------------------------------
# 3. TRANSFORMATION FUNCTION
# -----------------------------------------

def transform_dataset(dataset_name):

    input_file = INPUT_DIR / f"{dataset_name}.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_file}"
        )

    print(f"\nProcessing: {dataset_name}")

    df = pd.read_csv(input_file)

    original_rows = len(df)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    # Remove exact duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()

    # Missing-value analysis
    missing_values = df.isnull().sum().sum()

    # Save processed dataset
    output_file = OUTPUT_DIR / f"{dataset_name}_cleaned.csv"

    df.to_csv(output_file, index=False)

    print(f"Original rows: {original_rows}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Missing values: {missing_values}")
    print(f"Final rows: {len(df)}")
    print(f"Saved: {output_file}")

    return {
        "Dataset": dataset_name,
        "Original_Rows": original_rows,
        "Duplicates_Removed": duplicate_count,
        "Missing_Values": missing_values,
        "Final_Rows": len(df)
    }


# -----------------------------------------
# 4. MAIN PIPELINE
# -----------------------------------------

def main():

    print("=" * 55)
    print("ATMOSYNC DATA TRANSFORMATION PIPELINE")
    print("=" * 55)

    results = []

    for dataset in DATASETS:
        result = transform_dataset(dataset)
        results.append(result)

    # Generate quality report
    report = pd.DataFrame(results)

    report_file = OUTPUT_DIR / "data_quality_report.csv"

    report.to_csv(report_file, index=False)

    print("\nDATA QUALITY REPORT")
    print(report.to_string(index=False))

    print("\nTransformation completed successfully!")


if __name__ == "__main__":
    main()
