
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

DATASETS = [
    "containers_cleaned.csv",
    "weather_raw_cleaned.csv",
    "commodity_market_price_cleaned.csv",
    "market_routes_cleaned.csv"
]


def inspect_dataset(filename):

    file_path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"DATASET: {filename}")
    print("=" * 65)

    if not file_path.exists():
        print("File not found:", file_path)
        return

    df = pd.read_csv(file_path)

    print("\nShape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample records:")
    print(df.head(3).to_string(index=False))


def main():

    for filename in DATASETS:
        inspect_dataset(filename)


if __name__ == "__main__":
    main()
