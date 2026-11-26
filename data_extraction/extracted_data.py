
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Add project root to Python path
sys.path.insert(0, str(BASE_DIR))

# Import reusable Snowflake connection
from snowflake_connection.snowflake_connection import get_connection


# --------------------------------------------------
# 2. OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = BASE_DIR / "data" / "extracted"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. TABLES TO EXTRACT
# --------------------------------------------------

TABLES = [
    "CONTAINERS",
    "WEATHER_RAW",
    "COMMODITY_MARKET_PRICE",
    "MARKET_ROUTES"
]


# --------------------------------------------------
# 4. DATA EXTRACTION FUNCTION
# --------------------------------------------------

def extract_table(conn, table_name):

    """
    Extract a Snowflake table into a Pandas DataFrame
    and save it as a CSV file.
    """

    cursor = None

    try:

        print(f"\nExtracting table: {table_name}")

        cursor = conn.cursor()

        query = f"SELECT * FROM {table_name}"

        cursor.execute(query)

        # Fetch data into Pandas DataFrame
        df = cursor.fetch_pandas_all()

        # Output CSV path
        output_file = OUTPUT_DIR / f"{table_name.lower()}.csv"

        # Save extracted data
        df.to_csv(output_file, index=False)

        print(f"Successfully extracted: {table_name}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Saved file: {output_file}")

        return df

    except Exception as error:

        print(f"Error extracting {table_name}: {error}")

        raise

    finally:

        if cursor:
            cursor.close()


# --------------------------------------------------
# 5. MAIN EXTRACTION PIPELINE
# --------------------------------------------------

def main():

    conn = None

    try:

        print("=" * 55)
        print("ATMOSYNC - SNOWFLAKE DATA EXTRACTION PIPELINE")
        print("=" * 55)

        # Establish Snowflake connection
        conn = get_connection()

        print("\nSnowflake connection successful!")

        extraction_summary = []

        # Extract all tables
        for table_name in TABLES:

            df = extract_table(conn, table_name)

            extraction_summary.append({
                "Table": table_name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Status": "SUCCESS"
            })

        # Display extraction summary
        summary_df = pd.DataFrame(extraction_summary)

        print("\n" + "=" * 55)
        print("EXTRACTION SUMMARY")
        print("=" * 55)

        print(summary_df.to_string(index=False))

        print("\nAll tables extracted successfully!")

        print(f"\nOutput directory: {OUTPUT_DIR}")

    except Exception as error:

        print("\nDATA EXTRACTION PIPELINE FAILED")
        print("Error:", error)

        raise

    finally:

        if conn:
            conn.close()
            print("\nSnowflake connection closed.")


# --------------------------------------------------
# 6. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    main()
