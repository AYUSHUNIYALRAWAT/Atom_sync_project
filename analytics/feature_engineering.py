
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" / "weather_raw_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip().str.upper()

    required_columns = [
        "TIMESTAMP",
        "CONTAINER_ID",
        "TEMPERATURE_C",
        "HUMIDITY_PCT",
        "VIBRATION_G",
        "SENSOR_STATUS",
        "TEMPERATURE_DEVIATION_C",
        "RISK_LABEL"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["TIMESTAMP", "CONTAINER_ID"]
    )

    df = df.sort_values(
        ["CONTAINER_ID", "TIMESTAMP"]
    ).reset_index(drop=True)

    return df


# ============================================================
# 3. CREATE TIME FEATURES
# ============================================================

def create_time_features(df):

    df["HOUR"] = df["TIMESTAMP"].dt.hour
    df["DAY"] = df["TIMESTAMP"].dt.day
    df["MONTH"] = df["TIMESTAMP"].dt.month

    df["DAY_OF_WEEK"] = (
        df["TIMESTAMP"].dt.day_of_week
    )

    df["IS_WEEKEND"] = (
        df["DAY_OF_WEEK"] >= 5
    ).astype(int)

    df["HOUR_SIN"] = np.sin(
        2 * np.pi * df["HOUR"] / 24
    )

    df["HOUR_COS"] = np.cos(
        2 * np.pi * df["HOUR"] / 24
    )

    return df


# ============================================================
# 4. CREATE ROLLING SENSOR FEATURES
# ============================================================

def create_rolling_features(df):

    sensor_columns = [
        "TEMPERATURE_C",
        "HUMIDITY_PCT",
        "VIBRATION_G",
        "TEMPERATURE_DEVIATION_C"
    ]

    grouped = df.groupby(
        "CONTAINER_ID",
        sort=False
    )

    for column in sensor_columns:

        df[f"{column}_ROLLING_MEAN_4"] = (
            grouped[column]
            .transform(
                lambda x: x.rolling(
                    window=4,
                    min_periods=1
                ).mean()
            )
        )

        df[f"{column}_ROLLING_STD_4"] = (
            grouped[column]
            .transform(
                lambda x: x.rolling(
                    window=4,
                    min_periods=2
                ).std()
            )
        )

    return df


# ============================================================
# 5. CREATE SENSOR CHANGE FEATURES
# ============================================================

def create_change_features(df):

    grouped = df.groupby(
        "CONTAINER_ID",
        sort=False
    )

    change_columns = [
        "TEMPERATURE_C",
        "HUMIDITY_PCT",
        "VIBRATION_G",
        "TEMPERATURE_DEVIATION_C"
    ]

    for column in change_columns:

        previous_column = f"{column}_PREVIOUS"
        change_column = f"{column}_CHANGE"

        df[previous_column] = (
            grouped[column].shift(1)
        )

        df[change_column] = (
            df[column] - df[previous_column]
        )

    return df


# ============================================================
# 6. ADVANCED TEMPERATURE DEVIATION FEATURES
# ============================================================

def create_advanced_temperature_features(df):

    grouped = df.groupby(
        "CONTAINER_ID",
        sort=False
    )["TEMPERATURE_DEVIATION_C"]

    # Previous temperature deviation readings
    df["TEMP_DEV_LAG_2"] = grouped.shift(2)

    df["TEMP_DEV_LAG_3"] = grouped.shift(3)

    # Rolling maximum
    df["TEMP_DEV_ROLLING_MAX_4"] = (
        grouped.transform(
            lambda x: x.rolling(
                window=4,
                min_periods=1
            ).max()
        )
    )

    # Rolling minimum
    df["TEMP_DEV_ROLLING_MIN_4"] = (
        grouped.transform(
            lambda x: x.rolling(
                window=4,
                min_periods=1
            ).min()
        )
    )

    # Absolute temperature deviation change
    df["TEMP_DEV_ABSOLUTE_CHANGE"] = (
        df["TEMPERATURE_DEVIATION_C_CHANGE"].abs()
    )

    # Acceleration of temperature deviation
    df["TEMP_DEV_ACCELERATION"] = (
        df.groupby(
            "CONTAINER_ID",
            sort=False
        )["TEMPERATURE_DEVIATION_C_CHANGE"].diff()
    )

    print("\nAdvanced temperature features created.")

    return df


# ============================================================
# 7. SENSOR STATUS FEATURES
# ============================================================

def create_sensor_features(df):

    df["SENSOR_STATUS"] = (
        df["SENSOR_STATUS"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Actual values in the dataset are OK and MISSING
    df["SENSOR_OPERATIONAL_FLAG"] = (
        df["SENSOR_STATUS"] == "OK"
    ).astype(int)

    return df


# ============================================================
# 8. RUN FEATURE ENGINEERING
# ============================================================

def main():

    print("=" * 60)
    print("ATMOSYNC ADVANCED FEATURE ENGINEERING")
    print("=" * 60)

    df = load_data()

    print("\nInput rows:", len(df))
    print("Input columns:", len(df.columns))

    # Step 1: Time features
    df = create_time_features(df)

    # Step 2: Rolling sensor features
    df = create_rolling_features(df)

    # Step 3: Previous readings and changes
    df = create_change_features(df)

    # Step 4: Advanced temperature features
    df = create_advanced_temperature_features(df)

    # Step 5: Sensor status features
    df = create_sensor_features(df)

    # Step 6: Save engineered dataset
    output_file = (
        OUTPUT_DIR / "weather_features.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print("\nFinal rows:", len(df))
    print("Final columns:", len(df.columns))

    print("\nNew advanced features:")

    advanced_features = [
        "TEMP_DEV_LAG_2",
        "TEMP_DEV_LAG_3",
        "TEMP_DEV_ROLLING_MAX_4",
        "TEMP_DEV_ROLLING_MIN_4",
        "TEMP_DEV_ABSOLUTE_CHANGE",
        "TEMP_DEV_ACCELERATION"
    ]

    print(df[advanced_features].head())

    print("\nOutput saved:")
    print(output_file)


if __name__ == "__main__":
    main()
