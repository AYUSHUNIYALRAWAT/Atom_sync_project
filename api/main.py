
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import reusable ML prediction pipeline
from models.prediction_pipeline import predict_risk


# Initialize FastAPI
app = FastAPI(
    title="AtmoSync Risk Prediction API",
    description=(
        "Hierarchical XGBoost container risk prediction "
        "using engineered sensor features."
    ),
    version="1.0.0",
)


# Input schema: 27 engineered features
class FeatureInput(BaseModel):

    TEMPERATURE_C: float | None
    HUMIDITY_PCT: float | None
    VIBRATION_G: float | None
    TEMPERATURE_DEVIATION_C: float | None

    TEMPERATURE_C_ROLLING_MEAN_4: float | None
    HUMIDITY_PCT_ROLLING_MEAN_4: float | None
    VIBRATION_G_ROLLING_MEAN_4: float | None
    TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4: float | None

    TEMPERATURE_C_ROLLING_STD_4: float | None
    HUMIDITY_PCT_ROLLING_STD_4: float | None
    VIBRATION_G_ROLLING_STD_4: float | None
    TEMPERATURE_DEVIATION_C_ROLLING_STD_4: float | None

    TEMPERATURE_C_PREVIOUS: float | None
    TEMPERATURE_C_CHANGE: float | None
    HUMIDITY_PCT_PREVIOUS: float | None
    HUMIDITY_PCT_CHANGE: float | None
    VIBRATION_G_PREVIOUS: float | None
    VIBRATION_G_CHANGE: float | None

    HOUR: float | None
    DAY_OF_WEEK: float | None

    TEMP_DEV_LAG_2: float | None
    TEMP_DEV_LAG_3: float | None
    TEMP_DEV_ROLLING_MAX_4: float | None
    TEMP_DEV_ROLLING_MIN_4: float | None
    TEMP_DEV_ABSOLUTE_CHANGE: float | None
    TEMP_DEV_ACCELERATION: float | None

    SENSOR_STATUS: Literal["OK", "MISSING"]


# Batch input schema
class BatchPredictionRequest(BaseModel):
    records: list[FeatureInput] = Field(
        min_length=1,
        max_length=500
    )


# Health endpoint
@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "project": "AtmoSync",
        "model": "Hierarchical XGBoost",
        "feature_count": 27
    }


# Single prediction endpoint
@app.post("/predict")
def predict_single(record: FeatureInput):

    try:
        input_df = pd.DataFrame(
            [record.model_dump()]
        )

        result = predict_risk(input_df)

        return {
            "prediction": result.to_dict(
                orient="records"
            )[0]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):

    try:
        input_df = pd.DataFrame(
            [
                record.model_dump()
                for record in request.records
            ]
        )

        results = predict_risk(input_df)

        return {
            "total_records": len(results),
            "predictions": results.to_dict(
                orient="records"
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# Run FastAPI server
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        reload=False
    )
