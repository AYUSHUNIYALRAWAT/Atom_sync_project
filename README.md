# AtmoSync

AtmoSync analyzes synthetic cold-chain sensor data and predicts container risk from temperature, humidity, vibration, and related engineered features.

## Project contents

- `AtmoSync_Synthetic_Dataset/`: synthetic container, sensor, route, and commodity price data (already present on the repository's main branch).
- `data/`, `data_extraction/`, and `data_transformation/`: data assets and preparation scripts.
- `models/`: risk analysis, model training, validation, and prediction pipeline code.
- `api/`: FastAPI prediction service.
- `dashboard/`: Streamlit dashboard.
- `data_preparation_and_validation/`: Snowflake SQL used to prepare and profile data.

## Setup

Use Python 3.10 or newer, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirement.txt
```

Configure Snowflake connection settings in a local `.env` file when using the Snowflake scripts. Do not commit credentials.

## Run

Start the API from the repository root with:

```bash
python api/main.py
```

Start the dashboard with:

```bash
streamlit run dashboard/app.py
```

Generated analysis CSVs and trained model binaries are excluded from this upload. The prediction pipeline requires trained model bundles at the paths configured in `models/prediction_pipeline.py`.
