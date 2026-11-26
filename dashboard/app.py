
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AtmoSync | Risk Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# ATMOSYNC — PREMIUM LIGHT THEME
# ==========================================

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #F5F7FB;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Main headings */
    h1 {
        color: #14213D !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #1E293B !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] p {
        color: #334155;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #14213D;
        font-weight: 750;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        transition: 0.2s ease;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: #E2E8F0;
    }

    /* Captions */
    .stCaption {
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)





# ==================================================
# AUTOMATIC DASHBOARD REFRESH
# ==================================================

REFRESH_INTERVAL = 60_000  # 60 seconds

refresh_count = st_autorefresh(
    interval=REFRESH_INTERVAL,
    limit=None,
    key="atmosync_auto_refresh"
)

last_refresh_time = pd.Timestamp.now()

st.caption(
    f"Last dashboard refresh: "
    f"{last_refresh_time.strftime('%d %b %Y, %I:%M:%S %p')}"
)

if st.button("Refresh Dashboard Now", key="manual_refresh"):
    st.cache_data.clear()
    st.rerun()



# ==================================================
# 2. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    PROJECT_ROOT
    / "models"
    / "prediction_pipeline_results"
    / "risk_predictions.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "analytics"
    / "weather_features.csv"
)

HISTORY_DIR = (
    PROJECT_ROOT
    / "models"
    / "prediction_history"
)

HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"

API_URL = "http://127.0.0.1:8001/predict"
API_HEALTH_URL = "http://127.0.0.1:8001/health"

# ==================================================
# FASTAPI HEALTH MONITORING
# ==================================================

st.markdown("### System Status")

try:

    health_response = requests.get(
        API_HEALTH_URL,
        timeout=5
    )

    health_response.raise_for_status()

    health_data = health_response.json()

    col1, col2, col3 = st.columns(3)

    col1.success("FastAPI: Connected")

    col2.metric(
        "Backend Status",
        health_data.get("status", "Unknown").title()
    )

    col3.metric(
        "Model",
        health_data.get("model", "Unknown")
    )

except requests.exceptions.RequestException:

    st.error(
        "FastAPI is unavailable. "
        "Start the backend on port 8001."
    )

st.divider()


# ==================================================
# 3. MODEL INPUT FEATURES
# ==================================================

MODEL_FEATURES = [
    "TEMPERATURE_C",
    "HUMIDITY_PCT",
    "VIBRATION_G",
    "TEMPERATURE_DEVIATION_C",
    "TEMPERATURE_C_ROLLING_MEAN_4",
    "HUMIDITY_PCT_ROLLING_MEAN_4",
    "VIBRATION_G_ROLLING_MEAN_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_MEAN_4",
    "TEMPERATURE_C_ROLLING_STD_4",
    "HUMIDITY_PCT_ROLLING_STD_4",
    "VIBRATION_G_ROLLING_STD_4",
    "TEMPERATURE_DEVIATION_C_ROLLING_STD_4",
    "TEMPERATURE_C_PREVIOUS",
    "TEMPERATURE_C_CHANGE",
    "HUMIDITY_PCT_PREVIOUS",
    "HUMIDITY_PCT_CHANGE",
    "VIBRATION_G_PREVIOUS",
    "VIBRATION_G_CHANGE",
    "HOUR",
    "DAY_OF_WEEK",
    "TEMP_DEV_LAG_2",
    "TEMP_DEV_LAG_3",
    "TEMP_DEV_ROLLING_MAX_4",
    "TEMP_DEV_ROLLING_MIN_4",
    "TEMP_DEV_ABSOLUTE_CHANGE",
    "TEMP_DEV_ACCELERATION"
]

REQUIRED_COLUMNS = MODEL_FEATURES + ["SENSOR_STATUS"]


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

@st.cache_data(ttl=60)
def load_data():

    predictions = pd.read_csv(PREDICTION_FILE)

    features = None

    if FEATURE_FILE.exists():

        features = pd.read_csv(FEATURE_FILE)

    return predictions, features


def save_prediction_history(record):
    """
    Save a live prediction safely.

    Rewrites the CSV with headers to avoid problems
    with empty or corrupted history files.
    """

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    new_record = pd.DataFrame([record])

    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:

        try:

            existing_history = pd.read_csv(HISTORY_FILE)

            if not existing_history.empty:

                history_df = pd.concat(
                    [existing_history, new_record],
                    ignore_index=True
                )

            else:

                history_df = new_record

        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError
        ):

            history_df = new_record

    else:

        history_df = new_record

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )

HISTORY_DIR = PROJECT_ROOT / "models" / "prediction_history"
HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"

HISTORY_COLUMNS = [
    "PREDICTION_TIME",
    "RECORD_INDEX",
    "PREDICTED_RISK",
    "STAGE1_PROBABILITY_HIGH",
    "STAGE1_PROBABILITY_LOW",
    "STAGE1_PROBABILITY_MEDIUM",
    "STAGE2_MEDIUM_PROBABILITY"
]





def load_prediction_history():
    """
    Load prediction history and repair CSV files
    that were previously saved without headers.
    """

    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    if HISTORY_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        # First, try reading the CSV normally.
        history_df = pd.read_csv(HISTORY_FILE)

        if list(history_df.columns) == HISTORY_COLUMNS:
            return history_df

        # If headers are missing, read the file without headers.
        history_df = pd.read_csv(
            HISTORY_FILE,
            header=None
        )

        # Validate the number of columns.
        if len(history_df.columns) != len(HISTORY_COLUMNS):
            raise ValueError(
                "Unexpected number of columns in prediction history."
            )

        # Assign the correct column names.
        history_df.columns = HISTORY_COLUMNS

        return history_df

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError
    ) as error:

        st.error(
            f"Unable to load prediction history: {error}"
        )

        return pd.DataFrame(columns=HISTORY_COLUMNS)



def format_probability(value):

    if value is None:

        return "N/A"

    try:

        if pd.isna(value):

            return "N/A"

        return f"{float(value):.2%}"

    except (ValueError, TypeError):

        return "N/A"


def normalize_risk(value):

    return str(value).strip().title()


# ==================================================
# 5. HEADER
# ==================================================

# ==========================================
# ATMOSYNC — PREMIUM DASHBOARD HEADER
# ==========================================

st.html("""
<div style="
    background: linear-gradient(120deg, #14213D, #243B64);
    padding: 30px 34px;
    border-radius: 16px;
    margin-bottom: 22px;
    box-shadow: 0 5px 18px rgba(20, 33, 61, 0.12);
">
    <div style="
        color: #93C5FD;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    ">
        MICRO-CLIMATE INTELLIGENCE PLATFORM
    </div>

    <div style="
        color: #FFFFFF;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -1px;
    ">
        🌍 ATMOSYNC
    </div>

    <div style="
        color: #DCE7F7;
        font-size: 15px;
        margin-top: 8px;
    ">
        Micro-Climate Arbitrage Analytics
    </div>

    <div style="
        color: #B9C9DF;
        font-size: 12px;
        margin-top: 14px;
    ">
        CONTAINER RISK INTELLIGENCE
        | HIERARCHICAL XGBOOST
        | DECISION SUPPORT
    </div>
</div>
""")



# ==================================================
# 6. LOAD DATA
# ==================================================

if not PREDICTION_FILE.exists():

    st.error(
        f"Prediction file not found: {PREDICTION_FILE}"
    )

    st.stop()


try:

    predictions, features = load_data()

except Exception as error:

    st.error(
        f"Unable to load project data: {error}"
    )

    st.stop()


# ==================================================
# 7. VALIDATE PREDICTION DATA
# ==================================================

if "PREDICTED_RISK" not in predictions.columns:

    st.error(
        "PREDICTED_RISK column is missing "
        "from risk_predictions.csv."
    )

    st.write(
        "Available columns:",
        predictions.columns.tolist()
    )

    st.stop()


#
# ==================================================
# 8. INTERACTIVE GLOBAL FILTERS
# ==================================================

st.sidebar.title("Dashboard Controls")

# ------------------------------------------
# RISK CATEGORY FILTER
# ------------------------------------------

st.sidebar.markdown("### Risk Category")

risk_options = sorted(
    predictions["PREDICTED_RISK"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_risks = st.sidebar.multiselect(
    "Select Risk Categories",
    options=risk_options,
    default=risk_options,
    key="global_risk_filter"
)

# ------------------------------------------
# CONTAINER FILTER
# ------------------------------------------

st.sidebar.markdown("### Container Selection")

if "CONTAINER_ID" in predictions.columns:

    container_options = sorted(
        predictions["CONTAINER_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_containers = st.sidebar.multiselect(
        "Select Containers",
        options=container_options,
        default=container_options,
        key="global_container_filter"
    )

else:
    selected_containers = None

    st.sidebar.info(
        "Container ID is unavailable."
    )

# ------------------------------------------
# DATE FILTER
# ------------------------------------------

st.sidebar.markdown("### Date Range")

time_column = next(
    (
        col
        for col in ["TARGET_TIMESTAMP", "TIMESTAMP"]
        if col in predictions.columns
    ),
    None
)

if time_column:

    date_values = pd.to_datetime(
        predictions[time_column],
        errors="coerce"
    ).dropna()

    if not date_values.empty:

        min_date = date_values.min().date()
        max_date = date_values.max().date()

        selected_dates = st.sidebar.date_input(
            "Select Analysis Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="global_date_filter"
        )

    else:
        selected_dates = None

else:
    selected_dates = None

    st.sidebar.info(
        "No timestamp column is available for date filtering."
    )

# ------------------------------------------
# APPLY ALL GLOBAL FILTERS
# ------------------------------------------

filtered_data = predictions.copy()

# Risk filter.
filtered_data = filtered_data[
    filtered_data["PREDICTED_RISK"]
    .astype(str)
    .isin(selected_risks)
]

# Container filter.
if selected_containers is not None:

    filtered_data = filtered_data[
        filtered_data["CONTAINER_ID"]
        .astype(str)
        .isin(selected_containers)
    ]

# Date filter.
if time_column and selected_dates:

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_timestamps = pd.to_datetime(
            filtered_data[time_column],
            errors="coerce"
        )

        filtered_data = filtered_data[
            filtered_timestamps.dt.date.between(
                start_date,
                end_date
            )
        ]



# ==========================================
# ATMOSYNC — COMPLETE SIDEBAR
# ==========================================

# BRANDING
st.sidebar.title("ATMOSYNC")
st.sidebar.caption("Micro-Climate Arbitrage Analytics")
st.sidebar.divider()

# NAVIGATION
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Executive Overview",
        "Risk Intelligence",
        "Container Monitoring",
        "Environmental Analytics",
        "Live Prediction",
        "Prediction History",
        "Data Explorer"
    ],
    key="atmosync_main_navigation"
)

st.sidebar.divider()

# DATA INFORMATION
st.sidebar.markdown("### Data Information")

st.sidebar.metric(
    "Total Records",
    f"{len(predictions):,}"
)

st.sidebar.metric(
    "Filtered Records",
    f"{len(filtered_data):,}"
)

if features is not None:
    st.sidebar.metric(
        "Feature Records",
        f"{len(features):,}"
    )

st.sidebar.caption(
    "Filters apply to the dashboard analytics."
)

st.sidebar.divider()

# MODEL INFORMATION
st.sidebar.markdown("### Model Information")

st.sidebar.caption("Model: Hierarchical XGBoost")
st.sidebar.caption("Input Features: 27")
st.sidebar.caption("Backend: FastAPI")

st.sidebar.divider()

# FOOTER
st.sidebar.caption("ATMOSYNC | Version 1.0")
st.sidebar.caption("Micro-Climate Intelligence Platform")

if page == "Executive Overview":
    
    # ==================================================
    # 9. DYNAMIC EXECUTIVE OVERVIEW
    # ==================================================

    st.markdown("## Executive Overview")

# ------------------------------------------
# CALCULATE DYNAMIC METRICS
# ------------------------------------------

total_records = len(filtered_data)

if "CONTAINER_ID" in filtered_data.columns:
    total_containers = filtered_data["CONTAINER_ID"].nunique()
else:
    total_containers = 0

risk_labels = (
    filtered_data["PREDICTED_RISK"]
    .astype(str)
    .str.strip()
    .str.lower()
)

high_risk = risk_labels.eq("high").sum()
medium_risk = risk_labels.eq("medium").sum()
low_risk = risk_labels.eq("low").sum()

high_percentage = (
    high_risk / total_records * 100
    if total_records else 0
)

medium_percentage = (
    medium_risk / total_records * 100
    if total_records else 0
)

low_percentage = (
    low_risk / total_records * 100
    if total_records else 0
)

# ------------------------------------------
# KPI CARDS
# ------------------------------------------

st.caption("Live analytics based on selected dashboard filters.")



# ==========================================
# PREMIUM KPI CARDS — FIXED HTML RENDERING
# ==========================================

kpi_columns = st.columns(5, gap="medium")

kpi_data = [
    ("TOTAL PREDICTIONS", f"{total_records:,}", "All filtered records"),
    ("CONTAINERS", f"{total_containers:,}", "Unique containers"),
    ("HIGH RISK", f"{high_risk:,}", f"{high_percentage:.1f}% of records"),
    ("MEDIUM RISK", f"{medium_risk:,}", f"{medium_percentage:.1f}% of records"),
    ("LOW RISK", f"{low_risk:,}", f"{low_percentage:.1f}% of records")
]

for column, (label, value, description) in zip(kpi_columns, kpi_data):

    with column:

        st.html(f"""
        <div style="
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 22px 16px;
            min-height: 125px;
            box-shadow: 0 3px 12px rgba(15,23,42,0.05);
        ">

            <div style="
                color: #64748B;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.7px;
            ">
                {label}
            </div>

            <div style="
                color: #14213D;
                font-size: 29px;
                font-weight: 800;
                margin-top: 14px;
            ">
                {value}
            </div>

            <div style="
                color: #94A3B8;
                font-size: 11px;
                margin-top: 7px;
            ">
                {description}
            </div>

        </div>
        """)


# ==========================================
# PREMIUM RISK COMPOSITION
# ==========================================

st.markdown("### Risk Composition")

risk_cards = [
    ("HIGH RISK", high_risk, high_percentage, "#DC5555"),
    ("MEDIUM RISK", medium_risk, medium_percentage, "#D69B32"),
    ("LOW RISK", low_risk, low_percentage, "#299D70")
]

risk_html = ""

for label, count, percentage, color in risk_cards:

    risk_html += f"""
    <div style="
        background:#FFFFFF;
        padding:20px;
        border:1px solid #E2E8F0;
        border-radius:12px;
        margin-bottom:14px;
    ">
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:12px;
        ">
            <span style="
                color:#334155;
                font-size:13px;
                font-weight:700;
            ">{label}</span>

            <span style="
                color:#14213D;
                font-size:18px;
                font-weight:800;
            ">{count:,} records</span>
        </div>

        <div style="
            background:#E2E8F0;
            height:11px;
            border-radius:20px;
            overflow:hidden;
        ">
            <div style="
                width:{percentage:.2f}%;
                background:{color};
                height:11px;
                border-radius:20px;
            "></div>
        </div>

        <div style="
            color:#64748B;
            font-size:12px;
            margin-top:9px;
        ">{percentage:.2f}% of filtered predictions</div>
    </div>
    """

st.html(risk_html)

st.divider()


# ==================================================
# 10. INTERACTIVE RISK INTELLIGENCE
# ==================================================

st.markdown("## Risk Intelligence")

if filtered_data.empty:

    st.info("No records available for the selected filters.")

else:

    risk_counts = (
        filtered_data["PREDICTED_RISK"]
        .astype(str)
        .str.strip()
        .str.title()
        .value_counts()
        .rename_axis("Risk Category")
        .reset_index(name="Records")
    )

    risk_colors = {
        "High": "#DC5555",
        "Medium": "#D69B32",
        "Low": "#299D70"
    }

    risk_counts["Color"] = (
        risk_counts["Risk Category"].map(risk_colors)
    )

    total = risk_counts["Records"].sum()

    risk_counts["Percentage"] = (
        risk_counts["Records"] / total * 100
    )

    col1, col2 = st.columns(2)

    # ------------------------------------------
    # INTERACTIVE DONUT CHART
    # ------------------------------------------

    with col1:

        st.markdown("### Risk Distribution")

        fig = px.pie(
            risk_counts,
            names="Risk Category",
            values="Records",
            hole=0.58,
            color="Risk Category",
            color_discrete_map=risk_colors,
            custom_data=["Percentage"],
        )

        fig.update_traces(
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Records: %{value:,}<br>"
                "Percentage: %{customdata[0]:.2f}%"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="Arial",
                color="#334155",
                size=13
            ),
            title_font=dict(
                size=18,
                color="#14213D"
            ),
            legend_title_text="Risk Category",
            margin=dict(t=45, b=25, l=25, r=25)
        )

        st.plotly_chart(
            fig,
            width="stretch",
            theme="streamlit",
            key="risk_donut_chart"
        )


    # ------------------------------------------
    # INTERACTIVE BAR CHART
    # ------------------------------------------

    with col2:

        st.markdown("### Risk Category Comparison")

        fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Records",
            color="Risk Category",
            color_discrete_map=risk_colors,
            text="Records",
            custom_data=["Percentage"],
        )

        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Records: %{y:,}<br>"
                "Percentage: %{customdata[0]:.2f}%"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_white",
            height=420,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="Arial",
                color="#334155",
                size=13
            ),
            title_font=dict(
                size=18,
                color="#14213D"
            ),
            showlegend=False,
            xaxis_title="Risk Category",
            yaxis_title="Number of Records",
            yaxis=dict(
                gridcolor="#E9EEF5",
                zeroline=False
            ),
            margin=dict(t=45, b=45, l=35, r=25)
        )

        st.plotly_chart(
            fig,
            width="stretch",
            theme="streamlit",
            key="risk_bar_chart"
        )

    st.divider()

    # ------------------------------------------
    # RISK SUMMARY TABLE
    # ------------------------------------------

    st.markdown("### Risk Summary")

    st.dataframe(
        risk_counts[
            ["Risk Category", "Records", "Percentage"]
        ],
        width="stretch",
        hide_index=True
    )


# ==================================================
# 11. TEMPORAL RISK ANALYSIS
# ==================================================

st.markdown("### Temporal Risk Analysis")

time_column = next(
    (
        col
        for col in ["TARGET_TIMESTAMP", "TIMESTAMP"]
        if col in filtered_data.columns
    ),
    None
)

if time_column:

    temporal_data = filtered_data.copy()

    temporal_data[time_column] = pd.to_datetime(
        temporal_data[time_column],
        errors="coerce"
    )

    temporal_data = temporal_data.dropna(
        subset=[time_column]
    )

    if not temporal_data.empty:

        temporal_data["TIME_INTERVAL"] = (
            temporal_data[time_column].dt.floor("h")
        )

        trend = (
            temporal_data
            .groupby(
                ["TIME_INTERVAL", "PREDICTED_RISK"]
            )
            .size()
            .reset_index(name="Records")
        )

        fig = px.line(
            trend,
            x="TIME_INTERVAL",
            y="Records",
            color="PREDICTED_RISK",
            markers=True,
            title="Risk Predictions Over Time"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "No valid timestamps available."
        )

else:

    st.info(
        "Temporal analysis requires TIMESTAMP "
        "or TARGET_TIMESTAMP in the prediction file."
    )

st.divider()


if page == "Container Monitoring":

    # ==================================================
    # 12. CONTAINER INTELLIGENCE & DRILL-DOWN
    # ==================================================

    st.markdown("## Container Intelligence")

    if "CONTAINER_ID" not in filtered_data.columns:

        st.warning(
            "CONTAINER_ID is unavailable in the prediction dataset."
        )

    elif filtered_data.empty:

        st.info("No containers match the selected global filters.")

    else:

        container_options = sorted(
            filtered_data["CONTAINER_ID"]
            .dropna()
            .unique()
            .tolist(),
            key=str
        )

        selected_container = st.selectbox(
            "Select Container for Detailed Analysis",
            options=container_options,
            key="container_drilldown"
        )

        container_data = filtered_data[
            filtered_data["CONTAINER_ID"] == selected_container
        ].copy()

        # ------------------------------------------
        # CONTAINER RISK METRICS
        # ------------------------------------------

        container_risks = (
            container_data["PREDICTED_RISK"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        container_total = len(container_data)

        container_high = container_risks.eq("high").sum()
        container_medium = container_risks.eq("medium").sum()
        container_low = container_risks.eq("low").sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Container ID", str(selected_container))
        col2.metric("Total Records", f"{container_total:,}")
        col3.metric("High Risk Records", f"{container_high:,}")
        col4.metric("Low Risk Records", f"{container_low:,}")

        # ------------------------------------------
        # CONTAINER RISK DISTRIBUTION
        # ------------------------------------------

        st.markdown("### Container Risk Distribution")

        container_summary = pd.DataFrame({
            "Risk Category": ["High", "Medium", "Low"],
            "Records": [
                container_high,
                container_medium,
                container_low
            ]
        })

        container_colors = {
            "High": "#DC5555",
            "Medium": "#D69B32",
            "Low": "#299D70"
        }

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                container_summary,
                names="Risk Category",
                values="Records",
                hole=0.55,
                color="Risk Category",
                color_discrete_map=container_colors,
                title="Selected Container Risk Profile"
            )

            fig.update_layout(
                template="plotly_white",
                height=380
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.bar(
                container_summary,
                x="Risk Category",
                y="Records",
                color="Risk Category",
                color_discrete_map=container_colors,
                text="Records",
                title="Risk Records by Category"
            )

            fig.update_layout(
                template="plotly_white",
                height=380,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        # ------------------------------------------
        # CONTAINER HISTORICAL RISK TREND
        # ------------------------------------------

        st.markdown("### Historical Risk Trend")

        container_time_column = next(
            (
                col
                for col in ["TARGET_TIMESTAMP", "TIMESTAMP"]
                if col in container_data.columns
            ),
            None
        )

        if container_time_column:

            trend_data = container_data.copy()

            trend_data[container_time_column] = pd.to_datetime(
                trend_data[container_time_column],
                errors="coerce"
            )

            trend_data = trend_data.dropna(
                subset=[container_time_column]
            )

            if not trend_data.empty:

                trend_data["TIME_INTERVAL"] = (
                    trend_data[container_time_column].dt.floor("h")
                )

                trend_summary = (
                    trend_data
                    .groupby(
                        ["TIME_INTERVAL", "PREDICTED_RISK"]
                    )
                    .size()
                    .reset_index(name="Records")
                )

                fig = px.line(
                    trend_summary,
                    x="TIME_INTERVAL",
                    y="Records",
                    color="PREDICTED_RISK",
                    markers=True,
                    title=f"Risk Trend — {selected_container}"
                )

                fig.update_layout(
                    template="plotly_white",
                    height=420,
                    xaxis_title="Time",
                    yaxis_title="Number of Records"
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            else:

                st.info("No valid timestamps for this container.")

        else:

            st.info(
                "Historical risk analysis requires a timestamp column."
            )

        # ------------------------------------------
        # CONTAINER RECORD EXPLORER
        # ------------------------------------------

        st.markdown("### Container Records")

        st.dataframe(
            container_data,
            width="stretch",
            hide_index=True
        )

    st.divider()


    # ==================================================
    # SELECTED CONTAINER SENSOR ANALYSIS
    # ==================================================

    st.markdown("### Selected Container Sensor Analytics")

    if (
        features is not None
        and "CONTAINER_ID" in features.columns
        and "CONTAINER_ID" in filtered_data.columns
        and not filtered_data.empty
    ):

        sensor_container_data = features[
            features["CONTAINER_ID"].astype(str)
            == str(selected_container)
        ].copy()

        sensor_metrics = [
            "TEMPERATURE_C",
            "HUMIDITY_PCT",
            "VIBRATION_G"
        ]

        available_metrics = [
            col
            for col in sensor_metrics
            if col in sensor_container_data.columns
        ]

        if not sensor_container_data.empty and available_metrics:

            selected_metric = st.selectbox(
                "Choose Sensor Metric",
                available_metrics,
                key="container_sensor_metric"
            )

            sensor_time_column = (
                "TIMESTAMP"
                if "TIMESTAMP" in sensor_container_data.columns
                else None
            )

            if sensor_time_column:

                sensor_container_data[sensor_time_column] = (
                    pd.to_datetime(
                        sensor_container_data[sensor_time_column],
                        errors="coerce"
                    )
                )

                sensor_container_data = (
                    sensor_container_data
                    .dropna(subset=[sensor_time_column])
                    .sort_values(sensor_time_column)
                )

                fig = px.line(
                    sensor_container_data,
                    x=sensor_time_column,
                    y=selected_metric,
                    title=f"{selected_metric} — {selected_container}",
                    markers=True
                )

                fig.update_layout(
                    template="plotly_white",
                    height=420,
                    xaxis_title="Timestamp",
                    yaxis_title=selected_metric
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

            else:

                st.info(
                    "TIMESTAMP is unavailable for sensor trend analysis."
                )

        else:

            st.info(
                "No matching sensor records or sensor metrics found."
            )

    else:

        st.info(
            "Container-level sensor analysis requires "
            "CONTAINER_ID in weather_features.csv."
        )

    st.divider()


if page == "Environmental Analytics":
    #
# ==================================================
# 13. ADVANCED ENVIRONMENTAL SENSOR ANALYTICS
# ==================================================
 
    st.markdown("## Environmental Sensor Intelligence")

if features is None:

    st.warning(
        "Weather feature dataset is unavailable."
    )

else:

    sensor_data = features.copy()

    # ------------------------------------------
    # AVAILABLE SENSOR METRICS
    # ------------------------------------------

    sensor_metrics = [
        "TEMPERATURE_C",
        "HUMIDITY_PCT",
        "VIBRATION_G",
        "TEMPERATURE_DEVIATION_C"
    ]

    available_sensors = [
        column
        for column in sensor_metrics
        if column in sensor_data.columns
    ]

    if not available_sensors:

        st.warning(
            "No expected sensor metrics were found."
        )

    else:

        # ------------------------------------------
        # SENSOR CONTROLS
        # ------------------------------------------

        st.markdown("### Sensor Controls")

        col1, col2 = st.columns(2)

        with col1:

            selected_sensor = st.selectbox(
                "Select Sensor Metric",
                available_sensors,
                key="advanced_sensor_metric"
            )

        with col2:

            chart_type = st.selectbox(
                "Select Visualization",
                [
                    "Time Series",
                    "Distribution",
                    "Box Plot"
                ],
                key="sensor_chart_type"
            )

        # Convert selected sensor to numeric.
        sensor_data[selected_sensor] = pd.to_numeric(
            sensor_data[selected_sensor],
            errors="coerce"
        )

        sensor_data = sensor_data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        sensor_data = sensor_data.dropna(
            subset=[selected_sensor]
        )

        # ------------------------------------------
        # SENSOR DATE FILTER
        # ------------------------------------------

        if "TIMESTAMP" in sensor_data.columns:

            sensor_data["TIMESTAMP"] = pd.to_datetime(
                sensor_data["TIMESTAMP"],
                errors="coerce"
            )

            sensor_data = sensor_data.dropna(
                subset=["TIMESTAMP"]
            )

            if not sensor_data.empty:

                min_sensor_date = (
                    sensor_data["TIMESTAMP"].min().date()
                )

                max_sensor_date = (
                    sensor_data["TIMESTAMP"].max().date()
                )

                sensor_date_range = st.date_input(
                    "Sensor Analysis Date Range",
                    value=(
                        min_sensor_date,
                        max_sensor_date
                    ),
                    min_value=min_sensor_date,
                    max_value=max_sensor_date,
                    key="sensor_date_filter"
                )

                if (
                    isinstance(sensor_date_range, tuple)
                    and len(sensor_date_range) == 2
                ):

                    start_date, end_date = sensor_date_range

                    sensor_data = sensor_data[
                        sensor_data["TIMESTAMP"].dt.date.between(
                            start_date,
                            end_date
                        )
                    ]

        # ------------------------------------------
        # SENSOR SUMMARY
        # ------------------------------------------

        st.markdown("### Sensor Statistics")

        if sensor_data.empty:

            st.info(
                "No sensor observations are available "
                "for the selected filters."
            )

        else:

            sensor_values = sensor_data[selected_sensor]

            sensor_mean = sensor_values.mean()
            sensor_min = sensor_values.min()
            sensor_max = sensor_values.max()
            sensor_std = sensor_values.std()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Average",
                f"{sensor_mean:.3f}"
            )

            col2.metric(
                "Minimum",
                f"{sensor_min:.3f}"
            )

            col3.metric(
                "Maximum",
                f"{sensor_max:.3f}"
            )

            col4.metric(
                "Standard Deviation",
                f"{sensor_std:.3f}"
                if pd.notna(sensor_std)
                else "N/A"
            )

            # ------------------------------------------
            # INTERACTIVE SENSOR VISUALIZATION
            # ------------------------------------------

            st.markdown("### Sensor Visualization")

            if chart_type == "Time Series":

                if "TIMESTAMP" in sensor_data.columns:

                    plot_data = sensor_data.sort_values(
                        "TIMESTAMP"
                    )

                    fig = px.line(
                        plot_data,
                        x="TIMESTAMP",
                        y=selected_sensor,
                        title=f"{selected_sensor} Over Time",
                        markers=False,
                        hover_data=[
                            col
                            for col in ["CONTAINER_ID"]
                            if col in plot_data.columns
                        ]
                    )

                    fig.update_layout(
                        template="plotly_white",
                        height=450,
                        xaxis_title="Timestamp",
                        yaxis_title=selected_sensor,
                        hovermode="x unified"
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                        key="sensor_time_series"
                    )

                else:

                    st.info(
                        "Time-series visualization requires TIMESTAMP."
                    )

            elif chart_type == "Distribution":

                fig = px.histogram(
                    sensor_data,
                    x=selected_sensor,
                    nbins=40,
                    title=f"{selected_sensor} Distribution",
                    marginal="box"
                )

                fig.update_layout(
                    template="plotly_white",
                    height=450,
                    xaxis_title=selected_sensor,
                    yaxis_title="Frequency"
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    key="sensor_distribution"
                )

            elif chart_type == "Box Plot":

                if "CONTAINER_ID" in sensor_data.columns:

                    fig = px.box(
                        sensor_data,
                        x="CONTAINER_ID",
                        y=selected_sensor,
                        title=f"{selected_sensor} by Container"
                    )

                else:

                    fig = px.box(
                        sensor_data,
                        y=selected_sensor,
                        title=f"{selected_sensor} Distribution"
                    )

                fig.update_layout(
                    template="plotly_white",
                    height=450,
                    xaxis_title="Container",
                    yaxis_title=selected_sensor
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    key="sensor_box_plot"
                )

            # ------------------------------------------
            # SENSOR DATA TABLE
            # ------------------------------------------

            st.markdown("### Sensor Data Explorer")

            display_columns = [
                column
                for column in [
                    "CONTAINER_ID",
                    "TIMESTAMP",
                    "TEMPERATURE_C",
                    "HUMIDITY_PCT",
                    "VIBRATION_G",
                    "TEMPERATURE_DEVIATION_C"
                ]
                if column in sensor_data.columns
            ]

            st.dataframe(
                sensor_data[display_columns].head(500),
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Showing up to 500 sensor observations. "
                "Statistics are calculated from the selected sensor data."
            )

st.divider()



# ==================================================
# 14. LIVE RISK PREDICTION
# ==================================================

st.markdown("### Live Risk Prediction")

st.caption(
    "Generate a prediction by sending an engineered "
    "sensor record to the AtmoSync FastAPI backend."
)

if features is None:

    st.error(
        "Engineered feature dataset is unavailable."
    )

else:

    missing_columns = [
        col
        for col in REQUIRED_COLUMNS
        if col not in features.columns
    ]

    if missing_columns:

        st.error(
            "Required model features are missing "
            "from weather_features.csv:"
        )

        st.write(missing_columns)

    else:

        valid_features = features.copy()

        # Convert numeric model features.
        for column in MODEL_FEATURES:

            valid_features[column] = pd.to_numeric(
                valid_features[column],
                errors="coerce"
            )

        # Remove infinite values.
        valid_features[MODEL_FEATURES] = (
            valid_features[MODEL_FEATURES]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        # Normalize sensor status.
        valid_features["SENSOR_STATUS"] = (
            valid_features["SENSOR_STATUS"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        # Validate sensor status.
        valid_features = valid_features[
            valid_features["SENSOR_STATUS"].isin(
                ["OK", "MISSING"]
            )
        ]

        # Remove incomplete records.
        valid_features = (
            valid_features
            .dropna(subset=REQUIRED_COLUMNS)
            .reset_index(drop=True)
        )

        if valid_features.empty:

            st.error(
                "No complete engineered records "
                "are available for prediction."
            )

        else:

            st.caption(
                f"Available complete records: "
                f"{len(valid_features):,} "
                f"out of {len(features):,}"
            )

            row_number = st.number_input(
                "Select valid engineered record",
                min_value=0,
                max_value=len(valid_features) - 1,
                value=0,
                step=1
            )

            selected_row = valid_features.iloc[
                int(row_number)
            ]

            st.markdown(
                "#### Selected Sensor Record"
            )

            # Convert mixed data types to strings
            # before displaying the feature table.
            display_record = (
                selected_row[REQUIRED_COLUMNS]
                .astype(str)
                .to_frame(name="Feature Value")
            )

            st.dataframe(
                display_record,
                width="stretch"
            )

            if st.button(
                "Generate Live Risk Prediction",
                type="primary"
            ):

                response = None

                try:

                    # Prepare API payload.
                    payload = {
                        column: float(selected_row[column])
                        for column in MODEL_FEATURES
                    }

                    payload["SENSOR_STATUS"] = str(
                        selected_row["SENSOR_STATUS"]
                    )

                    # Call FastAPI.
                    with st.spinner(
                        "Sending record to AtmoSync FastAPI..."
                    ):

                        response = requests.post(
                            API_URL,
                            json=payload,
                            timeout=30
                        )

                    response.raise_for_status()

                    result = response.json()

                    prediction = result.get(
                        "prediction",
                        result
                    )

                    if not isinstance(prediction, dict):

                        raise ValueError(
                            "Unexpected prediction response format."
                        )

                    risk = prediction.get(
                        "PREDICTED_RISK",
                        "Unknown"
                    )

                    if normalize_risk(risk) not in [
                        "High",
                        "Medium",
                        "Low"
                    ]:

                        raise ValueError(
                            "API did not return a valid PREDICTED_RISK."
                        )

                    st.success(
                        "Live prediction generated successfully!"
                    )

                    # Prediction result.
                    st.markdown(
                        "#### Prediction Result"
                    )

                    st.metric(
                        "Predicted Container Risk",
                        normalize_risk(risk)
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Stage 1 High Probability",
                        format_probability(
                            prediction.get(
                                "STAGE1_PROBABILITY_HIGH"
                            )
                        )
                    )

                    col2.metric(
                        "Stage 1 Low Probability",
                        format_probability(
                            prediction.get(
                                "STAGE1_PROBABILITY_LOW"
                            )
                        )
                    )

                    col3.metric(
                        "Stage 1 Medium Probability",
                        format_probability(
                            prediction.get(
                                "STAGE1_PROBABILITY_MEDIUM"
                            )
                        )
                    )

                    stage2_probability = prediction.get(
                        "STAGE2_MEDIUM_PROBABILITY"
                    )

                    if stage2_probability is not None:

                        st.metric(
                            "Stage 2 Medium Probability",
                            format_probability(
                                stage2_probability
                            )
                        )

                    st.markdown(
                        "#### Complete API Response"
                    )

                    st.json(result)

                    # Save prediction history.
                    history_record = {
                        "PREDICTION_TIME": (
                            pd.Timestamp.now().isoformat()
                        ),

                        "RECORD_INDEX": int(row_number),

                        "PREDICTED_RISK": normalize_risk(risk),

                        "STAGE1_PROBABILITY_HIGH": (
                            prediction.get(
                                "STAGE1_PROBABILITY_HIGH"
                            )
                        ),

                        "STAGE1_PROBABILITY_LOW": (
                            prediction.get(
                                "STAGE1_PROBABILITY_LOW"
                            )
                        ),

                        "STAGE1_PROBABILITY_MEDIUM": (
                            prediction.get(
                                "STAGE1_PROBABILITY_MEDIUM"
                            )
                        ),

                        "STAGE2_MEDIUM_PROBABILITY": (
                            prediction.get(
                                "STAGE2_MEDIUM_PROBABILITY"
                            )
                        )
                    }

                    save_prediction_history(
                        history_record
                    )

                    st.success(
                        "Prediction saved to history successfully."
                    )

                    # Analyst recommendation.
                    st.markdown(
                        "#### Recommended Analyst Action"
                    )

                    normalized_risk = normalize_risk(risk)

                    if normalized_risk == "High":

                        st.error(
                            "HIGH RISK: Prioritize this container "
                            "for inspection. Review recent sensor "
                            "readings and investigate potential "
                            "environmental abnormalities."
                        )

                    elif normalized_risk == "Medium":

                        st.warning(
                            "MEDIUM RISK: Increase monitoring "
                            "frequency. Review recent sensor trends "
                            "and investigate unusual changes."
                        )

                    elif normalized_risk == "Low":

                        st.success(
                            "LOW RISK: Continue routine monitoring "
                            "and periodic sensor checks."
                        )

                    st.caption(
                        "Recommendations are prototype "
                        "decision-support rules based on "
                        "synthetic project data, not validated "
                        "operational safety procedures."
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Could not connect to FastAPI. "
                        "Make sure the API server is running "
                        "on port 8001."
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The API request timed out. "
                        "Please try again."
                    )

                except requests.exceptions.HTTPError as error:

                    st.error(
                        f"FastAPI returned an HTTP error: {error}"
                    )

                    if response is not None:

                        try:

                            st.json(response.json())

                        except ValueError:

                            st.write(response.text)

                except (ValueError, TypeError) as error:

                    st.error(
                        f"Invalid prediction input or response: {error}"
                    )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )


st.divider()


# ==================================================
# 15. PREDICTION HISTORY
# ==================================================

st.markdown("### Prediction History")

history_data = load_prediction_history()

if history_data.empty:

    st.info(
        "No live prediction history available yet. "
        "Generate a live prediction to create your first record."
    )

elif "PREDICTED_RISK" not in history_data.columns:

    st.error(
        "Prediction history is missing the "
        "PREDICTED_RISK column."
    )

    st.write(
        "Available columns:",
        history_data.columns.tolist()
    )

else:

    # Normalize risk labels.
    history_data["PREDICTED_RISK"] = (
        history_data["PREDICTED_RISK"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # History summary.
    history_total = len(history_data)

    history_high = (
        history_data["PREDICTED_RISK"]
        .eq("High")
        .sum()
    )

    history_medium = (
        history_data["PREDICTED_RISK"]
        .eq("Medium")
        .sum()
    )

    history_low = (
        history_data["PREDICTED_RISK"]
        .eq("Low")
        .sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Live Predictions",
        f"{history_total:,}"
    )

    col2.metric(
        "High Risk",
        f"{history_high:,}"
    )

    col3.metric(
        "Medium Risk",
        f"{history_medium:,}"
    )

    col4.metric(
        "Low Risk",
        f"{history_low:,}"
    )

    st.markdown(
        "#### Filter Prediction History"
    )

    history_risks = sorted(
        history_data["PREDICTED_RISK"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_history_risks = st.multiselect(
        "Filter by predicted risk",
        options=history_risks,
        default=history_risks,
        key="history_risk_filter"
    )

    filtered_history = history_data[
        history_data["PREDICTED_RISK"].isin(
            selected_history_risks
        )
    ].copy()

    # Newest records first.
    if "PREDICTION_TIME" in filtered_history.columns:

        filtered_history["PREDICTION_TIME"] = pd.to_datetime(
            filtered_history["PREDICTION_TIME"],
            errors="coerce"
        )

        filtered_history = filtered_history.sort_values(
            "PREDICTION_TIME",
            ascending=False,
            na_position="last"
        )

    else:

        filtered_history = filtered_history.iloc[::-1]

    st.markdown(
        "#### Recorded Predictions"
    )

    if filtered_history.empty:

        st.info(
            "No records match the selected risk filters."
        )

    else:

        st.dataframe(
            filtered_history,
            width="stretch",
            hide_index=True
        )

    # Download complete history.
    history_csv = history_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prediction History CSV",
        data=history_csv,
        file_name="atmosync_prediction_history.csv",
        mime="text/csv",
        key="download_prediction_history"
    )


st.divider()


# ==================================================
# 16. PREDICTION DATA EXPLORER
# ==================================================

st.markdown(
    "### Prediction Data Explorer"
)

st.dataframe(
    filtered_data.head(500),
    width="stretch",
    hide_index=True
)

st.caption(
    "Showing up to 500 records. "
    "Predictions are generated by the selected hierarchical model."
)


# ==================================================
# 17. FOOTER
# ==================================================

st.divider()

st.markdown(
    "**AtmoSync | Micro-Climate Arbitrage Analytics**"
)

st.caption(
    "Research and portfolio project using synthetic data. "
    "Predictions should not be interpreted as validated "
    "real-world operational risk assessments."
)
