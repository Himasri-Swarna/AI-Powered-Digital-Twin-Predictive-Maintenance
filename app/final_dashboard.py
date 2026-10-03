from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# PROJECT PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ==================================================
# PROJECT MODULES
# ==================================================

from data_loader import load_fd001, add_training_rul

from digital_twin import (
    get_digital_twin_state,
    calculate_health_score,
    determine_machine_status,
    rul_model,
    rul_features,
    failure_model,
    failure_features,
    anomaly_model,
    anomaly_features,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI-Powered Digital Twin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# SIMPLE DASHBOARD STYLING
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 15px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 15px;
    }

    .small-note {
        opacity: 0.75;
        font-size: 0.90rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# LOAD NASA DATA
# ==================================================

@st.cache_data
def load_project_data():

    train_df, test_df, true_rul_df = load_fd001()

    train_rul_df = add_training_rul(
        train_df
    )

    return (
        train_rul_df,
        test_df,
        true_rul_df
    )


train_df, test_df, true_rul_df = (
    load_project_data()
)


# ==================================================
# HEALTHY RUL REFERENCE
# ==================================================

HEALTHY_RUL_REFERENCE = float(
    train_df["RUL"].quantile(0.90)
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "⚙️ Digital Twin Controls"
)

engine_ids = sorted(
    test_df["engine_id"]
    .unique()
    .astype(int)
    .tolist()
)


selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids,
    index=0,
)


engine_data = (
    test_df[
        test_df["engine_id"]
        == selected_engine
    ]
    .sort_values("cycle")
    .copy()
)


min_cycle = int(
    engine_data["cycle"].min()
)

max_cycle = int(
    engine_data["cycle"].max()
)


selected_cycle = st.sidebar.slider(
    "Operating Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=max_cycle,
    step=1,
)


st.sidebar.info(
    "NASA C-MAPSS telemetry is replayed "
    "as a dataset-driven Digital Twin simulation."
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Dataset: NASA C-MAPSS FD001"
)

st.sidebar.caption(
    "RUL Model: Random Forest Regressor"
)

st.sidebar.caption(
    "Failure Risk: Random Forest Classifier"
)

st.sidebar.caption(
    "Anomaly Detection: Isolation Forest"
)


# ==================================================
# CURRENT DIGITAL TWIN STATE
# ==================================================

state = get_digital_twin_state(
    engine_id=int(selected_engine),
    cycle=int(selected_cycle),
)


# ==================================================
# BUILD HISTORICAL DIGITAL TWIN STATES
# ==================================================

history = (
    engine_data[
        engine_data["cycle"]
        <= selected_cycle
    ]
    .copy()
    .reset_index(drop=True)
)


# --------------------------------------------------
# RUL predictions
# --------------------------------------------------

history_rul = rul_model.predict(
    history[rul_features]
)

history_rul = np.maximum(
    history_rul,
    0
)


# --------------------------------------------------
# Failure probability
# --------------------------------------------------

history_failure_probability = (
    failure_model.predict_proba(
        history[failure_features]
    )[:, 1]
)


# --------------------------------------------------
# Anomaly predictions
# --------------------------------------------------

history_anomaly_prediction = (
    anomaly_model.predict(
        history[anomaly_features]
    )
)


history_anomaly_score = (
    anomaly_model.decision_function(
        history[anomaly_features]
    )
)


# --------------------------------------------------
# Health Score history
# --------------------------------------------------

history_health = []

for predicted_rul, failure_probability in zip(
    history_rul,
    history_failure_probability,
):

    score = calculate_health_score(
        predicted_rul,
        failure_probability,
        HEALTHY_RUL_REFERENCE,
    )

    history_health.append(
        score
    )


# --------------------------------------------------
# Status + maintenance history
# --------------------------------------------------

history_status = []
history_recommendation = []

for predicted_rul, failure_probability in zip(
    history_rul,
    history_failure_probability,
):

    status, recommendation = (
        determine_machine_status(
            predicted_rul,
            failure_probability,
        )
    )

    history_status.append(
        status
    )

    history_recommendation.append(
        recommendation
    )


# --------------------------------------------------
# Add AI results into historical dataframe
# --------------------------------------------------

history["predicted_rul"] = history_rul

history["failure_probability"] = (
    history_failure_probability * 100
)

history["health_score"] = (
    history_health
)

history["status"] = (
    history_status
)

history["recommendation"] = (
    history_recommendation
)

history["anomaly_status"] = np.where(
    history_anomaly_prediction == -1,
    "ANOMALY",
    "NORMAL",
)

history["anomaly_score"] = (
    history_anomaly_score
)


# ==================================================
# PAGE HEADER
# ==================================================

st.title(
    "AI-Powered Digital Twin for Predictive Maintenance"
)

st.caption(
    "Machine Health Scoring • Failure Risk Prediction • "
    "Remaining Useful Life • Anomaly Detection • "
    "Maintenance Alerts"
)

st.divider()


# ==================================================
# MACHINE OVERVIEW
# ==================================================

st.subheader(
    "Machine Overview"
)


overview1, overview2, overview3, overview4 = (
    st.columns(4)
)


overview1.markdown(
    f"**Asset**  \n"
    f"Turbofan Engine {selected_engine}"
)


overview2.markdown(
    f"**Operating Cycle**  \n"
    f"{selected_cycle}"
)


overview3.markdown(
    f"**Digital Twin Type**  \n"
    f"Dataset-Driven Simulation"
)


overview4.markdown(
    f"**Telemetry Source**  \n"
    f"NASA C-MAPSS FD001"
)


# ==================================================
# CURRENT MACHINE CONDITION
# ==================================================

st.subheader(
    "Current Machine Condition"
)


kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)


kpi1.metric(
    "Machine Health",
    f"{state['health_score']:.1f}%",
)


kpi2.metric(
    "Failure Probability",
    f"{state['failure_probability']:.1f}%",
)


kpi3.metric(
    "Predicted RUL",
    f"{state['predicted_rul']:.1f}",
    help="Remaining operating cycles",
)


kpi4.metric(
    "Machine Status",
    state["status"],
)


kpi5.metric(
    "Anomaly Status",
    state["anomaly_status"],
)


st.caption(
    "Predicted RUL is expressed in operating cycles, "
    "not calendar days."
)


# ==================================================
# GAUGES
# ==================================================

gauge1, gauge2, gauge3 = (
    st.columns(3)
)


# --------------------------------------------------
# Health gauge
# --------------------------------------------------

health_value = state[
    "health_score"
]

if health_value >= 75:

    health_color = "#22c55e"

elif health_value >= 50:

    health_color = "#f59e0b"

else:

    health_color = "#ef4444"


health_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=health_value,
        number={
            "suffix": "%"
        },
        title={
            "text": "Machine Health Score"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": health_color
            },
            "steps": [
                {
                    "range": [0, 50],
                    "color": "rgba(239,68,68,0.12)",
                },
                {
                    "range": [50, 75],
                    "color": "rgba(245,158,11,0.12)",
                },
                {
                    "range": [75, 100],
                    "color": "rgba(34,197,94,0.12)",
                },
            ],
        },
    )
)


health_gauge.update_layout(
    height=280,
    margin=dict(
        l=25,
        r=25,
        t=50,
        b=20,
    ),
)


with gauge1:

    st.plotly_chart(
        health_gauge,
        use_container_width=True,
    )


# --------------------------------------------------
# Failure risk gauge
# --------------------------------------------------

failure_value = state[
    "failure_probability"
]


failure_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=failure_value,
        number={
            "suffix": "%"
        },
        title={
            "text": "Failure Risk"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": "#ef4444"
            },
            "steps": [
                {
                    "range": [0, 30],
                    "color": "rgba(34,197,94,0.12)",
                },
                {
                    "range": [30, 60],
                    "color": "rgba(245,158,11,0.12)",
                },
                {
                    "range": [60, 100],
                    "color": "rgba(239,68,68,0.12)",
                },
            ],
        },
    )
)


failure_gauge.update_layout(
    height=280,
    margin=dict(
        l=25,
        r=25,
        t=50,
        b=20,
    ),
)


with gauge2:

    st.plotly_chart(
        failure_gauge,
        use_container_width=True,
    )


# --------------------------------------------------
# RUL gauge
# --------------------------------------------------

rul_value = state[
    "predicted_rul"
]


rul_max = max(
    150,
    int(
        HEALTHY_RUL_REFERENCE
    ),
)


rul_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=rul_value,
        number={
            "suffix": " cycles"
        },
        title={
            "text": "Estimated Remaining Useful Life"
        },
        gauge={
            "axis": {
                "range": [
                    0,
                    rul_max
                ]
            },
            "bar": {
                "color": "#3b82f6"
            },
        },
    )
)


rul_gauge.update_layout(
    height=280,
    margin=dict(
        l=25,
        r=25,
        t=50,
        b=20,
    ),
)


with gauge3:

    st.plotly_chart(
        rul_gauge,
        use_container_width=True,
    )


# ==================================================
# MAINTENANCE ADVISORY
# ==================================================

st.subheader(
    "Maintenance Advisory"
)


status = state[
    "status"
]


if status == "CRITICAL":

    st.error(
        "🔴 CRITICAL — "
        + state[
            "maintenance_recommendation"
        ]
    )


elif status == "HIGH RISK":

    st.error(
        "🟠 HIGH RISK — "
        + state[
            "maintenance_recommendation"
        ]
    )


elif status == "WARNING":

    st.warning(
        "🟡 WARNING — "
        + state[
            "maintenance_recommendation"
        ]
    )


else:

    st.success(
        "🟢 HEALTHY — "
        + state[
            "maintenance_recommendation"
        ]
    )


# ==================================================
# AI PREDICTION + ANOMALY
# ==================================================

prediction_col, anomaly_col = (
    st.columns(2)
)


with prediction_col:

    st.subheader(
        "AI Failure Prediction"
    )

    failure_percentage = float(
        state[
            "failure_probability"
        ]
    )

    healthy_percentage = (
        100
        - failure_percentage
    )


    st.write(
        f"Healthy / Not Imminent Failure: "
        f"**{healthy_percentage:.1f}%**"
    )

    st.progress(
        max(
            0.0,
            min(
                healthy_percentage / 100,
                1.0,
            ),
        )
    )


    st.write(
        f"Within Final 30 Cycles: "
        f"**{failure_percentage:.1f}%**"
    )

    st.progress(
        max(
            0.0,
            min(
                failure_percentage / 100,
                1.0,
            ),
        )
    )


    st.caption(
        "The classifier estimates whether current "
        "telemetry resembles an engine within its "
        "final 30 operating cycles."
    )


with anomaly_col:

    st.subheader(
        "Anomaly Detection"
    )


    anomaly_status = state[
        "anomaly_status"
    ]


    anomaly_score = state[
        "anomaly_decision_score"
    ]


    if anomaly_status == "ANOMALY":

        st.error(
            "⚠️ ANOMALY DETECTED"
        )

    else:

        st.success(
            "✅ Sensor pattern appears NORMAL"
        )


    st.metric(
        "Isolation Forest Decision Score",
        f"{anomaly_score:.4f}",
    )


    st.caption(
        "Positive scores indicate more normal behaviour; "
        "negative scores indicate more anomalous behaviour. "
        "This score is not a probability."
    )


# ==================================================
# CURRENT SENSOR READINGS
# ==================================================

st.subheader(
    "Current Sensor Readings"
)


sensor1, sensor2, sensor3, sensor4, sensor5 = (
    st.columns(5)
)


sensor1.metric(
    "Sensor 2",
    f"{state['sensor_2']:.2f}",
)


sensor2.metric(
    "Sensor 3",
    f"{state['sensor_3']:.2f}",
)


sensor3.metric(
    "Sensor 4",
    f"{state['sensor_4']:.2f}",
)


sensor4.metric(
    "Sensor 11",
    f"{state['sensor_11']:.2f}",
)


sensor5.metric(
    "Sensor 15",
    f"{state['sensor_15']:.4f}",
)


st.caption(
    "NASA C-MAPSS provides anonymized sensor channels, "
    "so the dashboard retains their original sensor IDs "
    "instead of assigning unsupported physical labels."
)


# ==================================================
# DIGITAL TWIN TRENDS
# ==================================================

st.subheader(
    "Digital Twin Trends"
)


trend1, trend2 = (
    st.columns(2)
)


# --------------------------------------------------
# Health Score Trend
# --------------------------------------------------

health_trend = px.line(
    history,
    x="cycle",
    y="health_score",
    title="Machine Health Score Trend",
    labels={
        "cycle":
            "Operating Cycle",
        "health_score":
            "Health Score (%)",
    },
)


health_trend.add_hline(
    y=50,
    line_dash="dash",
    annotation_text="Warning region",
)


with trend1:

    st.plotly_chart(
        health_trend,
        use_container_width=True,
    )


# --------------------------------------------------
# Predicted RUL Trend
# --------------------------------------------------

rul_trend = px.line(
    history,
    x="cycle",
    y="predicted_rul",
    title="Predicted Remaining Useful Life",
    labels={
        "cycle":
            "Operating Cycle",
        "predicted_rul":
            "Predicted RUL (cycles)",
    },
)


rul_trend.add_hline(
    y=30,
    line_dash="dash",
    annotation_text="High-risk boundary",
)


with trend2:

    st.plotly_chart(
        rul_trend,
        use_container_width=True,
    )


# ==================================================
# FAILURE PROBABILITY TREND
# ==================================================

failure_trend = px.line(
    history,
    x="cycle",
    y="failure_probability",
    title="Failure Probability Trend",
    labels={
        "cycle":
            "Operating Cycle",
        "failure_probability":
            "Failure Probability (%)",
    },
)


failure_trend.add_hline(
    y=60,
    line_dash="dash",
    annotation_text="High Risk",
)


failure_trend.add_hline(
    y=80,
    line_dash="dash",
    annotation_text="Critical",
)


st.plotly_chart(
    failure_trend,
    use_container_width=True,
)


# ==================================================
# SENSOR TREND
# ==================================================

st.subheader(
    "Sensor Trend"
)


available_sensor_choices = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_20",
    "sensor_21",
]


selected_sensor = st.selectbox(
    "Select sensor to visualize",
    available_sensor_choices,
)


sensor_chart = px.line(
    history,
    x="cycle",
    y=selected_sensor,
    title=(
        f"{selected_sensor.replace('_', ' ').title()} "
        f"Trend — Engine {selected_engine}"
    ),
    labels={
        "cycle":
            "Operating Cycle",
        selected_sensor:
            "Sensor Value",
    },
)


st.plotly_chart(
    sensor_chart,
    use_container_width=True,
)


# ==================================================
# RECENT ALERTS
# ==================================================

st.subheader(
    "Recent Alerts"
)


alerts = history[
    (
        history["status"]
        != "HEALTHY"
    )
    |
    (
        history["anomaly_status"]
        == "ANOMALY"
    )
].copy()


if alerts.empty:

    st.success(
        "No warning, high-risk, critical, "
        "or anomaly events detected up to this cycle."
    )


else:

    alerts["Event"] = alerts.apply(
        lambda row:
            (
                f"{row['status']} + ANOMALY"
                if (
                    row["status"] != "HEALTHY"
                    and row["anomaly_status"]
                    == "ANOMALY"
                )
                else (
                    row["status"]
                    if row["status"] != "HEALTHY"
                    else "ANOMALY"
                )
            ),
        axis=1,
    )


    alert_table = (
        alerts[
            [
                "cycle",
                "Event",
                "predicted_rul",
                "failure_probability",
                "health_score",
                "anomaly_score",
            ]
        ]
        .tail(8)
        .copy()
    )


    alert_table = alert_table.rename(
        columns={
            "cycle":
                "Cycle",
            "predicted_rul":
                "Predicted RUL",
            "failure_probability":
                "Failure Probability %",
            "health_score":
                "Health Score %",
            "anomaly_score":
                "Anomaly Score",
        }
    )


    alert_table[
        "Predicted RUL"
    ] = (
        alert_table[
            "Predicted RUL"
        ]
        .round(2)
    )


    alert_table[
        "Failure Probability %"
    ] = (
        alert_table[
            "Failure Probability %"
        ]
        .round(2)
    )


    alert_table[
        "Health Score %"
    ] = (
        alert_table[
            "Health Score %"
        ]
        .round(2)
    )


    alert_table[
        "Anomaly Score"
    ] = (
        alert_table[
            "Anomaly Score"
        ]
        .round(4)
    )


    st.dataframe(
        alert_table,
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# HEALTH SCORE INTERPRETATION
# ==================================================

st.subheader(
    "Machine Health Score Interpretation"
)


st.progress(
    max(
        0.0,
        min(
            state[
                "health_score"
            ] / 100,
            1.0,
        ),
    )
)


st.caption(
    "The Machine Health Score is a project-level "
    "composite heuristic: 70% model-derived failure-risk "
    "component plus 30% normalized predicted-RUL component. "
    "It is not a ground-truth NASA target."
)


# ==================================================
# MODEL VALIDATION INFORMATION
# ==================================================

with st.expander(
    "Model Validation Information"
):

    actual_rul = state[
        "actual_rul_for_validation"
    ]


    predicted_rul = state[
        "predicted_rul"
    ]


    absolute_error = abs(
        predicted_rul
        - actual_rul
    )


    val1, val2, val3 = (
        st.columns(3)
    )


    val1.metric(
        "Predicted RUL",
        f"{predicted_rul:.1f} cycles",
    )


    val2.metric(
        "True RUL",
        f"{actual_rul} cycles",
    )


    val3.metric(
        "Absolute Error",
        f"{absolute_error:.1f} cycles",
    )


    st.markdown(
        """
        **RUL model evaluation**

        Engine-held-out validation:

        - MAE: **23.81 cycles**
        - RMSE: **31.35 cycles**
        - R²: **0.772**

        NASA FD001 official test evaluation:

        - MAE: **19.85 cycles**
        - RMSE: **26.88 cycles**

        **Failure-risk classifier validation**

        - Accuracy: **95.2%**
        - Precision: **78.6%**
        - Recall: **94.5%**
        - F1 Score: **85.8%**
        """
    )


    st.caption(
        "True RUL is displayed only for academic "
        "validation. In a real deployment, future "
        "failure time would normally be unknown."
    )


# ==================================================
# DIGITAL TWIN TECHNICAL STATE
# ==================================================

with st.expander(
    "Digital Twin Technical State"
):

    st.json(
        state
    )


# ==================================================
# SYSTEM ARCHITECTURE
# ==================================================

with st.expander(
    "Project Architecture"
):

    st.code(
        """
NASA C-MAPSS FD001
        ↓
Data Loading + Preprocessing
        ↓
Random Forest RUL Regressor
        ↓
Random Forest Failure-Risk Classifier
        ↓
Isolation Forest Anomaly Detection
        ↓
Machine Health Score
        ↓
Digital Twin State
        ↓
Maintenance Alert Engine
        ↓
Interactive Streamlit Dashboard
        """,
        language="text",
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()


st.caption(
    "Dataset: NASA C-MAPSS FD001  |  "
    "RUL: Random Forest Regressor  |  "
    "Failure Risk: Random Forest Classifier  |  "
    "Anomaly Detection: Isolation Forest  |  "
    "Digital Twin: Dataset-driven simulation"
)