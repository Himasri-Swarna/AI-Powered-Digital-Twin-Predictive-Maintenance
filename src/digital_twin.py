from pathlib import Path
import joblib
import numpy as np

from data_loader import load_fd001, add_training_rul


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUL_MODEL_FILE = (
    PROJECT_ROOT / "models" / "rul_rf_model.joblib"
)

FAILURE_MODEL_FILE = (
    PROJECT_ROOT / "models" / "failure_risk_model.joblib"
)

ANOMALY_MODEL_FILE = (
    PROJECT_ROOT / "models" / "anomaly_model.joblib"
)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

rul_package = joblib.load(RUL_MODEL_FILE)

failure_package = joblib.load(FAILURE_MODEL_FILE)

anomaly_package = joblib.load(ANOMALY_MODEL_FILE)


rul_model = rul_package["model"]

rul_features = rul_package["features"]


failure_model = failure_package["model"]

failure_features = failure_package["features"]

FAILURE_WINDOW = failure_package["failure_window"]


anomaly_model = anomaly_package["model"]

anomaly_features = anomaly_package["features"]


# --------------------------------------------------
# HEALTH SCORE
# --------------------------------------------------

def calculate_health_score(
    predicted_rul,
    failure_probability,
    healthy_rul_reference
):
    """
    Composite heuristic Machine Health Score.

    Components:

    70% = model-derived failure-risk component

    30% = normalized predicted-RUL component

    NOTE:
    This is not a ground-truth NASA variable.
    It is a project-level health indicator.
    """

    # If failure probability is 0,
    # risk component becomes 100.
    #
    # If failure probability is 1,
    # risk component becomes 0.

    risk_component = (
        1.0 - failure_probability
    ) * 100


    # Convert predicted RUL into a 0-100 score.

    rul_component = (
        predicted_rul
        / healthy_rul_reference
    ) * 100


    # Ensure score stays between 0 and 100.

    rul_component = np.clip(
        rul_component,
        0,
        100
    )


    # Final health score.

    health_score = (
        0.70 * risk_component
        + 0.30 * rul_component
    )


    return float(
        np.clip(
            health_score,
            0,
            100
        )
    )


# --------------------------------------------------
# MACHINE STATUS + ALERT ENGINE
# --------------------------------------------------

def determine_machine_status(
    predicted_rul,
    failure_probability
):
    """
    Determines machine condition and maintenance recommendation.

    Thresholds:

    CRITICAL:
        failure probability >= 80%
        OR predicted RUL <= 15 cycles

    HIGH RISK:
        failure probability >= 60%
        OR predicted RUL <= 30 cycles

    WARNING:
        failure probability >= 30%
        OR predicted RUL <= 60 cycles

    HEALTHY:
        otherwise
    """


    if (
        failure_probability >= 0.80
        or predicted_rul <= 15
    ):

        return (
            "CRITICAL",
            "Immediate maintenance inspection recommended."
        )


    elif (
        failure_probability >= 0.60
        or predicted_rul <= FAILURE_WINDOW
    ):

        return (
            "HIGH RISK",
            "Schedule maintenance as soon as possible."
        )


    elif (
        failure_probability >= 0.30
        or predicted_rul <= 60
    ):

        return (
            "WARNING",
            "Monitor closely and prepare maintenance."
        )


    else:

        return (
            "HEALTHY",
            "No immediate maintenance action required."
        )


# --------------------------------------------------
# BUILD DIGITAL TWIN STATE
# --------------------------------------------------

def get_digital_twin_state(
    engine_id=1,
    cycle=None
):
    """
    Creates one current dataset-driven Digital Twin state
    using NASA C-MAPSS FD001 test telemetry.

    Outputs include:

    - predicted RUL
    - true RUL for academic validation
    - failure probability
    - Machine Health Score
    - machine status
    - anomaly status
    - anomaly decision score
    - maintenance recommendation
    - important sensor values
    """


    # --------------------------------------------------
    # LOAD NASA DATA
    # --------------------------------------------------

    train_df, test_df, true_rul_df = load_fd001()

    train_with_rul = add_training_rul(
        train_df
    )


    # --------------------------------------------------
    # HEALTHY RUL REFERENCE
    # --------------------------------------------------

    # Use the 90th percentile of real training RUL
    # as a healthy-life reference for Health Score.

    healthy_rul_reference = float(
        train_with_rul["RUL"].quantile(0.90)
    )


    # --------------------------------------------------
    # SELECT ENGINE
    # --------------------------------------------------

    engine_data = (
        test_df[
            test_df["engine_id"] == engine_id
        ]
        .sort_values("cycle")
        .copy()
    )


    if engine_data.empty:

        raise ValueError(
            f"Engine {engine_id} does not exist."
        )


    # --------------------------------------------------
    # SELECT CURRENT CYCLE
    # --------------------------------------------------

    if cycle is None:

        # If no cycle is supplied,
        # use the latest available cycle.

        current_row = engine_data.iloc[-1]


    else:

        available = engine_data[
            engine_data["cycle"] <= cycle
        ]


        if available.empty:

            raise ValueError(
                f"No data available for cycle {cycle}."
            )


        current_row = available.iloc[-1]


    # --------------------------------------------------
    # RUL PREDICTION
    # --------------------------------------------------

    rul_input = current_row[
        rul_features
    ].to_frame().T


    predicted_rul = float(
        rul_model.predict(
            rul_input
        )[0]
    )


    # RUL cannot be negative.

    predicted_rul = max(
        predicted_rul,
        0
    )


    # --------------------------------------------------
    # FAILURE PROBABILITY
    # --------------------------------------------------

    failure_input = current_row[
        failure_features
    ].to_frame().T


    failure_probability = float(
        failure_model.predict_proba(
            failure_input
        )[0, 1]
    )


    # --------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------

    anomaly_input = current_row[
        anomaly_features
    ].to_frame().T


    # Isolation Forest returns:
    #
    #  1  = normal
    # -1  = anomaly

    anomaly_prediction = int(
        anomaly_model.predict(
            anomaly_input
        )[0]
    )


    # decision_function:
    #
    # positive value = more normal
    # negative value = more anomalous

    anomaly_decision_score = float(
        anomaly_model.decision_function(
            anomaly_input
        )[0]
    )


    if anomaly_prediction == -1:

        anomaly_status = "ANOMALY"

    else:

        anomaly_status = "NORMAL"


    # --------------------------------------------------
    # MACHINE HEALTH SCORE
    # --------------------------------------------------

    health_score = calculate_health_score(
        predicted_rul,
        failure_probability,
        healthy_rul_reference
    )


    # --------------------------------------------------
    # MACHINE STATUS + MAINTENANCE ALERT
    # --------------------------------------------------

    status, recommendation = (
        determine_machine_status(
            predicted_rul,
            failure_probability
        )
    )


    # --------------------------------------------------
    # TRUE RUL FOR ACADEMIC VALIDATION
    # --------------------------------------------------

    # The NASA test set is truncated.
    #
    # RUL_FD001 gives the true RUL
    # at the final observed cycle.

    final_observed_cycle = int(
        engine_data["cycle"].max()
    )


    final_true_rul = int(
        true_rul_df.iloc[
            engine_id - 1
        ]["true_rul"]
    )


    current_cycle = int(
        current_row["cycle"]
    )


    # True RUL at the selected historical cycle.

    actual_rul = (
        final_true_rul
        + final_observed_cycle
        - current_cycle
    )


    # --------------------------------------------------
    # DIGITAL TWIN STATE
    # --------------------------------------------------

    state = {

        "engine_id":
            int(engine_id),

        "cycle":
            current_cycle,


        # ----------------------------------------------
        # RUL
        # ----------------------------------------------

        "predicted_rul": round(
            predicted_rul,
            2
        ),

        "actual_rul_for_validation": int(
            actual_rul
        ),


        # ----------------------------------------------
        # FAILURE RISK
        # ----------------------------------------------

        "failure_probability": round(
            failure_probability * 100,
            2
        ),


        # ----------------------------------------------
        # HEALTH SCORE
        # ----------------------------------------------

        "health_score": round(
            health_score,
            2
        ),


        # ----------------------------------------------
        # MACHINE CONDITION
        # ----------------------------------------------

        "status":
            status,


        # ----------------------------------------------
        # ANOMALY DETECTION
        # ----------------------------------------------

        "anomaly_status":
            anomaly_status,

        "anomaly_decision_score": round(
            anomaly_decision_score,
            4
        ),


        # ----------------------------------------------
        # MAINTENANCE
        # ----------------------------------------------

        "maintenance_recommendation":
            recommendation,


        # ----------------------------------------------
        # SELECTED SENSOR VALUES
        # ----------------------------------------------

        "sensor_2": float(
            current_row["sensor_2"]
        ),

        "sensor_3": float(
            current_row["sensor_3"]
        ),

        "sensor_4": float(
            current_row["sensor_4"]
        ),

        "sensor_11": float(
            current_row["sensor_11"]
        ),

        "sensor_15": float(
            current_row["sensor_15"]
        ),

        "sensor_20": float(
            current_row["sensor_20"]
        ),

        "sensor_21": float(
            current_row["sensor_21"]
        ),
    }


    return state


# --------------------------------------------------
# TEST DIGITAL TWIN
# --------------------------------------------------

if __name__ == "__main__":

    state = get_digital_twin_state(
        engine_id=1
    )


    print(
        "\n========== DIGITAL TWIN STATE ==========\n"
    )


    for key, value in state.items():

        print(
            f"{key}: {value}"
        )


    print(
        "\n========================================\n"
    )