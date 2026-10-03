from pathlib import Path
import joblib

from sklearn.ensemble import IsolationForest

from data_loader import load_fd001, add_training_rul


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "anomaly_model.joblib"


ANOMALY_FEATURES = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_20",
    "sensor_21"
]


def train_anomaly_model():

    print("\n========== ANOMALY MODEL TRAINING ==========\n")

    train_df, _, _ = load_fd001()

    train_df = add_training_rul(train_df)

    # Use high-RUL telemetry as our healthy baseline.
    healthy_data = train_df[
        train_df["RUL"] > 120
    ].copy()

    print(
        f"Healthy baseline rows: {len(healthy_data)}"
    )

    print(
        f"Features used: {ANOMALY_FEATURES}"
    )

    X_healthy = healthy_data[
        ANOMALY_FEATURES
    ]

    model = IsolationForest(
        n_estimators=150,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")

    model.fit(X_healthy)

    predictions = model.predict(X_healthy)

    anomaly_count = (
        predictions == -1
    ).sum()

    anomaly_percent = (
        anomaly_count / len(predictions)
    ) * 100

    print(
        f"\nBaseline anomalies detected: "
        f"{anomaly_count}"
    )

    print(
        f"Baseline anomaly rate: "
        f"{anomaly_percent:.2f}%"
    )

    package = {
        "model": model,
        "features": ANOMALY_FEATURES,
        "baseline_definition":
            "NASA FD001 training records with RUL > 120 cycles",
        "method":
            "Isolation Forest"
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        package,
        MODEL_FILE
    )

    print(
        f"\nAnomaly model saved to:\n{MODEL_FILE}"
    )

    print("\n============================================\n")


if __name__ == "__main__":
    train_anomaly_model()