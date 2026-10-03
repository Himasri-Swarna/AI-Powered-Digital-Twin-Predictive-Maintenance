from pathlib import Path

from data_loader import load_fd001, add_training_rul
from digital_twin import get_digital_twin_state


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_tests():

    print("\n========== FINAL PROJECT TEST ==========\n")

    # --------------------------------------------------
    # TEST 1 — DATASET
    # --------------------------------------------------

    train_df, test_df, true_rul_df = load_fd001()

    train_df = add_training_rul(train_df)

    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(true_rul_df) == 100
    assert train_df["RUL"].min() == 0

    print("✅ Dataset loading")
    print("✅ Training RUL calculation")


    # --------------------------------------------------
    # TEST 2 — MODEL FILES
    # --------------------------------------------------

    required_models = [
        "rul_rf_model.joblib",
        "failure_risk_model.joblib",
        "anomaly_model.joblib",
    ]

    for model_name in required_models:

        model_path = (
            PROJECT_ROOT
            / "models"
            / model_name
        )

        assert model_path.exists(), (
            f"Missing model: {model_name}"
        )

    print("✅ All trained models exist")


    # --------------------------------------------------
    # TEST 3 — HEALTHY ENGINE
    # --------------------------------------------------

    healthy = get_digital_twin_state(
        engine_id=1
    )

    assert healthy["status"] == "HEALTHY"
    assert healthy["health_score"] >= 0
    assert healthy["health_score"] <= 100
    assert healthy["failure_probability"] >= 0
    assert healthy["failure_probability"] <= 100

    print("✅ Healthy Digital Twin state")


    # --------------------------------------------------
    # TEST 4 — CRITICAL ENGINE
    # --------------------------------------------------

    critical = get_digital_twin_state(
        engine_id=34,
        cycle=203
    )

    assert critical["status"] == "CRITICAL"
    assert critical["predicted_rul"] >= 0
    assert critical["failure_probability"] >= 80

    print("✅ Critical Digital Twin state")


    # --------------------------------------------------
    # TEST 5 — ANOMALY
    # --------------------------------------------------

    assert healthy["anomaly_status"] in [
        "NORMAL",
        "ANOMALY"
    ]

    assert critical["anomaly_status"] in [
        "NORMAL",
        "ANOMALY"
    ]

    print("✅ Anomaly detection output")


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\nHealthy engine:")
    print(
        f"Engine {healthy['engine_id']} | "
        f"Health {healthy['health_score']}% | "
        f"Failure {healthy['failure_probability']}% | "
        f"RUL {healthy['predicted_rul']} | "
        f"{healthy['status']} | "
        f"{healthy['anomaly_status']}"
    )

    print("\nCritical engine:")
    print(
        f"Engine {critical['engine_id']} | "
        f"Health {critical['health_score']}% | "
        f"Failure {critical['failure_probability']}% | "
        f"RUL {critical['predicted_rul']} | "
        f"{critical['status']} | "
        f"{critical['anomaly_status']}"
    )

    print(
        "\n🎉 ALL FINAL TESTS PASSED"
    )

    print(
        "\n========================================\n"
    )


if __name__ == "__main__":
    run_tests()