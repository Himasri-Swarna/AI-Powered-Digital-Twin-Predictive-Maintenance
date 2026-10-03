from data_loader import load_fd001
from digital_twin import get_digital_twin_state


def scan_engines():

    _, test_df, _ = load_fd001()

    engine_ids = sorted(
        test_df["engine_id"].unique()
    )

    results = []

    print("\n========== DIGITAL TWIN RISK SCAN ==========\n")

    for engine_id in engine_ids:

        try:
            state = get_digital_twin_state(
                engine_id=int(engine_id)
            )

            results.append({
                "engine_id": engine_id,
                "cycle": state["cycle"],
                "predicted_rul": state["predicted_rul"],
                "failure_probability":
                    state["failure_probability"],
                "health_score":
                    state["health_score"],
                "status":
                    state["status"]
            })

        except Exception as error:
            print(
                f"Engine {engine_id} failed: {error}"
            )

    # Highest-risk engines first
    results = sorted(
        results,
        key=lambda x: (
            -x["failure_probability"],
            x["predicted_rul"]
        )
    )

    print("Top 15 highest-risk engines:\n")

    print(
        f"{'Engine':<8}"
        f"{'Cycle':<8}"
        f"{'RUL':<12}"
        f"{'Fail %':<12}"
        f"{'Health':<12}"
        f"{'Status'}"
    )

    print("-" * 70)

    for item in results[:15]:

        print(
            f"{item['engine_id']:<8}"
            f"{item['cycle']:<8}"
            f"{item['predicted_rul']:<12.2f}"
            f"{item['failure_probability']:<12.2f}"
            f"{item['health_score']:<12.2f}"
            f"{item['status']}"
        )

    print("\n============================================\n")


if __name__ == "__main__":
    scan_engines()