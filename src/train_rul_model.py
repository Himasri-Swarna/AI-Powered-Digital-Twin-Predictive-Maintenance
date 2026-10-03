from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from data_loader import load_fd001, add_training_rul


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "rul_rf_model.joblib"


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

def prepare_features(train_df):
    """
    Selects operational and sensor features.

    engine_id is NOT used because it only identifies the engine.
    RUL is the prediction target.
    """

    candidate_features = (
        ["cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Remove columns that never change.
    feature_columns = [
        column
        for column in candidate_features
        if train_df[column].nunique() > 1
    ]

    return feature_columns


# --------------------------------------------------
# TRAIN AND VALIDATE
# --------------------------------------------------

def train_model():
    print("\n========== RUL MODEL TRAINING ==========\n")

    train_df, test_df, true_rul_df = load_fd001()

    train_df = add_training_rul(train_df)

    feature_columns = prepare_features(train_df)

    print(f"Features being used: {len(feature_columns)}")
    print(feature_columns)

    X = train_df[feature_columns]
    y = train_df["RUL"]

    # IMPORTANT:
    # Split by ENGINE, not random individual rows.
    # This prevents rows from the same engine appearing
    # in both training and validation sets.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_index, validation_index = next(
        splitter.split(
            X,
            y,
            groups=train_df["engine_id"]
        )
    )

    X_train = X.iloc[train_index]
    X_validation = X.iloc[validation_index]

    y_train = y.iloc[train_index]
    y_validation = y.iloc[validation_index]

    print(
        f"\nTraining rows: {len(X_train)}"
    )
    print(
        f"Validation rows: {len(X_validation)}"
    )

    print(
        f"Training engines: "
        f"{train_df.iloc[train_index]['engine_id'].nunique()}"
    )

    print(
        f"Validation engines: "
        f"{train_df.iloc[validation_index]['engine_id'].nunique()}"
    )

    # --------------------------------------------------
    # RANDOM FOREST RUL REGRESSOR
    # --------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_predictions = np.maximum(
        validation_predictions,
        0
    )

    mae = mean_absolute_error(
        y_validation,
        validation_predictions
    )

    rmse = root_mean_squared_error(
        y_validation,
        validation_predictions
    )

    r2 = r2_score(
        y_validation,
        validation_predictions
    )

    print("\n----- Engine-Held-Out Validation -----")
    print(f"MAE  : {mae:.2f} cycles")
    print(f"RMSE : {rmse:.2f} cycles")
    print(f"R²   : {r2:.3f}")

    # --------------------------------------------------
    # NASA TEST ENGINES
    # --------------------------------------------------

    # One prediction is required for each test engine.
    # We therefore use its LAST AVAILABLE telemetry row.
    final_test_rows = (
        test_df
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
    )

    test_predictions = model.predict(
        final_test_rows[feature_columns]
    )

    test_predictions = np.maximum(
        test_predictions,
        0
    )

    actual_test_rul = (
        true_rul_df["true_rul"]
        .to_numpy()
    )

    test_mae = mean_absolute_error(
        actual_test_rul,
        test_predictions
    )

    test_rmse = root_mean_squared_error(
        actual_test_rul,
        test_predictions
    )

    print("\n----- NASA FD001 Test Evaluation -----")
    print(f"Test MAE  : {test_mae:.2f} cycles")
    print(f"Test RMSE : {test_rmse:.2f} cycles")

    print("\nFirst 5 test predictions:")

    comparison = pd.DataFrame({
        "engine_id": final_test_rows["engine_id"].head(5),
        "actual_RUL": actual_test_rul[:5],
        "predicted_RUL": np.round(
            test_predictions[:5],
            2
        )
    })

    print(
        comparison.to_string(index=False)
    )

    # --------------------------------------------------
    # TRAIN FINAL MODEL ON ALL TRAINING DATA
    # --------------------------------------------------

    print("\nTraining final model on all 100 engines...")

    final_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(
        X,
        y
    )

    # Save model + exact feature list together.
    model_package = {
        "model": final_model,
        "features": feature_columns,
        "model_name": "Random Forest RUL Regressor",
        "dataset": "NASA C-MAPSS FD001",
        "rul_unit": "operating cycles"
    }

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model_package,
        MODEL_FILE
    )

    print(
        f"\nModel saved successfully to:\n{MODEL_FILE}"
    )

    print("\n========================================\n")


if __name__ == "__main__":
    train_model()