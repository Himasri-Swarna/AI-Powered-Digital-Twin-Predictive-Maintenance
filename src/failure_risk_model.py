from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from data_loader import load_fd001, add_training_rul


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "failure_risk_model.joblib"


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

def get_feature_columns(df):

    candidates = (
        ["cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    return [
        column
        for column in candidates
        if df[column].nunique() > 1
    ]


# --------------------------------------------------
# TRAIN FAILURE-RISK MODEL
# --------------------------------------------------

def train_failure_model():

    print("\n========== FAILURE RISK MODEL ==========\n")

    train_df, _, _ = load_fd001()

    train_df = add_training_rul(train_df)

    # Failure window:
    # Engine is considered at imminent risk
    # when remaining life <= 30 cycles.
    FAILURE_WINDOW = 30

    train_df["failure_soon"] = (
        train_df["RUL"] <= FAILURE_WINDOW
    ).astype(int)

    features = get_feature_columns(train_df)

    X = train_df[features]

    y = train_df["failure_soon"]

    print("Failure definition:")
    print(f"RUL <= {FAILURE_WINDOW} cycles")

    print("\nClass distribution:")
    print(y.value_counts())

    print("\nFailure percentage:")
    print(f"{y.mean() * 100:.2f}%")

    # ----------------------------------------------
    # ENGINE-LEVEL SPLIT
    # ----------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=train_df["engine_id"]
        )
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    print(
        f"\nTraining engines: "
        f"{train_df.iloc[train_idx]['engine_id'].nunique()}"
    )

    print(
        f"Validation engines: "
        f"{train_df.iloc[val_idx]['engine_id'].nunique()}"
    )

    # ----------------------------------------------
    # RANDOM FOREST CLASSIFIER
    # ----------------------------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining failure-risk classifier...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # ----------------------------------------------
    # EVALUATION
    # ----------------------------------------------

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    print("\n----- Validation Results -----")

    print(
        f"Accuracy  : {accuracy:.3f}"
    )

    print(
        f"Precision : {precision:.3f}"
    )

    print(
        f"Recall    : {recall:.3f}"
    )

    print(
        f"F1 Score  : {f1:.3f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            predictions
        )
    )

    # Show sample probabilities
    sample = pd.DataFrame({
        "Actual": y_val.head(10).values,
        "Predicted": predictions[:10],
        "Failure_Probability": np.round(
            probabilities[:10] * 100,
            2
        )
    })

    print("\nExample predictions:")
    print(
        sample.to_string(index=False)
    )

    # ----------------------------------------------
    # FINAL MODEL
    # ----------------------------------------------

    print("\nTraining final classifier...")

    final_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(
        X,
        y
    )

    package = {
        "model": final_model,
        "features": features,
        "failure_window": FAILURE_WINDOW,
        "description": (
            "Probability that telemetry resembles "
            "an engine within its final 30 operating cycles"
        )
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
        f"\nFailure-risk model saved to:\n{MODEL_FILE}"
    )

    print("\n========================================\n")


if __name__ == "__main__":
    train_failure_model()