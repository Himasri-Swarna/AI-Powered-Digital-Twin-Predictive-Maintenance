from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset" / "CMAPSSData"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"


# NASA C-MAPSS columns
COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001():
    """Load NASA C-MAPSS FD001 train, test and test-RUL files."""

    train_df = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    test_df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES
    )

    true_rul_df = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None,
        names=["true_rul"]
    )

    return train_df, test_df, true_rul_df


def add_training_rul(train_df):
    """
    Training engines run until failure.

    RUL = engine's final cycle - current cycle
    """

    df = train_df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    df = df.merge(max_cycles, on="engine_id", how="left")

    df["RUL"] = df["max_cycle"] - df["cycle"]

    df.drop(columns=["max_cycle"], inplace=True)

    return df


def validate_data(train_df, test_df, true_rul_df):

    print("\n========== FD001 DATA VALIDATION ==========\n")

    print("Training shape:")
    print(train_df.shape)

    print("\nTesting shape:")
    print(test_df.shape)

    print("\nNumber of training engines:")
    print(train_df["engine_id"].nunique())

    print("\nNumber of testing engines:")
    print(test_df["engine_id"].nunique())

    print("\nNumber of test RUL labels:")
    print(len(true_rul_df))

    print("\nMissing values in training data:")
    print(train_df.isnull().sum().sum())

    print("\nRUL range:")
    print(f"Minimum RUL = {train_df['RUL'].min()} cycles")
    print(f"Maximum RUL = {train_df['RUL'].max()} cycles")

    print("\nFirst 5 rows:")
    print(
        train_df[
            ["engine_id", "cycle", "sensor_2", "sensor_3", "RUL"]
        ].head()
    )

    print("\n===========================================\n")


if __name__ == "__main__":

    train, test, true_rul = load_fd001()

    train = add_training_rul(train)

    validate_data(train, test, true_rul)