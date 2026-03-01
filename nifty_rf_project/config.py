"""Central configuration for the Nifty Random Forest project."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "nifty_rf_model.joblib"

DATA_FILE = DATA_DIR / "nifty_5min.csv"

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "volume_z",
    "dist_to_vwap",
    "rsi_14",
]
TARGET_COLUMN = "target_up"

TARGET_PCT = 0.006
STOP_LOSS_PCT = 0.004
TRIGGER_PCT = -0.005
DEFAULT_THRESHOLD = 0.60

TRAIN_SPLIT = 0.80
RANDOM_STATE = 42

BROKERAGE_BPS = 2


def ensure_dirs() -> None:
    """Create required directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
