"""Central configuration for the MLOps Heart Disease project."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_FILE: Path = DATA_DIR / "heart_disease_uci.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_FILE: Path = MODELS_DIR / "heart_model.joblib"
METRICS_FILE: Path = MODELS_DIR / "metrics.json"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

MLFLOW_TRACKING_URI: str = f"file://{PROJECT_ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT: str = "heart-disease-classification"

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "target"           # engineered from `num`
RAW_TARGET_COLUMN: str = "num"          # 0..4 in UCI dataset
DROP_COLUMNS: list[str] = ["id"]        # non-informative
NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES: list[str] = [
    "sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal",
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_SPLITS: int = 5

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_TITLE: str = "Heart Disease Risk API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Predicts the risk of heart disease for a patient given clinical features. "
    "Built for the AIMLCZG523 MLOps Assignment 01."
)
