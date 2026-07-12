"""Shared pytest fixtures."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make ``src`` importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data_preprocessing import load_raw_dataset, prepare_dataset  # noqa: E402


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    return load_raw_dataset()


@pytest.fixture(scope="session")
def xy(raw_dataset):
    X, y = prepare_dataset(raw_dataset)
    return X, y


@pytest.fixture(scope="session")
def tiny_xy(xy):
    """Small stratified slice for fast unit tests."""
    X, y = xy
    df = X.copy()
    df["_y"] = y.values
    sample = df.groupby("_y", group_keys=False).apply(
        lambda g: g.sample(n=min(30, len(g)), random_state=0)
    )
    return sample.drop(columns=["_y"]), sample["_y"].astype(int)


@pytest.fixture()
def tmp_model_path(tmp_path: Path) -> Path:
    """Return a temporary path for saving a joblib model."""
    return tmp_path / "model.joblib"


@pytest.fixture(autouse=True, scope="session")
def _clean_mlruns():
    """Prevent tests from polluting the main MLflow directory."""
    yield
    stray = config.PROJECT_ROOT / "mlruns_test"
    if stray.exists():
        shutil.rmtree(stray, ignore_errors=True)
