"""Data loading and preprocessing pipeline for the Heart Disease dataset.

The public surface is intentionally tiny so it can be re-used by:
- training script (``src.train``)
- inference API (``src.api.main``)
- unit tests (``tests/``)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_raw_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Return the raw UCI heart-disease CSV as a DataFrame."""
    path = Path(path) if path else config.DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python data/download_data.py`."
        )
    return pd.read_csv(path)


def _coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """The UCI CSV stores fbs/exang as ``TRUE``/``FALSE`` strings; normalise."""
    for col in ("fbs", "exang"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(
                {"TRUE": True, "FALSE": False, "True": True, "False": False}
            )
    return df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, engineer target, and return (X, y).

    - Drops non-informative columns (``id``).
    - Binarises the target: presence of any heart disease (``num >= 1``).
    - Coerces boolean strings.
    - Leaves imputation/scaling to the sklearn ``Pipeline``.
    """
    df = df.copy()
    df = _coerce_booleans(df)

    # Engineer binary target BEFORE dropping the raw column
    if config.RAW_TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Expected column '{config.RAW_TARGET_COLUMN}' in dataset, "
            f"got {list(df.columns)}"
        )
    df[config.TARGET_COLUMN] = (df[config.RAW_TARGET_COLUMN] >= 1).astype(int)

    cols_to_drop = [c for c in config.DROP_COLUMNS + [config.RAW_TARGET_COLUMN] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Ensure all expected features exist (fill with NaN if missing)
    for col in config.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Return a ``ColumnTransformer`` that imputes, scales, and encodes."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(estimator) -> Pipeline:
    """Wrap the preprocessor + a scikit-learn compatible estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )
