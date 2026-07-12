"""Tests for the data loading and preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    load_raw_dataset,
    make_pipeline,
    prepare_dataset,
)


def test_raw_dataset_loads():
    df = load_raw_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 920, "Expected the merged 4-hospital UCI dataset (920 rows)"
    assert {"age", "sex", "cp", "num"} <= set(df.columns)


def test_prepare_dataset_binarises_target(raw_dataset):
    X, y = prepare_dataset(raw_dataset)
    assert set(np.unique(y)) <= {0, 1}, "Target must be binary"
    assert (y == 1).sum() > 0 and (y == 0).sum() > 0, "Both classes must be present"
    assert "num" not in X.columns and "id" not in X.columns
    assert list(X.columns) == config.FEATURE_COLUMNS


def test_prepare_dataset_missing_raw_target_raises():
    df = pd.DataFrame({"age": [1, 2], "sex": ["Male", "Female"]})
    with pytest.raises(KeyError):
        prepare_dataset(df)


def test_preprocessor_handles_missing_and_categorical(tiny_xy):
    X, _ = tiny_xy
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)
    # After OHE the feature dimension must be > raw feature count
    assert transformed.shape[1] > len(config.FEATURE_COLUMNS)
    assert not np.isnan(transformed).any(), "Preprocessor must impute all NaNs"


def test_pipeline_end_to_end_on_tiny_slice(tiny_xy):
    from sklearn.linear_model import LogisticRegression

    X, y = tiny_xy
    pipe = make_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    proba = pipe.predict_proba(X)[:, 1]
    assert ((0.0 <= proba) & (proba <= 1.0)).all()
