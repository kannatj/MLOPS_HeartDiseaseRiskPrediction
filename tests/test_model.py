"""Tests for training-time metrics and evaluation helpers."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_preprocessing import make_pipeline
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve


def _fit_small(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pipe = make_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    return pipe, Xte, yte, proba


def test_compute_metrics_returns_expected_keys(tiny_xy):
    X, y = tiny_xy
    _, _, yte, proba = _fit_small(X, y)
    preds = (proba >= 0.5).astype(int)
    metrics = compute_metrics(yte, preds, proba)
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_roc_auc_beats_random(xy):
    """The full-dataset LR should easily beat a coin flip on ROC-AUC."""
    X, y = xy
    _, _, yte, proba = _fit_small(X, y)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(yte, proba) > 0.7


def test_random_forest_pipeline_serialises(tiny_xy, tmp_model_path: Path):
    X, y = tiny_xy
    pipe = make_pipeline(RandomForestClassifier(n_estimators=25, random_state=0))
    pipe.fit(X, y)
    joblib.dump(pipe, tmp_model_path)
    loaded = joblib.load(tmp_model_path)
    np.testing.assert_array_equal(loaded.predict(X), pipe.predict(X))


def test_plots_are_written(tiny_xy, tmp_path):
    X, y = tiny_xy
    _, _, yte, proba = _fit_small(X, y)
    preds = (proba >= 0.5).astype(int)
    cm = plot_confusion_matrix(yte, preds, tmp_path / "cm.png", "cm")
    roc = plot_roc_curve(yte, proba, tmp_path / "roc.png", "roc")
    assert cm.exists() and cm.stat().st_size > 0
    assert roc.exists() and roc.stat().st_size > 0
