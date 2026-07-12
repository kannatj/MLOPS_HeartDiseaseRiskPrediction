"""Train, tune, and log candidate heart-disease classifiers to MLflow.

Run:
    python -m src.train                 # full grid-search training
    python -m src.train --fast          # small grid for CI smoke tests

The script produces:
    - MLflow experiment ``heart-disease-classification`` with one run per model.
    - A registered "best" model saved to ``models/heart_model.joblib``.
    - ``models/metrics.json`` summarising every candidate.
    - Confusion-matrix and ROC plots under ``reports/figures/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
    _XGB_IMPORT_ERROR: str | None = None
except Exception as _xgb_exc:  # ImportError / OSError / XGBoostError (missing libomp)
    XGBClassifier = None  # type: ignore[assignment,misc]
    XGBOOST_AVAILABLE = False
    _XGB_IMPORT_ERROR = str(_xgb_exc)

from src import config
from src.data_preprocessing import load_raw_dataset, make_pipeline, prepare_dataset
from src.evaluate import (
    compute_metrics,
    dump_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Candidate models + hyper-parameter grids
# ---------------------------------------------------------------------------
def build_candidates(fast: bool) -> dict[str, tuple[object, dict[str, list]]]:
    grids: dict[str, tuple[object, dict[str, list]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear"),
            {
                "classifier__C": [0.1, 1.0] if fast else [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"] if fast else ["l1", "l2"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [100] if fast else [200, 400],
                "classifier__max_depth": [None, 8] if fast else [None, 6, 10, 16],
                "classifier__min_samples_split": [2] if fast else [2, 4, 8],
            },
        ),
    }
    if XGBOOST_AVAILABLE:
        grids["xgboost"] = (
            XGBClassifier(
                random_state=config.RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [200] if fast else [200, 400, 600],
                "classifier__max_depth": [4] if fast else [3, 4, 6, 8],
                "classifier__learning_rate": [0.1] if fast else [0.03, 0.1, 0.2],
                "classifier__subsample": [0.9] if fast else [0.7, 0.9, 1.0],
            },
        )
    else:
        log.warning(
            "XGBoost is unavailable (%s). Skipping xgboost candidate. "
            "Install `libomp` on macOS (`brew install libomp`) to enable it.",
            _XGB_IMPORT_ERROR,
        )
    return grids


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def train(fast: bool = False) -> dict:
    log.info("Loading dataset from %s", config.DATA_FILE)
    df = load_raw_dataset()
    X, y = prepare_dataset(df)
    log.info("Dataset shape: X=%s  y=%s (positives=%d)", X.shape, y.shape, int(y.sum()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    results: dict[str, dict] = {}
    best_pipeline = None
    best_name: str | None = None
    best_score = -np.inf

    candidates = build_candidates(fast=fast)
    for name, (estimator, param_grid) in candidates.items():
        pipeline = make_pipeline(estimator)
        run_name = f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "model": name,
                    "framework": "scikit-learn" if name != "xgboost" else "xgboost",
                    "fast_mode": str(fast),
                }
            )

            log.info("[%s] running GridSearchCV over %d params",
                     name, int(np.prod([len(v) for v in param_grid.values()])))
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)

            cv_best = float(search.best_score_)
            best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}

            # Evaluate on held-out test set
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)
            metrics["cv_roc_auc"] = cv_best

            # Plots
            cm_path = config.FIGURES_DIR / f"{name}_confusion_matrix.png"
            roc_path = config.FIGURES_DIR / f"{name}_roc_curve.png"
            plot_confusion_matrix(y_test, y_pred, cm_path, f"{name} — Confusion Matrix")
            plot_roc_curve(y_test, y_proba, roc_path, f"{name} — ROC Curve")

            # ---- Log to MLflow ------------------------------------------------
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

            model_path = config.MODELS_DIR / f"{name}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(search.best_estimator_, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model_pickle")
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")

            results[name] = {
                "best_params": best_params,
                "metrics": metrics,
                "model_path": str(model_path),
            }

            log.info(
                "[%s] cv_auc=%.4f  test_auc=%.4f  acc=%.4f  f1=%.4f",
                name,
                cv_best,
                metrics["roc_auc"],
                metrics["accuracy"],
                metrics["f1"],
            )

            score = metrics["roc_auc"]
            if score > best_score:
                best_score = score
                best_pipeline = search.best_estimator_
                best_name = name

    # ---- Persist final "best" artefact -----------------------------------
    assert best_pipeline is not None and best_name is not None
    joblib.dump(best_pipeline, config.MODEL_FILE)
    log.info("Best model = %s (roc_auc=%.4f) -> %s", best_name, best_score, config.MODEL_FILE)

    summary = {
        "best_model": best_name,
        "best_roc_auc": best_score,
        "candidates": results,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    dump_metrics(summary, config.METRICS_FILE)
    log.info("Wrote summary metrics -> %s", config.METRICS_FILE)

    # Also stash a "best_model" MLflow run
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.set_tags({"role": "best_model", "model": best_name})
        mlflow.log_metrics(results[best_name]["metrics"])
        mlflow.log_params(results[best_name]["best_params"])
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
        mlflow.log_artifact(str(config.MODEL_FILE), artifact_path="model_pickle")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Small grid (used by CI smoke tests).",
    )
    args = parser.parse_args()
    summary = train(fast=args.fast)
    print(json.dumps({"best_model": summary["best_model"], "best_roc_auc": summary["best_roc_auc"]}, indent=2))


if __name__ == "__main__":
    main()
