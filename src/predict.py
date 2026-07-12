"""Standalone command-line inference for the trained heart-disease model.

This script satisfies the "inference" deliverable independently of the API:
it loads the persisted ``sklearn.Pipeline`` and scores one or more patients.

Examples
--------
Run on the bundled sample patient::

    python -m src.predict

Score a single patient from a JSON string::

    python -m src.predict --json '{"age": 63, "sex": "Male", "cp": "typical angina"}'

Score a batch from a JSON file (object or list of objects)::

    python -m src.predict --input patients.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config

# A representative, high-risk sample used when no input is supplied.
SAMPLE_PATIENT: dict = {
    "age": 63, "sex": "Male", "dataset": "Cleveland", "cp": "typical angina",
    "trestbps": 145, "chol": 233, "fbs": True, "restecg": "lv hypertrophy",
    "thalch": 150, "exang": False, "oldpeak": 2.3, "slope": "downsloping",
    "ca": 0, "thal": "fixed defect",
}


def load_model(model_path: Path):
    """Load the trained pipeline, failing loudly with a clear message."""
    if not model_path.exists():
        sys.exit(
            f"ERROR: model file not found at {model_path}. "
            "Train first with `python -m src.train`."
        )
    return joblib.load(model_path)


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Align arbitrary patient dicts to the training-time feature columns."""
    df = pd.DataFrame(records)
    for col in config.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[config.FEATURE_COLUMNS]


def predict(model, records: list[dict]) -> list[dict]:
    """Return a list of prediction dicts for the given patient records."""
    proba = model.predict_proba(to_frame(records))[:, 1]
    results = []
    for p in proba:
        pred = int(p >= 0.5)
        results.append(
            {
                "prediction": pred,
                "label": "disease" if pred == 1 else "no_disease",
                "probability": round(float(p), 4),
                "confidence": round(float(p if pred == 1 else 1.0 - p), 4),
            }
        )
    return results


def _load_records(args: argparse.Namespace) -> list[dict]:
    if args.input:
        data = json.loads(Path(args.input).read_text())
    elif args.json:
        data = json.loads(args.json)
    else:
        print("No input supplied — scoring the bundled sample patient.\n")
        data = SAMPLE_PATIENT
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart-disease CLI inference")
    parser.add_argument("--input", help="Path to a JSON file (object or list)")
    parser.add_argument("--json", help="Inline JSON string for one patient")
    parser.add_argument(
        "--model", default=str(config.MODEL_FILE), help="Path to the .joblib model"
    )
    args = parser.parse_args()

    model = load_model(Path(args.model))
    records = _load_records(args)
    results = predict(model, records)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
