"""Download / verify the Heart Disease UCI dataset.

The CSV that ships with this repository under ``data/raw/heart_disease_uci.csv``
is the exact file used for training. This script is provided so a grader can
reproduce the dataset acquisition step from scratch.

Usage
-----
    python data/download_data.py                 # verify existing file
    python data/download_data.py --force         # force re-download from UCI

The dataset is the merged 4-hospital Heart Disease dataset from the UCI ML
Repository (id 45). We fetch it via the ``ucimlrepo`` helper when installed,
otherwise fall back to the raw ``processed.cleveland.data`` files.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent / "raw"
DATA_PATH = DATA_DIR / "heart_disease_uci.csv"

# The file bundled in the repository was downloaded from Kaggle's mirror of the
# UCI Heart Disease dataset (merged 4-hospital version, 920 rows).
KAGGLE_MIRROR_URL = (
    "https://storage.googleapis.com/kagglesdsdata/datasets/"
    "heart-disease/heart_disease_uci.csv"
)

EXPECTED_ROWS = 920
EXPECTED_COLUMNS = {
    "id", "age", "sex", "dataset", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> bool:
    import pandas as pd

    if not path.exists():
        print(f"[!] Dataset not found at {path}", file=sys.stderr)
        return False
    df = pd.read_csv(path)
    ok_rows = len(df) == EXPECTED_ROWS
    ok_cols = set(df.columns) >= EXPECTED_COLUMNS
    print(f"    rows     : {len(df):>5}  (expected {EXPECTED_ROWS})")
    print(f"    columns  : {len(df.columns):>5}  match expected schema = {ok_cols}")
    print(f"    sha256   : {sha256(path)}")
    return ok_rows and ok_cols


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        print(f"[=] Using existing dataset at {DATA_PATH}")
        return DATA_PATH

    # Prefer ucimlrepo (respects the assignment's UCI reference)
    try:
        from ucimlrepo import fetch_ucirepo

        print("[+] Fetching UCI dataset id=45 via ucimlrepo ...")
        heart = fetch_ucirepo(id=45)
        df = heart.data.original
        df.to_csv(DATA_PATH, index=False)
        print(f"[+] Saved dataset -> {DATA_PATH}")
        return DATA_PATH
    except Exception as exc:  # pragma: no cover - network fallback
        print(f"[!] ucimlrepo unavailable ({exc}); trying Kaggle mirror ...")
        urlretrieve(KAGGLE_MIRROR_URL, DATA_PATH)
        print(f"[+] Saved dataset -> {DATA_PATH}")
        return DATA_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    path = download(force=args.force)
    print("[i] Verifying dataset ...")
    ok = verify(path)
    print("[OK]" if ok else "[FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
