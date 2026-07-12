# Validation Report — MLOps Assignment 01 (Heart Disease Risk API)

**Validated by:** Code Puppy (Leo) · **Date:** 2026-07-11
**Repo:** `/Users/k0j089g/BITS/Assignment/MLOPS`

This document maps every assignment requirement to concrete, reproduced evidence.
All raw command output is in `screenshots/evidence/*.txt`.

---

## Requirement → Evidence matrix (50 marks)

| # | Task (marks) | Status | Evidence |
|---|--------------|--------|----------|
| 1 | Data Acquisition & EDA (5) |  PASS | `data/download_data.py` (download+verify, SHA-256, 920 rows), `notebooks/01_eda.ipynb` (missing-value analysis, histograms, class balance, correlation heatmap) |
| 2 | Feature Engineering & Models (8) |  PASS | `src/data_preprocessing.py` (ColumnTransformer: impute+scale+OHE), `src/train.py` (LR + RandomForest + XGBoost, GridSearchCV, StratifiedKFold 5-fold). See `03_train_fast.txt` |
| 3 | Experiment Tracking — MLflow (5) |  PASS | `src/train.py` logs params, 7 metrics, confusion-matrix + ROC plots, joblib + sklearn model per run. `mlruns/` has 14 runs. See `03_train_fast.txt` |
| 4 | Packaging & Reproducibility (7) |  PASS | single `sklearn.Pipeline` → `models/heart_model.joblib`, pinned `requirements.txt`, seed in `config.py`. |
| 5 | CI/CD & Testing (8) |  PASS | `.github/workflows/ci.yml` (ruff → pytest+cov → smoke-train → docker build+probe). 16/16 tests pass, ruff clean. See `01_ruff_lint.txt`, `02_pytest.txt` |
| 6 | Containerization (5) |  PASS | multi-stage `docker/Dockerfile`, image built (`da8a66e188bf`, 989MB), container serves `/predict` returning prediction+confidence. See `06_docker_build.txt`, `07_docker_run.txt`, `08_docker_logs.txt` |
| 7 | Production Deployment (7) |  PASS | Deployed to local k3d/k3s cluster: 2/2 pods Running, Service, Ingress, **HPA active** (metrics-server). Live `/predict` served through the Service. See `09_k8s_validation.txt`, `10_helm.txt`, `11_k8s_deploy.txt`, `12_k8s_predict.txt`. |
| 8 | Monitoring & Logging (3) |  PASS | Live stack (API+Prometheus+Grafana): both targets UP, 215 predictions tracked (125 no_disease/90 disease), p95 latency 24ms, HTTP 285x2xx+8x4xx, Grafana datasource `uid=prometheus` + provisioned "Heart Disease API" dashboard, structured request logging (296 log lines). See `04_api_curl.txt`, `05_api_logging.txt`, `13_monitoring.txt`, `14_request_logging.txt`. |
| 9 | Documentation & Reporting (2) |  PARTIAL | `docs/report.md` (12 sections + arch diagram `docs/architecture.mmd`). **Needs personal details filled + PDF export + real screenshots.** |

---

## Verified runs (this session)

1. **Lint** — `ruff check src tests` → *All checks passed!*
2. **Tests** — `pytest --cov=src` → *16 passed*, 66% coverage (train.py exercised separately).
3. **Training** — `python -m src.train --fast` → 3 models trained + logged to MLflow.
   Best = **random_forest**, test ROC-AUC **0.9287**.
4. **API (local)** — `/health` ok, `/predict` returns prediction+probability+confidence,
   invalid input → 422, `/metrics` exposes Prometheus counters, request logging works.
5. **Docker** — image built (18 stages), container healthy, `/predict` + `/predict/batch`
   verified from inside the container.
6. **K8s/Helm** — all 5 manifests structurally valid; `helm lint` clean; `helm template` renders.
7. **K8s live deploy (k3d/k3s v1.30.4)** — `kubectl apply -f k8s/` → Deployment 2/2, Service,
   Ingress, HPA (cpu 1%/70%, 2 replicas). Live `/health` + `/predict` verified through
   `svc/heart-api` via port-forward. Image loaded with `k3d image import heart-api:latest`.

## Model comparison (models/metrics.json)

| Model | CV ROC-AUC | Test ROC-AUC | Test F1 | Test Acc |
|-------|-----------|--------------|---------|----------|
| Logistic Regression | 0.8921 | 0.9175 | 0.8692 | 0.8478 |
| **Random Forest (best)** | 0.8874 | **0.9287** | **0.8704** | 0.8478 |
| XGBoost | 0.8595 | 0.9026 | 0.8612 | 0.8424 |

---

##  Remaining manual TODOs before submission

- [ ] Fill personal fields in `docs/report.md` (name, BITS ID, GitHub repo URL, video link).
- [ ] Export `docs/report.md` → PDF (≈10 pages).
- [ ] Capture real screenshots into `screenshots/`:
      MLflow UI, GitHub Actions run, K8s pods running, Prometheus targets, Grafana dashboard.
- [x] (Task 7) Deployed to local k3d/k3s cluster — evidence in `11_k8s_deploy.txt` + `12_k8s_predict.txt`.
      (Take a terminal screenshot of these for the report's screenshots/ folder.)
- [ ] Record the short pipeline walkthrough video.
- [ ] Push to a clean GitHub repo (ensure `mlruns/` size is acceptable or gitignored).
