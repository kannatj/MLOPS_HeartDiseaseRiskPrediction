# MLOps Assignment 01 — Heart Disease Risk Prediction

End-to-end MLOps pipeline for AIMLCZG523 Assignment 01. It trains classifiers on the
UCI Heart Disease dataset, tracks experiments in **MLflow**, serves predictions from a
**FastAPI** service, ships as a **Docker** image, runs on **Kubernetes** (Docker Desktop /
Minikube / any cluster), is monitored via **Prometheus + Grafana**, and is CI-tested via
**GitHub Actions**.

## Architecture at a glance

```
       ┌──────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
       │ heart_disease│──►│ Preprocessing │──►│ Training       │──►│ MLflow (mlruns/) │
       │  _uci.csv    │    │ Pipeline      │    │ LR / RF / XGB │    └──────────────────┘
       └──────────────┘    └──────────────┘    └──────┬────────┘
                                                      ▼
                                            ┌──────────────────┐   docker/K8s
                                            │ heart_model.joblib│──────────────┐
                                            └──────────────────┘              ▼
                                                                    ┌──────────────────┐
                                              curl /predict ──────►│ FastAPI service  │──► Prometheus ► Grafana
                                                                    └──────────────────┘
                                              GitHub Actions: lint → test → train → build image
```

## Repository layout

```
mlops-heart-disease/
├── data/                        # dataset + download/verify script
├── src/                         # library code (config, preprocessing, training, api)
├── notebooks/                   # 01_eda.ipynb, 02_model_dev.ipynb
├── tests/                       # pytest suite (preprocessing / model / api)
├── docker/Dockerfile            # multi-stage image for the API
├── docker-compose.yml           # API + Prometheus + Grafana (local demo)
├── k8s/                         # Deployment, Service, Ingress, HPA
├── helm/heart-api/              # equivalent Helm chart
├── monitoring/                  # prometheus.yml + Grafana provisioning
├── .github/workflows/ci.yml     # GitHub Actions pipeline
├── docs/report.md               # written report (10 pages)
├── screenshots/                 # deliverable screenshots
└── models/                      # trained joblib artefacts
```

## Quick start

Requires Python 3.11+, Docker, and optionally Docker Desktop Kubernetes.

```bash
# 0. clone & set up virtualenv
make install                     # or: pip install -r requirements-dev.txt

# 1. sanity-check dataset
python data/download_data.py

# 2. run tests + linter
make lint
make test

# 3. train models (writes MLflow runs + models/heart_model.joblib)
make train

# 4. inspect MLflow UI
mlflow ui --backend-store-uri ./mlruns   # then open http://127.0.0.1:5000

# 5. run the API locally (auto-reload)
make serve                       # http://127.0.0.1:8000/docs

# 6. Docker image
make docker-build
make docker-run                  # http://127.0.0.1:8000/docs

# 7. Full local stack (API + Prometheus + Grafana)
make compose-up
# Grafana:    http://127.0.0.1:3000  (anonymous viewer)
# Prometheus: http://127.0.0.1:9090
make compose-down

# 8. Kubernetes (Docker Desktop / Minikube)
kubectl apply -f k8s/
kubectl -n heart-disease port-forward svc/heart-api 8000:80
```

## Sample `/predict` call

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "age": 63, "sex": "Male", "dataset": "Cleveland",
        "cp": "typical angina", "trestbps": 145, "chol": 233,
        "fbs": true, "restecg": "lv hypertrophy", "thalch": 150,
        "exang": false, "oldpeak": 2.3, "slope": "downsloping",
        "ca": 0, "thal": "fixed defect"
      }' | python -m json.tool
```

Response:

```json
{
  "prediction": 1,
  "label": "disease",
  "probability": 0.89,
  "confidence": 0.89,
  "model_version": "xgboost@2026-...",
  "threshold": 0.5
}
```

## What each task maps to

| Task | Marks | Where |
|---|---|---|
| 1. Acquisition + EDA | 5 | `data/download_data.py`, [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) |
| 2. Features + models + tuning | 8 | `src/data_preprocessing.py`, `src/train.py` |
| 3. MLflow tracking | 5 | `src/train.py` (params, metrics, artefacts, plots) |
| 4. Packaging + reproducibility | 7 | `requirements.txt`, `models/heart_model.joblib`, sklearn Pipeline |
| 5. CI/CD + tests | 8 | `tests/`, `.github/workflows/ci.yml` |
| 6. Containerisation | 5 | `docker/Dockerfile`, `docker-compose.yml` |
| 7. Kubernetes deployment | 7 | `k8s/`, `helm/heart-api/` |
| 8. Monitoring + logging | 3 | `src/api/main.py` (logging + Prometheus), `monitoring/` |
| 9. Report | 2 | `docs/report.md` |

See [docs/report.md](docs/report.md) for the written submission report.
