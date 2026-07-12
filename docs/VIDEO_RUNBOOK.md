# Video Recording Runbook — MLOps Assignment 01

A ~6–8 minute screen recording that walks through the full pipeline.
Every command below has been **pre-flighted** and produces clean output.

> Record with **Cmd + Shift + 5 → Record Entire Screen**. The `.mov` saves to your Desktop.

---

## Before you hit record (setup — do NOT record this)

1. Open **2 terminal tabs**, both `cd /Users/k0j089g/BITS/Assignment/MLOPS`:
   - **Terminal A** = commands
   - **Terminal B** = live logs → `docker compose logs -f api`
2. Confirm everything is up:
   ```bash
   docker compose ps           # api + prometheus + grafana healthy
   kubectl --context k3d-heart -n heart-disease get pods   # 2/2 Running
   ```
3. Open **4 browser tabs** (ready, not recorded):
   - http://localhost:8000/docs  (Swagger)
   - http://localhost:3000       (Grafana — admin/admin)
   - http://localhost:9090/targets (Prometheus)
   - http://localhost:9090/graph  (Prometheus graph)

>  **Do NOT run `make serve` / a local `uvicorn`.** The compose container already
> owns port 8000; a second local instance causes a metrics split-brain.

---

## SCENE 1 — Intro & repo tour (~30s)
> "End-to-end MLOps pipeline for heart-disease risk prediction."
```bash
cd /Users/k0j089g/BITS/Assignment/MLOPS
ls
git log --oneline -5
```

## SCENE 2 — Data, EDA, Models (~40s)
> "Task 1 & 2 — data, EDA, three tuned models."
```bash
ls data/raw/
head -3 data/raw/heart_disease_uci.csv     # NOTE: data/raw/ , not data/
ls notebooks/
cat models/metrics.json
```
Point out: Random Forest wins, test ROC-AUC ~0.93. Optionally open `notebooks/01_eda.ipynb`.

## SCENE 3 — MLflow tracking (~40s)
> "Task 3 — experiment tracking."
```bash
source .venv/bin/activate
mlflow ui --backend-store-uri ./mlruns --port 5001 &   # 5001 avoids macOS AirPlay :5000
sleep 4
```
Open http://localhost:5001 → experiment `heart-disease-classification` → click a run →
show params, metrics, artifacts (confusion matrix, ROC). Then stop it:
```bash
pkill -f "mlflow ui"
```

## SCENE 4 — Tests & CI (~40s)
> "Task 5 — automated tests and CI/CD."
```bash
ruff check src tests
pytest -q
cat .github/workflows/ci.yml | head -40
```
> "On every push, GitHub Actions runs lint → tests → smoke-train → Docker build + probe."

## SCENE 5 — Docker + live API / Swagger (~60s)
> "Task 6 — containerized API via docker-compose."
```bash
docker compose ps
```
In **Swagger** (http://localhost:8000/docs): POST /predict → Try it out → paste:
```json
{"age":63,"sex":"Male","dataset":"Cleveland","cp":"typical angina","trestbps":145,"chol":233,"fbs":true,"restecg":"lv hypertrophy","thalch":150,"exang":false,"oldpeak":2.3,"slope":"downsloping","ca":0,"thal":"fixed defect"}
```
→ Execute → show response. Glance at **Terminal B** — the request appears in the logs.

You can also show the standalone CLI inference:
```bash
python -m src.predict --json '{"age":63,"sex":"Male","cp":"typical angina"}'
```

## SCENE 6 — Monitoring: Prometheus + Grafana (~80s)
> "Task 8 — logging and monitoring. Generating traffic:"
```bash
bash scripts/generate_traffic.sh 120
```
Watch **Terminal B** stream request logs. Wait ~15s, then:
- **Prometheus** http://localhost:9090/targets → `heart-api` = **UP**
- **Prometheus graph** http://localhost:9090/graph → query `heart_predictions_total` → Graph tab
- **Grafana** http://localhost:3000 (admin/admin) → Dashboards → **Heart Disease API** →
  set range **Last 15 minutes**, refresh. Show all 4 panels populated.

## SCENE 7 — Kubernetes deployment (~60s)
> "Task 7 — deployed on a local Kubernetes cluster, 2 replicas + Service + HPA."
```bash
kubectl config use-context k3d-heart
kubectl get nodes
kubectl -n heart-disease get deploy,pods,svc,hpa
```
Prove live inference through the Service (port-forward on **8080** to avoid the compose :8000):
```bash
kubectl -n heart-disease port-forward svc/heart-api 8080:80 &
sleep 4
curl -s http://localhost:8080/health
curl -s -X POST http://localhost:8080/predict -H "Content-Type: application/json" --data-binary @/tmp/patient.json
kill %1     # stop port-forward
helm lint helm/heart-api
```

## SCENE 8 — Wrap up (~20s)
> "Data → EDA → three tuned models tracked in MLflow → tested & CI/CD-gated →
> containerized → deployed to Kubernetes → monitored with Prometheus & Grafana. Thank you."

**Stop the recording** (stop button in the menu bar).

---

## Gotchas (so nothing breaks on camera)
| Risk | Avoid it |
|------|----------|
| Port 8000 conflict | Never run a local `uvicorn`/`make serve`; the compose container owns 8000. |
| Grafana "No data" | Run `scripts/generate_traffic.sh` first + wait ~15s + set range "Last 15m". |
| K8s port-forward | Uses **8080** on purpose (compose uses 8000). |
| MLflow port | Use **5001** (5000 clashes with macOS AirPlay). |
| Missing sample file | `/tmp/patient.json` is created by the traffic script; if absent, use the JSON body from Scene 5. |
