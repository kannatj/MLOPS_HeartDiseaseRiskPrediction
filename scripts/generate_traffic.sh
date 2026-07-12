#!/usr/bin/env bash
# generate_traffic.sh — fire varied prediction traffic at the running API
# so the Grafana dashboard + Prometheus have data to show.
#
# Usage:
#   bash scripts/generate_traffic.sh          # default: 150 requests
#   bash scripts/generate_traffic.sh 300      # custom count
#
# Requires the compose stack to be up:  docker compose up -d
set -euo pipefail

API="${API_URL:-http://localhost:8000}"
COUNT="${1:-150}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- patient profiles (mix of low- and high-risk) -------------------------
cat > "$TMP/p1.json" <<'JSON'
{"age":41,"sex":"Female","dataset":"Cleveland","cp":"non-anginal","trestbps":112,"chol":180,"fbs":false,"restecg":"normal","thalch":172,"exang":false,"oldpeak":0.0,"slope":"upsloping","ca":0,"thal":"normal"}
JSON
cat > "$TMP/p2.json" <<'JSON'
{"age":35,"sex":"Female","dataset":"Cleveland","cp":"atypical angina","trestbps":120,"chol":198,"fbs":false,"restecg":"normal","thalch":180,"exang":false,"oldpeak":0.2,"slope":"upsloping","ca":0,"thal":"normal"}
JSON
cat > "$TMP/p3.json" <<'JSON'
{"age":45,"sex":"Female","dataset":"Cleveland","cp":"asymptomatic"}
JSON
cat > "$TMP/p4.json" <<'JSON'
{"age":67,"sex":"Male","dataset":"Cleveland","cp":"asymptomatic","trestbps":160,"chol":286,"fbs":false,"restecg":"lv hypertrophy","thalch":108,"exang":true,"oldpeak":1.5,"slope":"flat","ca":3,"thal":"normal"}
JSON
cat > "$TMP/p5.json" <<'JSON'
{"age":62,"sex":"Male","dataset":"Cleveland","cp":"asymptomatic","trestbps":140,"chol":268,"fbs":false,"restecg":"lv hypertrophy","thalch":160,"exang":false,"oldpeak":3.6,"slope":"downsloping","ca":2,"thal":"reversable defect"}
JSON
cat > "$TMP/p6.json" <<'JSON'
{"age":58,"sex":"Male","dataset":"Cleveland","cp":"typical angina","trestbps":150,"chol":270,"fbs":true,"restecg":"lv hypertrophy","thalch":111,"exang":true,"oldpeak":0.8,"slope":"flat","ca":0,"thal":"reversable defect"}
JSON

echo "Firing $COUNT predictions at $API ..."
for i in $(seq 1 "$COUNT"); do
  n=$(( (i % 6) + 1 ))
  curl -s -X POST "$API/predict" \
    -H "Content-Type: application/json" \
    --data-binary "@$TMP/p${n}.json" > /dev/null
done

# a few intentionally-invalid requests to populate the 4xx HTTP-status panel
for i in $(seq 1 5); do
  curl -s -X POST "$API/predict" -H "Content-Type: application/json" \
    --data-binary '{"age":"notanumber"}' > /dev/null
done

echo "Done. Wait ~15s for Prometheus to scrape, then refresh Grafana."
