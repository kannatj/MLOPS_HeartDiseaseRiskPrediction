PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help venv install lint format test train serve docker-build docker-run compose-up compose-down k8s-apply k8s-delete clean

help:
	@echo "Targets:"
	@echo "  venv          - create local virtualenv"
	@echo "  install       - install dev + runtime dependencies"
	@echo "  lint          - run ruff"
	@echo "  format        - run black"
	@echo "  test          - run pytest with coverage"
	@echo "  train         - train models and log to MLflow"
	@echo "  serve         - run FastAPI locally on :8000"
	@echo "  docker-build  - build API docker image"
	@echo "  docker-run    - run docker image on :8000"
	@echo "  compose-up    - API + Prometheus + Grafana"
	@echo "  compose-down  - tear down compose stack"
	@echo "  k8s-apply     - apply k8s manifests"
	@echo "  k8s-delete    - delete k8s resources"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

lint:
	$(BIN)/ruff check src tests

format:
	$(BIN)/black src tests

test:
	$(BIN)/pytest --cov=src --cov-report=term-missing

train:
	$(BIN)/python -m src.train

serve:
	$(BIN)/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile -t heart-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-api:latest

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down -v

k8s-apply:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete -f k8s/ --ignore-not-found

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
