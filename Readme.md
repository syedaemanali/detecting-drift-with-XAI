# Explainability Drift Detection Pipeline

This project compares three drift detection methods — SHAP-based explainability drift, PSI, and KS test — across four types of distribution shift, built on top of the UCI Bank Marketing dataset.

## Setup

pip install -r requirements.txt
python main.py

For the full monitoring stack (MLflow, Prometheus, Grafana):

docker compose up --build

Then in a second terminal:

python main.py

## What it does

Trains four classifiers, picks the best one, then simulates drift on the test stream and runs all three detectors. Results, plots, and a paper-ready table are written to `results/` and `plots/` automatically.

## Services

| Service | URL |
|---|---|
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |
| Metrics API | http://localhost:8000/metrics |

## Running tests

pytest tests/ -v

## Dataset

UCI Bank Marketing : https://archive.ics.uci.edu/ml/datasets/bank+marketing

Downloaded automatically on first run.
