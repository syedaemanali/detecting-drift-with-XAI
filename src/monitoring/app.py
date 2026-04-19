import logging
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()

# one gauge per metric per detector
GAUGES = {
    "latency":   Gauge("drift_detection_latency",   "Detection latency in windows", ["detector"]),
    "fpr":       Gauge("drift_fpr",                 "False positive rate",          ["detector"]),
    "fnr":       Gauge("drift_fnr",                 "False negative rate",          ["detector"]),
    "precision": Gauge("drift_precision",            "Precision",                    ["detector"]),
    "recall":    Gauge("drift_recall",               "Recall",                       ["detector"]),
    "f1":        Gauge("drift_f1",                   "F1 score",                     ["detector"]),
    "cost":      Gauge("drift_cost_score",           "Weighted cost score",          ["detector"]),
}


def refresh_gauges():
    if not config.METRICS_FILE.exists():
        log.info("Metrics file not found yet, nothing to export")
        return

    df = pd.read_csv(config.METRICS_FILE)

    for _, row in df.iterrows():
        detector = row["detector"]
        for metric, gauge in GAUGES.items():
            gauge.labels(detector=detector).set(row[metric])

    log.info("Prometheus gauges refreshed with latest metrics")


@app.get("/metrics")
def metrics_endpoint():
    refresh_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}