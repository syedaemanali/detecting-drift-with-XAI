from pathlib import Path
import os 
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_PATH   = DATA_DIR / "bank-additional-full.csv"
PROCESSED_PATH  = DATA_DIR / "processed.parquet"
MODELS_DIR      = ROOT_DIR / "models"
RESULTS_DIR     = ROOT_DIR / "results"
PLOTS_DIR       = ROOT_DIR / "plots"
SHAP_DIR        = RESULTS_DIR / "shap"
METRICS_FILE    = RESULTS_DIR / "metrics_summary.csv"

UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00222/bank-additional.zip"
)

TARGET_COL   = "y"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

FEATURE_COLS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed",
]

MODELS_TO_TRAIN  = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
SELECTION_METRIC = "f1"

MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": RANDOM_STATE,
        "verbose": -1
    },
}
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "explainability-drift"
MLFLOW_MODEL_NAME      = "bank-marketing-classifier"

STREAM_WINDOW_SIZE = 200
DRIFT_START_FRAC   = 0.40
DRIFT_MAGNITUDE    = 2.0
DRIFT_TYPES        = ["sudden", "gradual", "recurring", "incremental"]

# experiment controls
WARMUP_WINDOWS = 2
CALIBRATION_WINDOWS = 3
SMOTE_EXPERIMENT_SUFFIX = "smote"
SMOTE_EXPERIMENT_NAME = "explainability-drift-smote"

# detector calibration and smoothing
CALIBRATION_QUANTILE = 0.95
PSI_WARNING_QUANTILE = 0.85
PSI_ALERT_QUANTILE   = 0.95
TEMPORAL_CONFIRM_WINDOW = 3
TEMPORAL_CONFIRM_MIN_ALERTS = 2
WINDOW_DRIFT_LABEL_THRESHOLD = 0.50

SHAP_DRIFT_THRESHOLD    = 0.15
PSI_WARNING_THRESHOLD   = 0.10
PSI_ALERT_THRESHOLD     = 0.20
KS_ALPHA                = 0.05
SHAP_BACKGROUND_SAMPLES = 100

METRICS_SERVER_PORT = 8000
PROMETHEUS_PORT     = 9090
GRAFANA_PORT        = 3000

PALETTE = {
    "shap_drift": "#7B9CDA",
    "psi":        "#F4A896",
    "ks":         "#A8D5A2",
    "no_drift":   "#D3D3D3",
    "alert":      "#E8735A",
    "warning":    "#F6C567",
    "background": "#FAFAFA",
}

FIGURE_DPI = 150