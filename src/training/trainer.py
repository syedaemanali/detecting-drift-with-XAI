import logging
import joblib

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest":       RandomForestClassifier,
    "xgboost":             XGBClassifier,
    "lightgbm":            LGBMClassifier,
}


def build_model(model_name):
    cls = MODEL_REGISTRY[model_name]
    return cls(**config.MODEL_PARAMS[model_name])


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":         accuracy_score(y_test, preds),
        "f1":               f1_score(y_test, preds),
        "precision":        precision_score(y_test, preds),
        "recall":           recall_score(y_test, preds),
        "roc_auc":          roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }


def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
    experiment_name=None,
    model_name_suffix=None,
    run_name_prefix=None,
):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or config.MLFLOW_EXPERIMENT_NAME)

    results = {}
    suite_label = model_name_suffix or "baseline"

    log.info("Starting model suite: %s", suite_label)

    for model_name in config.MODELS_TO_TRAIN:
        log.info("[%s] Training %s", suite_label, model_name)

        run_name = model_name if not run_name_prefix else f"{run_name_prefix}_{model_name}"

        with mlflow.start_run(run_name=run_name):
            model = build_model(model_name)
            model.fit(X_train, y_train)

            metrics = evaluate(model, X_test, y_test)
            loggable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}

            mlflow.log_params(config.MODEL_PARAMS[model_name])
            mlflow.log_metrics(loggable)

            # MLflow auto-increments version on each registration
            registered_name = f"{config.MLFLOW_MODEL_NAME}_{model_name}"
            if model_name_suffix:
                registered_name = f"{registered_name}_{model_name_suffix}"

            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=registered_name,
            )

            joblib.dump(model, config.MODELS_DIR / f"{model_name}.joblib")
            results[model_name] = {"model": model, "metrics": metrics}

            log.info(
                "[%s] %s done with F1 %.4f and AUC %.4f",
                suite_label,
                model_name,
                metrics["f1"],
                metrics["roc_auc"]
            )

    log.info("Finished model suite: %s", suite_label)

    return results


def pick_best_model(results):
    best_name = max(
        results,
        key=lambda name: results[name]["metrics"][config.SELECTION_METRIC]
    )
    best_score = results[best_name]["metrics"][config.SELECTION_METRIC]
    log.info("Best model is %s with %s %.4f", best_name, config.SELECTION_METRIC, best_score)
    return best_name, results[best_name]["model"]


def save_best_model(model_name, model):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / "best_model.joblib"
    joblib.dump({"model_name": model_name, "model": model}, path)
    log.info("Best model saved to %s", path)
    return path


def load_best_model():
    payload = joblib.load(config.MODELS_DIR / "best_model.joblib")
    return payload["model_name"], payload["model"]