import logging
import numpy as np
import shap
from scipy.spatial.distance import cosine

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


def build_explainer(model, background_data):
    """Build a SHAP explainer. TreeExplainer for tree models, KernelExplainer as fallback."""
    try:
        explainer = shap.TreeExplainer(model)
        log.info("Using TreeExplainer for SHAP")
    except Exception:
        background_sample = shap.sample(background_data, config.SHAP_BACKGROUND_SAMPLES)
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        log.info("Using KernelExplainer for SHAP")
    return explainer


def compute_mean_shap_vector(explainer, window_data):
    """Returns the mean absolute SHAP value per feature for a window of samples."""
    shap_values = explainer.shap_values(window_data)

    # TreeExplainer returns a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.mean(np.abs(shap_values), axis=0)


def cosine_distance(vector_a, vector_b):
    # cosine distance of 0 means identical, 1 means completely different
    return cosine(vector_a, vector_b)


def _apply_temporal_confirmation(boolean_flags):
    window = config.TEMPORAL_CONFIRM_WINDOW
    min_alerts = config.TEMPORAL_CONFIRM_MIN_ALERTS

    smoothed = []
    for i in range(len(boolean_flags)):
        start = max(0, i - window + 1)
        votes = np.sum(boolean_flags[start : i + 1])
        smoothed.append(votes >= min_alerts)
    return np.array(smoothed, dtype=bool)


def detect_shap_drift(
    model,
    reference_data,
    stream_data,
    threshold=None,
    apply_confirmation=True,
    warmup_windows=0,
):
    explainer = build_explainer(model, reference_data)
    reference_shap = compute_mean_shap_vector(explainer, reference_data)

    window_size = config.STREAM_WINDOW_SIZE
    n_windows = len(stream_data) // window_size

    distances = []
    raw_flags = []

    active_threshold = config.SHAP_DRIFT_THRESHOLD if threshold is None else threshold

    for i in range(n_windows):
        window = stream_data[i * window_size : (i + 1) * window_size]
        window_shap = compute_mean_shap_vector(explainer, window)
        dist = cosine_distance(reference_shap, window_shap)

        distances.append(dist)
        raw_flags.append(dist > active_threshold)

    drift_flags = np.array(raw_flags, dtype=bool)
    if apply_confirmation:
        drift_flags = _apply_temporal_confirmation(drift_flags)

    if warmup_windows > 0:
        drift_flags[:warmup_windows] = False

    for i, (dist, flag) in enumerate(zip(distances, drift_flags)):
        log.info("SHAP window %d cosine distance %.4f drift %s", i, dist, flag)

    return np.array(distances), np.array(drift_flags), reference_shap
