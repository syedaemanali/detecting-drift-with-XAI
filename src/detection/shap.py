import logging
import numpy as np
import shap
from scipy.spatial.distance import cosine

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
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

    # TreeExplainer returns a list [class0, class1] for classifiers
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.mean(np.abs(shap_values), axis=0)


def cosine_distance(vector_a, vector_b):
    # cosine distance of 0 means identical, 1 means completely different
    return cosine(vector_a, vector_b)


def detect_shap_drift(model, reference_data, stream_data):
    """
    Slide a window over stream_data and compare its mean SHAP vector
    against the reference SHAP vector using cosine distance.
    Returns per-window distances and a boolean drift flag array.
    """
    explainer = build_explainer(model, reference_data)
    reference_shap = compute_mean_shap_vector(explainer, reference_data)

    window_size = config.STREAM_WINDOW_SIZE
    n_windows = len(stream_data) // window_size

    distances = []
    drift_flags = []

    for i in range(n_windows):
        window = stream_data[i * window_size : (i + 1) * window_size]
        window_shap = compute_mean_shap_vector(explainer, window)
        dist = cosine_distance(reference_shap, window_shap)

        distances.append(dist)
        drift_flags.append(dist > config.SHAP_DRIFT_THRESHOLD)

        log.info("SHAP window %d cosine distance %.4f drift %s", i, dist, drift_flags[-1])

    return np.array(distances), np.array(drift_flags), reference_shap